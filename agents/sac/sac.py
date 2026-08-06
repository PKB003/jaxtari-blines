# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
import os

# Limit JAX GPU memory usage to avoid OOM / fragmentation issues.
# Must be set before JAX initializes its allocator.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import random
import subprocess
import time
from functools import partial
from typing import Optional

import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxatari.wrappers import (
    AtariWrapper,
    PixelObsWrapper,
    ObjectCentricWrapper,
    LogWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    ContinuousActionWrapper,
)
from jaxatari import spaces

from rtpt import RTPT
from agents.sac.sac_eval import evaluate


def get_gpu_stats():
    """Return (memory_used_MB, memory_total_MB, utilization_percent) for the first GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        used, total, util = [int(x) for x in line.split(",")]
        return used, total, util
    except Exception as e:
        return -1, -1, -1


def make_env(
    env_id: str,
    mods: Optional[list] = None,
    pixel_based: bool = True,
    native_downscaling: bool = True,
    eval: bool = False,
    continuous_action: bool = True,
    tau: float = 0.5,
):
    if mods is None:
        mods = []
    if not eval and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        import jaxatari
        env = jaxatari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=continuous_action,
        )

        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )

        env = LogWrapper(env)
        if continuous_action:
            env = ContinuousActionWrapper(env, tau=tau)

        return env
    return thunk


# ---------- Networks ----------
class Network(nn.Module):
    """CNN for pixel observations."""
    @nn.compact
    def __call__(self, x):
        if x.ndim == 6:
            # (batch, env, stack, H, W, C)
            b, n_env, stack, h, w, c = x.shape
            # Move the stack axis next to the channel axis before reshaping,
            # so the channel axis becomes (stack * C) of the SAME pixel rather
            # than stacking consecutive spatial pixels (which would corrupt the
            # image for the CNN).
            x = jnp.transpose(x, (0, 1, 3, 4, 2, 5))
            x = x.reshape((b * n_env, h, w, stack * c))

        elif x.ndim == 5:
            # (batch, stack, H, W, C)
            b, stack, h, w, c = x.shape
            # Transpose (batch, H, W, stack, C) then reshape: channel axis =
            # (stack * C) at the same spatial pixel (frame stacking order).
            x = jnp.transpose(x, (0, 2, 3, 1, 4))
            x = x.reshape((b, h, w, stack * c))

        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class MLP_Network(nn.Module):
    """MLP for object-centric observations."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Actor(nn.Module):
    """Gaussian policy for continuous actions."""
    action_dim: int
    log_std_min: float = -5.0   # CleanRL default
    log_std_max: float = 2.0    # CleanRL default

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        # CleanRL-style tanh scaling: map tanh output [-1,1] -> [log_std_min, log_std_max]
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mean, log_std


class SoftQNetwork(nn.Module):
    """Q(s,a) network: takes hidden features + action, concatenates them."""
    action_dim: int

    @nn.compact
    def __call__(self, x, a):
        # Concatenate hidden features with action
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0))(x)


@flax.struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


def single_run(config: dict):
    # Convert config to uppercase
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )

    # Seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, network_key, actor_key, qf1_key, qf2_key = jax.random.split(key, 5)

    # Environment
    env = make_env(
        config["ENV_ID"],
        list(config.get("TRAIN_MODS", [])),
        config["PIXEL_BASED"],
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()
    obs_space = env.observation_space()
    assert isinstance(obs_space, spaces.Box), "SAC requires Box observation space."
    action_space = env.action_space()
    assert isinstance(action_space, spaces.Box), "ContinuousActionWrapper should give Box action space."
    action_dim = action_space.shape[0]
    low = jnp.array(action_space.low)
    high = jnp.array(action_space.high)
    # CleanRL action_scale = (high - low) / 2, used in the tanh log-prob
    # correction (log-det-Jacobian). Confirmed by compare_sac_math.py as the
    # first divergence from CleanRL.
    action_scale = (high - low) / 2.0

    # Vectorized environment wrappers
    @jax.jit
    def vmap_reset(key):
        obs, state = jax.vmap(env.reset)(key)
        return obs, state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs, state, reward, next_done, info

    # Networks
    NetworkClass = Network if config["PIXEL_BASED"] else MLP_Network
    network = NetworkClass()
    actor = Actor(action_dim=action_dim)
    qf1 = SoftQNetwork(action_dim=action_dim)
    qf2 = SoftQNetwork(action_dim=action_dim)

    sample_obs = jnp.zeros(
        (1,) + obs_space.shape,
        dtype=jnp.uint8
    )
    network_params = network.init(network_key, sample_obs)
    hidden = network.apply(network_params, sample_obs)
    actor_params = actor.init(actor_key, hidden)
    # For Q networks, we need a dummy action to init
    dummy_action = jnp.zeros((1, action_dim))
    qf1_params = qf1.init(qf1_key, hidden, dummy_action)
    qf2_params = qf2.init(qf2_key, hidden, dummy_action)

    # Target networks (initialized as copies)
    qf1_target_params = qf1_params
    qf2_target_params = qf2_params

    # Separate optimizers
    # Q-network optimizer
    q_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config["Q_LR"], eps=1e-5),
    )
    # Policy + network encoder optimizer
    policy_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config["POLICY_LR"], eps=1e-5),
    )

    # We use separate TrainStates for Q and policy
    # Q networks share one optimizer
    qf1_state = TrainState.create(apply_fn=None, params=qf1_params, tx=q_tx)
    qf2_state = TrainState.create(apply_fn=None, params=qf2_params, tx=q_tx)
    # Actor + network share another optimizer
    actor_state = TrainState.create(apply_fn=None, params=actor_params, tx=policy_tx)
    network_state = TrainState.create(apply_fn=None, params=network_params, tx=policy_tx)

    # Automatic entropy tuning
    if config["AUTOTUNE"]:
        target_entropy = -action_dim
        log_alpha = jnp.zeros(())
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=config.get("ALPHA_LR", config["POLICY_LR"]), eps=1e-5),
        )
    else:
        alpha = config["ALPHA"]
        alpha_state = None

    # ---------- JIT functions ----------

    @jax.jit
    def sample_action(network_params, actor_params, obs, key):
        """Sample action from the policy (stochastic)."""
        hidden = network.apply(network_params, obs)
        mean, log_std = actor.apply(actor_params, hidden)
        std = jnp.exp(log_std)
        key, subkey = jax.random.split(key)
        z = mean + std * jax.random.normal(subkey, shape=mean.shape)
        action = jnp.tanh(z)
        action_rescaled = low + (action + 1.0) * (high - low) / 2.0
        return action_rescaled, key

    @jax.jit
    def get_det_action(network_params, actor_params, obs):
        """Deterministic action (mean, no noise) for evaluation."""
        hidden = network.apply(network_params, obs)
        mean, _ = actor.apply(actor_params, hidden)
        action = jnp.tanh(mean)
        action_rescaled = low + (action + 1.0) * (high - low) / 2.0
        return action_rescaled

    # ---------- SAC update functions ----------

    def remove_env_dim(x):
        # Flashbax adds an env dimension when add_batches=True:
        # (batch, num_envs, ...). Handle both NUM_ENVS=1 and NUM_ENVS>1.
        if x.ndim >= 2 and x.shape[1] > 1:
            return x.reshape((-1,) + x.shape[2:])
        if x.ndim >= 2 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    @jax.jit
    def update_qf(
        network_state, qf1_state, qf2_state,
        qf1_target_params, qf2_target_params,
        actor_state, alpha, batch, key,
    ):
        """Update Q-networks and shared encoder using MSE loss against target Q values."""
        # next_hidden is stop-gradient (used for target computation only)
        next_hidden = jax.lax.stop_gradient(network.apply(network_state.params, batch.next_obs))

        # Compute target Q (with torch.no_grad() equivalent)
        next_mean, next_log_std = actor.apply(actor_state.params, next_hidden)
        next_std = jnp.exp(next_log_std)
        key, subkey = jax.random.split(key)
        next_z = next_mean + next_std * jax.random.normal(subkey, shape=next_mean.shape)
        next_action = jnp.tanh(next_z)

        # Log-prob of the Gaussian before tanh
        next_log_prob = -0.5 * (((next_z - next_mean) / (next_std + 1e-8)) ** 2
                                 + 2 * jnp.log(next_std + 1e-8) + jnp.log(2 * jnp.pi))
        next_log_prob = next_log_prob.sum(axis=-1)
        # CleanRL tanh log-prob correction incl. action_scale (log-det-Jacobian)
        next_log_prob -= jnp.log(action_scale * (1 - next_action**2) + 1e-6).sum(axis=-1)

        # The Q network takes the raw tanh action (in [-1, 1])
        qf1_next_target = qf1.apply(qf1_target_params, next_hidden, next_action).squeeze(-1)
        qf2_next_target = qf2.apply(qf2_target_params, next_hidden, next_action).squeeze(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target) - alpha * next_log_prob
        next_q_value = batch.reward + config["GAMMA"] * (1 - batch.done) * min_qf_next_target
        next_q_value = jax.lax.stop_gradient(next_q_value)

        # Current Q values
        # The action in the buffer is already rescaled to [low, high], so we need to
        # convert it back to [-1, 1] for the Q network.
        # Guard against degenerate action spaces (high == low) and clip to avoid
        # NaN from float32 rounding pushing the value slightly outside [-1, 1].
        action_range = jnp.where(high > low, high - low, 1.0)
        action_tanh = jnp.clip(
            2.0 * (batch.action - low) / action_range - 1.0,
            -0.999999,
            0.999999,
        )

        # Compute loss and gradients for encoder + both Q-networks in a single pass.
        # Gradient flows through the encoder from both Q-losses (like CleanRL where
        # each Q-network updates its own encoder; here the encoder is shared).
        def qf_loss_fn(params):
            network_params, qf1_params, qf2_params = params
            hidden = network.apply(network_params, batch.obs)
            qf1_a = qf1.apply(qf1_params, hidden, action_tanh).squeeze(-1)
            qf2_a = qf2.apply(qf2_params, hidden, action_tanh).squeeze(-1)
            qf1_loss = ((qf1_a - next_q_value) ** 2).mean()
            qf2_loss = ((qf2_a - next_q_value) ** 2).mean()
            return qf1_loss + qf2_loss, (qf1_loss, qf2_loss, qf1_a.mean(), qf2_a.mean())

        (qf_loss, (qf1_loss, qf2_loss, qf1_values, qf2_values)), \
            (network_grads, qf1_grads, qf2_grads) = jax.value_and_grad(qf_loss_fn, has_aux=True)(
                (network_state.params, qf1_state.params, qf2_state.params)
            )

        new_network_state = network_state.apply_gradients(grads=network_grads)
        new_qf1_state = qf1_state.apply_gradients(grads=qf1_grads)
        new_qf2_state = qf2_state.apply_gradients(grads=qf2_grads)

        return (new_qf1_state, new_qf2_state, new_network_state, qf_loss, qf1_loss, qf2_loss,
                qf1_values, qf2_values, next_q_value.mean(), key)

    @jax.jit
    def update_actor_and_alpha(
        network_params, actor_state, qf1_state, qf2_state,
        alpha_state, alpha, batch, key,
    ):
        """Update actor and alpha. The encoder is FROZEN (stop-gradient) here,
        so it is only trained by the Q-loss in update_qf (standard practice for
        shared-encoder SAC variants such as DrQ / SAC-AE)."""
        # Freeze encoder representation for the policy update to avoid
        # representation drift from two competing gradient sources.
        hidden = jax.lax.stop_gradient(network.apply(network_params, batch.obs))
        # Split the key so each actor update inside the delayed-policy loop
        # uses fresh noise (previously the same key was reused across iterations).
        key, noise_key = jax.random.split(key)

        def actor_loss_fn(actor_params):
            mean, log_std = actor.apply(actor_params, hidden)
            std = jnp.exp(log_std)
            noise = jax.random.normal(noise_key, shape=mean.shape)
            z = mean + std * noise
            action = jnp.tanh(z)
            log_prob = -0.5 * (((z - mean) / (std + 1e-8)) ** 2
                                + 2 * jnp.log(std + 1e-8) + jnp.log(2 * jnp.pi))
            log_prob = log_prob.sum(axis=-1)
            # CleanRL tanh log-prob correction incl. action_scale (log-det-Jacobian)
            log_prob -= jnp.log(action_scale * (1 - action**2) + 1e-6).sum(axis=-1)

            qf1_pi = qf1.apply(qf1_state.params, hidden, action).squeeze(-1)
            qf2_pi = qf2.apply(qf2_state.params, hidden, action).squeeze(-1)
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
            actor_loss = (alpha * log_prob - min_qf_pi).mean()
            return actor_loss, log_prob

        (actor_loss, log_prob), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            actor_state.params
        )
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        # ---- TEMPORARY diagnostics (Stage 5 actor entropy failure isolation) ----
        # Recompute sampled action / log_std / log_prob with current params
        # for monitoring only (no gradient effect).
        mean_diag, log_std_diag = actor.apply(new_actor_state.params, hidden)
        std_diag = jnp.exp(log_std_diag)
        key_diag, noise_key_diag = jax.random.split(key)
        z_diag = mean_diag + std_diag * jax.random.normal(noise_key_diag, shape=mean_diag.shape)
        action_diag = jnp.tanh(z_diag)

        # Gaussian log prob (pre-tanh)
        gauss_log_prob_diag = -0.5 * (((z_diag - mean_diag) / (std_diag + 1e-8)) ** 2
                                      + 2 * jnp.log(std_diag + 1e-8) + jnp.log(2 * jnp.pi))
        gauss_log_prob_diag = gauss_log_prob_diag.sum(axis=-1)
        # Tanh correction (log-det-Jacobian) with action_scale
        tanh_corr_diag = -jnp.log(action_scale * (1 - action_diag**2) + 1e-6).sum(axis=-1)
        log_prob_diag = gauss_log_prob_diag + tanh_corr_diag
        action_rescaled = low + (action_diag + 1.0) * (high - low) / 2.0

        # Actor loss decomposition (entropy_term = alpha*log_prob, q_term = -min_q)
        qf1_pi_diag = qf1.apply(qf1_state.params, hidden, action_diag).squeeze(-1)
        qf2_pi_diag = qf2.apply(qf2_state.params, hidden, action_diag).squeeze(-1)
        min_qf_pi_diag = jnp.minimum(qf1_pi_diag, qf2_pi_diag)
        entropy_term_diag = alpha * log_prob_diag
        q_term_diag = -min_qf_pi_diag

        debug = {
            # Actor distribution
            "actor_mean_mean": mean_diag.mean(),
            "actor_mean_min": mean_diag.min(),
            "actor_mean_max": mean_diag.max(),
            "log_std_mean": log_std_diag.mean(),
            "log_std_min": log_std_diag.min(),
            "log_std_max": log_std_diag.max(),
            "std_mean": std_diag.mean(),
            # Pre-tanh latent z
            "z_mean": z_diag.mean(),
            "z_std": z_diag.std(),
            "z_min": z_diag.min(),
            "z_max": z_diag.max(),
            "fraction_abs_z_gt_5": (jnp.abs(z_diag) > 5.0).mean(),
            "fraction_abs_z_gt_10": (jnp.abs(z_diag) > 10.0).mean(),
            # Tanh saturation
            "mean_abs_tanh": jnp.abs(action_diag).mean(),
            "fraction_abs_tanh_gt_0p99": (jnp.abs(action_diag) > 0.99).mean(),
            "fraction_abs_tanh_gt_0p999": (jnp.abs(action_diag) > 0.999).mean(),
            # Log prob decomposition
            "gaussian_log_prob_mean": gauss_log_prob_diag.mean(),
            "tanh_correction_mean": tanh_corr_diag.mean(),
            "log_prob_mean": log_prob_diag.mean(),
            # Actor loss decomposition
            "entropy_term_mean": entropy_term_diag.mean(),
            "q_term_mean": q_term_diag.mean(),
            # Alpha
            "log_alpha": jnp.log(alpha),
            "alpha": alpha,
            # Existing extra diagnostics
            "action_tanh_mean": action_diag.mean(),
            "action_tanh_abs_mean": jnp.abs(action_diag).mean(),
            "action_tanh_max": jnp.abs(action_diag).max(),
            "entropy_gap": (log_prob_diag + target_entropy).mean(),
            "env_action_mean_dim0": action_rescaled[..., 0].mean(),
            "env_action_mean_dim1": action_rescaled[..., 1].mean(),
            "env_action_mean_dim2": action_rescaled[..., 2].mean(),
        }

        # Alpha update (if autotune)
        if config["AUTOTUNE"]:
            def alpha_loss_fn(p):
                alpha_val = jnp.exp(p["log_alpha"])
                a_loss = -(p["log_alpha"] * (log_prob + target_entropy)).mean()
                return a_loss, alpha_val

            (alpha_loss, new_alpha), a_grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(
                alpha_state.params
            )
            new_alpha_state = alpha_state.apply_gradients(grads=a_grads)
        else:
            new_alpha_state = alpha_state
            alpha_loss = jnp.array(0.0)
            new_alpha = alpha

        return (new_actor_state, new_alpha_state, new_alpha,
                actor_loss, log_prob.mean(), alpha_loss, debug, key)

    @jax.jit
    def target_update(qf1_state, qf2_state, qf1_target_params, qf2_target_params, tau):
        """Soft-update target networks (jitted to avoid per-call kernel dispatch)."""
        new_qf1_target_params = optax.incremental_update(
            qf1_state.params, qf1_target_params, tau
        )
        new_qf2_target_params = optax.incremental_update(
            qf2_state.params, qf2_target_params, tau
        )
        return new_qf1_target_params, new_qf2_target_params

    # ---------- Save and eval function ----------
    def save_and_eval(iteration, network_state, actor_state, qf1_state, qf2_state):
        if config.get("SAVE_PATH") is not None:
            model_path = f'{config["SAVE_PATH"]}/{run_name}/{config["EXP_NAME"]}_{iteration}_{time.time()}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            config,
                            [
                                network_state.params,
                                actor_state.params,
                                qf1_state.params,
                                qf2_state.params,
                            ],
                        ]
                    )
                )
            print(f"model saved to {model_path}")
        else:
            model_path = None

        # Evaluate across all mods
        eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
        eval_configs = [([], "default")]
        if len(eval_mods) > 0:
            mods_list = list(eval_mods)
            for mod in mods_list:
                mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
                mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
                eval_configs.append((mods_config, mod_label))

        metrics = {}
        for mods_config, mod_label in eval_configs:
            print(f"Evaluating on {mod_label} ...")
            if model_path is None:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".cleanrl_model") as tmp:
                    tmp_path = tmp.name
                    tmp.write(
                        flax.serialization.to_bytes(
                            [
                                config,
                                [
                                    network_state.params,
                                    actor_state.params,
                                    qf1_state.params,
                                    qf2_state.params,
                                ],
                            ]
                        )
                    )
                eval_model_path = tmp_path
            else:
                eval_model_path = model_path

            episodic_returns, env_states = evaluate(
                eval_model_path,
                partial(
                    make_env,
                    mods=mods_config,
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=(Network, Actor, SoftQNetwork) if config["PIXEL_BASED"] else (MLP_Network, Actor, SoftQNetwork),
                seed=config["SEED"] + 42,
            )
            mean_return = np.mean(jax.device_get(episodic_returns))
            metrics[mod_label] = mean_return
            wandb.log({f"eval/episodic_return_{mod_label}": mean_return}, step=iteration)

            if config.get("CAPTURE_VIDEO", False):
                import jaxatari
                clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_config).renderer
                frames = jax.vmap(clean_renderer.render)(env_states)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=iteration)
                print(f"Video (eval) logged with {frames.shape[0]} frames.")

            if model_path is None:
                os.remove(eval_model_path)

        return metrics

    # ---------- Training loop ----------
    num_envs = config["NUM_ENVS"]
    total_timesteps = config["TOTAL_TIMESTEPS"]
    learning_starts = config["LEARNING_STARTS"]

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))

    # Keep the replay buffer on CPU by default to avoid exhausting GPU VRAM.
    buffer_on_cpu = config.get("BUFFER_ON_CPU", True)
    cpu_device = jax.devices("cpu")[0] if buffer_on_cpu else None

    # Replay buffer with flashbax
    dummy_transition = Transition(
        obs=obs,
        action=jnp.zeros((config["NUM_ENVS"], action_dim), dtype=jnp.float32),
        reward=jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32),
        next_obs=obs,
        done=jnp.zeros((config["NUM_ENVS"],), dtype=jnp.bool_),
    )
    # Move the dummy transition to CPU BEFORE buffer.init so flashbax allocates
    # the large buffer directly on CPU. This avoids materializing the buffer on
    # GPU and then copying it to CPU (which would temporarily use ~1.4 GB VRAM).
    if buffer_on_cpu:
        dummy_transition = jax.tree.map(
            lambda x: jax.device_put(x, cpu_device), dummy_transition
        )

    buffer = fbx.make_item_buffer(
        max_length=config["BUFFER_SIZE"],
        min_length=config["BATCH_SIZE"],
        sample_batch_size=config["BATCH_SIZE"],
        add_batches=True,
        add_sequences=False,
    )
    buffer_state = buffer.init(dummy_transition)

    if buffer_on_cpu:
        @jax.jit(device=cpu_device)
        def buffer_add(state, transition):
            return buffer.add(state, transition)

        @jax.jit(device=cpu_device)
        def buffer_sample(state, key):
            return buffer.sample(state, key)

        @jax.jit(device=cpu_device)
        def buffer_can_sample(state):
            return buffer.can_sample(state)
    else:
        buffer_add = buffer.add
        buffer_sample = buffer.sample
        buffer_can_sample = buffer.can_sample

    # Fill buffer with random actions (initial exploration).
    # Batch multiple transitions on GPU, then move the whole batch to CPU and
    # call buffer.add once per batch. This avoids a synchronous GPU->CPU device
    # transfer + full-buffer scatter-add for every single transition (which was
    # the dominant cost: ~102ms/step vs ~2.7ms for the env step).
    print("Filling replay buffer with random actions...")
    steps_to_fill = int(learning_starts // num_envs) + 1
    fill_batch_size = config.get("FILL_BATCH_SIZE", 256)
    fill_t_step = 0.0
    fill_t_transition = 0.0
    fill_t_add = 0.0
    n_flushes = 0
    fill_start = time.perf_counter()

    pending = []  # transitions kept on GPU until we flush a batch

    def flush_pending(pending, buffer_state):
        nonlocal n_flushes, fill_t_add
        if not pending:
            return buffer_state
        # Stack (N, num_envs, ...) — matches flashbax item_buffer.add(add_batches=True).
        batch = Transition(
            obs=jnp.stack([t.obs for t in pending], axis=0),
            action=jnp.stack([t.action for t in pending], axis=0),
            reward=jnp.stack([t.reward for t in pending], axis=0),
            next_obs=jnp.stack([t.next_obs for t in pending], axis=0),
            done=jnp.stack([t.done for t in pending], axis=0),
        )
        # Move the whole batch to CPU in a single transfer.
        batch = jax.tree.map(lambda x: jax.device_put(x, cpu_device), batch)
        t2 = time.perf_counter()
        new_state = buffer_add(buffer_state, batch)
        t3 = time.perf_counter()
        fill_t_add += t3 - t2
        n_flushes += 1
        return new_state

    for _ in range(steps_to_fill):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (num_envs, action_dim), minval=low, maxval=high)
        t0 = time.perf_counter()
        next_obs, env_state, reward, next_done, info = vmap_step(env_state, action)
        t1 = time.perf_counter()
        transition = Transition(
            obs,
            action,
            reward.astype(jnp.float32),
            next_obs,
            next_done.astype(jnp.bool_)
        )
        t2 = time.perf_counter()
        fill_t_step += t1 - t0
        fill_t_transition += t2 - t1
        pending.append(transition)
        if len(pending) >= fill_batch_size:
            buffer_state = flush_pending(pending, buffer_state)
            pending = []
        obs = next_obs

    # Flush any remaining transitions.
    buffer_state = flush_pending(pending, buffer_state)

    fill_elapsed = time.perf_counter() - fill_start
    print(f"[FILL] steps={steps_to_fill} elapsed={fill_elapsed:.2f}s "
          f"step={fill_t_step:.2f}s transition={fill_t_transition:.2f}s "
          f"add={fill_t_add:.2f}s flushes={n_flushes} "
          f"avg_add={fill_t_add / max(n_flushes, 1) * 1000:.3f}ms")
    used_mb, total_mb, util = get_gpu_stats()
    print(f"[FILL] GPU mem={used_mb}/{total_mb}MB util={util}%")

    print("Starting training...")
    global_step = 0
    start_time = time.time()

    # Batched training buffer adds: accumulate transitions on GPU, then flush the
    # whole batch to CPU with a single buffer.add. This avoids a synchronous
    # GPU->CPU transfer + full-buffer scatter-add per transition (which dominated
    # training time before batching). Batching mirrors the fill-loop fix.
    train_add_batch_size = config.get("TRAIN_ADD_BATCH_SIZE", 8)
    train_pending = []

    def flush_train_pending(pending, buffer_state):
        if not pending:
            return buffer_state
        # Stack (N, num_envs, ...) — matches flashbax item_buffer.add(add_batches=True).
        batch = Transition(
            obs=jnp.stack([t.obs for t in pending], axis=0),
            action=jnp.stack([t.action for t in pending], axis=0),
            reward=jnp.stack([t.reward for t in pending], axis=0),
            next_obs=jnp.stack([t.next_obs for t in pending], axis=0),
            done=jnp.stack([t.done for t in pending], axis=0),
        )
        # Move the whole batch to CPU in a single transfer.
        batch = jax.tree.map(lambda x: jax.device_put(x, cpu_device), batch)
        return buffer_add(buffer_state, batch)

    # RTPT
    total_iterations = total_timesteps // (num_envs * config["SCAN_STEPS"]) + 1
    rtpt = RTPT(name_initials=config.get("NAME_INITIALS", "SA"), experiment_name=run_name, max_iterations=total_iterations)
    rtpt.start()

    steps_per_iteration = config["SCAN_STEPS"]
    num_iterations = total_timesteps // (num_envs * steps_per_iteration) + 1

    for iteration in range(num_iterations):
        rtpt.step()

        # Do SCAN_STEPS environment steps
        for local_step in range(steps_per_iteration):
            # Sample action
            key, subkey = jax.random.split(key)
            action, key = sample_action(network_state.params, actor_state.params, obs, subkey)

            # Step environment
            next_obs, env_state, reward, next_done, info = vmap_step(env_state, action)

            # Add to buffer (batched: append on GPU, flush every TRAIN_ADD_BATCH_SIZE)
            transition = Transition(
                obs.astype(jnp.uint8),
                action.astype(jnp.float32),
                reward.astype(jnp.float32),
                next_obs.astype(jnp.uint8),
                next_done.astype(jnp.bool_)
            )
            train_pending.append(transition)
            if len(train_pending) >= train_add_batch_size:
                buffer_state = flush_train_pending(train_pending, buffer_state)
                train_pending = []

            obs = next_obs
            global_step += num_envs

            # Update if buffer has enough samples
            if buffer_can_sample(buffer_state):
                key, sample_key = jax.random.split(key)
                # Sample on CPU (replay buffer lives on CPU to save VRAM).
                batch = buffer_sample(buffer_state, sample_key).experience
                # Remove env dims on CPU first (reduces transfer size).
                batch = Transition(
                    remove_env_dim(batch.obs),
                    remove_env_dim(batch.action),
                    remove_env_dim(batch.reward),
                    remove_env_dim(batch.next_obs),
                    remove_env_dim(batch.done),
                )
                # Move the sampled batch to GPU explicitly for learning.
                # Obs stays uint8 during transfer to minimize data movement;
                # conversion to float32 happens on GPU below.
                if buffer_on_cpu:
                    batch = jax.device_put(batch, jax.devices("gpu")[0])
                batch = Transition(
                    batch.obs.astype(jnp.float32),
                    batch.action.astype(jnp.float32),
                    batch.reward.astype(jnp.float32),
                    batch.next_obs.astype(jnp.float32),
                    batch.done.astype(jnp.float32),
                )
                # Get current alpha
                if config["AUTOTUNE"]:
                    current_alpha = jnp.exp(alpha_state.params["log_alpha"])
                else:
                    current_alpha = config["ALPHA"]

                # Q update (every step after learning starts, like CleanRL)
                # Encoder is also updated here (gradient from Q-loss).
                (qf1_state, qf2_state, network_state, qf_loss, qf1_loss, qf2_loss,
                 qf1_values, qf2_values, next_q_values, key) = update_qf(
                    network_state, qf1_state, qf2_state,
                    qf1_target_params, qf2_target_params,
                    actor_state, current_alpha, batch, key,
                )

                # Actor + Alpha update (delayed: every POLICY_FREQUENCY steps)
                # The loop matches CleanRL exactly: update the actor POLICY_FREQUENCY
                # times on the same batch to compensate for the delayed update.
                # The encoder is NOT updated here (see update_actor_and_alpha).
                if global_step % config["POLICY_FREQUENCY"] == 0:
                    for _ in range(config["POLICY_FREQUENCY"]):
                        (
                            actor_state,
                            alpha_state,
                            current_alpha,
                            actor_loss,
                            log_prob_mean,
                            alpha_loss,
                            debug,
                            key,
                        ) = update_actor_and_alpha(
                            network_state.params,
                            actor_state,
                            qf1_state,
                            qf2_state,
                            alpha_state,
                            current_alpha,
                            batch,
                            key,
                        )
                else:
                    actor_loss = jnp.array(0.0)
                    log_prob_mean = jnp.array(0.0)
                    alpha_loss = jnp.array(0.0)
                    debug = {
                        "action_tanh_mean": jnp.array(0.0),
                        "action_tanh_abs_mean": jnp.array(0.0),
                        "action_tanh_max": jnp.array(0.0),
                        "log_std_mean": jnp.array(0.0),
                        "log_std_min": jnp.array(0.0),
                        "log_std_max": jnp.array(0.0),
                        "log_prob_mean": jnp.array(0.0),
                        "entropy_gap": jnp.array(0.0),
                        "alpha": current_alpha,
                        "env_action_mean_dim0": jnp.array(0.0),
                        "env_action_mean_dim1": jnp.array(0.0),
                        "env_action_mean_dim2": jnp.array(0.0),
                        "actor_mean_mean": jnp.array(0.0),
                        "actor_mean_min": jnp.array(0.0),
                        "actor_mean_max": jnp.array(0.0),
                        "std_mean": jnp.array(0.0),
                        "z_mean": jnp.array(0.0),
                        "z_std": jnp.array(0.0),
                        "z_min": jnp.array(0.0),
                        "z_max": jnp.array(0.0),
                        "fraction_abs_z_gt_5": jnp.array(0.0),
                        "fraction_abs_z_gt_10": jnp.array(0.0),
                        "mean_abs_tanh": jnp.array(0.0),
                        "fraction_abs_tanh_gt_0p99": jnp.array(0.0),
                        "fraction_abs_tanh_gt_0p999": jnp.array(0.0),
                        "gaussian_log_prob_mean": jnp.array(0.0),
                        "tanh_correction_mean": jnp.array(0.0),
                        "entropy_term_mean": jnp.array(0.0),
                        "q_term_mean": jnp.array(0.0),
                        "log_alpha": jnp.array(0.0),
                    }

                # Target network update (every TARGET_NETWORK_FREQUENCY steps)
                if global_step % config["TARGET_NETWORK_FREQUENCY"] == 0:
                    qf1_target_params, qf2_target_params = target_update(
                        qf1_state, qf2_state, qf1_target_params, qf2_target_params, config["TAU"]
                    )

        # Flush any transitions still pending in this iteration so they are
        # committed to the buffer (keeps the buffer fresh across iterations).
        buffer_state = flush_train_pending(train_pending, buffer_state)
        train_pending = []

        # Logging
        if iteration % 1 == 0:
            avg_return = info["returned_episode_returns"].mean() if "returned_episode_returns" in info else 0.0
            avg_length = info["returned_episode_lengths"].mean() if "returned_episode_lengths" in info else 0.0

            # Gather all device tensors in a single host transfer to avoid
            # repeated device->host copies (each float() triggers one).
            metrics = jax.device_get({
                "charts/avg_episodic_return": avg_return,
                "charts/avg_episodic_length": avg_length,
                "losses/qf1_loss": qf1_loss,
                "losses/qf2_loss": qf2_loss,
                "losses/qf_loss": qf_loss / 2.0,
                "losses/qf1_values": qf1_values,
                "losses/qf2_values": qf2_values,
                "losses/actor_loss": actor_loss,
                "losses/log_prob_mean": log_prob_mean,
                "losses/alpha_loss": alpha_loss,
                "losses/alpha": current_alpha,
                "reward": jnp.mean(batch.reward),
                "charts/SPS": global_step / (time.time() - start_time + 1e-8),
                "charts/global_step": global_step,
                "charts/iteration": iteration,
                # ---- TEMPORARY diagnostics (A/B validation) ----
                # ---- TEMPORARY diagnostics (Stage 5 actor entropy) ----
                "debug/action_tanh_mean": debug["action_tanh_mean"],
                "debug/action_tanh_abs_mean": debug["action_tanh_abs_mean"],
                "debug/action_tanh_max": debug["action_tanh_max"],
                "debug/actor_mean_mean": debug["actor_mean_mean"],
                "debug/actor_mean_min": debug["actor_mean_min"],
                "debug/actor_mean_max": debug["actor_mean_max"],
                "debug/log_std_mean": debug["log_std_mean"],
                "debug/log_std_min": debug["log_std_min"],
                "debug/log_std_max": debug["log_std_max"],
                "debug/std_mean": debug["std_mean"],
                "debug/z_mean": debug["z_mean"],
                "debug/z_std": debug["z_std"],
                "debug/z_min": debug["z_min"],
                "debug/z_max": debug["z_max"],
                "debug/fraction_abs_z_gt_5": debug["fraction_abs_z_gt_5"],
                "debug/fraction_abs_z_gt_10": debug["fraction_abs_z_gt_10"],
                "debug/mean_abs_tanh": debug["mean_abs_tanh"],
                "debug/fraction_abs_tanh_gt_0p99": debug["fraction_abs_tanh_gt_0p99"],
                "debug/fraction_abs_tanh_gt_0p999": debug["fraction_abs_tanh_gt_0p999"],
                "debug/gaussian_log_prob_mean": debug["gaussian_log_prob_mean"],
                "debug/tanh_correction_mean": debug["tanh_correction_mean"],
                "debug/log_prob_mean": debug["log_prob_mean"],
                "debug/entropy_term_mean": debug["entropy_term_mean"],
                "debug/q_term_mean": debug["q_term_mean"],
                "debug/log_alpha": debug["log_alpha"],
                "debug/entropy_gap": debug["entropy_gap"],
                "debug/alpha": debug["alpha"],
                "debug/env_action_mean_dim0": debug["env_action_mean_dim0"],
                "debug/env_action_mean_dim1": debug["env_action_mean_dim1"],
                "debug/env_action_mean_dim2": debug["env_action_mean_dim2"],
            })
            wandb.log(metrics, step=global_step)

        # Evaluation
        if config.get("EVAL_DURING_TRAIN", False) and iteration > 0 and iteration % config.get("EVAL_EVERY", 50) == 0:
            save_and_eval(iteration, network_state, actor_state, qf1_state, qf2_state)

    # Final eval
    print("Evaluating final model ...")
    metrics = save_and_eval(iteration + 1, network_state, actor_state, qf1_state, qf2_state)
    wandb.finish()
    print("Training finished.")
    train_elapsed = time.time() - start_time
    print(f"[TRAIN] elapsed={train_elapsed:.2f}s SPS={global_step / (train_elapsed + 1e-8):.1f}")
    used_mb, total_mb, util = get_gpu_stats()
    print(f"[TRAIN] GPU mem={used_mb}/{total_mb}MB util={util}%")

    return metrics