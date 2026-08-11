# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
import os

# Limit JAX GPU memory usage to avoid OOM / fragmentation issues.
# Must be set before JAX initializes its allocator.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import random
import subprocess
import time
from collections import deque
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
class CNNEncoder(nn.Module):
    """CNN Encoder for pixel observations.

    Accepts the jaxatari observation layout:
      (B, stack, H, W, C)  e.g. (1, 4, 84, 84, 1)
    and reshapes to (B, H, W, stack*C) before the conv layers.
    Also tolerates a plain 4D (B, C, H, W) input.
    """
    @nn.compact
    def __call__(self, x):
        if x.ndim == 5:
            # (B, stack, H, W, C) -> (B, H, W, stack*C)
            b, stack, h, w, c = x.shape
            x = jnp.transpose(x, (0, 2, 3, 1, 4))
            x = x.reshape((b, h, w, stack * c))
        elif x.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            x = jnp.transpose(x, (0, 2, 3, 1))
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


class MLPEncoder(nn.Module):
    """MLP for object-centric observations.

    The input dimension is inferred from the observation at call time
    (Dense infers its input features), so no hardcoded obs dim is used.
    Output dimension is 512 to match the CNN encoder.
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
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
    """Q(s,a) network (Critic): takes hidden features + action, concatenates them."""
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
    key, actor_encoder_key, critic1_encoder_key, critic2_encoder_key, actor_key, qf1_key, qf2_key = jax.random.split(key, 7)

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
    action_scale = (high - low) / 2.0
    action_bias = (high + low) / 2.0
    print(f"[ENV] action_dim={action_dim}")
    print(f"[ENV] action_low={action_space.low}")
    print(f"[ENV] action_high={action_space.high}")
    print(f"[ENV] action_scale={action_scale.tolist()}")
    print(f"[ENV] action_bias={action_bias.tolist()}")

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
    encoder_cls = CNNEncoder if config["PIXEL_BASED"] else MLPEncoder

    critic1_encoder = encoder_cls()
    critic2_encoder = encoder_cls()
    actor_encoder = encoder_cls()

    actor = Actor(action_dim=action_dim)

    qf1 = SoftQNetwork()
    qf2 = SoftQNetwork()
    qf1_target = SoftQNetwork()
    qf2_target = SoftQNetwork()

    # Dummy inputs
    sample_obs = jnp.zeros(
        (1,) + obs_space.shape,
        dtype=jnp.uint8 if config["PIXEL_BASED"] else jnp.float32,
    )
    dummy_action = jnp.zeros(
        (1, action_dim),
        dtype=jnp.float32,
    )

    # Initialize Actor Encoder + Actor
    actor_encoder_params = actor_encoder.init(
        actor_encoder_key,
        sample_obs,
    )

    actor_hidden = actor_encoder.apply(
        actor_encoder_params,
        sample_obs,
    )

    actor_params = actor.init(actor_key, actor_hidden)

    # Initialize Critic 1 Encoder + Q1
    critic1_encoder_params = critic1_encoder.init(
        critic1_encoder_key,
        sample_obs,
    )

    critic1_hidden = critic1_encoder.apply(
        critic1_encoder_params,
        sample_obs,
    )

    qf1_params = qf1.init(
        qf1_key,
        critic1_hidden,
        dummy_action,
    )

    # Initialize Critic 2 Encoder + Q2
    critic2_encoder_params = critic2_encoder.init(
        critic2_encoder_key,
        sample_obs,
    )

    critic2_hidden = critic2_encoder.apply(
        critic2_encoder_params,
        sample_obs,
    )

    qf2_params = qf2.init(
        qf2_key,
        critic2_hidden,
        dummy_action,
    )

    print(f"[NET] actor_encoder hidden shape: {tuple(actor_hidden.shape)} (feature dim should end in 512)")
    print(f"[NET] critic1_encoder hidden shape: {tuple(critic1_hidden.shape)}")
    print(f"[NET] critic2_encoder hidden shape: {tuple(critic2_hidden.shape)}")
    print(f"[NET] actor mean/log_std output dim: {action_dim}")
    print(f"[NET] Q input dim (hidden + action): {critic1_hidden.shape[-1] + action_dim}")

    # Separate optimizers
    # Q-network optimizer
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(
            learning_rate=config["Q_LR"],
            eps=1e-8,
        ),
    )
    # Actor optimizer
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(
            learning_rate=config["POLICY_LR"],
            eps=1e-8,
        ),
    )

    # Pack encoder + network parameters
    actor_train_params = {
        "encoder": actor_encoder_params,
        "actor": actor_params,
    }

    critic1_train_params = {
        "encoder": critic1_encoder_params,
        "qf": qf1_params,
    }

    critic2_train_params = {
        "encoder": critic2_encoder_params,
        "qf": qf2_params,
    }
    # Separate TrainStates for Q and Actor
    # Q networks share one optimizer
    # Critic 1 + Critic 1 Encoder
    qf1_state = TrainState.create(
        apply_fn=None,
        params=critic1_train_params,
        tx=q_optimizer,
    )
    # Critic 2 + Critic 2 Encoder
    qf2_state = TrainState.create(
        apply_fn=None,
        params=critic2_train_params,
        tx=q_optimizer,
    )
    # Actor use another optimizer
    # Actor + Actor Encoder
    actor_state = TrainState.create(
        apply_fn=None,
        params=actor_train_params,
        tx=actor_optimizer,
    )
    # Target Critic parameters
    qf1_target_params = {
        "encoder": critic1_encoder_params,
        "qf": qf1_params,
    }

    qf2_target_params = {
        "encoder": critic2_encoder_params,
        "qf": qf2_params,
    }
    # Automatic entropy tuning
    if config["AUTOTUNE"]:
        target_entropy = -float(0.5 * action_dim)

        alpha_state = TrainState.create(
            apply_fn=None,
            params={
                "log_alpha": jnp.zeros(
                    (1,),
                    dtype=jnp.float32,
                )
            },
            tx=optax.adam(
                learning_rate=config.get(
                    "ALPHA_LR",
                    config["Q_LR"],
                ),
                eps=1e-8,
            ),
        )
        alpha = jnp.exp(alpha_state.params["log_alpha"])
    else:
        alpha = jnp.asarray(
            config["ALPHA"],
            dtype=jnp.float32,
        )
        alpha_state = None

    # ---------- CALE action transformation helpers ----------
    # Dopamine/CALE-compatible squashed Gaussian policy:
    #   u ~ Normal(mean, std) -> y = tanh(u) -> action = action_bias + action_scale * y
    # The final env action lies in [low, high] (e.g. [0,1] x [-pi,pi] x [0,1]).
    # The log-probability is for the fully transformed env action and includes
    # the tanh Jacobian (stable softplus form) plus the affine scaling Jacobian.

    def gaussian_log_prob(x, mean, log_std):
        """Log density of Normal(mean, exp(log_std)) evaluated at x (per-dim)."""
        std = jnp.exp(log_std)
        return -0.5 * (
            ((x - mean) / (std + 1e-8)) ** 2
            + 2.0 * jnp.log(std + 1e-8)
            + jnp.log(2.0 * jnp.pi)
        )

    def tanh_affine_log_jacobian(u):
        """log J = sum_i log( action_scale_i * (1 - tanh(u_i)^2 ) ).

        Stable identity: log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u)).
        """
        tanh_corr = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        return tanh_corr + jnp.log(action_scale)

    def sample_env_action(mean, log_std, key):
        """Reparameterized sample of the CALE-domain env action.

        Returns:
          action_env: env-domain action in [low, high]
          log_prob:    log pi(a_env | s) with full tanh + affine Jacobian
          pre_tanh:    the latent u (for diagnostics)
        """
        noise = jax.random.normal(key, shape=mean.shape)
        u = mean + jnp.exp(log_std) * noise
        action_tanh = jnp.tanh(u)
        action_env = action_bias + action_scale * action_tanh
        log_prob = gaussian_log_prob(u, mean, log_std).sum(axis=-1)
        log_prob = log_prob - tanh_affine_log_jacobian(u).sum(axis=-1)
        return action_env, log_prob, u

    def deterministic_env_action(mean):
        """Deterministic / evaluation action (tanh of mean + CALE affine)."""
        return action_bias + action_scale * jnp.tanh(mean)


    target_entropy = -0.5 * action_dim

    # ---------- JIT functions ----------

    @jax.jit
    def sample_action(actor_state, obs, key):
        """Sample a stochastic CALE-domain action from the current policy."""
        hidden = actor_encoder.apply(actor_state.params["encoder"], obs)
        mean, log_std = actor.apply(actor_state.params["actor"], hidden)
        key, subkey = jax.random.split(key)
        action_env, _, _ = sample_env_action(mean, log_std, subkey)
        return action_env, key

    @jax.jit
    def get_det_action(actor_state, obs):
        """Deterministic action (mean) for evaluation."""
        hidden = actor_encoder.apply(actor_state.params["encoder"], obs)
        mean, _ = actor.apply(actor_state.params["actor"], hidden)
        return deterministic_env_action(mean)

    # ---------- SAC update functions ----------

    def remove_env_dim(x):
        # Flashbax adds an env dimension when add_batches=True:
        # (batch, num_envs, ...). Handle both NUM_ENVS=1 and NUM_ENVS>1.
        if x.ndim >= 2 and x.shape[1] > 1:
            return x.reshape((-1,) + x.shape[2:])
        if x.ndim >= 2 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    def _g_norm(*pytrees):
        """Global L2 norm of flattened gradient pytrees (for debug logging)."""
        flat, _ = jax.flatten_util.ravel_pytree(pytrees)
        return jnp.sqrt(jnp.sum(flat ** 2))

    def _to_float(x):
        """Safely convert a (possibly 0-d or 1-d) jax/numpy array to a python float."""
        return float(np.asarray(x).reshape(-1)[0])

    def discrete_action_id(action, tau=0.5):
        """Map CALE continuous (r, theta, fire) to a discrete id (0..17).

        Mirrors the ContinuousActionWrapper tau discretisation; used to measure
        action diversity of the current policy from real env-step actions.
        """
        if action.shape[-1] != 3:
            return jnp.zeros(action.shape[:-1], dtype=jnp.int32)
        r = action[..., 0]
        theta = action[..., 1]
        fire = action[..., 2]
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        x_idx = (x > tau).astype(jnp.int32) - (x < -tau).astype(jnp.int32) + 1
        y_idx = (y > tau).astype(jnp.int32) - (y < -tau).astype(jnp.int32) + 1
        fire_idx = (fire > tau).astype(jnp.int32)
        return x_idx * 6 + y_idx * 2 + fire_idx

    @jax.jit
    def update_qf(
        qf1_state, qf2_state,
        qf1_target_params, qf2_target_params,
        actor_state, alpha, batch, key,
    ):
        """Update Critic 1 (encoder + Q1) and Critic 2 (encoder + Q2) independently.

        Gradient isolation:
          - qf1_state.params ({"encoder", "qf"}) receives gradients from Q1 loss only
          - qf2_state.params ({"encoder", "qf"}) receives gradients from Q2 loss only
          - the actor is used only for the target action / log-prob and is NOT
            differentiated (target is stop_gradient; actor params are not argnums).
        """
        # ---- sample next action from the CURRENT actor (no target actor) ----
        next_hidden = actor_encoder.apply(actor_state.params["encoder"], batch.next_obs)
        next_mean, next_log_std = actor.apply(actor_state.params["actor"], next_hidden)
        key, subkey = jax.random.split(key)
        next_action, next_log_prob, _ = sample_env_action(next_mean, next_log_std, subkey)

        # ---- target Q with target critics (each own target encoder + Q head) ----
        # The target critics receive the CALE-domain next action.
        z1_next = critic1_encoder.apply(qf1_target_params["encoder"], batch.next_obs)
        z2_next = critic2_encoder.apply(qf2_target_params["encoder"], batch.next_obs)
        qf1_next_target = qf1.apply(qf1_target_params["qf"], z1_next, next_action).squeeze(-1)
        qf2_next_target = qf2.apply(qf2_target_params["qf"], z2_next, next_action).squeeze(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target) - alpha * next_log_prob
        next_q_value = batch.reward + config["GAMMA"] * (1.0 - batch.done) * min_qf_next_target
        next_q_value = jax.lax.stop_gradient(next_q_value)

        # ---- current Q values (critics receive the CALE-domain buffer action) ----
        def critic_loss_fn(params1, params2):
            z1 = critic1_encoder.apply(params1["encoder"], batch.obs)
            z2 = critic2_encoder.apply(params2["encoder"], batch.obs)
            qf1_a = qf1.apply(params1["qf"], z1, batch.action).squeeze(-1)
            qf2_a = qf2.apply(params2["qf"], z2, batch.action).squeeze(-1)
            qf1_loss = ((qf1_a - next_q_value) ** 2).mean()
            qf2_loss = ((qf2_a - next_q_value) ** 2).mean()
            return qf1_loss + qf2_loss, (qf1_loss, qf2_loss, qf1_a.mean(), qf2_a.mean())

        (qf_loss, (qf1_loss, qf2_loss, qf1_values, qf2_values)), \
            (qf1_grads, qf2_grads) = jax.value_and_grad(
                critic_loss_fn, argnums=(0, 1), has_aux=True
            )(qf1_state.params, qf2_state.params)

        new_qf1_state = qf1_state.apply_gradients(grads=qf1_grads)
        new_qf2_state = qf2_state.apply_gradients(grads=qf2_grads)

        # ---- DEBUG: gradient norms (per encoder / per Q head) ----
        qf1_grad_norm = _g_norm(qf1_grads)
        qf2_grad_norm = _g_norm(qf2_grads)
        qf1_enc_grad_norm = _g_norm(qf1_grads["encoder"])
        qf1_q_grad_norm = _g_norm(qf1_grads["qf"])
        qf2_enc_grad_norm = _g_norm(qf2_grads["encoder"])
        qf2_q_grad_norm = _g_norm(qf2_grads["qf"])

        return (new_qf1_state, new_qf2_state, qf_loss, qf1_loss, qf2_loss,
                qf1_values, qf2_values, next_q_value.mean(),
                qf1_grad_norm, qf2_grad_norm,
                qf1_enc_grad_norm, qf1_q_grad_norm,
                qf2_enc_grad_norm, qf2_q_grad_norm,
                key)

    @jax.jit
    def update_actor_and_alpha(
            actor_state,
            qf1_state,
            qf2_state,
            alpha_state,
            alpha,
            batch,
            key,
    ):
        # ============================================================
        # 1. ACTOR UPDATE
        # ============================================================

        key, actor_key = jax.random.split(key)

        def actor_loss_fn(actor_params):
            z_actor = actor_encoder.apply(
                actor_params["encoder"],
                batch.obs,
            )

            mean, log_std = actor.apply(
                actor_params["actor"],
                z_actor,
            )

            action_env, log_prob, _ = sample_env_action(
                mean,
                log_std,
                actor_key,
            )

            # Critic encoders/features are treated as constants
            z1 = jax.lax.stop_gradient(
                critic1_encoder.apply(
                    qf1_state.params["encoder"],
                    batch.obs,
                )
            )

            z2 = jax.lax.stop_gradient(
                critic2_encoder.apply(
                    qf2_state.params["encoder"],
                    batch.obs,
                )
            )

            qf1_pi = qf1.apply(
                qf1_state.params["qf"],
                z1,
                action_env,
            ).squeeze(-1)

            qf2_pi = qf2.apply(
                qf2_state.params["qf"],
                z2,
                action_env,
            ).squeeze(-1)

            min_qf_pi = jnp.minimum(
                qf1_pi,
                qf2_pi,
            )

            # Alpha is constant from actor's perspective
            alpha_for_actor = jax.lax.stop_gradient(alpha)

            actor_loss = (
                    alpha_for_actor * log_prob
                    - min_qf_pi
            ).mean()

            return actor_loss

        actor_loss, actor_grads = jax.value_and_grad(
            actor_loss_fn
        )(actor_state.params)

        # IMPORTANT:
        # actor parameters are updated BEFORE alpha update
        new_actor_state = actor_state.apply_gradients(
            grads=actor_grads
        )

        # ============================================================
        # 2. ALPHA UPDATE
        # ============================================================

        if config["AUTOTUNE"]:

            # New random key, because CleanRL calls actor.get_action()
            # again after the actor optimizer step.
            key, alpha_key = jax.random.split(key)

            def get_log_prob(actor_params):
                z_actor = actor_encoder.apply(
                    actor_params["encoder"],
                    batch.obs,
                )

                mean, log_std = actor.apply(
                    actor_params["actor"],
                    z_actor,
                )

                _, log_prob, _ = sample_env_action(
                    mean,
                    log_std,
                    alpha_key,
                )

                return log_prob

            # IMPORTANT:
            # use UPDATED actor parameters
            log_prob_alpha = get_log_prob(
                new_actor_state.params
            )

            # Equivalent to torch.no_grad()
            log_prob_alpha = jax.lax.stop_gradient(
                log_prob_alpha
            )

            def alpha_loss_fn(alpha_params):
                log_alpha = alpha_params["log_alpha"]
                alpha_value = jnp.exp(log_alpha)

                alpha_loss = -(
                        alpha_value *
                        (log_prob_alpha + target_entropy)
                ).mean()

                return alpha_loss

            alpha_loss, alpha_grads = jax.value_and_grad(
                alpha_loss_fn
            )(alpha_state.params)

            # Update log_alpha
            new_alpha_state = alpha_state.apply_gradients(
                grads=alpha_grads
            )

            # IMPORTANT:
            # recompute alpha AFTER optimizer update
            new_alpha = jnp.exp(
                new_alpha_state.params["log_alpha"]
            )

            log_prob_mean = log_prob_alpha.mean()

        else:

            new_alpha_state = alpha_state
            new_alpha = alpha
            alpha_loss = jnp.array(0.0)
            log_prob_mean = jnp.array(0.0)

        # ---- DEBUG: actor / alpha gradient norms ----
        actor_grad_norm = _g_norm(actor_grads)
        actor_enc_grad_norm = _g_norm(actor_grads["encoder"])
        actor_mlp_grad_norm = _g_norm(actor_grads["actor"])

        if config["AUTOTUNE"]:
            alpha_grad_norm = _g_norm(alpha_grads)
        else:
            alpha_grad_norm = jnp.array(0.0)

        return (
            new_actor_state,
            new_alpha_state,
            new_alpha,
            actor_loss,
            log_prob_mean,
            alpha_loss,
            actor_grad_norm,
            actor_enc_grad_norm,
            actor_mlp_grad_norm,
            alpha_grad_norm,
            key,
        )

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
    def save_and_eval(iteration, actor_state, qf1_state, qf2_state):
        if config.get("SAVE_PATH") is not None:
            model_path = f'{config["SAVE_PATH"]}/{run_name}/{config["EXP_NAME"]}_{iteration}_{time.time()}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            config,
                            [
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
                Model=(CNNEncoder, Actor, SoftQNetwork) if config["PIXEL_BASED"] else (MLPEncoder, Actor, SoftQNetwork),
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
        obs_dtype = jnp.uint8 if config["PIXEL_BASED"] else jnp.float32
        transition = Transition(
            obs.astype(obs_dtype),
            action,
            reward.astype(jnp.float32),
            next_obs.astype(obs_dtype),
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

    # DEBUG: track recent env-step actions to measure policy diversity.
    recent_actions = deque(maxlen=200)

    for iteration in range(num_iterations):
        rtpt.step()

        # Do SCAN_STEPS environment steps
        for local_step in range(steps_per_iteration):
            # Sample action
            key, subkey = jax.random.split(key)
            action, key = sample_action(actor_state, obs, subkey)

            # DEBUG: record env-step actions (after action.squeeze, shape (3,))
            recent_actions.append(np.asarray(action.squeeze()))

            # Step environment
            next_obs, env_state, reward, next_done, info = vmap_step(env_state, action)

            # Add to buffer (batched: append on GPU, flush every TRAIN_ADD_BATCH_SIZE).
            # Pixel obs are stored as uint8; vector/object-centric obs as float32 —
            # must match dummy_transition's dtype or flashbax raises a dtype mismatch.
            obs_dtype = jnp.uint8 if config["PIXEL_BASED"] else jnp.float32
            transition = Transition(
                obs.astype(obs_dtype),
                action.astype(jnp.float32),
                reward.astype(jnp.float32),
                next_obs.astype(obs_dtype),
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
                # Pixel obs stay uint8 during transfer to minimize data movement;
                # conversion to float32 happens on GPU below. Object-centric obs
                # are already float32 in the buffer.
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
                    # alpha is stored as shape (1,); squeeze to scalar so float() works
                    current_alpha = jnp.exp(alpha_state.params["log_alpha"]).squeeze()
                else:
                    current_alpha = config["ALPHA"]

                # Q update (every step after learning starts, like CleanRL)
                (qf1_state, qf2_state, qf_loss, qf1_loss, qf2_loss,
                 qf1_values, qf2_values, next_q_values,
                 qf1_grad_norm, qf2_grad_norm,
                 qf1_enc_grad_norm, qf1_q_grad_norm,
                 qf2_enc_grad_norm, qf2_q_grad_norm,
                 key) = update_qf(
                    qf1_state, qf2_state,
                    qf1_target_params, qf2_target_params,
                    actor_state, current_alpha, batch, key,
                )

                # Actor + Alpha update (delayed: every POLICY_FREQUENCY steps)
                if global_step % config["POLICY_FREQUENCY"] == 0:
                    for _ in range(config["POLICY_FREQUENCY"]):
                        (
                            actor_state,
                            alpha_state,
                            current_alpha,
                            actor_loss,
                            log_prob_mean,
                            alpha_loss,
                            actor_grad_norm,
                            actor_enc_grad_norm,
                            actor_mlp_grad_norm,
                            alpha_grad_norm,
                            key,
                        ) = update_actor_and_alpha(
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
                    actor_grad_norm = jnp.array(0.0)
                    actor_enc_grad_norm = jnp.array(0.0)
                    actor_mlp_grad_norm = jnp.array(0.0)
                    alpha_grad_norm = jnp.array(0.0)

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

            # DEBUG: action diversity over the last 200 env-step actions.
            recent_as = np.stack([np.asarray(a) for a in recent_actions]) if recent_actions else np.zeros((1, action_dim))
            recent_ids = np.asarray(jax.device_get(discrete_action_id(jnp.array(recent_as, dtype=jnp.float32))))
            unique_discrete = int(np.unique(recent_ids).size)
            frac_cont_unique = float(len(np.unique(np.round(recent_as, 4), axis=0))) / max(len(recent_as), 1)
            action_abs_mean = float(np.abs(recent_as).mean()) if len(recent_as) else 0.0

            # DEBUG: verify the full action pipeline
            #   actor output -> [0,1]x[-pi,pi]x[0,1] -> replay buffer -> critic.
            # The env-step actions in `recent_actions` are the actor's CALE-domain
            # samples (identical transform to next_action inside update_qf), so we
            # print them here per-iteration instead of inside the jitted hot loop
            # (which would spam ~150 lines/sec).
            actor_a = recent_as
            print(
                "actor action min:",
                actor_a.min(axis=0),
                "max:",
                actor_a.max(axis=0),
                "mean:",
                actor_a.mean(axis=0),
            )

            # Replay-buffer action: exactly the tensor the critics receive.
            buf_a = np.asarray(jax.device_get(batch.action))
            print(
                "buffer action min:",
                buf_a.min(axis=0),
                "max:",
                buf_a.max(axis=0),
                "mean:",
                buf_a.mean(axis=0),
            )

            # Q-sensitivity of Critic 1 to the action input: do extreme CALE-domain
            # actions give distinct Q values, and is |dQ/da| meaningful? If
            # mean|dQ/da| ~ 0, the critic ignores the action -> actor gets no policy
            # gradient -> frozen policy (exactly what the current log suggests:
            # |g_act| collapses to ~0.01 while Q drifts flat negative).
            obs0 = batch.obs[:1]
            mz = critic1_encoder.apply(qf1_state.params["encoder"], obs0)
            a1 = jnp.array([[0.0, -jnp.pi, 0.0]], dtype=jnp.float32)
            a2 = jnp.array([[0.5, 0.0, 0.5]], dtype=jnp.float32)
            a3 = jnp.array([[1.0, jnp.pi, 1.0]], dtype=jnp.float32)
            qv1 = qf1.apply(qf1_state.params["qf"], mz, a1).squeeze()
            qv2 = qf1.apply(qf1_state.params["qf"], mz, a2).squeeze()
            qv3 = qf1.apply(qf1_state.params["qf"], mz, a3).squeeze()
            q_range_corners = _to_float(qv3) - _to_float(qv1)
            print("Q(a1) =", _to_float(qv1), " Q(a2) =", _to_float(qv2), " Q(a3) =", _to_float(qv3))

            def q_of_action(action):
                z_c = critic1_encoder.apply(qf1_state.params["encoder"], obs0)
                return qf1.apply(qf1_state.params["qf"], z_c, action).sum()

            dq_da = jax.grad(q_of_action)(a2)
            dq_da_np = np.asarray(jax.device_get(dq_da))
            mean_abs_dqda = _to_float(jnp.mean(jnp.abs(dq_da)))
            print("dQ/da =", dq_da_np)
            print("mean |dQ/da| =", mean_abs_dqda)

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
                # ---- DEBUG metrics ----
                "debug/unique_discrete_actions": unique_discrete,
                "debug/fraction_cont_unique": frac_cont_unique,
                "debug/action_abs_mean": action_abs_mean,
                "debug/q_range_corners": q_range_corners,
                "debug/mean_abs_dqda": mean_abs_dqda,
                "debug/actor_grad_norm": actor_grad_norm,
                "debug/actor_enc_grad_norm": actor_enc_grad_norm,
                "debug/actor_mlp_grad_norm": actor_mlp_grad_norm,
                "debug/alpha_grad_norm": alpha_grad_norm,
                "debug/qf1_grad_norm": qf1_grad_norm,
                "debug/qf2_grad_norm": qf2_grad_norm,
                "debug/qf1_enc_grad_norm": qf1_enc_grad_norm,
                "debug/qf1_q_grad_norm": qf1_q_grad_norm,
                "debug/qf2_enc_grad_norm": qf2_enc_grad_norm,
                "debug/qf2_q_grad_norm": qf2_q_grad_norm,
            })
            wandb.log(metrics, step=global_step)

            # DEBUG: compact per-iteration terminal line.
            print(
                f"[ITER {iteration:3d}] step={global_step:6d} ret={_to_float(avg_return):6.1f} "
                f"len={_to_float(avg_length):5.0f} alpha={_to_float(current_alpha):.4f} "
                f"logp={_to_float(log_prob_mean):6.2f} q1={_to_float(qf1_values):7.2f} "
                f"q2={_to_float(qf2_values):7.2f} qf_loss={_to_float(qf_loss):.4f} "
                f"act_loss={_to_float(actor_loss):6.3f} "
                f"|g_act|={_to_float(actor_grad_norm):6.2f} |g_enc_act|={_to_float(actor_enc_grad_norm):6.2f} "
                f"|g_q1|={_to_float(qf1_grad_norm):6.2f} |g_q2|={_to_float(qf2_grad_norm):6.2f} "
                f"uniq_disc={unique_discrete}/18 frac_cont_uniq={frac_cont_unique:.3f} "
                f"|a|_mean={action_abs_mean:.3f} |dQ/da|={mean_abs_dqda:.4f} Qrange={q_range_corners:.3f}"
            )

        # Evaluation
        if config.get("EVAL_DURING_TRAIN", False) and iteration > 0 and iteration % config.get("EVAL_EVERY", 50) == 0:
            save_and_eval(iteration, actor_state, qf1_state, qf2_state)

    # Final eval
    print("Evaluating final model ...")
    metrics = save_and_eval(iteration + 1, actor_state, qf1_state, qf2_state)
    wandb.finish()
    print("Training finished.")
    train_elapsed = time.time() - start_time
    print(f"[TRAIN] elapsed={train_elapsed:.2f}s SPS={global_step / (train_elapsed + 1e-8):.1f}")
    used_mb, total_mb, util = get_gpu_stats()
    print(f"[TRAIN] GPU mem={used_mb}/{total_mb}MB util={util}%")

    return metrics