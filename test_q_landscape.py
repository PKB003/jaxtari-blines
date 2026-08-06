"""Stage 7 — Boundary action Q-landscape audit.

Determines whether Q overestimation spikes specifically near tanh action
boundaries. No fixes, no clipping, no action redesign.

Inputs (pick one):
  --checkpoint PATH   : real saved model [config,[network,actor,qf1,qf2]] (use real Pong)
  (default: random init, synthetic obs — validates the harness; run with
   --checkpoint on the GPU server for the real answer)

Evaluates, on each observation batch:
  1. actor sampled action
  2. actor mean action
  3. random uniform actions in [-1,1]^3
  4. boundary actions (+/-1 corners and mixed)

Logs Q1/Q2 for each, ranking (is actor_Q > random_Q? is boundary_Q > actor_Q?),
the % of states where boundary_Q / actor_Q is in the top 5% of (random+actor)
Q values, and continuous->discrete action-id mapping stats.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

# ---------------- networks (copied VERBATIM from agents/sac/sac.py) ----------------
class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        if x.ndim == 6:
            b, n_env, stack, h, w, c = x.shape
            x = jnp.transpose(x, (0, 1, 3, 4, 2, 5))
            x = x.reshape((b * n_env, h, w, stack * c))
        elif x.ndim == 5:
            b, stack, h, w, c = x.shape
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


class Actor(nn.Module):
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mean, log_std


class SoftQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x, a):
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0))(x)


# ---------------- config ----------------
ACTION_DIM = 3
OBS_SHAPE = (4, 84, 84, 1)   # real Pong obs shape
SEED = 0
N_STATES = 256               # number of obs in the batch
N_RANDOM = 64                # random actions per state
TAU_DISCRETE = 0.5           # ContinuousActionWrapper tau


def discrete_action_id(a, tau=TAU_DISCRETE):
    r = a[..., 0]
    theta = a[..., 1]
    fire = a[..., 2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x_idx = (x > tau).astype(np.int32) - (x < -tau).astype(np.int32) + 1
    y_idx = (y > tau).astype(np.int32) - (y < -tau).astype(np.int32) + 1
    fire_idx = (fire > tau).astype(np.int32)
    return x_idx * 6 + y_idx * 2 + fire_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to saved model. If omitted, uses random init + synthetic obs.")
    parser.add_argument("--obs-npy", type=str, default=None,
                        help="Optional .npy of real observations (uint8). If omitted, synthetic.")
    parser.add_argument("--n-states", type=int, default=N_STATES)
    parser.add_argument("--n-random", type=int, default=N_RANDOM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # ---- networks & params ----
    network = Network(); actor = Actor(action_dim=ACTION_DIM)
    qf1 = SoftQNetwork(action_dim=ACTION_DIM); qf2 = SoftQNetwork(action_dim=ACTION_DIM)
    key = jax.random.PRNGKey(args.seed)
    key, nk, ak, k1, k2 = jax.random.split(key, 5)
    sample = jnp.zeros((1,) + OBS_SHAPE, dtype=jnp.uint8)
    np_params = network.init(nk, sample)
    hidden0 = network.apply(np_params, sample)
    act_params = actor.init(ak, hidden0)
    da = jnp.zeros((1, ACTION_DIM))
    q1_params = qf1.init(k1, hidden0, da)
    q2_params = qf2.init(k2, hidden0, da)

    if args.checkpoint:
        import flax
        with open(args.checkpoint, "rb") as f:
            _, (np_params, act_params, q1_params, q2_params) = flax.serialization.from_bytes(
                (None, (np_params, act_params, q1_params, q2_params)), f.read())
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint: using RANDOM init + SYNTHETIC observations (harness validation only).")

    # ---- observations ----
    if args.obs_npy:
        obs = np.load(args.obs_npy).astype(np.uint8)
        print(f"Loaded obs {obs.shape} from {args.obs_npy}")
    else:
        rng = np.random.default_rng(args.seed)
        obs = rng.integers(0, 256, size=(args.n_states,) + OBS_SHAPE, dtype=np.uint8)
        print(f"SYNTHETIC obs {obs.shape}")

    obs_j = jnp.array(obs, dtype=jnp.float32)

    # ---- hidden, actor actions ----
    hidden = network.apply(np_params, obs_j)                    # (N,512)
    # actor sampled action + mean + log_std
    key, ak2 = jax.random.split(key)
    mean, log_std = actor.apply(act_params, hidden)
    std = jnp.exp(log_std)
    z = mean + std * jax.random.normal(ak2, shape=mean.shape)
    actor_sampled = jnp.tanh(z)                                 # [-1,1]^3
    actor_mean = jnp.tanh(mean)                                 # [-1,1]^3

    # random uniform actions in [-1,1]^3
    key, rk = jax.random.split(key)
    random_actions = jax.random.uniform(rk, (args.n_states, args.n_random, ACTION_DIM),
                                        minval=-1.0, maxval=1.0)  # (N,n_random,3)

    # helper Q eval: Q(obs, action) min of q1,q2
    def q_eval(a):
        # a: (N, d) or (N, n_random, d)
        q1v = qf1.apply(q1_params, hidden[:, None] if a.ndim == 3 else hidden, a)
        return q1v

    q_actor_sampled = qf1.apply(q1_params, hidden, actor_sampled).squeeze(-1)   # (N,)
    q_actor_mean = qf1.apply(q1_params, hidden, actor_mean).squeeze(-1)         # (N,)
    # tile hidden (N,512) -> (N, n_random, 512) to match random_actions (N,n_random,3)
    hidden_tiled = jnp.tile(hidden[:, None, :], (1, args.n_random, 1))          # (N,n_random,512)
    q_random = qf1.apply(q1_params, hidden_tiled, random_actions).squeeze(-1)   # (N,n_random)
    min_actor_s = jnp.minimum(q_actor_sampled, qf2.apply(q2_params, hidden, actor_sampled).squeeze(-1))
    max_actor_s = jnp.maximum(q_actor_sampled, qf2.apply(q2_params, hidden, actor_sampled).squeeze(-1))

    # boundary actions: all 8 corners +/-1
    corners = jnp.array([
        [1, 1, 1], [-1, -1, -1], [1, 1, -1], [1, -1, 1],
        [-1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=jnp.float32)  # (8,3)
    # tile hidden (N,512) -> (N, 8, 512) and corners -> (N, 8, 3)
    hidden_corners = jnp.tile(hidden[:, None, :], (1, corners.shape[0], 1))   # (N,8,512)
    corners_tiled = jnp.broadcast_to(corners[None, :, :], (hidden.shape[0],) + corners.shape)  # (N,8,3)
    q_corners = qf1.apply(q1_params, hidden_corners, corners_tiled).squeeze(-1)  # (N,8)

    # ---- rank actor vs random ----
    q_random_max = q_random.max(axis=-1)           # best random per state
    actor_vs_random = (min_actor_s > q_random_max).mean()   # P(actor_Q > best_random_Q)
    # actor vs boundary top 5%
    combined = jnp.concatenate([q_corners, q_random], axis=-1)   # (N, 8+n_random)
    top5 = jnp.argpartition(combined, kth=-max(1, int(0.05 * combined.shape[-1])), axis=-1)[:, -max(1, int(0.05 * combined.shape[-1])):]
    is_top5 = jnp.zeros_like(combined, dtype=bool)
    is_top5 = is_top5.at[jnp.arange(combined.shape[0])[:, None], top5].set(True)

    # actor in top5 of (corners+random)?
    actor_in_top5 = False  # actor not in combined; compute separately
    # boundary (best corner) in top5%?
    # build a per-state (N, 8) bool where corner in top5
    corner_in_top5 = jnp.stack([is_top5[i, jnp.arange(8)] for i in range(combined.shape[0])], axis=0)  # (N,8)
    corner_best_in_top5 = corner_in_top5.max(axis=-1).mean()  # P(best corner in top5%)

    # actor_Q vs corner_Q
    corner_max = q_corners.max(axis=-1)             # best corner Q per state
    boundary_gt_actor = (corner_max > max_actor_s).mean()  # P(boundary beats actor)
    actor_gt_corner = (min_actor_s > corner_max).mean()

    # ---- continuous -> discrete mapping ----
    a_samples_np = np.asarray(actor_sampled)  # in [-1,1]; convert to env range
    # env-range: low=[0,-pi,0], high=[1,pi,1]
    low = np.array([0.0, -np.pi, 0.0]); high = np.array([1.0, np.pi, 1.0])
    env_a = low + (a_samples_np + 1.0) * (high - low) / 2.0
    disc = discrete_action_id(env_a)
    unique_cont = len(np.unique(np.round(a_samples_np, 3), axis=0))
    unique_disc = len(np.unique(disc))

    # ---- print diagnostics ----
    print("\n" + "=" * 70)
    print("Q-LANDSCAPE DIAGNOSTICS")
    print("=" * 70)
    print(f"states={args.n_states}  random_per_state={args.n_random}")
    print(f"[Actor actions] mean={float(actor_sampled.mean()):.3f} std={float(actor_sampled.std()):.3f} "
          f"frac|a|>0.95={float((jnp.abs(actor_sampled) > 0.95).mean()):.4f}")
    print(f"\n[Q distribution]")
    print(f"  Q(actor_sampled): mean={float(min_actor_s.mean()):.3f} max={float(min_actor_s.max()):.3f}")
    print(f"  Q(actor_mean):    mean={float(q_actor_mean.mean()):.3f}")
    print(f"  Q(random):        mean={float(q_random.mean()):.3f} max_per_state_mean={float(q_random_max.mean()):.3f}")
    print(f"  Q(boundary_max):  mean={float(corner_max.mean()):.3f} max={float(corner_max.max()):.3f}")
    print(f"\n[Ranking]")
    print(f"  P(actor_Q > best_random_Q) = {float(actor_vs_random):.4f}")
    print(f"  P(best_boundary_Q > actor_Q) = {float(boundary_gt_actor):.4f}")
    print(f"  P(actor_Q > best_boundary_Q) = {float(actor_gt_corner):.4f}")
    print(f"  P(best_boundary_Q in top5% of (corners+random)) = {float(corner_best_in_top5):.4f}")
    print(f"\n[Discrete mapping]")
    print(f"  unique continuous actions (3dp): {unique_cont}  unique discrete ids: {unique_disc}")
    print(f"  continuous->discrete compression: {unique_cont}/{max(unique_disc,1)}")
    print("\n[Interpretation]")
    if float(boundary_gt_actor) > 0.4 and float(corner_best_in_top5) > 0.4:
        print("  A) Q has boundary spikes (boundary actions are overestimated).")
    elif float(q_actor_sampled.mean()) > float(q_random.mean()) and float(boundary_gt_actor) < 0.2:
        print("  B) Q uniformly overestimated across actions (no boundary spike).")
    elif float(boundary_gt_actor) > 0.4 and float(corner_best_in_top5) < 0.4:
        print("  B/C) boundary neither spiked nor top5: Q broad, but log_prob could still be wrong.")
    else:
        print("  Check raw numbers; likely C) Q correct but log_prob is wrong, or D) other.")

if __name__ == "__main__":
    main()