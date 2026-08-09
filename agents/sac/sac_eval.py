from typing import Callable, Tuple, Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: tuple,
    seed=1,
) -> Tuple[jnp.ndarray, Any]:
    """
    Evaluate a trained SAC agent.
    Args:
        model_path: path to the saved model file
        make_env: function that creates the environment
        env_id: environment id
        eval_episodes: number of episodes to run
        Model: tuple of (EncoderClass, ActorClass, SoftQNetworkClass)
        seed: random seed
    Returns:
        episodic_returns: (eval_episodes,) array of total rewards per episode
        env_states_until_done: environment states for the first episode (for video)
    """
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    _Encoder, _Actor, _SoftQNetwork = Model
    key = jax.random.key(seed)

    @jax.jit
    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        if next_obs.ndim == 3:
            next_obs = next_obs[None, ..., None]
        else:
            next_obs = next_obs[None, ...]

        return next_obs, state

    @jax.jit
    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        if next_obs.ndim == 3:
            next_obs = next_obs[None, ..., None]
        else:
            next_obs = next_obs[None, ...]

        return next_obs, next_state, reward, done, info

    key, reset_key = jax.random.split(key)
    actor_encoder = _Encoder()
    actor = _Actor(action_dim=env.action_space().shape[0])
    critic1_encoder = _Encoder()
    critic2_encoder = _Encoder()
    qf1 = _SoftQNetwork()
    qf2 = _SoftQNetwork()
    key, actor_encoder_key, actor_key, critic1_encoder_key, qf1_key, critic2_encoder_key, qf2_key = (
        jax.random.split(key, 7)
    )
    sample_obs = env.observation_space().sample(jax.random.PRNGKey(0))
    if sample_obs.ndim == 3:
        sample_obs = sample_obs[None, ..., None]
    else:
        sample_obs = sample_obs[None, ...]

    sample_obs = sample_obs.astype(jnp.float32)

    # Actor: encoder + Gaussian policy head
    actor_encoder_params = actor_encoder.init(actor_encoder_key, sample_obs)
    actor_hidden = actor_encoder.apply(actor_encoder_params, sample_obs)
    actor_params = actor.init(actor_key, actor_hidden)

    # Critic 1: encoder + Q1 head
    critic1_encoder_params = critic1_encoder.init(critic1_encoder_key, sample_obs)
    critic1_hidden = critic1_encoder.apply(critic1_encoder_params, sample_obs)
    dummy_action = jnp.zeros((1, env.action_space().shape[0]))
    qf1_params = qf1.init(qf1_key, critic1_hidden, dummy_action)

    # Critic 2: encoder + Q2 head
    critic2_encoder_params = critic2_encoder.init(critic2_encoder_key, sample_obs)
    critic2_hidden = critic2_encoder.apply(critic2_encoder_params, sample_obs)
    qf2_params = qf2.init(qf2_key, critic2_hidden, dummy_action)

    actor_params = {"encoder": actor_encoder_params, "actor": actor_params}
    critic1_params = {"encoder": critic1_encoder_params, "qf": qf1_params}
    critic2_params = {"encoder": critic2_encoder_params, "qf": qf2_params}

    # Load model: saved as [config, [actor_params, critic1_params, critic2_params]]
    with open(model_path, "rb") as f:
        (args, (actor_params, critic1_params, critic2_params)) = flax.serialization.from_bytes(
            (None, (actor_params, critic1_params, critic2_params)), f.read()
        )

    low = jnp.array(env.action_space().low)
    high = jnp.array(env.action_space().high)

    @jax.jit
    def get_action(actor_params, next_obs, key):
        """Deterministic action (mean, no noise) for evaluation."""
        if next_obs.ndim == 4:
            next_obs = next_obs[None, ...]

        hidden = actor_encoder.apply(actor_params["encoder"], next_obs)
        hidden = hidden.squeeze(0)
        mean, _ = actor.apply(actor_params["actor"], hidden)
        action_tanh = jnp.tanh(mean)
        action = low + (action_tanh + 1.0) * (high - low) / 2.0
        return action, key

    def step_fn(carry, _):
        next_obs, env_state, keys = carry
        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0))(
            actor_params, next_obs, keys
        )
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, actions)
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (next_obs, env_state, keys), (first_states, done, reward, actions)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    _, (first_states, dones, rewards, actions) = jax.lax.scan(
        step_fn, (next_obs, env_states, reset_keys), None, length=27_000
    )

    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(rewards, axis=0)

    # For video capture, we take the first episode's states
    env_states_until_done = jax.tree.map(lambda x: x[:first_done[0] + 1], first_states.atari_state.atari_state.env_state)
    return episodic_returns, env_states_until_done