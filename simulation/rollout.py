"""
Collect episode rollouts from a trained PPO agent.

Returns raw trajectory dicts for LSTM dataset construction.
Each dict: {t, y, vy, target_y, mass_fraction, x, z}
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env


def collect_rollouts(model_path: str, vecnorm_path: str,
                     env_factory, n_episodes: int = 200,
                     seed_offset: int = 0) -> list[dict]:
    """
    Run n_episodes with the given model and collect trajectory data.

    Args:
        model_path:   path to PPO .zip (without extension)
        vecnorm_path: path to VecNormalize .pkl
        env_factory:  callable(seed=int) returning a Gymnasium env
        n_episodes:   number of episodes to collect
        seed_offset:  offset episode seeds to avoid overlap across agents

    Returns:
        List of trajectory dicts, one per episode.
    """
    model = PPO.load(model_path, device="cpu")

    trajectories = []
    for ep in range(n_episodes):
        seed = seed_offset + ep

        # Fresh raw env for trajectory recording
        raw_env = env_factory(seed=seed)
        obs_raw, _ = raw_env.reset(seed=seed)

        # Fresh vec env for normalisation (load once per episode — light pkl load)
        _s = seed  # capture for lambda
        vec_env = make_vec_env(lambda: env_factory(seed=_s), n_envs=1)
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env.reset()

        ts, ys, vys, target_ys, mass_fracs = [], [], [], [], []
        xs, zs = [], []

        done = False
        while not done:
            # Normalise obs for model inference
            obs_norm = vec_env.normalize_obs(obs_raw[np.newaxis])[0]
            action, _ = model.predict(obs_norm, deterministic=True)

            # Record current state BEFORE stepping
            state = raw_env._state
            t_now = raw_env._t

            # Get target — 2D env has self._target, 3D has _get_target()
            if hasattr(raw_env, '_get_target'):
                target = raw_env._get_target()
            else:
                target = raw_env._target

            if hasattr(target, 'query3d'):
                ty = target.query3d(t_now)[0]
            else:
                ty = target.query(t_now)[0]

            mass = float(state[-1]) if len(state) == 7 else float(state[4])
            rocket = raw_env._rocket
            fuel_total = rocket.mass_wet - rocket.mass_dry
            fuel_rem   = max(0.0, mass - rocket.mass_dry)
            mass_frac  = fuel_rem / fuel_total if fuel_total > 0 else 0.0

            ts.append(t_now)
            ys.append(float(state[1]))
            vys.append(float(state[4]))
            target_ys.append(float(ty))
            mass_fracs.append(float(mass_frac))
            xs.append(float(state[0]))
            zs.append(float(state[2]) if len(state) == 7 else 0.0)

            # Step both envs with same action
            obs_raw, _, terminated, truncated, _ = raw_env.step(action)
            vec_env.step(action[np.newaxis])
            done = terminated or truncated

        vec_env.close()
        trajectories.append({
            "t":        np.array(ts,         dtype=np.float32),
            "y":        np.array(ys,         dtype=np.float32),
            "vy":       np.array(vys,        dtype=np.float32),
            "target_y": np.array(target_ys,  dtype=np.float32),
            "mass_frac":np.array(mass_fracs, dtype=np.float32),
            "x":        np.array(xs,         dtype=np.float32),
            "z":        np.array(zs,         dtype=np.float32),
        })

        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes")

    return trajectories
