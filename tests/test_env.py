"""Tests for the Gymnasium RL environment."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from simulation.env import RocketEnv


def make_env(randomize=False) -> RocketEnv:
    return RocketEnv(randomize=randomize, seed=0)


def test_gymnasium_check():
    """Official Gymnasium environment checker — catches API violations."""
    env = make_env()
    check_env(env, warn=True, skip_render_check=True)


def test_reset_returns_valid_obs():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape == (10,)
    assert env.observation_space.contains(obs)


def test_step_returns_correct_shape():
    env = make_env()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates():
    """A full episode must end — no infinite loop."""
    env = make_env()
    env.reset()
    for _ in range(40_000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return
    pytest.fail("Episode never terminated within 40000 steps")


def test_no_randomization_is_deterministic():
    """Same seed + fixed actions → identical reward sequences."""
    env = make_env(randomize=False)
    fixed_actions = [np.array([v]) for v in np.linspace(-1, 1, 100)]
    rewards_a, rewards_b = [], []

    env.reset(seed=1)
    for a in fixed_actions:
        _, r, done, trunc, _ = env.step(a)
        rewards_a.append(r)
        if done or trunc:
            break

    env.reset(seed=1)
    for a in fixed_actions:
        _, r, done, trunc, _ = env.step(a)
        rewards_b.append(r)
        if done or trunc:
            break

    assert rewards_a == rewards_b


def test_reward_has_expected_keys():
    env = make_env()
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0]))
    for key in ("r_alt", "r_vel", "r_fuel", "r_smooth", "t", "y"):
        assert key in info
