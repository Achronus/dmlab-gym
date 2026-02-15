"""Tests for DmLabEnv via gym.make and gym.make_vec."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from tests.conftest import HEIGHT, WIDTH


class TestGymMake:
    """Single environment via gym.make (RGB_INTERLEAVED, channels-last)."""

    def test_creates_dmlab_env(self, env):
        from dmlab_gym import DmLabEnv

        assert isinstance(env.unwrapped, DmLabEnv)

    def test_observation_space_is_box(self, env):
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (HEIGHT, WIDTH, 3)
        assert env.observation_space.dtype == np.uint8

    def test_action_space_is_box(self, env):
        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (7,)
        assert env.action_space.dtype == np.intc

    def test_reset_returns_array_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (HEIGHT, WIDTH, 3)
        assert isinstance(info, dict)

    def test_step_returns_tuple(self, env):
        env.reset()
        action = np.zeros(7, dtype=np.intc)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_truncates(self, env):
        from gymnasium.wrappers import TimeLimit

        wrapped = TimeLimit(env, max_episode_steps=10)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        for _ in range(10):
            _, _, terminated, truncated, _ = wrapped.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_close(self, env):
        env.reset()


class TestGymMakeVec:
    """Vectorized environments via gym.make_vec."""

    def test_num_envs(self, vec_env):
        assert vec_env.num_envs == 2

    def test_reset_returns_batched_obs(self, vec_env):
        obs, info = vec_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, HEIGHT, WIDTH, 3)

    def test_step_returns_batched_results(self, vec_env):
        vec_env.reset()
        actions = np.zeros((2, 7), dtype=np.intc)
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        assert obs.shape == (2, HEIGHT, WIDTH, 3)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
