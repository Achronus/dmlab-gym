"""Tests for environment memory lifecycle and subprocess isolation."""

from __future__ import annotations

import tracemalloc

import gymnasium as gym
import numpy as np
import pytest

deepmind_lab = pytest.importorskip("deepmind_lab", reason="deepmind_lab not installed")

import dmlab_gym  # noqa: E402
from dmlab_gym.wrappers import SubprocessEnv  # noqa: E402

LEVEL = "lt_chasm"
WIDTH = 84
HEIGHT = 84
ENV_ID = f"dmlab_gym/{LEVEL}-v0"

if ENV_ID not in gym.registry:
    dmlab_gym.register(LEVEL, renderer="software", width=WIDTH, height=HEIGHT)


def _make_env():
    return gym.make(ENV_ID)


def test_create_close_no_leak():
    """Creating and closing environments should not accumulate memory."""
    env = _make_env()
    env.reset()
    env.close()

    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    for _ in range(3):
        env = _make_env()
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
        env.close()

    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = after.compare_to(baseline, "lineno")
    leaked = sum(s.size_diff for s in stats if s.size_diff > 0)
    assert leaked < 5 * 1024 * 1024, f"Leaked {leaked / 1024 / 1024:.1f} MB"


def test_sequential_reuse():
    """Environments can be created sequentially after closing the previous one."""
    for i in range(3):
        env = _make_env()
        obs, info = env.reset(seed=i)
        assert obs is not None
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs is not None
        env.close()


class TestSubprocessEnv:
    """Tests for the SubprocessEnv wrapper."""

    def test_reset_step_close(self):
        """Basic lifecycle works through the subprocess boundary."""
        env = SubprocessEnv(ENV_ID)
        try:
            obs, info = env.reset(seed=0)
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (HEIGHT, WIDTH, 3)
            assert isinstance(info, dict)

            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
        finally:
            env.close()

    def test_observation_space(self):
        """Observation space is correctly forwarded from child."""
        env = SubprocessEnv(ENV_ID)
        try:
            assert isinstance(env.observation_space, gym.spaces.Box)
            assert env.observation_space.shape == (HEIGHT, WIDTH, 3)
            assert env.observation_space.dtype == np.uint8
        finally:
            env.close()

    def test_action_space(self):
        """Action space is correctly forwarded from child."""
        env = SubprocessEnv(ENV_ID)
        try:
            assert isinstance(env.action_space, gym.spaces.Box)
            assert env.action_space.shape == (7,)
            assert env.action_space.dtype == np.intc
        finally:
            env.close()

    def test_multiple_simultaneous(self):
        """Multiple SubprocessEnv instances can coexist."""
        envs = [SubprocessEnv(ENV_ID) for _ in range(3)]
        try:
            observations = []
            for env in envs:
                obs, _ = env.reset(seed=0)
                observations.append(obs)

            assert len(observations) == 3
            for obs in observations:
                assert obs.shape == (HEIGHT, WIDTH, 3)

            for env in envs:
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                assert obs.shape == (HEIGHT, WIDTH, 3)
        finally:
            for env in envs:
                env.close()

    def test_standby_state_preserved(self):
        """An environment on standby keeps its episode state."""
        env = SubprocessEnv(ENV_ID)
        try:
            env.reset(seed=42)
            for _ in range(5):
                env.step(env.action_space.sample())

            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert obs.shape == (HEIGHT, WIDTH, 3)
            assert obs.dtype == np.uint8
        finally:
            env.close()

    def test_close_cleans_up_process(self):
        """Child process is terminated after close."""
        env = SubprocessEnv(ENV_ID)
        env.reset()
        env.close()
        assert not env._process.is_alive()

    def test_double_close_is_safe(self):
        """Calling close() twice does not raise."""
        env = SubprocessEnv(ENV_ID)
        env.reset()
        env.close()
        env.close()

    def test_factory_function(self):
        """SubprocessEnv accepts a callable factory."""

        def make_env():
            import dmlab_gym
            dmlab_gym.register(LEVEL, renderer="software", width=WIDTH, height=HEIGHT)
            return gym.make(ENV_ID)

        env = SubprocessEnv(make_env)
        try:
            obs, _ = env.reset()
            assert obs.shape == (HEIGHT, WIDTH, 3)
        finally:
            env.close()
