"""Smoke tests for all DmLab levels via gym.make and gym.make_vec."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

deepmind_lab = pytest.importorskip("deepmind_lab", reason="deepmind_lab not installed")

import dmlab_gym  # noqa: E402
from dmlab_gym import ALL_LEVELS
from tests.conftest import HEIGHT, WIDTH

# Levels that require external datasets or special test-only decorators.
_SKIP_LEVELS: set[str] = {
    "psychlab_arbitrary_visuomotor_mapping",
    "psychlab_continuous_recognition",
    "psychlab_sequential_comparison",
    "psychlab_visual_search",
    "rooms_collect_good_objects_test",
    "rooms_exploit_deferred_effects_test",
}

LEVELS = [l for l in ALL_LEVELS if l not in _SKIP_LEVELS]

# Register all levels once at module level.
for _level in LEVELS:
    _id = f"dmlab_gym/{_level}-v0"
    if _id not in gym.registry:
        dmlab_gym.register(
            _level, renderer="software", width=WIDTH, height=HEIGHT
        )


def _env_id(level: str) -> str:
    return f"dmlab_gym/{level}-v0"


# -- Tests: gym.make ----------------------------------------------------------


class TestGymMakeAllLevels:
    """Every level can be created, reset, and stepped with gym.make."""

    @pytest.mark.parametrize("level", LEVELS)
    def test_reset(self, level):
        env = gym.make(_env_id(level))
        try:
            obs, info = env.reset()
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (HEIGHT, WIDTH, 3)
            assert obs.dtype == np.uint8
        finally:
            env.close()

    @pytest.mark.parametrize("level", LEVELS)
    def test_step(self, level):
        env = gym.make(_env_id(level))
        try:
            env.reset()
            action = np.zeros(7, dtype=np.intc)
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (HEIGHT, WIDTH, 3)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
        finally:
            env.close()


# -- Tests: gym.make_vec ------------------------------------------------------


class TestGymMakeVecAllLevels:
    """Every level can be created, reset, and stepped with gym.make_vec."""

    @pytest.mark.parametrize("level", LEVELS)
    def test_reset(self, level):
        vec_env = gym.make_vec(_env_id(level), num_envs=2)
        try:
            obs, info = vec_env.reset()
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (2, HEIGHT, WIDTH, 3)
            assert obs.dtype == np.uint8
        finally:
            vec_env.close()

    @pytest.mark.parametrize("level", LEVELS)
    def test_step(self, level):
        vec_env = gym.make_vec(_env_id(level), num_envs=2)
        try:
            vec_env.reset()
            actions = np.zeros((2, 7), dtype=np.intc)
            obs, rewards, terminated, truncated, infos = vec_env.step(actions)
            assert obs.shape == (2, HEIGHT, WIDTH, 3)
            assert rewards.shape == (2,)
            assert terminated.shape == (2,)
            assert truncated.shape == (2,)
        finally:
            vec_env.close()
