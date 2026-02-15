"""Tests for environment memory lifecycle and subprocess isolation."""

from __future__ import annotations

import tracemalloc

import gymnasium as gym
import numpy as np
import pytest

deepmind_lab = pytest.importorskip("deepmind_lab", reason="deepmind_lab not installed")

import dmlab_gym  # noqa: E402

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


