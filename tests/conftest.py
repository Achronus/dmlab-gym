"""Shared test fixtures for dmlab-gym tests.

Tests require the ``deepmind_lab`` native extension to be installed
(via ``dmlab-gym build``). They will be skipped if it is not available.
"""

from __future__ import annotations

import gymnasium as gym
import pytest

deepmind_lab = pytest.importorskip("deepmind_lab", reason="deepmind_lab not installed")

import dmlab_gym  # noqa: E402

LEVEL = "lt_chasm"
WIDTH = 84
HEIGHT = 84

# Default observation is RGB_INTERLEAVED (H, W, 3) — channels-last.
ENV_ID = f"dmlab_gym/{LEVEL}-v0"
# RGBD observation is channels-first (4, H, W) — for RGBD-specific tests.
RGBD_ID = f"dmlab_gym/{LEVEL}_rgbd-v0"

if ENV_ID not in gym.registry:
    dmlab_gym.register(LEVEL, renderer="software", width=WIDTH, height=HEIGHT)

if RGBD_ID not in gym.registry:
    gym.register(
        id=RGBD_ID,
        entry_point="dmlab_gym.env:DmLabEnv",
        kwargs={
            "level_name": LEVEL,
            "observations": ["RGBD"],
            "renderer": "software",
            "width": WIDTH,
            "height": HEIGHT,
        },
    )


@pytest.fixture
def env():
    """RGB_INTERLEAVED environment (channels-last, shape: H×W×3)."""
    e = gym.make(ENV_ID)
    yield e
    e.close()


@pytest.fixture
def rgbd_env():
    """RGBD environment (channels-first, shape: 4×H×W)."""
    e = gym.make(RGBD_ID)
    yield e
    e.close()


@pytest.fixture
def render_env():
    """RGB_INTERLEAVED environment with render_mode='rgb_array'."""
    e = gym.make(ENV_ID, render_mode="rgb_array")
    yield e
    e.close()


@pytest.fixture
def vec_env():
    """Vectorized RGB_INTERLEAVED environment (2 envs)."""
    ve = gym.make_vec(ENV_ID, num_envs=2)
    yield ve
    ve.close()
