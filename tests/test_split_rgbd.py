"""Tests for SplitRGBD custom wrapper."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from dmlab_gym.wrappers import SplitRGBD
from tests.conftest import HEIGHT, WIDTH


class TestSplitRGBDChannelsFirst:
    """SplitRGBD on an RGBD environment (channels-first: 4×H×W)."""

    def test_observation_space_is_dict(self, rgbd_env):
        wrapped = SplitRGBD(rgbd_env)
        assert isinstance(wrapped.observation_space, spaces.Dict)
        assert set(wrapped.observation_space.spaces) == {"RGB", "Depth"}

    def test_rgb_shape_and_dtype(self, rgbd_env):
        wrapped = SplitRGBD(rgbd_env)
        assert wrapped.observation_space["RGB"].shape == (3, HEIGHT, WIDTH)
        assert wrapped.observation_space["RGB"].dtype == np.uint8

    def test_depth_shape_and_dtype(self, rgbd_env):
        wrapped = SplitRGBD(rgbd_env)
        assert wrapped.observation_space["Depth"].shape == (1, HEIGHT, WIDTH)
        assert wrapped.observation_space["Depth"].dtype == np.uint8

    def test_reset_splits_channels(self, rgbd_env):
        wrapped = SplitRGBD(rgbd_env)
        obs, _ = wrapped.reset()
        assert isinstance(obs, dict)
        assert obs["RGB"].shape == (3, HEIGHT, WIDTH)
        assert obs["Depth"].shape == (1, HEIGHT, WIDTH)

    def test_step_splits_channels(self, rgbd_env):
        wrapped = SplitRGBD(rgbd_env)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        obs, _, _, _, _ = wrapped.step(action)
        assert obs["RGB"].shape == (3, HEIGHT, WIDTH)
        assert obs["Depth"].shape == (1, HEIGHT, WIDTH)
