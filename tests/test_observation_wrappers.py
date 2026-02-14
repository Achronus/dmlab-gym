"""Tests for official Gymnasium observation wrappers with DmLab."""

from __future__ import annotations

import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

from tests.conftest import HEIGHT, WIDTH


# -- GrayscaleObservation ------------------------------------------------------


class TestGrayscaleObservation:
    """Convert RGB observations to grayscale."""

    def test_observation_shape(self, env):
        wrapped = GrayscaleObservation(env)
        assert wrapped.observation_space.shape == (HEIGHT, WIDTH)

    def test_reset_returns_grayscale(self, env):
        wrapped = GrayscaleObservation(env)
        obs, _ = wrapped.reset()
        assert obs.shape == (HEIGHT, WIDTH)
        assert obs.dtype == np.uint8

    def test_step_returns_grayscale(self, env):
        wrapped = GrayscaleObservation(env)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        obs, _, _, _, _ = wrapped.step(action)
        assert obs.shape == (HEIGHT, WIDTH)

    def test_keep_dim(self, env):
        wrapped = GrayscaleObservation(env, keep_dim=True)
        obs, _ = wrapped.reset()
        assert obs.shape == (HEIGHT, WIDTH, 1)


# -- ResizeObservation ---------------------------------------------------------


class TestResizeObservation:
    """Resize image observations."""

    def test_observation_shape(self, env):
        wrapped = ResizeObservation(env, shape=(42, 42))
        assert wrapped.observation_space.shape == (42, 42, 3)

    def test_reset_returns_resized(self, env):
        wrapped = ResizeObservation(env, shape=(42, 42))
        obs, _ = wrapped.reset()
        assert obs.shape == (42, 42, 3)

    def test_step_returns_resized(self, env):
        wrapped = ResizeObservation(env, shape=(42, 42))
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        obs, _, _, _, _ = wrapped.step(action)
        assert obs.shape == (42, 42, 3)


# -- FrameStackObservation -----------------------------------------------------


class TestFrameStackObservation:
    """Stack consecutive frames."""

    def test_observation_shape(self, env):
        wrapped = FrameStackObservation(env, stack_size=4)
        assert wrapped.observation_space.shape == (4, HEIGHT, WIDTH, 3)

    def test_reset_returns_stacked(self, env):
        wrapped = FrameStackObservation(env, stack_size=4)
        obs, _ = wrapped.reset()
        assert obs.shape == (4, HEIGHT, WIDTH, 3)

    def test_step_returns_stacked(self, env):
        wrapped = FrameStackObservation(env, stack_size=4)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        obs, _, _, _, _ = wrapped.step(action)
        assert obs.shape == (4, HEIGHT, WIDTH, 3)


# -- Composition ---------------------------------------------------------------


class TestObservationComposition:
    """Combining observation wrappers."""

    def test_grayscale_then_resize(self, env):
        wrapped = ResizeObservation(
            GrayscaleObservation(env, keep_dim=True), shape=(42, 42)
        )
        obs, _ = wrapped.reset()
        # ResizeObservation squeezes the trailing dim=1 channel.
        assert obs.shape in ((42, 42, 1), (42, 42))

    def test_resize_then_frame_stack(self, env):
        wrapped = FrameStackObservation(
            ResizeObservation(env, shape=(42, 42)), stack_size=4
        )
        obs, _ = wrapped.reset()
        assert obs.shape == (4, 42, 42, 3)
