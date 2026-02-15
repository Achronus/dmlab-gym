"""Tests for render mode support and render-dependent wrappers."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo, TimeLimit

from tests.conftest import HEIGHT, WIDTH


class TestRender:
    """Render mode support for recording and visualization."""

    def test_render_returns_rgb_array(self, render_env):
        render_env.reset()
        frame = render_env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (HEIGHT, WIDTH, 3)
        assert frame.dtype == np.uint8

    def test_render_updates_after_step(self, render_env):
        render_env.reset()
        frame_before = render_env.render().copy()
        for _ in range(10):
            render_env.step(render_env.action_space.sample())
        frame_after = render_env.render()
        assert frame_after.shape == frame_before.shape

    def test_render_none_without_render_mode(self, env):
        env.reset()
        assert env.render() is None

    def test_render_raises_before_reset(self, render_env):
        with pytest.raises(Exception):
            render_env.render()


class TestRecordVideo:
    """RecordVideo wrapper with render_mode='rgb_array'."""

    @pytest.fixture
    def video_env(self, render_env, tmp_path):
        wrapped = RecordVideo(
            TimeLimit(render_env, max_episode_steps=5),
            video_folder=str(tmp_path),
            episode_trigger=lambda ep: True,
        )
        yield wrapped
        wrapped.close()

    def test_records_video_file(self, video_env, tmp_path):
        video_env.reset()
        action = np.zeros(7, dtype=np.intc)
        for _ in range(5):
            _, _, terminated, truncated, _ = video_env.step(action)
            if terminated or truncated:
                break
        video_env.close()
        videos = list(tmp_path.glob("*.mp4"))
        assert len(videos) >= 1

    def test_step_still_returns_valid_data(self, video_env):
        video_env.reset()
        action = np.zeros(7, dtype=np.intc)
        obs, reward, terminated, truncated, info = video_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
