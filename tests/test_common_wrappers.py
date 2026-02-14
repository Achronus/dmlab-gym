"""Tests for official Gymnasium common and reward wrappers with DmLab."""

from __future__ import annotations

import numpy as np
from gymnasium.wrappers import (
    ClipReward,
    NormalizeReward,
    RecordEpisodeStatistics,
    TimeLimit,
)


# -- TimeLimit -----------------------------------------------------------------


class TestTimeLimit:
    """Truncate episodes after a fixed number of steps."""

    def test_truncates_at_limit(self, env):
        wrapped = TimeLimit(env, max_episode_steps=5)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        for i in range(5):
            _, _, terminated, truncated, _ = wrapped.step(action)
            if terminated:
                break
        if not terminated:
            assert truncated

    def test_does_not_truncate_before_limit(self, env):
        wrapped = TimeLimit(env, max_episode_steps=100)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        _, _, terminated, truncated, _ = wrapped.step(action)
        if not terminated:
            assert not truncated


# -- RecordEpisodeStatistics ---------------------------------------------------


class TestRecordEpisodeStatistics:
    """Track episode return and length."""

    def test_info_contains_episode_on_done(self, env):
        wrapped = RecordEpisodeStatistics(TimeLimit(env, max_episode_steps=10))
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        episode_found = False
        for _ in range(10):
            _, _, terminated, truncated, info = wrapped.step(action)
            if terminated or truncated:
                assert "episode" in info
                assert "r" in info["episode"]
                assert "l" in info["episode"]
                episode_found = True
                break
        assert episode_found


# -- ClipReward ----------------------------------------------------------------


class TestClipReward:
    """Clip reward values."""

    def test_reward_is_clipped(self, env):
        wrapped = ClipReward(env, min_reward=-1.0, max_reward=1.0)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        _, reward, _, _, _ = wrapped.step(action)
        assert -1.0 <= reward <= 1.0


# -- NormalizeReward -----------------------------------------------------------


class TestNormalizeReward:
    """Normalize rewards using running statistics."""

    def test_returns_float_reward(self, env):
        wrapped = NormalizeReward(env)
        wrapped.reset()
        action = np.zeros(7, dtype=np.intc)
        _, reward, _, _, _ = wrapped.step(action)
        assert isinstance(reward, (float, np.floating))
