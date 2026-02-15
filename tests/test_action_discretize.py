"""Tests for ActionDiscretize custom wrapper."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from dmlab_gym.wrappers import DEFAULT_ACTION_TABLE, ActionDiscretize


class TestActionDiscretizeSingle:
    """ActionDiscretize on a single environment."""

    def test_action_space_is_discrete(self, env):
        wrapped = ActionDiscretize(env)
        assert isinstance(wrapped.action_space, spaces.Discrete)
        assert wrapped.action_space.n == len(DEFAULT_ACTION_TABLE)

    def test_step_with_discrete_action(self, env):
        wrapped = ActionDiscretize(env)
        wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

    def test_all_default_actions_valid(self, env):
        wrapped = ActionDiscretize(env)
        wrapped.reset()
        for action_idx in range(wrapped.action_space.n):
            obs, _, terminated, _, _ = wrapped.step(action_idx)
            if terminated:
                wrapped.reset()

    def test_custom_action_table(self, env):
        table = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0],
        ], dtype=np.intc)
        wrapped = ActionDiscretize(env, action_table=table)
        assert wrapped.action_space.n == 2
        wrapped.reset()
        wrapped.step(0)
        wrapped.step(1)


