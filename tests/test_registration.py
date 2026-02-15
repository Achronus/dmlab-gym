"""Tests for environment registration."""

from __future__ import annotations

import gymnasium as gym

import dmlab_gym
from dmlab_gym import DmLabEnv


class TestRegistration:
    """dmlab_gym.register() and gym.make() integration."""

    def test_register_returns_env_id(self):
        env_id = dmlab_gym.register("seekavoid_arena_01", version=1)
        assert env_id == "dmlab_gym/seekavoid_arena_01-v1"

    def test_registered_env_in_registry(self):
        env_id = dmlab_gym.register("nav_maze_static_01", version=2)
        assert env_id in gym.registry

    def test_make_returns_dmlab_env(self):
        dmlab_gym.register("nav_maze_random_goal_01", version=3, renderer="software")
        env = gym.make("dmlab_gym/nav_maze_random_goal_01-v3")
        assert isinstance(env.unwrapped, DmLabEnv)
        env.close()

    def test_kwargs_forwarded(self):
        dmlab_gym.register(
            "lt_hallway_slope",
            version=4,
            renderer="software",
            width=64,
            height=48,
        )
        env = gym.make("dmlab_gym/lt_hallway_slope-v4")
        assert env.observation_space.shape == (48, 64, 3)
        env.close()
