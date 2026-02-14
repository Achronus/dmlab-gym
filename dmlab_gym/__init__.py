"""DeepMind Lab with Bazel 8, Python 3.13, and Gymnasium support."""

import gymnasium as gym

from dmlab_gym.env import DmLabEnv, _DMLAB30_LEVELS
from dmlab_gym.wrappers import ActionDiscretize, SplitRGBD

CORE_LEVELS: list[str] = [
    "lt_chasm",
    "lt_hallway_slope",
    "lt_horseshoe_color",
    "lt_space_bounce_hard",
    "nav_maze_random_goal_01",
    "nav_maze_random_goal_02",
    "nav_maze_random_goal_03",
    "nav_maze_static_01",
    "nav_maze_static_02",
    "nav_maze_static_03",
    "seekavoid_arena_01",
    "stairway_to_melon",
]

DMLAB30_LEVELS: list[str] = sorted(_DMLAB30_LEVELS)

ALL_LEVELS: list[str] = CORE_LEVELS + DMLAB30_LEVELS

__all__ = [
    "ALL_LEVELS",
    "ActionDiscretize",
    "CORE_LEVELS",
    "DMLAB30_LEVELS",
    "DmLabEnv",
    "SplitRGBD",
    "register",
]


def register(level_name: str, *, version: int = 0, **kwargs: str) -> str:
    """Register a DeepMind Lab level as a Gymnasium environment.

    Args:
        level_name: Level to register (e.g. ``"lt_chasm"``).
        version: Environment version number.
        **kwargs: Default config passed to :class:`DmLabEnv`.

    Returns:
        The registered environment ID.
    """
    env_id = f"dmlab_gym/{level_name}-v{version}"
    gym.register(
        id=env_id,
        entry_point="dmlab_gym.env:DmLabEnv",
        kwargs={"level_name": level_name, **kwargs},
    )
    return env_id
