"""Gymnasium wrappers for DeepMind Lab environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# IMPALA discrete action set for DMLab (Table D.2, Espeholt et al. 2018).
# Columns: [look_lr, look_ud, strafe_lr, move_bf, fire, jump, crouch]
# fmt: off
DEFAULT_ACTION_TABLE = np.array([
    [  0, 0,  0,  1, 0, 0, 0],  # 0: forward
    [  0, 0,  0, -1, 0, 0, 0],  # 1: backward
    [  0, 0, -1,  0, 0, 0, 0],  # 2: strafe left
    [  0, 0,  1,  0, 0, 0, 0],  # 3: strafe right
    [-20, 0,  0,  0, 0, 0, 0],  # 4: look left
    [ 20, 0,  0,  0, 0, 0, 0],  # 5: look right
    [-20, 0,  0,  1, 0, 0, 0],  # 6: forward + look left
    [ 20, 0,  0,  1, 0, 0, 0],  # 7: forward + look right
    [  0, 0,  0,  0, 1, 0, 0],  # 8: fire
], dtype=np.intc)
# fmt: on


class SplitRGBD(gym.ObservationWrapper):
    """Split an RGBD observation into separate RGB and Depth channels.

    Supports both channels-first ``(4, H, W)`` and channels-last ``(H, W, 4)``
    layouts. The channel dimension is detected automatically from the shape.

    Produces a Dict observation with ``"RGB"`` and ``"Depth"`` keys.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box), (
            f"SplitRGBD requires Box observation space, got {type(obs_space).__name__}"
        )

        shape = obs_space.shape
        if shape[0] == 4:
            self._channels_first = True
        elif shape[-1] == 4:
            self._channels_first = False
        else:
            raise ValueError(
                f"SplitRGBD requires 4-channel observations, got shape {shape}"
            )

        if self._channels_first:
            rgb_shape = (3, *shape[1:])
            depth_shape = (1, *shape[1:])
            self.observation_space = spaces.Dict({
                "RGB": spaces.Box(
                    obs_space.low[:3], obs_space.high[:3],
                    shape=rgb_shape, dtype=obs_space.dtype,
                ),
                "Depth": spaces.Box(
                    obs_space.low[3:], obs_space.high[3:],
                    shape=depth_shape, dtype=obs_space.dtype,
                ),
            })
        else:
            rgb_shape = (*shape[:-1], 3)
            depth_shape = (*shape[:-1], 1)
            self.observation_space = spaces.Dict({
                "RGB": spaces.Box(
                    obs_space.low[..., :3], obs_space.high[..., :3],
                    shape=rgb_shape, dtype=obs_space.dtype,
                ),
                "Depth": spaces.Box(
                    obs_space.low[..., 3:], obs_space.high[..., 3:],
                    shape=depth_shape, dtype=obs_space.dtype,
                ),
            })

    def observation(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        if self._channels_first:
            return {"RGB": observation[:3], "Depth": observation[3:]}
        return {"RGB": observation[..., :3], "Depth": observation[..., 3:]}


class ActionDiscretize(gym.ActionWrapper):
    """Convert a Box action space to Discrete using an action lookup table.

    Each discrete action index maps to a predefined action vector from the
    IMPALA paper (Table D.2, Espeholt et al. 2018) by default.

    Args:
        env: The environment to wrap.
        action_table: Array of shape ``(N, action_dim)`` mapping discrete
            indices to action vectors. Defaults to the 9-action IMPALA set.
    """

    def __init__(
        self,
        env: gym.Env,
        action_table: np.ndarray | None = None,
    ) -> None:
        super().__init__(env)

        if action_table is None:
            action_table = DEFAULT_ACTION_TABLE

        self._action_table = np.asarray(action_table, dtype=np.intc)
        assert self._action_table.ndim == 2, "action_table must be 2D"
        self.action_space = spaces.Discrete(len(self._action_table))

    def action(self, action: int) -> np.ndarray:
        return self._action_table[action]
