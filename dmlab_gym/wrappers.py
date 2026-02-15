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
