"""Gymnasium wrapper for DeepMind Lab."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# DMLab-30 levels require a path prefix internally.  This mapping lets users
# pass just the bare name (e.g. ``"explore_goal_locations_large"``) and have
# it resolved automatically.
_DMLAB30_LEVELS: set[str] = {
    "explore_goal_locations_large",
    "explore_goal_locations_small",
    "explore_object_locations_large",
    "explore_object_locations_small",
    "explore_object_rewards_few",
    "explore_object_rewards_many",
    "explore_obstructed_goals_large",
    "explore_obstructed_goals_small",
    "language_answer_quantitative_question",
    "language_execute_random_task",
    "language_select_described_object",
    "language_select_located_object",
    "lasertag_one_opponent_large",
    "lasertag_one_opponent_small",
    "lasertag_three_opponents_large",
    "lasertag_three_opponents_small",
    "natlab_fixed_large_map",
    "natlab_varying_map_randomized",
    "natlab_varying_map_regrowth",
    "psychlab_arbitrary_visuomotor_mapping",
    "psychlab_continuous_recognition",
    "psychlab_sequential_comparison",
    "psychlab_visual_search",
    "rooms_collect_good_objects_test",
    "rooms_collect_good_objects_train",
    "rooms_exploit_deferred_effects_test",
    "rooms_exploit_deferred_effects_train",
    "rooms_keys_doors_puzzle",
    "rooms_select_nonmatching_object",
    "rooms_watermaze",
    "skymaze_irreversible_path_hard",
    "skymaze_irreversible_path_varied",
}


def _resolve_level(level_name: str) -> str:
    """Resolve a bare level name to its engine path."""
    if "/" in level_name:
        return level_name
    if level_name in _DMLAB30_LEVELS:
        return f"contributed/dmlab30/{level_name}"
    return level_name


class DmLabEnv(gym.Env):
    """Gymnasium environment wrapping the DeepMind Lab C API.

    Args:
        level_name: Level to load (e.g. ``"lt_chasm"``).
        observations: Observation names to request from the engine.
        renderer: ``"software"`` (headless OSMesa) or ``"hardware"`` (OpenGL).
        width: Observation pixel width.
        height: Observation pixel height.
        fps: Frames per second.
        max_num_steps: Maximum steps per episode (0 = unlimited).
        render_mode: Gymnasium render mode (unused, reserved for compatibility).
        **config: Extra key-value config forwarded to ``deepmind_lab.Lab``.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        level_name: str,
        observations: list[str] | None = None,
        renderer: str = "software",
        width: int = 240,
        height: int = 320,
        fps: int = 60,
        max_num_steps: int = 0,
        render_mode: str | None = None,
        **config: str,
    ) -> None:
        super().__init__()

        import deepmind_lab

        if observations is None:
            observations = ["RGB_INTERLEAVED"]

        self._obs_names = observations
        self._single_obs = len(observations) == 1
        self._max_num_steps = max_num_steps
        self._num_steps = 0
        self.render_mode = render_mode

        lab_config = {"width": str(width), "height": str(height), "fps": str(fps)}
        lab_config.update({k: str(v) for k, v in config.items()})

        self._lab = deepmind_lab.Lab(
            _resolve_level(level_name),
            observations,
            config=lab_config,
            renderer=renderer,
        )

        # Build observation space from engine specs.
        obs_specs = {s["name"]: s for s in self._lab.observation_spec()}
        obs_spaces = {}
        for name in observations:
            spec = obs_specs[name]
            shape = tuple(spec["shape"])
            dtype = spec["dtype"]
            if dtype == np.uint8:
                obs_spaces[name] = spaces.Box(0, 255, shape=shape, dtype=np.uint8)
            else:
                obs_spaces[name] = spaces.Box(
                    -np.inf, np.inf, shape=shape, dtype=np.float64
                )

        if self._single_obs:
            self.observation_space = obs_spaces[observations[0]]
        else:
            self.observation_space = spaces.Dict(obs_spaces)

        # Build action space from engine specs.
        action_specs = self._lab.action_spec()
        lows = np.array([s["min"] for s in action_specs], dtype=np.intc)
        highs = np.array([s["max"] for s in action_specs], dtype=np.intc)
        self.action_space = spaces.Box(low=lows, high=highs, dtype=np.intc)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        super().reset(seed=seed, options=options)
        self._num_steps = 0
        self._lab.reset(seed=seed if seed is not None else -1)
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict]:
        reward = self._lab.step(np.asarray(action, dtype=np.intc))
        self._num_steps += 1

        terminated = not self._lab.is_running()
        truncated = (
            not terminated
            and self._max_num_steps > 0
            and self._num_steps >= self._max_num_steps
        )

        obs = self._obs() if not terminated else self._last_obs
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        self._lab.close()

    def _obs(self) -> np.ndarray | dict[str, np.ndarray]:
        raw = self._lab.observations()
        if self._single_obs:
            self._last_obs = raw[self._obs_names[0]]
        else:
            self._last_obs = {name: raw[name] for name in self._obs_names}
        return self._last_obs
