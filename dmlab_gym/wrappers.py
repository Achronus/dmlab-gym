"""Gymnasium wrappers for DeepMind Lab environments."""

from __future__ import annotations

import multiprocessing
import traceback
from typing import Any, Callable

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
            self.observation_space = spaces.Dict(
                {
                    "RGB": spaces.Box(
                        obs_space.low[:3],
                        obs_space.high[:3],
                        shape=rgb_shape,
                        dtype=obs_space.dtype,
                    ),
                    "Depth": spaces.Box(
                        obs_space.low[3:],
                        obs_space.high[3:],
                        shape=depth_shape,
                        dtype=obs_space.dtype,
                    ),
                }
            )
        else:
            rgb_shape = (*shape[:-1], 3)
            depth_shape = (*shape[:-1], 1)
            self.observation_space = spaces.Dict(
                {
                    "RGB": spaces.Box(
                        obs_space.low[..., :3],
                        obs_space.high[..., :3],
                        shape=rgb_shape,
                        dtype=obs_space.dtype,
                    ),
                    "Depth": spaces.Box(
                        obs_space.low[..., 3:],
                        obs_space.high[..., 3:],
                        shape=depth_shape,
                        dtype=obs_space.dtype,
                    ),
                }
            )

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


def _get_mp_context() -> Any:
    """Return a safe multiprocessing context.

    Prefers ``forkserver`` (avoids copying parent address space, safer with
    shared libraries like the DMLab C extension) and falls back to ``spawn``.
    """
    for method in ("forkserver", "spawn"):
        try:
            return multiprocessing.get_context(method)
        except ValueError:
            continue
    return multiprocessing.get_context()


def _worker(pipe: Any, env_fn_bytes: bytes) -> None:
    """Child-process event loop that owns a single Gymnasium environment."""
    import pickle

    env: gym.Env | None = None
    try:
        env_fn: Callable[[], gym.Env] = pickle.loads(env_fn_bytes)
        env = env_fn()

        # Send spaces back to parent so it can expose them.
        pipe.send(("spaces", env.observation_space, env.action_space))

        while True:
            cmd, data = pipe.recv()

            if cmd == "reset":
                result = env.reset(**data)
                pipe.send(("ok", result))

            elif cmd == "step":
                result = env.step(data)
                pipe.send(("ok", result))

            elif cmd == "close":
                pipe.send(("ok", None))
                break

            else:
                raise RuntimeError(f"Unknown command: {cmd!r}")

    except Exception:
        tb = traceback.format_exc()
        try:
            pipe.send(("error", tb))
        except BrokenPipeError:
            pass
    finally:
        if env is not None:
            env.close()
        pipe.close()


class SubprocessEnv(gym.Env):
    """Run a single Gymnasium environment in a dedicated subprocess.

    This is necessary for environments backed by C engines with global state
    (like DMLab's Quake 3 engine) where only one instance can exist per
    process. Each ``SubprocessEnv`` gets its own address space.

    Args:
        env_or_fn: Either a registered environment ID (``str``) or a
            zero-argument callable that returns a ``gym.Env``.

    Example::

        env = SubprocessEnv("dmlab_gym/lt_chasm-v0")
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
    """

    def __init__(self, env_or_fn: str | Callable[[], gym.Env]) -> None:
        super().__init__()

        self._closed = True
        self._parent_pipe = None
        self._process = None

        if isinstance(env_or_fn, str):
            env_id = env_or_fn
            spec = gym.spec(env_id)
            entry_point = spec.entry_point
            spec_kwargs = dict(spec.kwargs) if spec.kwargs else {}

            def env_fn() -> gym.Env:
                if env_id not in gym.registry:
                    gym.register(
                        id=env_id,
                        entry_point=entry_point,
                        kwargs=spec_kwargs,
                    )
                return gym.make(env_id)
        else:
            env_fn = env_or_fn

        import cloudpickle

        env_fn_bytes = cloudpickle.dumps(env_fn)

        ctx = _get_mp_context()
        self._parent_pipe, child_pipe = ctx.Pipe()
        self._process = ctx.Process(
            target=_worker,
            args=(child_pipe, env_fn_bytes),
            daemon=True,
        )
        self._process.start()
        child_pipe.close()

        self._closed = False

        msg = self._parent_pipe.recv()
        if msg[0] == "error":
            self.close()
            raise RuntimeError(
                f"Environment subprocess failed during init:\n{msg[1]}"
            )
        _, self.observation_space, self.action_space = msg

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict]:
        return self._call("reset", {"seed": seed, "options": options})

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        return self._call("step", action)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._parent_pipe is not None:
            try:
                self._parent_pipe.send(("close", None))
                self._parent_pipe.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                self._parent_pipe.close()
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)

    def __del__(self) -> None:
        self.close()

    def _call(self, cmd: str, data: Any) -> Any:
        if self._closed:
            raise RuntimeError("Environment is closed")
        self._parent_pipe.send((cmd, data))
        tag, result = self._parent_pipe.recv()
        if tag == "error":
            raise RuntimeError(f"Environment subprocess error:\n{result}")
        return result
