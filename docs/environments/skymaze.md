# SkyMaze

|                       |                                                        |
| --------------------- | ------------------------------------------------------ |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)`                       |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`                        |
| **Levels**            | 2                                                      |
| **Import**            | `dmlab_gym.register("skymaze_irreversible_path_hard")` |

## Description

The SkyMaze levels require the agent to reach a goal located at a distance from the starting position. The goal and start are connected by a sequence of platforms at different heights. Jumping is disabled, so higher platforms are unreachable once the agent drops down. The agent must plan a route to avoid becoming stuck.

## Levels

### skymaze_irreversible_path_hard

Fixed high difficulty. The agent must navigate a sequence of platforms where wrong choices are irreversible.

### skymaze_irreversible_path_varied

Variable difficulty. Each episode selects a map layout of random difficulty.

## Action Space

| Index | Action            | Range                    |
| ----- | ----------------- | ------------------------ |
| 0     | Look left/right   | -512 to 512              |
| 1     | Look up/down      | -512 to 512              |
| 2     | Strafe left/right | -1 to 1                  |
| 3     | Move back/forward | -1 to 1                  |
| 4     | Fire              | 0 to 1                   |
| 5     | Jump              | 0 to 1 (disabled — NOOP) |
| 6     | Crouch            | 0 to 1                   |

## Observation Space

Default observation is `RGB_INTERLEAVED` — a `Box(0, 255, (H, W, 3), uint8)` image. The engine observation spec is RGBD; pass `observations=["RGBD"]` for 4-channel RGBD.

## Episode End

The episode terminates when the engine signals completion. Episodes can also be truncated via `max_num_steps`.

## Usage

```python
import gymnasium as gym
import dmlab_gym

dmlab_gym.register("skymaze_irreversible_path_hard", renderer="software")
env = gym.make("dmlab_gym/skymaze_irreversible_path_hard-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
