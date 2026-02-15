# NatLab

|                       |                                                |
| --------------------- | ---------------------------------------------- |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)`               |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`                |
| **Levels**            | 3                                              |
| **Import**            | `dmlab_gym.register("natlab_fixed_large_map")` |

## Description

The NatLab levels are mushroom foraging tasks in naturalistic terrain environments. The agent must collect mushrooms to maximise score. The three variants test different aspects of memory and navigation: long-term memory on a fixed map, short-term memory with regrowing mushrooms, and generalisation across randomised maps. The time of day is randomised (day, dawn, night) across all variants.

## Levels

### natlab_fixed_large_map

Long-term memory variant. Mushrooms do not regrow. The map is a fixed large environment. Each episode the spawn location is picked randomly from a set of potential locations.

### natlab_varying_map_regrowth

Short-term memory variant. Mushrooms regrow after around one minute in the same location. The map is a randomised small environment. Topographical variation, shrubs, cacti, rocks, and spawn location are all randomised.

### natlab_varying_map_randomized

Randomised variant. Mushrooms do not regrow. The map is randomly generated at intermediate size. Topographical variation, vegetation, mushroom locations, and spawn location are all randomised.

## Action Space

| Index | Action            | Range       |
| ----- | ----------------- | ----------- |
| 0     | Look left/right   | -512 to 512 |
| 1     | Look up/down      | -512 to 512 |
| 2     | Strafe left/right | -1 to 1     |
| 3     | Move back/forward | -1 to 1     |
| 4     | Fire              | 0 to 1      |
| 5     | Jump              | 0 to 1      |
| 6     | Crouch            | 0 to 1      |

## Observation Space

Default observation is `RGB_INTERLEAVED` — a `Box(0, 255, (H, W, 3), uint8)` image. The engine observation spec is RGBD; pass `observations=["RGBD"]` for 4-channel RGBD.

## Episode End

The episode terminates when the engine signals completion (120-second episodes). Episodes can also be truncated via `max_num_steps`.

## Usage

```python
import gymnasium as gym
import dmlab_gym

dmlab_gym.register("natlab_fixed_large_map", renderer="software")
env = gym.make("dmlab_gym/natlab_fixed_large_map-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
