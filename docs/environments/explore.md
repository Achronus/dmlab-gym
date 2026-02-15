# Explore

|                       |                                                      |
| --------------------- | ---------------------------------------------------- |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)`                     |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`                      |
| **Levels**            | 8                                                    |
| **Import**            | `dmlab_gym.register("explore_goal_locations_large")` |

## Description

The Explore levels test navigation and exploration in procedurally generated mazes. Tasks involve finding goals, collecting objects, and navigating obstructed paths. Each level comes in small and large variants (larger maps with longer episodes). Layouts, themes, and spawn locations are randomised per episode.

## Levels

### explore_goal_locations_small

Find the goal object as fast as possible. After the goal is found, the level restarts. Goal location, layout, and theme are randomised per episode.

### explore_goal_locations_large

Same as Goal Locations Small but with a larger map and longer episode duration.

### explore_object_locations_small

Collect apples placed in rooms within the maze. Upon collecting all apples, the level resets. Apple locations, layout, and theme are randomised per episode.

### explore_object_locations_large

Same as Object Locations Small but with a larger map and longer episode duration.

### explore_obstructed_goals_small

Find the goal as fast as possible, but with randomly opened and closed doors. A path to the goal always exists. After the goal is found, the level restarts. Door states are randomised per reset.

### explore_obstructed_goals_large

Same as Obstructed Goals Small but with a larger map and longer episode duration.

### explore_object_rewards_few

Collect human-recognisable objects placed around a room. Some objects are positive, some negative. After all positive objects are collected, the level restarts. Object categories and reward values are randomised per episode.

### explore_object_rewards_many

A more difficult variant of Object Rewards Few with an increased number of goal objects and longer episode duration.

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

The episode terminates when the engine signals completion. Episodes can also be truncated via `max_num_steps`.

## Usage

```python
import gymnasium as gym
import dmlab_gym

dmlab_gym.register("explore_goal_locations_large", renderer="software")
env = gym.make("dmlab_gym/explore_goal_locations_large-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
