# Rooms

|                       |                                                 |
| --------------------- | ----------------------------------------------- |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)`                |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`                 |
| **Levels**            | 7                                               |
| **Import**            | `dmlab_gym.register("rooms_keys_doors_puzzle")` |

## Description

The Rooms levels test object interaction, memory, and planning in multi-room environments. Tasks range from collecting the correct objects to solving key-door puzzles and spatial memory challenges.

## Levels

### rooms_collect_good_objects_train

The agent must learn to collect good objects and avoid bad objects. During training, only some combinations of objects/environments are shown, creating a correlational structure. The agent must learn that the environment appearance does not determine which objects are good.

For more details: [Higgins et al., "DARLA: Improving Zero-Shot Transfer in Reinforcement Learning" (2017)](https://arxiv.org/abs/1707.08475).

### rooms_collect_good_objects_test

Test set for Collect Good Objects. Consists of held-out combinations of objects/environments never seen during training.

### rooms_exploit_deferred_effects_train

The agent must make a conceptual leap from picking up a special object to getting access to more rewards later on, even though this is never shown in a single environment and is costly. Expected to be hard for model-free agents but simpler with model-based strategies.

### rooms_exploit_deferred_effects_test

Test set for Exploit Deferred Effects. Tested in a room configuration never seen during training, where picking up a special object suddenly becomes useful.

### rooms_select_nonmatching_object

The agent is placed in a small room containing an out-of-reach object and a teleport pad. Touching the pad awards 1 point and teleports the agent to a second room with two objects — one matching and one non-matching.

- Collect matching object: **-10 points**
- Collect non-matching object: **+10 points**

### rooms_watermaze

The agent must find a hidden platform which, when found, generates a reward. Difficult to find the first time, but in subsequent trials the agent should remember the location. Tests episodic memory and navigation ability.

### rooms_keys_doors_puzzle

A procedural planning puzzle. The agent must reach a goal object blocked by coloured doors. Single-use coloured keys open matching doors (one key at a time). The objective is to figure out the correct sequence — visiting rooms or collecting keys in the wrong order can make the goal unreachable.

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

dmlab_gym.register("rooms_keys_doors_puzzle", renderer="software")
env = gym.make("dmlab_gym/rooms_keys_doors_puzzle-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
