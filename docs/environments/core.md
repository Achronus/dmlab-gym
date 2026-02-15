# Core

|                       |                                  |
| --------------------- | -------------------------------- |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)` |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`  |
| **Levels**            | 12                               |
| **Import**            | `dmlab_gym.register("lt_chasm")` |

## Description

The core levels are standalone environments shipped with DeepMind Lab. They cover two main categories: laser tag arenas with bot opponents, and maze navigation with goal-seeking or seek-and-avoid objectives.

## Levels

### lt_chasm

Laser tag in a map featuring a chasm. The agent fights 4 bots across a divided arena.

### lt_hallway_slope

Laser tag in a hallway with sloped terrain. The agent fights 4 bots in a corridor environment.

### lt_horseshoe_color

Laser tag in a horseshoe-shaped map with colour-coded elements. The agent fights 4 bots.

### lt_space_bounce_hard

A challenging laser tag level in a space-themed environment with bouncing mechanics. The agent fights 4 bots at maximum skill level.

### nav_maze_random_goal_01

Navigate a small procedurally-defined maze to find randomly spawning goal objects. Episode length: 60 seconds.

### nav_maze_random_goal_02

Navigate a medium maze to find randomly spawning goal objects. Episode length: 150 seconds.

### nav_maze_random_goal_03

Navigate a large maze to find randomly spawning goal objects. Episode length: 300 seconds.

### nav_maze_static_01

Navigate a small maze with static goal and obstacle placements. The agent must seek positive objects while avoiding negative ones. Episode length: 60 seconds.

### nav_maze_static_02

Navigate a medium maze with static goal and obstacle placements. Episode length: 150 seconds.

### nav_maze_static_03

Navigate a large maze with static goal and obstacle placements. Episode length: 300 seconds.

### seekavoid_arena_01

An open arena where the agent must seek positive objects while avoiding negative ones. Episode length: 20 seconds.

### stairway_to_melon

A seek-and-avoid level featuring stairway navigation leading to a goal. Episode length: 60 seconds.

## Action Space

All core levels use the standard 7D integer action space:

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

Default observation is `RGB_INTERLEAVED` — a `Box(0, 255, (H, W, 3), uint8)` image. Pass `observations=["RGBD"]` for 4-channel RGBD.

## Rewards

Rewards are level-specific and returned directly from the engine. Laser tag levels reward tagging opponents. Navigation levels reward reaching goals and penalise collecting bad objects.

## Episode End

The episode terminates when the engine signals completion (`is_running()` returns false). Episodes can also be truncated via the `max_num_steps` parameter.

## Usage

```python
import gymnasium as gym
import dmlab_gym

dmlab_gym.register("lt_chasm", renderer="software")
env = gym.make("dmlab_gym/lt_chasm-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
