# LaserTag

|                       |                                                     |
| --------------------- | --------------------------------------------------- |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)`                    |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`                     |
| **Levels**            | 4                                                   |
| **Import**            | `dmlab_gym.register("lasertag_one_opponent_small")` |

## Description

The LaserTag levels require the agent to play laser tag in procedurally generated maps containing random gadgets and power-ups. The agent begins each episode with the default Rapid Gadget and a limit of 100 tags. The agent's Shield starts at 125 and slowly drops to 100. Gadgets, power-ups, and map layout are randomised per episode.

The four variants differ in map size (small/large) and opponent count (1/3 bots at difficulty level 4).

## Levels

### lasertag_one_opponent_small

Small map, 1 opponent bot.

### lasertag_three_opponents_small

Small map, 3 opponent bots.

### lasertag_one_opponent_large

Large map, 1 opponent bot.

### lasertag_three_opponents_large

Large map, 3 opponent bots.

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

dmlab_gym.register("lasertag_one_opponent_small", renderer="software")
env = gym.make("dmlab_gym/lasertag_one_opponent_small-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
