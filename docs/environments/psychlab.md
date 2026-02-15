# PsychLab

|                       |                                                |
| --------------------- | ---------------------------------------------- |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)`               |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`                |
| **Levels**            | 4                                              |
| **Import**            | `dmlab_gym.register("psychlab_visual_search")` |

## Description

The PsychLab levels are cognitive psychology experiments implemented as first-person 3D environments. They test visuomotor learning, familiarity memory, working memory, and visual attention.

For details: [Leibo et al., "Psychlab: A Psychology Laboratory for Deep Reinforcement Learning Agents" (2018)](https://arxiv.org/abs/1801.08116).

## Levels

### psychlab_arbitrary_visuomotor_mapping

The agent is shown consecutive images and must remember associations with specific movement patterns (locations to point at). Images are drawn from a set of ~2500, and associations are randomly generated per episode.

### psychlab_continuous_recognition

Tests familiarity memory. Consecutive images are shown and the agent must indicate whether it has seen each image before during the episode. Looking at the left square indicates "no", right indicates "yes". Images (~2500) are shown in a different random order each episode.

### psychlab_sequential_comparison

Two consecutive patterns are shown. The agent must indicate whether the two patterns are identical. The delay between the study pattern and test pattern is variable.

### psychlab_visual_search

A collection of shapes are shown. The agent must identify whether a specific shape (pink "T") is present. Two black squares at the bottom of the screen are used for "yes" and "no" responses.

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

dmlab_gym.register("psychlab_visual_search", renderer="software")
env = gym.make("dmlab_gym/psychlab_visual_search-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
