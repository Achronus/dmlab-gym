# Language

|                       |                                                          |
| --------------------- | -------------------------------------------------------- |
| **Action Space**      | `Box(shape=(7,), dtype=np.intc)`                         |
| **Observation Space** | `Box(0, 255, (H, W, 3), uint8)`                          |
| **Levels**            | 4                                                        |
| **Import**            | `dmlab_gym.register("language_select_described_object")` |

## Description

The Language levels test grounded language understanding. The agent receives natural language instructions and must act on them in a 3D environment — selecting described objects, following spatial directions, or answering questions.

For details: [Hermann, Hill et al., "Grounded Language Learning in a Simulated 3D World" (2017)](https://arxiv.org/abs/1706.06551).

## Levels

### language_select_described_object

The agent is placed in a small room with two objects. An instruction describes one of them. The agent must collect the described object.

### language_select_located_object

The agent is asked to collect a specified coloured object in a specified coloured room (e.g. "Pick the red object in the blue room"). There are four variants with 2-6 rooms — more rooms means more distractors.

### language_execute_random_task

The agent receives one of seven possible tasks, each with a different type of language instruction (e.g. "Get the red hat from the blue room"). The agent is rewarded for collecting the correct object and penalised for the wrong one. The level restarts after any object is collected.

### language_answer_quantitative_question

The agent receives a yes/no question about object colours and counts (e.g. "Are all cars blue?", "Is any car blue?"). The agent answers by collecting a white sphere (yes) or black sphere (no).

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

Default observation is `RGB_INTERLEAVED` — a `Box(0, 255, (H, W, 3), uint8)` image. The engine observation spec includes both RGBD and language channels.

## Episode End

The episode terminates when the engine signals completion. Episodes can also be truncated via `max_num_steps`.

## Usage

```python
import gymnasium as gym
import dmlab_gym

dmlab_gym.register("language_select_described_object", renderer="software")
env = gym.make("dmlab_gym/language_select_described_object-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
