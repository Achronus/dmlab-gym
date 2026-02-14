# dmlab-gym

A fork of [DeepMind Lab](https://github.com/google-deepmind/lab) updated for Bazel 8, Python 3.13, and wrapped with [Gymnasium](https://gymnasium.farama.org/).

## Platform Support

| Platform       | Status        |
| -------------- | ------------- |
| Linux (x86_64) | Supported     |
| macOS          | Not supported |
| Windows        | Not supported |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python environment management)
- [Podman](https://podman.io/) or [Docker](https://www.docker.com/) (container builds)
- Python 3.13+
- OSMesa (runtime dependency for headless rendering) — installed automatically by `dmlab-gym build` if missing

<details>
<summary>Manual OSMesa install</summary>

| Distribution        | Command                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| Fedora / RHEL       | `sudo dnf install mesa-libOSMesa` or `mesa-compat-libOSMesa`            |
| Bluefin / immutable | `sudo dnf install --transient mesa-compat-libOSMesa` (resets on reboot) |
| Ubuntu / Debian     | `sudo apt install libosmesa6`                                           |
| Arch                | `sudo pacman -S mesa`                                                   |

</details>

## Installation

```bash
pip install dmlab-gym
```

## Building the Native Extension

After installing, build the DeepMind Lab native extension:

### Quick Start

```bash
dmlab-gym build                        # build and install deepmind-lab
dmlab-gym build -o ~/my_output         # custom output directory
dmlab-gym build --no-install           # build only, skip install
```

This auto-detects Podman or Docker, builds the native extension inside a container, installs the wheel, and verifies the import automatically.

<details>
<summary>Example output</summary>

```
$ dmlab-gym build

Building deepmind-lab (podman)
  Source: /home/user/.local/share/dmlab-gym/source
  Output: /home/user/my-project/lab

  ✓ Building container image
  ✓ Compiling native extension (this may take a while)
  ✓ Installing wheel
  ✓ deepmind_lab is importable

  Done! deepmind-lab installed from deepmind_lab-1.0-py3-none-any.whl
```

</details>

## Usage

```python
import gymnasium as gym
import dmlab_gym

# Register a level
dmlab_gym.register("lt_chasm")

# Single environment
env = gym.make("dmlab_gym/lt_chasm-v0", renderer="software")
obs, info = env.reset()       # obs: (320, 240, 3) uint8
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()

# Vectorized
vec_env = gym.make_vec("dmlab_gym/lt_chasm-v0", num_envs=4, renderer="software")
obs, info = vec_env.reset()   # obs: (4, 320, 240, 3) uint8
```

Observations follow the standard Gymnasium convention — channels-last `(H, W, 3)` uint8 Box space using `RGB_INTERLEAVED`.

### Wrappers

Two custom wrappers are included:

- **`ActionDiscretize`** — converts the native 7-dimensional continuous action space to `Discrete(9)` using the IMPALA action set ([Espeholt et al., 2018](https://arxiv.org/abs/1802.01561), Table D.2).
- **`SplitRGBD`** — splits an RGBD observation into a Dict with separate `RGB` and `Depth` keys.

```python
from dmlab_gym.wrappers import ActionDiscretize, SplitRGBD

env = ActionDiscretize(gym.make("dmlab_gym/lt_chasm-v0", renderer="software"))
env.action_space  # Discrete(9)
```

All standard [Gymnasium wrappers](https://gymnasium.farama.org/main/api/wrappers/) are compatible:

| Wrapper | Compatible | Notes |
| --- | --- | --- |
| `GrayscaleObservation` | Yes | RGB to grayscale conversion |
| `ResizeObservation` | Yes | Resize to any resolution (requires `gymnasium[other]`) |
| `FrameStackObservation` | Yes | Stack N consecutive frames |
| `MaxAndSkipObservation` | Yes | Frame skipping with max pooling |
| `ReshapeObservation` | Yes | Reshape observation arrays |
| `RescaleObservation` | Yes | Best used after `DtypeObservation(float)` |
| `DtypeObservation` | Yes | Convert uint8 to float32/float64 |
| `FlattenObservation` | Yes | Flatten to 1D |
| `NormalizeObservation` | Yes | Running mean/std normalisation |
| `TransformObservation` | Yes | Custom observation function |
| `TimeLimit` | Yes | Truncate after N steps |
| `ClipReward` | Yes | Bound rewards to a range |
| `NormalizeReward` | Yes | Normalise rewards via running stats |
| `TransformReward` | Yes | Custom reward function |
| `RecordEpisodeStatistics` | Yes | Track episode returns and lengths |
| `RecordVideo` | Yes | Requires `render_mode="rgb_array"` and `moviepy` |
| `HumanRendering` | Yes | Requires `render_mode="rgb_array"`, `pygame`, and `opencv` |
| `FilterObservation` | No | Box observation space (not Dict) |
| `ClipAction` | No | Use `ActionDiscretize` instead |
| `RescaleAction` | No | Use `ActionDiscretize` instead |

## License

This project is a derivative work of DeepMind Lab and is distributed under the same licenses:

- **GPLv2** -- `engine/`, `q3map2/`, `assets_oa/`
- **CC BY 4.0** -- `assets/`
- **Custom academic license** -- `engine/code/tools/lcc/`
- **BSD 3-Clause** -- `q3map2/libs/{ddslib,picomodel}`

See [LICENSE](LICENSE) for full details.

## Attribution

Based on [DeepMind Lab](https://github.com/google-deepmind/lab) by DeepMind.

If you use DeepMind Lab in your research, please cite the [DeepMind Lab paper](https://arxiv.org/abs/1612.03801).
