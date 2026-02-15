# dmlab-gym

A [Gymnasium](https://gymnasium.farama.org/) port of [DeepMind Lab](https://github.com/google-deepmind/lab) — 42 first-person 3D environments for Reinforcement Learning research, updated for Bazel 8 and Python 3.13.

[[Original Paper]](https://arxiv.org/abs/1612.03801)

## What is this?

This project wraps the original [DeepMind Lab](https://github.com/google-deepmind/lab) C engine (Quake 3 based) as standard [Gymnasium](https://gymnasium.farama.org/) environments.

The C game code remains **unchanged** with deprecation warnings removed. This package only adds a Python interface layer around it.

### Key features

- Standard `gym.make()` API for all 42 levels
- Compatible with all standard Gymnasium wrappers
- Vectorized environments via `gym.make_vec()`
- Python 3.13+ support
- Managed with `uv` and `pyproject.toml`

## Installation

```bash
pip install dmlab-gym
```

**Requirements:** Python 3.13+ (64-bit), Linux x86_64.

| Platform       | Status        |
| -------------- | ------------- |
| Linux (x86_64) | Supported     |
| macOS          | Not supported |
| Windows        | Not supported |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python environment management)
- [Podman](https://podman.io/) or [Docker](https://www.docker.com/) (container builds)
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

## Building the Native Extension

After installing, build the DeepMind Lab native extension:

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

## Quick Start

### Single environment (standard Gymnasium API)

```python
import gymnasium as gym
import dmlab_gym  # auto-registers all 42 levels

env = gym.make("dmlab_gym/lt_chasm-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Vectorized environments (parallel training)

DMLab's C engine only supports one instance per process. Use `gym.make_vec` to run N copies in separate subprocesses — each gets its own memory-isolated process and stays alive between steps:

```python
import gymnasium as gym
import dmlab_gym

vec_env = gym.make_vec(
    "dmlab_gym/lt_chasm-v0",
    num_envs=8,
    vectorization_mode="async",
)

obs, infos = vec_env.reset()
obs, rewards, terminated, truncated, infos = vec_env.step(vec_env.action_space.sample())
vec_env.close()
```

To customise environment options (resolution, renderer, etc.), use `dmlab_gym.register()` to re-register with different settings.

## Environments

All environments produce `(H, W, 3)` RGB observations and use a 7D integer action space `Box(shape=(7,), dtype=np.intc)`. See [docs/environments/](docs/environments/) for detailed per-group documentation.

| Group                                     | Count | Levels                                            | Description                               |
| ----------------------------------------- | ----- | ------------------------------------------------- | ----------------------------------------- |
| [Core](docs/environments/core.md)         | 12    | `lt_chasm`, `seekavoid_arena_01`, ...             | Laser tag arenas and maze navigation      |
| [Rooms](docs/environments/rooms.md)       | 7     | `rooms_keys_doors_puzzle`, `rooms_watermaze`, ... | Object interaction, memory, and planning  |
| [Language](docs/environments/language.md) | 4     | `language_select_described_object`, ...           | Grounded language understanding           |
| [LaserTag](docs/environments/lasertag.md) | 4     | `lasertag_one_opponent_small`, ...                | Procedural laser tag with bots            |
| [NatLab](docs/environments/natlab.md)     | 3     | `natlab_fixed_large_map`, ...                     | Mushroom foraging in naturalistic terrain |
| [SkyMaze](docs/environments/skymaze.md)   | 2     | `skymaze_irreversible_path_hard`, ...             | Irreversible platform navigation          |
| [PsychLab](docs/environments/psychlab.md) | 4     | `psychlab_visual_search`, ...                     | Cognitive psychology experiments          |
| [Explore](docs/environments/explore.md)   | 8     | `explore_goal_locations_large`, ...               | Maze exploration and object collection    |

Access level lists via `dmlab_gym.CORE_LEVELS`, `dmlab_gym.DMLAB30_LEVELS`, or `dmlab_gym.ALL_LEVELS`.

## Environment Options

All options can be passed as keyword arguments to `dmlab_gym.register()` or directly to `DmLabEnv`:

| Option          | Default               | Description                                             |
| --------------- | --------------------- | ------------------------------------------------------- |
| `renderer`      | `"software"`          | `"software"` (headless OSMesa) or `"hardware"` (OpenGL) |
| `width`         | `84`                  | Observation pixel width                                 |
| `height`        | `84`                  | Observation pixel height                                |
| `fps`           | `60`                  | Frames per second                                       |
| `max_num_steps` | `0`                   | Maximum steps per episode (0 = unlimited)               |
| `observations`  | `["RGB_INTERLEAVED"]` | Observation channels to request from the engine         |

## Wrappers

One custom wrapper included:

| Wrapper            | Description                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `ActionDiscretize` | Converts the 7D action space to `Discrete(9)` using the [IMPALA action set](https://arxiv.org/abs/1802.01561) |

```python
from dmlab_gym.wrappers import ActionDiscretize

env = ActionDiscretize(gym.make("dmlab_gym/lt_chasm-v0", renderer="software"))
env.action_space  # Discrete(9)
```

## Gymnasium Wrapper Compatibility

All standard [Gymnasium wrappers](https://gymnasium.farama.org/main/api/wrappers/) are compatible:

| Wrapper                   | Compatible | Notes                                                  |
| ------------------------- | ---------- | ------------------------------------------------------ |
| `GrayscaleObservation`    | Yes        | RGB to grayscale conversion                            |
| `ResizeObservation`       | Yes        | Resize to any resolution (requires `gymnasium[other]`) |
| `FrameStackObservation`   | Yes        | Stack N consecutive frames                             |
| `MaxAndSkipObservation`   | Yes        | Frame skipping with max pooling                        |
| `ReshapeObservation`      | Yes        | Reshape observation arrays                             |
| `RescaleObservation`      | Yes        | Best used after `DtypeObservation(float)`              |
| `DtypeObservation`        | Yes        | Convert uint8 to float32/float64                       |
| `FlattenObservation`      | Yes        | Flatten to 1D                                          |
| `NormalizeObservation`    | Yes        | Running mean/std normalisation                         |
| `TransformObservation`    | Yes        | Custom observation function                            |
| `TimeLimit`               | Yes        | Truncate after N steps                                 |
| `ClipReward`              | Yes        | Bound rewards to a range                               |
| `NormalizeReward`         | Yes        | Normalise rewards via running stats                    |
| `TransformReward`         | Yes        | Custom reward function                                 |
| `RecordEpisodeStatistics` | Yes        | Track episode returns and lengths                      |
| `RecordVideo`             | No         | No render modes supported                              |
| `HumanRendering`          | No         | No render modes supported                              |
| `FilterObservation`       | No         | Box observation space (not Dict)                       |
| `ClipAction`              | Yes        | Clips actions to Box bounds                            |
| `RescaleAction`           | No         | Float rescaling produces NaN with integer action space |

## License

This project is a derivative work of DeepMind Lab and is distributed under the same licenses:

- **GPLv2** -- `engine/`, `q3map2/`, `assets_oa/`
- **CC BY 4.0** -- `assets/`
- **Custom academic license** -- `engine/code/tools/lcc/`
- **BSD 3-Clause** -- `q3map2/libs/{ddslib,picomodel}`

See [LICENSE](LICENSE) for full details.

## Attribution

Based on [DeepMind Lab](https://github.com/google-deepmind/lab) by DeepMind.
