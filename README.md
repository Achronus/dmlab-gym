# dmlab-gym

A fork of [DeepMind Lab](https://github.com/google-deepmind/lab) updated for Bazel 8, Python 3.13, and wrapped with [Gymnasium](https://gymnasium.farama.org/) via [Shimmy](https://shimmy.farama.org/).

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
