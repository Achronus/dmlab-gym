# dmlab-gym

A fork of [DeepMind Lab](https://github.com/google-deepmind/lab) updated for Bazel 8, Python 3.13, and wrapped with [Gymnasium](https://gymnasium.farama.org/) via [Shimmy](https://shimmy.farama.org/).

## What This Is

DeepMind Lab is a 3D learning environment based on id Software's Quake III Arena. This fork modernizes the build toolchain so it compiles and runs on current systems:

- **Bazel 8** with `--enable_workspace` compatibility
- **Python 3.13** C extension and package support
- **Gymnasium/Shimmy** wrapper for standard RL interfaces

The engine builds inside a Podman container. The resulting wheel installs into a host-side `uv` environment.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python environment management)
- [Podman](https://podman.io/) or [Docker](https://www.docker.com/) (container builds)
- Python 3.13+
- OSMesa (runtime dependency for headless rendering)

Install OSMesa for your distribution:

| Distribution | Command |
|---|---|
| Fedora / RHEL | `sudo dnf install mesa-libOSMesa` or `mesa-compat-libOSMesa` |
| Bluefin / immutable | `rpm-ostree install mesa-compat-libOSMesa` (reboot required) |
| Ubuntu / Debian | `sudo apt install libosmesa6` |
| Arch | `sudo pacman -S mesa` |

## Building

### Quick Start

```bash
./build_wheel.sh              # builds to /tmp/dmlab_pkg/
./build_wheel.sh ~/my_output  # or specify a custom output directory
```

The script auto-detects Podman or Docker and handles the container build + wheel packaging.

### Install the Wheel

```bash
uv pip install /tmp/dmlab_pkg/deepmind_lab-1.0-py3-none-any.whl
```

### Verify

```bash
uv run python -c "import deepmind_lab; print('OK')"
```

### Manual Build

If you prefer to run the steps yourself:

**Podman:**

```bash
podman build -t dmlab-builder -f Dockerfile.build .

mkdir -p /tmp/dmlab_pkg
podman run --rm \
    -v ./:/build/lab:Z \
    -v /tmp/dmlab_pkg:/output:Z \
    dmlab-builder \
    bash -c "
        bazel build -c opt //python/pip_package:build_pip_package \
            --define headless=osmesa && \
        ./bazel-bin/python/pip_package/build_pip_package /output
    "
```

**Docker:**

```bash
docker build -t dmlab-builder -f Dockerfile.build .

mkdir -p /tmp/dmlab_pkg
docker run --rm \
    -v ./:/build/lab \
    -v /tmp/dmlab_pkg:/output \
    dmlab-builder \
    bash -c "
        bazel build -c opt //python/pip_package:build_pip_package \
            --define headless=osmesa && \
        ./bazel-bin/python/pip_package/build_pip_package /output
    "
```

## Project Structure

| Directory | Description |
|-----------|-------------|
| `engine/` | ioquake3 game engine (GPLv2) |
| `q3map2/` | Map compilation tools (GPLv2) |
| `assets/` | DeepMind Lab assets (CC BY 4.0) |
| `assets_oa/` | Open Arena assets (GPLv2) |
| `game_scripts/` | Lua game/level scripts |
| `python/` | Python C extension and pip package |
| `deepmind/` | Core DeepMind Lab code |
| `public/` | Public API headers |
| `third_party/` | Vendored dependencies |
| `bazel/` | Bazel build configuration |
| `testing/` | Test utilities |
| `examples/` | Example agent code |

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

## Upstream Sources

DeepMind Lab is built from the following open source projects:

- [ioquake3](https://github.com/ioquake/ioq3) -- game engine
- [bspc](https://github.com/TTimo/bspc) -- bot route compilation
- [GtkRadiant / q3map2](https://github.com/TTimo/GtkRadiant) -- map compilation
