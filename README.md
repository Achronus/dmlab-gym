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
- [Podman](https://podman.io/) (container builds)
- Python 3.13+

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
