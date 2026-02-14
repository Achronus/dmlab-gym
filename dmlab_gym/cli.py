"""CLI entry point for dmlab-gym."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Default output directory for the wheel.
DEFAULT_OUTPUT_DIR = "/tmp/dmlab_pkg"

# Container image name used for building.
IMAGE_NAME = "dmlab-builder"


def _find_runtime() -> tuple[str, str]:
    """Detect podman or docker and return (runtime, volume_flag)."""
    for name, flag in [("podman", ":Z"), ("docker", "")]:
        if not shutil.which(name):
            continue
        # Check the daemon/service is actually running.
        result = subprocess.run(
            [name, "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return name, flag
        print(
            f"Warning: {name} is installed but not running, skipping.",
            file=sys.stderr,
        )

    print(
        "Error: no working container runtime found.\n"
        "Install and start podman or docker, then try again.",
        file=sys.stderr,
    )
    sys.exit(1)


def _project_root() -> Path:
    """Return the project root (where Dockerfile.build lives)."""
    # Walk up from this file to find the repo root.
    path = Path(__file__).resolve().parent.parent
    if (path / "Dockerfile.build").exists():
        return path
    # Fallback: cwd
    cwd = Path.cwd()
    if (cwd / "Dockerfile.build").exists():
        return cwd
    print(
        "Error: cannot find Dockerfile.build. "
        "Run this from the dmlab-gym repo root.",
        file=sys.stderr,
    )
    sys.exit(1)


def _ensure_osmesa() -> None:
    """Check for libOSMesa and install it transiently if missing."""
    import ctypes

    try:
        ctypes.CDLL("libOSMesa.so.8")
        return
    except OSError:
        pass

    print("libOSMesa.so.8 not found. Attempting transient install...")

    # Detect the package manager and install command.
    if shutil.which("dnf"):
        cmd = ["sudo", "dnf", "install", "-y", "--transient", "mesa-compat-libOSMesa"]
    elif shutil.which("apt-get"):
        cmd = ["sudo", "apt-get", "install", "-y", "libosmesa6"]
    elif shutil.which("pacman"):
        cmd = ["sudo", "pacman", "-S", "--noconfirm", "mesa"]
    else:
        print(
            "Error: libOSMesa.so.8 is missing and no supported package manager found.\n"
            "Install OSMesa manually (see README).",
            file=sys.stderr,
        )
        sys.exit(1)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Error: failed to install OSMesa.", file=sys.stderr)
        sys.exit(1)

    # Verify it actually worked.
    try:
        ctypes.CDLL("libOSMesa.so.8")
        print("OSMesa installed successfully.")
    except OSError:
        print(
            "Error: OSMesa was installed but libOSMesa.so.8 still cannot be loaded.\n"
            "You may need to run ldconfig or restart your shell.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_build(args: argparse.Namespace) -> None:
    """Build the deepmind-lab wheel inside a container and install it."""
    _ensure_osmesa()
    runtime, volume_flag = _find_runtime()
    root = _project_root()
    output = Path(args.output).resolve()

    print(f"Using container runtime: {runtime}")
    print(f"Project root: {root}")
    print(f"Output directory: {output}")

    output.mkdir(parents=True, exist_ok=True)

    # Build the container image.
    subprocess.run(
        [runtime, "build", "-t", IMAGE_NAME, "-f", str(root / "Dockerfile.build"), str(root)],
        check=True,
    )

    # Run the build inside the container.
    subprocess.run(
        [
            runtime, "run", "--rm",
            "-v", f"{root}:/build/lab{volume_flag}",
            "-v", f"{output}:/output{volume_flag}",
            IMAGE_NAME,
            "bash", "-c",
            "bazel build -c opt //python/pip_package:build_pip_package "
            "--define headless=osmesa && "
            "./bazel-bin/python/pip_package/build_pip_package /output",
        ],
        check=True,
    )

    wheels = list(output.glob("*.whl"))
    if not wheels:
        print("Error: no wheel found in output directory", file=sys.stderr)
        sys.exit(1)

    wheel = wheels[0]
    print(f"\nWheel built: {wheel}")

    # Install the wheel.
    if args.no_install:
        print("Skipping install (--no-install).")
        return

    print("Installing wheel...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)],
        check=True,
    )
    print("Done. deepmind-lab is installed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dmlab-gym",
        description="DeepMind Lab build and management tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser(
        "build",
        help="Build and install the deepmind-lab wheel via container.",
    )
    build_parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the wheel (default: {DEFAULT_OUTPUT_DIR}).",
    )
    build_parser.add_argument(
        "--no-install",
        action="store_true",
        help="Only build the wheel, do not install it.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        cmd_build(args)
