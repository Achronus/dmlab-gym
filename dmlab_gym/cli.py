"""CLI entry point for dmlab-gym."""

import argparse
import shutil
import subprocess
import sys
import threading
from pathlib import Path

# Default output directory for the wheel (relative to cwd).
DEFAULT_OUTPUT_DIR = "./lab"

# Container image name used for building.
IMAGE_NAME = "dmlab-builder"

# Source repo for cloning.
REPO_URL = "https://github.com/Achronus/dmlab-gym.git"

# Cached clone location.
CACHE_DIR = Path.home() / ".local" / "share" / "dmlab-gym"

# Spinner characters.
_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _run_with_spinner(
    cmd: list[str],
    label: str,
    *,
    capture_errors: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command with a spinner, suppressing stdout.

    Stderr is captured and only shown if the command fails.
    """
    stop = threading.Event()

    def spin():
        i = 0
        while not stop.is_set():
            print(f"\r  {_SPINNER[i % len(_SPINNER)]} {label}", end="", flush=True)
            i += 1
            stop.wait(0.1)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_errors else subprocess.DEVNULL,
        text=True,
    )

    stop.set()
    spinner.join()

    if result.returncode != 0:
        print(f"\r  x {label} — failed", flush=True)
        if capture_errors and result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print(f"\r  ✓ {label}", flush=True)
    return result


def _find_runtime() -> tuple[str, str]:
    """Detect podman or docker and return (runtime, volume_flag)."""
    for name, flag in [("podman", ":Z"), ("docker", "")]:
        if not shutil.which(name):
            continue
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
    """Return the project root (where Dockerfile.build lives).

    Search order:
      1. Parent of this file (editable / dev install).
      2. Current working directory.
      3. Cached clone at ~/.local/share/dmlab-gym/source.
         Clones the repo if not present, pulls if it is.
    """
    # 1. Dev install — source tree contains the package.
    path = Path(__file__).resolve().parent.parent
    if (path / "Dockerfile.build").exists():
        return path

    # 2. CWD is the repo root.
    cwd = Path.cwd()
    if (cwd / "Dockerfile.build").exists():
        return cwd

    # 3. Cached clone.
    source = CACHE_DIR / "source"
    if (source / "Dockerfile.build").exists():
        _run_with_spinner(
            ["git", "-C", str(source), "pull", "--ff-only"],
            "Updating cached source",
        )
        return source

    # Clone fresh.
    if not shutil.which("git"):
        print(
            "Error: cannot find the dmlab-gym source tree and git is not "
            "installed to clone it.\nEither run from the repo root or "
            "install git.",
            file=sys.stderr,
        )
        sys.exit(1)

    source.parent.mkdir(parents=True, exist_ok=True)
    _run_with_spinner(
        ["git", "clone", "--depth", "1", REPO_URL, str(source)],
        "Cloning dmlab-gym source",
    )

    return source


def _ensure_osmesa() -> None:
    """Check for libOSMesa and install it transiently if missing."""
    import ctypes

    try:
        ctypes.CDLL("libOSMesa.so.8")
        return
    except OSError:
        pass

    print("libOSMesa.so.8 not found. Attempting transient install...")

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

    output.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding deepmind-lab ({runtime})")
    print(f"  Source: {root}")
    print(f"  Output: {output}\n")

    # Build the container image.
    _run_with_spinner(
        [runtime, "build", "-t", IMAGE_NAME, "-f", str(root / "Dockerfile.build"), str(root)],
        "Building container image",
    )

    # Run the build inside the container.
    _run_with_spinner(
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
        "Compiling native extension (this may take a while)",
    )

    wheels = list(output.glob("*.whl"))
    if not wheels:
        print("Error: no wheel found in output directory", file=sys.stderr)
        sys.exit(1)

    wheel = wheels[0]

    # Install the wheel.
    if args.no_install:
        print(f"\nWheel built: {wheel}")
        return

    if shutil.which("uv"):
        install_cmd = ["uv", "pip", "install", "--upgrade", "--force-reinstall", str(wheel)]
    else:
        install_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", str(wheel)]

    _run_with_spinner(install_cmd, "Installing wheel")

    # Verify the import works.
    result = subprocess.run(
        [sys.executable, "-c", "import deepmind_lab"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        print("  ✓ deepmind_lab is importable")
    else:
        print("  x deepmind_lab import failed", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print(f"\n  Done! deepmind-lab installed from {wheel.name}")


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
