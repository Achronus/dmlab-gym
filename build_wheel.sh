#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-/tmp/dmlab_pkg}"

# Detect container runtime
if command -v podman &>/dev/null; then
    RUNTIME=podman
    VOLUME_FLAG=":Z"
elif command -v docker &>/dev/null; then
    RUNTIME=docker
    VOLUME_FLAG=""
else
    echo "Error: neither podman nor docker found" >&2
    exit 1
fi

echo "Using container runtime: $RUNTIME"

mkdir -p "$OUTPUT_DIR"

"$RUNTIME" build -t dmlab-builder -f "$SCRIPT_DIR/Dockerfile.build" "$SCRIPT_DIR"

"$RUNTIME" run --rm \
    -v "$SCRIPT_DIR:/build/lab${VOLUME_FLAG}" \
    -v "$OUTPUT_DIR:/output${VOLUME_FLAG}" \
    dmlab-builder \
    bash -c "
        bazel build -c opt //python/pip_package:build_pip_package \
            --define headless=osmesa && \
        ./bazel-bin/python/pip_package/build_pip_package /output
    "

echo "Wheel built in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.whl
