#!/usr/bin/env bash
# Copy mlx-swift_Cmlx.bundle / default.metallib next to built momij so MLX loads.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer a local Xcode-built bundle from qwisp if present; else leave a clear error.
CANDIDATES=(
  "$HOME/repos/qwisp/swift/.xcode-build-rel/Build/Products/Release/mlx-swift_Cmlx.bundle"
  "$HOME/repos/qwisp/swift/.xcode-build/Build/Products/Debug/mlx-swift_Cmlx.bundle"
)
SRC=""
for c in "${CANDIDATES[@]}"; do
  if [[ -d "$c" ]]; then SRC="$c"; break; fi
done
if [[ -z "$SRC" ]]; then
  echo "error: no mlx-swift_Cmlx.bundle found (build qwisp with Xcode once, or install mlx metallib)" >&2
  exit 1
fi
for d in debug release; do
  DEST="$ROOT/swift/.build/arm64-apple-macosx/$d"
  [[ -d "$DEST" ]] || continue
  rm -rf "$DEST/mlx-swift_Cmlx.bundle"
  cp -R "$SRC" "$DEST/"
  cp "$SRC/Contents/Resources/default.metallib" "$DEST/default.metallib"
  echo "installed metallib → $DEST"
done
