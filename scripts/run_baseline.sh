#!/usr/bin/env bash
# Reproduce docs/baseline.md numbers. Stops omlx for the duration.
set -euo pipefail
MODEL="${MOMIJ_MODEL:-$HOME/models/deepgrove/maple-preview-2bit-mlx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
omlx stop 2>/dev/null || true
cleanup() { omlx start 2>/dev/null || true; }
trap cleanup EXIT
cd "$ROOT/swift"
swift build -c release
BIN="$ROOT/swift/.build/arm64-apple-macosx/release/momij"
echo "=== oracle exact ==="
"$BIN" bench --backend oracle --model "$MODEL" -p 128 -g 256 -n 3
echo "=== oracle flash-head ==="
"$BIN" bench --backend oracle --model "$MODEL" --flash-head -p 128 -g 256 -n 3
echo "=== seedless metal ==="
"$BIN" seedless-bench --model "$MODEL"
