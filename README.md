# momij

High-speed inference engine for Maple-Preview on Apple Silicon (M1 Max).
Primary product: an OpenAI-compatible server intended to **replace oMLX when serving Maple**.

## Approach

- **Stack**: Swift + raw Metal (Qwisp Seedless style). MLX is used for weight load and the canonical greedy reference path.
- **Model**: `~/models/deepgrove/maple-preview-2bit-mlx`
- **References**: `qwisp` (Seedless / Tell), `mlx-lm-deepgrove` (canonical Maple)

Details: [docs/research-catalog.md](docs/research-catalog.md) and [docs/baseline.md](docs/baseline.md).

## Setup

```bash
git config core.hooksPath git-hooks
cd swift
swift build -c release
```

Oracle benches (`mlx-lm-deepgrove`) need the DeepGrove venv:

```bash
# already at ~/repos/mlx-lm-deepgrove/.venv
```

## Commands

```bash
# Baseline-compatible bench (stop competing GPU users yourself)
omlx stop   # if running
swift run -c release momij -- bench --model ~/models/deepgrove/maple-preview-2bit-mlx -p 128 -g 256 -n 3
swift run -c release momij -- bench --model ~/models/deepgrove/maple-preview-2bit-mlx --flash-head -p 128 -g 256 -n 3

# Seedless Metal microbench
swift run -c release momij -- seedless-bench --model ~/models/deepgrove/maple-preview-2bit-mlx

# OpenAI server (default port 8742) — point clients here instead of oMLX for Maple
swift run -c release momij -- serve --model ~/models/deepgrove/maple-preview-2bit-mlx --port 8742

# SuffixSpec (env or flag)
MOMIJ_SUFFIX_SPEC=1 swift run -c release momij -- serve --model ~/models/deepgrove/maple-preview-2bit-mlx
```

## Build & Test

```bash
cd swift
swift build
swift test
```

## License

MIT. See `LICENSE`.
