# momij

Maple-Preview 専用の高速推論エンジン（Apple Silicon / M1 Max）。
**oMLX の Maple 置き換え**を想定した OpenAI 互換サーバを第1成果物とする。

## 方針

- **スタック**: Swift + raw Metal（Qwisp Seedless 型）。MLX はロード／正準 greedy 基板。
- **モデル**: `~/models/deepgrove/maple-preview-2bit-mlx`
- **参照**: `qwisp`（Seedless / Tell）、`mlx-lm-deepgrove`（正準 Maple）

詳細は [docs/research-catalog.md](docs/research-catalog.md) と [docs/baseline.md](docs/baseline.md)。

## Setup

```bash
git config core.hooksPath git-hooks
cd swift
swift build -c release
```

Oracle ベンチ（mlx-lm-deepgrove）には DeepGrove venv が必要:

```bash
# already at ~/repos/mlx-lm-deepgrove/.venv
```

## Commands

```bash
# Baseline-compatible bench (stops competing GPU users yourself)
omlx stop   # if running
swift run -c release momij -- bench --model ~/models/deepgrove/maple-preview-2bit-mlx -p 128 -g 256 -n 3
swift run -c release momij -- bench --model ~/models/deepgrove/maple-preview-2bit-mlx --flash-head -p 128 -g 256 -n 3

# Seedless Metal microbench
swift run -c release momij -- seedless-bench --model ~/models/deepgrove/maple-preview-2bit-mlx

# OpenAI server (default port 8742) — point evprtr / clients here instead of oMLX for Maple
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
