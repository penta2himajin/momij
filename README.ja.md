# momij

> Source: README.md @ 75f8598954ef799d0891b3664ecdae5701e3fdba

[English](./README.md)

Maple-Preview 専用の高速推論エンジン（Apple Silicon / M1 Max）。
**oMLX の Maple 置き換え**を想定した OpenAI 互換サーバを第1成果物とする。

## 方針

- **スタック**: Swift + raw Metal（Qwisp Seedless 型）。MLX はロード／正準 greedy 基板。
- **モデル**: `~/models/deepgrove/maple-preview-2bit-mlx`
- **参照**: `qwisp`（Seedless / Tell）、`mlx-lm-deepgrove`（正準 Maple）

詳細は [docs/research-catalog.md](docs/research-catalog.md) と [docs/baseline.md](docs/baseline.md)。

## oMLX Maple 差し替え（evprtr）

Maple 向けに oMLX の代わりへ momij を向ける例:

```bash
swift run -c release momij -- serve --model ~/models/deepgrove/maple-preview-2bit-mlx --port 8742
# EVPRTR_UPSTREAM_BASE_URL=http://127.0.0.1:8742/v1
```

契約の要点:

- 既定: `127.0.0.1:8742`。`--model-id` が `/v1/models` とレスポンス `model`（既定 `maple-preview`）。
- 余分なリクエスト欄（`tools` / `tool_choice` / `response_format` など）は**無視**して生成続行。
- 制約デコード（P1）: `structured_outputs.choice`、リテラル和の `grammar` / `guided_grammar`、加えて **xgrammar** による `structured_outputs.json`・フル EBNF・非リテラル `regex`。トップレベル `grammar` は無視（oMLX と同じ）。
- serve では HF `chat_template` **必須**（byte tokenizer フォールバックなし）。失敗時は 5xx / 起動失敗。
- `<tool_call>` を含む assistant 本文は `choices[0].message.content`。ネイティブ `tool_calls` は不要。
- `Authorization` は無視（401 にしない）。同時実行はプロセス内ロックで逐次。

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
