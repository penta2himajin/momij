# momij

[日本語](./README.ja.md)

High-speed inference engine for Maple-Preview on Apple Silicon (M1 Max).
Primary product: an OpenAI-compatible server intended to **replace oMLX when serving Maple**.

## Approach

- **Stack**: Swift + raw Metal (Qwisp Seedless style). MLX is used for weight load and the canonical greedy reference path.
- **Model**: `~/models/deepgrove/maple-preview-2bit-mlx`
- **References**: `qwisp` (Seedless / Tell), `mlx-lm-deepgrove` (canonical Maple)

Details: [docs/research-catalog.md](docs/research-catalog.md) and [docs/baseline.md](docs/baseline.md).

## oMLX Maple drop-in (evprtr)

Point OpenAI clients (e.g. evprtr) at momij instead of oMLX for Maple:

```bash
swift run -c release momij -- serve --model ~/models/deepgrove/maple-preview-2bit-mlx --port 8742
# EVPRTR_UPSTREAM_BASE_URL=http://127.0.0.1:8742/v1
```

Contract notes:

- Default listen: `127.0.0.1:8742` (`--host` / `--port`). `--model-id` sets `/v1/models` and response `model` (default `maple-preview`).
- Extra request fields (`tools`, `tool_choice`, `response_format`, …) are **ignored**; generation continues.
- Constrained decode (P1, oMLX insurance): `structured_outputs.choice`, literal-union `grammar` / `guided_grammar`, plus **xgrammar** for `structured_outputs.json`, full EBNF, and non-literal `regex`. Top-level `grammar` stays ignored (oMLX parity).
- HF `chat_template` is **required** on serve (no byte-tokenizer fallback). Template failure → 5xx / serve exit.
- Assistant text (including `<tool_call>` markup) is returned in `choices[0].message.content`. Native OpenAI `tool_calls` are not required.
- `Authorization` is ignored (no 401). One request at a time (in-process lock).

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

# SuffixSpec / sampled spec default on (`=0` to disable). Exact lm_head: `MOMIJ_EXACT_HEAD=1`
MOMIJ_SUFFIX_SPEC=0 swift run -c release momij -- serve --model ~/models/deepgrove/maple-preview-2bit-mlx
MOMIJ_EXACT_HEAD=1 swift run -c release momij -- serve --model ~/models/deepgrove/maple-preview-2bit-mlx
```

## Build & Test

```bash
cd swift
swift build
swift test
```

## License

MIT. See `LICENSE`.
