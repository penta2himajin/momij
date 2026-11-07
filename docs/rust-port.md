# Rust port (llama.cpp backend)

Branch `human/rust-rewrite`. The Rust workspace (`Cargo.toml`, `rust/`) re-implements everything in
`swift/` **except model execution**. The Swift Seedless/Metal and MLX engines stay where they are;
the Rust server instead talks to the official DeepGrove `llama.cpp` fork over HTTP.

## Run it

```bash
# 1. llama-server from https://github.com/deepgrove-ai/llama.cpp (Maple architecture lives only there)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF   # Metal is fine on Apple Silicon
cmake --build build -j --target llama-server
./build/bin/llama-server -m maple-preview-TQ2_0-head-Q4_K.gguf --jinja -np 1 -c 16384 --port 8080

# 2. momij in front of it
cargo run --release -p momij -- serve --llama-url http://127.0.0.1:8080 --port 8742
cargo run --release -p momij -- generate --prompt "The capital of France is"
cargo run --release -p momij -- bench -p 128 -g 256 -n 5
```

`MOMIJ_LLAMA_URL` replaces `--llama-url`. `--jinja` is required (`/apply-template`). Env knobs
(`MOMIJ_WORKSPACE`, `MOMIJ_PROTOCOL_TOOLS`, `MOMIJ_ENABLE_THINKING`, `MOMIJ_VERIFY_SANITIZE`,
`MOMIJ_TRACE_DIR`) behave as before but are read once at startup; `MOMIJ_CONCURRENCY` (default 1)
sets how many model calls run at once.

Checks: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.
Live check against a running server: `MOMIJ_LLAMA_URL=... cargo test -p momij -- --ignored`.

## Map

| Swift | Rust |
|---|---|
| `OpenAIChatCompat`, `StructuredOutputs`, `ToolMarkup`, `ArgRepair`, `PathPolicy`, `LoopBreaker`, `ProtocolAbsorb`, `ResponseVerify`, `ChatTemplatePatch`, `TraceStore` | `momij-core` modules of the same names |
| `LogitsProcessor`, `SpeculativeSampling`, `SuffixSpec`, `SuffixDraftIndex`, `TokenRecycleIndex`, `GenerateOptions` | `momij-core` (pure algorithms) |
| `Server.swift` (router, repair pipeline, SSE) | `momij::{server, pipeline, trace}` |
| `MomijCLI.swift` (`serve`, `generate`, `bench`) | `momij` binary |
| `SeedlessBackend`, `MapleMLXBackend`, `OracleBackend` | `momij::llama::LlamaServerBackend` |
| `XGrammarGuide` | llama.cpp samplers (`json_schema`, GBNF) |

Not ported (model execution, stays in Swift): `Seedless*`, `Maple{Engine,Attention,Fused,MoE,SeedlessMoE}`,
`QuantProj`, `WeightStore`, `AttentionMasks`, `seedless-bench`, and the Metal/MLX tests.

## Facts checked against the real Maple TQ2_0 GGUF

- Token ids match the Swift constants: `<think>` 151667, `</think>` 151668, `<|im_start|>` 151644,
  `<|im_end|>` 151645, `<tool_call>` 151657, `</tool_call>` 151658.
- The stock template always opens the assistant turn with `<think>\n` (`/apply-template`), so the
  non-thinking default strips that suffix from the rendered prompt and bans `<think>` and
  `<|im_start|>` with `logit_bias`.
- The `/completion` stream includes the EOS token in `tokens`; `stop_type` is `eos` or `limit`.
- llama-server's own sampler defaults (`top_k 40`, `top_p 0.95`, `min_p 0.05`) would silently
  change sampling, so every request overrides them (`top_k 0`, `min_p 0`, explicit `top_p`).
- Structured outputs map to llama.cpp samplers: `choice` and literal unions -> GBNF, JSON schema ->
  `json_schema`, EBNF -> GBNF passthrough. **Regex is rejected with a 400**: llama.cpp has no regex field.

## Deliberate differences from Swift

| Where | Swift | Rust | Why |
|---|---|---|---|
| `ToolMarkup.rewriteMessages` | closes the call object with `"{"` | closes with `"}"` | typo since the first commit; emitted malformed JSON into the model-visible history and no test noticed |
| `jsonDumps` | unordered keys, keys unescaped, `/` escaped by `JSONSerialization` | insertion order, escaped keys, `/` not escaped | Python `json.dumps` parity is the documented contract |
| streamed `tool_calls` | every delta has `index: 0` | `index` = position | clients merge deltas by index, so parallel calls were fused |
| `testLenientJSONWithLiteralNewlines` | outside its `XCTestCase` class (never ran); inner quotes unescaped | real test with the intended payload, plus a test pinning the original input as unrecoverable | the original input cannot pass |
| `findRepetition` offsets | UTF-16 units | code points | Python oracle parity |
| frequency ties (suffix tree, `ConstrainedPick`) | unordered dictionary/set iteration | lowest token id | determinism |
| trace timestamps | `CFAbsoluteTime` (since 2001) | Unix epoch seconds | evprtr records `time.time()` |
| stream vs non-stream repair | two copies of the logic | one `resolve_turn`; `Mode` selects the existing divergences | see `pipeline::Mode` |

Divergences between the two Swift paths that were **kept** (they may be bugs; decide before removing):
broken-call drop (stream: whenever still broken; non-stream: only after a failed whole-reply repair, so a
call with a missing required field is dropped on stream but sent on non-stream), repair-nudge wording,
and the opt-in sanitize running on non-stream only.

## Not covered / follow-ups

- `useSuffixSpec`/`draftK` are carried in `GenerateOptions` but llama-server speculation is configured on
  the server; the ported `SuffixSpec`/`SpeculativeSampling` are library code, not wired into the HTTP path.
- `repetition_penalty` maps to `repeat_penalty`, which llama.cpp applies over its `repeat_last_n` window
  (prompt included); the Swift engine used generated tokens only.
- No TLS for `--llama-url` (plain HTTP only).
- Throughput numbers from this port are **not comparable** with `docs/baseline.md`: that file is M1 Max
  (MLX/Metal); the sandbox used here is a 4-vCPU x86 CPU build (about 31 tok/s decode, 133 tok/s prefill,
  p128/g32, n=2). Re-measure on the M1 Max with `omlx stop` first.
- `CLAUDE.md` still states "Production decode target = Swift raw Metal". The Rust port keeps that engine
  out of scope, but the boundary text has not been updated.
