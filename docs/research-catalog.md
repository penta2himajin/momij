# Research catalog — Maple / momij speed levers

Policy: Swift + raw Metal (Qwisp-style); OpenAI server as oMLX Maple replacement.
Baseline: ~182 tok/s exact on M1 Max (see [baseline.md](baseline.md)).

## P0 — shipping path

| Lever | Status in momij | Notes |
|---|---|---|
| Raw Metal 1-CB decode (Seedless) | `SeedlessMetal` kernels landed; full-layer 1-CB wiring next | Qwisp notes/01: GPU ~35% busy = sync floor |
| Fused ternary expert block | `maple_fused_expert_topk` + `seedless-bench` | BitNet TL/I2_S inspiration; Metal rewrite |
| MLX exact greedy | `MapleEngine` / `MapleMLXBackend` | Correctness path |
| Oracle (= mlx-lm-deepgrove) | `OracleBackend` + `python/momij_oracle/worker.py` | Matches published baseline |

## P1 — serve / agentic

| Lever | Status | Notes |
|---|---|---|
| SuffixSpec / Tell | `SuffixSpec` + `MOMIJ_SUFFIX_SPEC=1` | No draft training; helps repetitive content |
| OpenAI API + continuous batch | `/v1/chat/completions` streaming; CB deferred | oMLX drop-in surface |
| P-EAGLE / DFlash | **deferred — needs draft training** | Maple has `num_nextn_predict_layers=0`. Draft must be ≫2.5× target speed on AS ([AtomGradient](https://atomgradient.github.io/apple-silicon-llm-inference/paper.pdf)). Sources: [P-EAGLE](https://arxiv.org/abs/2602.01469), [DFlash](https://arxiv.org/abs/2602.06036) |

## P2 — revisit later

| Lever | Why deferred |
|---|---|
| FlashAttention / FlashDecoding | SWA≤512 dominates; attn not primary decode cost on M1 Max today |
| FlashHead | ±0 vs exact at baseline; remeasure after Seedless raises effective BW |
| Expert streaming / bolt | Model ~7GB fits 64GB |
| Routing prediction across layers | Hard; Qwisp saw ~82% ceiling |
| External dense draft SD | Speed-ratio gate fails when MoE target is already fast |

## Sources

- Qwisp `notes/01-speedup-investigation.md`
- DeepGrove `mlx-lm-deepgrove` README (FlashHead M4/M5 numbers)
- [Bitnet.cpp](https://aclanthology.org/2025.acl-long.457.pdf)
- [P-EAGLE](https://vllm.ai/blog/2026-03-13-p-eagle)
- [DFlash](https://arxiv.org/abs/2602.06036)
- [AtomGradient Apple Silicon SD](https://atomgradient.github.io/apple-silicon-llm-inference/paper.pdf)
