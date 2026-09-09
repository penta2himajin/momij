# Research catalog — Maple / momij speed levers

Policy: Swift + raw Metal (Qwisp-style); OpenAI server as oMLX Maple replacement.
Baseline: ~182 tok/s exact on M1 Max (see [baseline.md](baseline.md)).
Phase profile: [bench/RESULTS.md](../bench/RESULTS.md) (2026-09-09).

## Measured bottlenecks (2026-09-09)

| Finding | Evidence |
|---|---|
| Hybrid host copy is not the bottleneck | host in+out ~0.02 ms vs router+metal ~1.5 ms |
| Per-layer sync dominates hybrid | 24×~(0.92+0.62) ms ≈ e2e ~25–30 tok/s |
| Metal experts already beat oracle switch under sync | metal 0.62 ms vs switch 0.77 ms |
| momij MLX ≪ oracle before Seedless | 97 vs 182 tok/s natural |
| Graph fusion is why oracle/MLX look fast | per-MoE `eval` drops MLX to ~30 tok/s |

## Ordered plan (do in order)

1. **Parity: momij MLX → oracle (~97→~182 tok/s)**  
   Port Maple fused router / SwitchGLU details; match decode graph+eval cadence. Re-measure e2e.
2. **Drop host-bridged hybrid as the decode hot path**  
   Keep Metal experts for 1-CB; do not ship `MOMIJ_SEEDLESS_MOE` host roundtrip.
3. **P0 Seedless: full 1-CB decode**  
   attn + norms + router + experts (+ optional lm_head) in one CB — no mid-layer MLX↔Metal sync.
4. **Then raise expert/attn kernels** against the 1-CB ceiling (BW ~970 tok/s).
5. **P1 serve**: SuffixSpec, continuous batch, oMLX drop-in hardening.
6. **P1 draft (later)**: P-EAGLE/DFlash only after draft ≫2.5× target on AS.

## P0 — shipping path

| Lever | Status in momij | Notes |
|---|---|---|
| Raw Metal 1-CB decode (Seedless) | kernels landed; hybrid host bridge measured-and-rejected for hot path | Qwisp notes/01: GPU ~35% busy = sync floor |
| Fused ternary expert block | real-weight ~1760 steps/s | faster than oracle switch under sync |
| MLX exact greedy | `MapleEngine` ~97 tok/s | must close gap to oracle before claiming Seedless wins |
| Oracle (= mlx-lm-deepgrove) | `OracleBackend` + profile via `MOMIJ_PROFILE_MOE=1` | ~182 tok/s; phase timers in worker |

## P1 — serve / agentic

| Lever | Status | Notes |
|---|---|---|
| SuffixSpec / Tell | `SuffixSpec` + `MOMIJ_SUFFIX_SPEC=1` | No draft training; helps repetitive content |
| OpenAI API + continuous batch | `/v1/chat/completions` streaming; CB deferred | oMLX drop-in surface |
| P-EAGLE / DFlash | **deferred — needs draft training** | Maple has `num_nextn_predict_layers=0`. Draft must be ≫2.5× target on AS ([AtomGradient](https://atomgradient.github.io/apple-silicon-llm-inference/paper.pdf)). Sources: [P-EAGLE](https://arxiv.org/abs/2602.01469), [DFlash](https://arxiv.org/abs/2602.06036) |

## P2 — revisit later

| Lever | Why deferred |
|---|---|
| FlashAttention / FlashDecoding | SWA≤512; attn ~0.86 ms sync vs moe ~1.61 ms — secondary until 1-CB |
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
