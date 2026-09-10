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

1. **Parity: momij MLX → oracle** — **plateau ~143 vs ~180**  
   Landed oracle maple.py fused path (async, add-norm, router, qkv, qk-norm-rope, chunked KV).  
   Oracle is **not** 1-CB Seedless. Residual ~20–25% → move to Seedless rather than more MLX glue.
2. **Drop host-bridged hybrid as the decode hot path**  
   Keep Metal experts for 1-CB; do not ship `MOMIJ_SEEDLESS_MOE` host roundtrip.
3. **P0 Seedless: full decode without mid-token host sync** ← **Milestone B landed**  
   A/C MoE 1-CB done. B: attn+MoE with **per-layer commit → one wait** (~182 tok/s @pos=0).  
   Next: wire into `MapleEngine`, SWA rotate, growing `pos`; if e2e stalls, revisit Python-MLX track.
4. **Then raise expert/attn kernels** against the 1-CB ceiling (BW ~970 tok/s).
5. **P1 serve**: SuffixSpec, continuous batch, oMLX drop-in hardening.
6. **P1 draft (later)**: P-EAGLE/DFlash only after draft ≫2.5× target on AS.

## P0 — shipping path

| Lever | Status in momij | Notes |
|---|---|---|
| Raw Metal 1-CB decode (Seedless) | **e2e wired** `SeedlessDecodeEngine` | **~169 gen tok/s** vs oracle ~177 (p128/g128); L0 rel_l2 1.9e-4; next=layers/lm_head |
| Fused ternary expert block | real-weight ~1600–2300 steps/s | building block inside 1-CB |
| MLX exact greedy | `MapleEngine` ~143 tok/s | oracle-aligned fused kernels; residual vs Python MLX |
| Oracle (= mlx-lm-deepgrove) | MLX graph + metal_kernel (not 1-CB) | ~182 tok/s; Seedless MoE+attn now matches this band at cold pos=0 |

## P1 — serve / agentic

| Lever | Status | Notes |
|---|---|---|
| SuffixSpec / Tell | `SuffixSpec` + suffix trie + FlashHead-approx Token Recycling | Tree+global+`MOMIJ_SPEC_ALPHA`; recycle via FlashHead top-k (`MOMIJ_SPEC_RECYCLE`). Open-chat recycle-only ≈0.16 accept/gen — full-vocab TR / EAGLE still deferred |
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
