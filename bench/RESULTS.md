# momij bench notes

## 2026-09-09 momij MLX → oracle parity

**Oracle architecture:** MLX lazy graph + fused `mx.fast.metal_kernel` ops + `async_eval` — **not** Seedless 1-CB.

M1 Max, AC, omlx stop, p128/g128:

| Step | momij MLX decode tok/s |
|---|---|
| Before | ~97 |
| + asyncEval / fused add-norm / fused router / chunked KV | ~121–143 |
| + fused qkv + qk-norm-rope (oracle maple.py) | **~143–145** (warm) |
| oracle same windows | **~170–186** |

Oracle-faithful ports landed; residual **~20–25%**. bf16-as-oracle was **slower** on this Mac (keep f16). `mlx.compile` / QuantizedEmb / deep-async chain regress. Remaining gap looks like Swift vs Python MLX graph fusion — proceed to **Seedless 1-CB** to beat oracle rather than chase the last MLX percent.

## 2026-09-09 Phase profile (M1 Max, AC, omlx stop, p64/g32 n=1)

Forced `mx.eval` / Metal `waitUntilCompleted` at phase boundaries (`MOMIJ_PROFILE_MOE=1`).
Natural e2e is a separate run without per-MoE sync.

| Path | Natural decode tok/s | Sync phase (ms / decode MoE or attn call) |
|---|---|---|
| oracle (mlx-lm-deepgrove) | **182** | router **0.45** / switch **0.77** / agg **0.39** → moe **1.61**; attn **0.86** |
| momij MLX | **97** (then raised — see above) | moe (eval each) **1.37** → e2e collapses to ~30 |
| hybrid Seedless MoE | **~24–32** | router **0.92** / metal **0.62** / host in+out **~0.02** |
| Seedless micro (real weights) | — | fused-expert **~1760 steps/s ≈ 0.57 ms** |

Interpretation (measured, not guessed):

1. Host activation copy is negligible (~0.02 ms).
2. Hybrid loses to per-layer dual-runtime sync (MLX router eval + Metal wait), not to the ternary kernel.
3. Under sync, momij Metal experts (~0.62 ms) beat oracle switch (~0.77 ms); momij router (~0.92 ms) is ~2× oracle fused router (~0.45 ms).
4. Oracle/MLX natural speed comes from amortizing sync across a fused graph; forcing per-MoE eval drops MLX to ~30 tok/s (same band as hybrid).
5. momij MLX is still behind oracle at natural e2e — close that gap before expecting Seedless e2e wins.

## 2026-09-09 Seedless Metal microbench (M1 Max, release)

```
seedless gqmm2 (H=2048→N=512,Ktop=1) kernel/s ≈ 2800
seedless fused-expert (E=256,K=8) steps/s ≈ 1700
seedless fused-expert real-weights steps/s ≈ 1600–2300
```

## 2026-09-09 Seedless Milestone A — MoE block 1-CB

`SeedlessMetal.moeBlockOneCB`: postNorm RMS + dense gate (TG-reduce) + top-8 + fused experts + resid, **one wait**. No host inds/scores roundtrip.

```
seedless moe-block-1cb (L0) blocks/s ≈ 1400  →  ~58 tok/s floor @24 layers (MoE only, no attn)
```

## 2026-09-09 Seedless Milestone A — why ~50 tok/s (profile)

M1 Max, AC, omlx stop. GPU time = `MTLCommandBuffer.gpuEndTime - gpuStartTime`. Timed loops have **no host memcpy**.

| Probe | wall ms | GPU ms | busy |
|---|---|---|---|
| empty CB wait | 0.023 | 0 | — |
| full MoE block (1 CB) | 0.786 | 0.521 | 66% |
| rms / gate / route (own CB) | 0.31 / 0.36 / 0.41 | 0.078 / 0.128 / 0.183 | launch-dominated |
| fused expert | 0.506 | 0.285 | 56% |
| gqmm2 up / down | 0.41 / 0.41 | 0.191 / 0.192 | 46% |
| **24× same block, 1 CB** | **4.29** | **3.94** | **92%** |

Streamed ≈ 8.1 MB/layer (8 experts × 2-bit up+down + dense gate) → **15.6 GB/s** on 1-layer GPU time, **~50 GB/s** on 24L-1CB (M1 Max peak ~400). Not bandwidth-bound.

MoE-only tok/s floor @24L: per-layer wait **~53** · 1-layer GPU **~80** · 24L-1CB **~254**.

Cause: short-CB launch/sync tax + batch=1 gather occupancy, **not** host copy (empty wait 23 µs) and **not** missing 1-CB on a single layer. Oracle e2e ~182 tok/s is a fused MLX graph; same kernels under per-op sync were already in the same band as hybrid.

## 2026-09-09 Seedless 24L MoE 1-CB re-measure (real weights)

`SeedlessMoEStack`: all 24 layers' real weights, **per-layer private scratch**, shared residual `h`, encode into **one** CB. Same process — relative numbers are the claim.

| Probe | wall ms | GPU ms | busy | MoE-only tok/s (wall / gpu) |
|---|---|---|---|---|
| 1L (L0) | 0.43 | 0.12 | 29% | — |
| 24× L0 repeat, 1 CB | 2.94 | 2.60 | 88% | 340 / 385 |
| **24L real, 1 CB** | **3.34** | **2.86** | **86%** | **300 / 349** |
| 24L × per-layer wait | 10.57 | 3.71 | 35% | 95 / 269 |

- Streamed ≈ 195 MB/token → **68 GB/s** on 24L-real GPU (peak ~400) — still occupancy/dispatch, not BW.
- Per-layer wait wall is **~3.2×** 1-CB wall; GPU busy rises 35% → 86% when layers share one CB.
- Real-weight 24L ≈ L0-repeat (2.86 vs 2.60 ms GPU) → earlier L0-stack proxy was not a hazard artifact.
- MoE-only 1-CB floor **~300 tok/s wall** leaves headroom vs oracle e2e ~180 **before** attn/embed/lm_head; next is Milestone B (attn in the same CB).

## 2026-09-09 Seedless Milestone B — attn+MoE (pos=0)

Kernels: `maple_qk_norm_rope`, `maple_write_kv`, `maple_sdpa_d128` + existing `gqmm2` for fused qkv/o. `SeedlessLayerStack` loads all 24 real layers.

| Probe | wall ms | GPU ms | busy | tok/s (wall / gpu) |
|---|---|---|---|---|
| moe-only L0 | 0.72 | 0.43 | 60% | — |
| attn+resid L0 | 0.60 | 0.29 | 49% | — |
| layer L0 (attn+MoE) | 1.20 | 0.92 | 76% | — |
| 24L **one encoder** | 17.4 | 6.3 | 36% | 58 / 158 — CPU encode-bound |
| **24L commit→1 wait** | **5.49** | **5.07** | **92%** | **182 / 197** |
| 24L × per-layer wait | 10.5 | 4.4 | 42% | 95 / 225 |

Claim: packing 24 full layers into **one** encoder is wrong for wall time (encode tax). Right pattern is **per-layer CB commit, single final wait** → MoE+attn decode floor **~182 tok/s wall** at pos=0 (N=1), matching oracle e2e band before embed/lm_head/growing KV. Next: wire into `MapleEngine`, SWA rotate, longer `pos`.

## Baseline (oracle / mlx-lm-deepgrove)

See `docs/baseline.md` (~182 tok/s exact decode). Recheck same day after omlx stop:
`generation_tps ≈ 169` (p128/g128, n=2) — same band.

## OpenAI server smoke (oracle backend, port 8742)

```
GET  /healthz → ok
GET  /v1/models → maple-preview
POST /v1/chat/completions → assistant tokens
```

Drop-in for Maple vs oMLX: point clients at `http://127.0.0.1:8742/v1`.
