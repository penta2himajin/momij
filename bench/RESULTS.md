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

## 2026-09-09 Seedless e2e (embed + layers + lm_head + SWA/pos)

Connected: memcpy embed, Metal attn+MoE with SWA rotate + growing `offset`, final RMS on last CB, MLX 4-bit lm_head+argmax. L0 parity **rel_l2 ≈ 1.9e-4**.

M1 Max, AC, omlx stop, p128/g128 n=3 (seedless) / n=2 (oracle, mlx):

| Backend | prompt tok/s | generation tok/s |
|---|---|---|
| **seedless** | **170** | **168–169** |
| oracle | 655 | **177** |
| momij MLX | 673 | 142 |

Phase ms/tok (seedless gen): embed **0.001** · layers **4.7–4.8** · head(lm_head+argmax) **1.18** · sum ≈ 5.96 → ~168 tok/s.

Bottleneck now: **layers (~80%)** then **lm_head (~20%)**. Embed solved. Gap to oracle ~5% (~0.3 ms/tok). Next levers: faster SDPA/gqmm2 under growing N; Metal/FlashHead for lm_head.

## 2026-09-09 Seedless → 200 tok/s chase (FlashHead Metal)

M1 Max, fans max, omlx stop, p128/g128 n=5. Path: `layersPerCB=4`, FlashHead centroids fused into final CB (`maple_batched_gemv`), CPU top-k, Metal `maple_qmm4_gather` (1 TG/probe).

| Config | gen tok/s | layers ms | head ms |
|---|---|---|---|
| seedless probes=96 | **~187** | ~4.87 | ~0.48 |
| seedless probes=64 | ~182 | ~5.0 | ~0.49 |
| exact lm_head (no FlashHead) | ~173 | ~4.75 | ~1.03 |
| oracle (same window) | ~166–171 | — | — |

Head dropped from ~1.2 (exact) / ~0.8 (MLX FlashHead) to **~0.48** via Metal gather. Remaining gap to 200 is **~0.5 ms/tok**, almost all in **layers** (GPU commit→1wait floor ~200–216 tok/s at pos=0; growing KV + encode tax keep e2e layers ~4.9 ms).

Do **not** fuse GPU serial/multi-round top-k into the layer CB (measured regression to ~25–35 tok/s). Next: gqmm2/MoE occupancy, or eliminate the gather CB wait (true single-wait FlashHead with a fast parallel top-k).

## 2026-09-09 up→SwiGLU fusion + ~200 tok/s

M1 Max, fans max, omlx stop, p128/g128 n=5, `layersPerCB=4`, FlashHead probes=96.

Added `gqmm2_up_swiglu` (parity-tested vs gqmm2 + `maple_clamped_swiglu`). Env: `MOMIJ_FUSE_UP_SWIGLU=1` to enable; **default off**.

| Path | gen tok/s (n=5) | notes |
|---|---|---|
| default (separate up + SwiGLU) | **198–200** (peak **200.3**) | layers ~4.5–4.9 ms |
| `MOMIJ_FUSE_UP_SWIGLU=1` | ~193–197 | expert micro ↑ (~2440 vs ~1910 steps/s) but e2e slightly ↓ |
| exact lm_head | ~185 | head ~1.0 ms |

Lessons:

1. Occupancy kernels with large `threadgroup` caches in the same metallib tanked even legacy gqmm2 (~3×) — reverted; not thermal.
2. up→SwiGLU fusion helps isolated expert throughput; e2e layers+attn prefer the lighter two-dispatch path.
3. **~200 gen tok/s** is reachable on this machine without enabling the fused up kernel.

## 2026-09-09 FlashHead B: kill second-wait tax

Profile of separate gather CB (probes=96): CPU top-k ~0.07 ms, gather **GPU ~0.08 ms**, gather **wait wall ~0.35 ms** (commit/wait tax dominates). Force-token overlap is negligible (3 tokens).

**Fix (default on):** hierarchical GPU top-k (`maple_flash_topk_chunk` → `merge`, `localTop=K` for worst-case correctness) + `qmm4` gather encoded **into the layer CB** after centroids. Env: `MOMIJ_FLASH_FUSE=0` to restore CPU top-k + second CB. Default probes **64** (`MOMIJ_FLASH_PROBES`).

| Config | gen tok/s | layers ms | head ms |
|---|---|---|---|
| fuse on (default), probes=64 | **~187–209** | ~4.8–5.3 | **~0.014** |
| fuse off, probes=64 | ~177–185 | ~4.8–5.0 | ~0.57–0.66 |

Not the failed serial full-E×K top-k. Chunk serial rounds are only over 256, merge over ~nChunks×K candidates. Unit-tested incl. adversarial “all top-K in chunk 0”.

## 2026-09-09 SuffixSpec (C) — Seedless wiring

`SeedlessDecodeEngine.generateSuffixSpec` + `benchmarkSuffixSpec` (`momij bench --backend seedless --suffix-spec` / `seedless-bench --suffix-spec`).

- Early-exit greedy verify + rolling gate (`MOMIJ_SPEC_GATE=0` to disable).
- Lossless vs greedy (same token stream).
- Sequential verify **does not beat greedy tok/s** (each accepted token still costs one forward).
- KV snapshot helpers (`snapshotCaches` / `restoreCaches`) for reject rollback.

## 2026-09-09 SuffixSpec C2 — draft-driven 1-CB chain verify

Not true M-row batched attention (still D+1 serial layer walks inside one CB). Interim = GPU `maple_embed_token` + one commit/wait for the whole chain (`stepChainFeeds` / `verifyDraftChain`).

| Probe (M1 Max, FlashHead fuse, p64/g64) | result |
|---|---|
| forced full-accept chain vs sequential | **chain1cb ≈ 201 tok/s** vs seq ≈ 186; **match=true**; **~1.08×** |
| cold motif drafts (`accept≈0`) with `MOMIJ_SPEC_BATCH=1` | **regress** (~124 vs greedy ~180) — restore+replay tax |

Policy: batch **off by default**; enable only with `MOMIJ_SPEC_BATCH=1` or rolling `meanAccept >= 1.5`. `MOMIJ_SPEC_BATCH=0` forces sequential. `MOMIJ_SPEC_MAX_M` caps feeds (default 9).

True M-row verify still needed for peak ~250; chain mainly amortizes CB wait when drafts hit.

## 2026-09-09 gqmm2 A — split-K / w16 (measured)

Goal: raise packed-CB GPU floor (~5.1 ms / ~194 gpu tok/s @24L) without TG-cache metallib bloat.

| Variant | env | micro (own CB) | 24L commit→1w GPU | e2e gen (p128/g128, interleaved×3) |
|---|---|---|---|---|
| default `gqmm2_rows` (8 rows/TG) | — | ~3.0k kernel/s | **~5.1 ms** | **~175–183** (mean ~178) |
| split-K + reduce | `MOMIJ_GQMM2_SPLITK=1` | ↑ (expert micro often +50%+) | **~6.1 ms worse** | **~146–154** (regress) |
| wide TG 16 rows | `MOMIJ_GQMM2_W16=1` | ~neutral | ~5.1 ms | **~181–184** (mean ~182, mild) |

Lessons:

1. Single-CB micro can lie: split-K fills wait bubbles; under packed layers the extra partials+reduce traffic loses.
2. No large `threadgroup` caches (prior occupancy attempt tanked legacy path ~3×).
3. Default stays stock `gqmm2_rows`. Both variants opt-in + parity-tested.
4. Next A candidates: software-pipeline inner K loop (no extra dispatch), CB double-buffer for wall, or SDPA/gqmm2 fuse — not another reduce-style split-K.

## 2026-09-10 gqmm2 A — inner-K software pipeline (no-go)

Tried `gqmm2_rows_pf` (double-buffer x + vectorized `half4`/`uint32` loads), parity OK vs stock.

| Probe | default | PF |
|---|---|---|
| 24L commit→1w GPU | ~5.5 ms | **~6.0 ms worse** |
| e2e gen (interleaved×3) | ~173–181 | **~155–161 regress** |

Not shipped (removed from metallib to avoid bloat). Same lesson as split-K: latency-hiding tricks that help single-CB micro do not raise the packed 1-CB floor.

## 2026-09-10 wall: greedy GPU token-feedback chain

Qwisp II-a style with force-tokens on GPU + `MTLSharedEvent` between tokens
(`layersPerCB` commits; no mega-CB). Short `generate` **ALL_MATCH** vs `MOMIJ_CHAIN_K=0`.

| Config (p128/g128, cold interleaved) | gen tok/s |
|---|---|
| sequential (default) | **~193–198** |
| `MOMIJ_CHAIN_K=8` (event-pipelined) | ~180 |

**Not defaulted:** chain is lossless but slower than today’s sequential floor. Mega-CB
variant previously looked like a win only when seq was ~183; under current ~197 seq it
regresses (CPU encode stall and/or event/argmax tax). Keep `MOMIJ_CHAIN_K` opt-in.

Stable ~200 is essentially the sequential Seedless path on this machine; peak 250 still
needs a lower GPU floor (next: per-op layer profile).

## 2026-09-10 decode-floor profile (`--profile-floor`)

Solo-CB per-op times include launch tax (noisy; `sdpaish` by subtraction can go ~0). Use **shares** and packed 24L.

| Probe | result |
|---|---|
| 24L commit→1w @pos≈0 | gpu **~5.1 ms** → ~196 tok/s |
| 24L commit→1w after ~32 steps | gpu **~4.3 ms** → **~231 tok/s** |
| e2e sequential p128/g128 | **~193–198** gen |

Rank (when shares are stable): **gqmm2 MoE up/down** and **gate** dominate MoE; attn qkv/o secondary; SDPA not the main lever at N≤128 on SWA.

Implication for peak 250 (~4.0 ms/tok): need ~20% less packed GPU work — focus MoE gather-qmv / gate, not chain or FlashHead.

## 2026-09-10 gate gemv: simd default

A/B `maple_gate_gemv` (256-TG reduce) vs `maple_batched_gemv` (1 simdgroup/row). Parity OK.

| Probe | TG reduce | simd (now default) |
|---|---|---|
| 24L commit→1w GPU | ~5.12 ms | **~4.90 ms** (~204 gpu tok/s) |
| e2e p128 (interleaved) | ~176–182 (noisy) | **~181–183** (steadier) |

Env: `MOMIJ_GATE_SIMD=0` restores TG path. Next: MoE `gqmm2` (up/down) without TG-cache / split-K.

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
