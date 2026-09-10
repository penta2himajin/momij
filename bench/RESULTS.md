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

## 2026-09-10 gqmm2_rows_vec — no-go

Tried vectorized `half4` x-load + `uint32` weight pack (`gqmm2_rows_vec`), parity OK.

| Probe | stock | vec |
|---|---|---|
| gqmm2 micro | ~3.5k kernel/s | **~2.8k worse** |
| 24L commit→1w GPU | ~5.05 ms | **~5.67 ms worse** |
| e2e p128 | ~184 | **~170–178 regress** |

Not shipped (removed from metallib). Same family as PF: micro/load tricks do not raise the packed MoE floor. Remaining MoE levers need a different angle (layout / fewer bytes / Qwisp diff), not load vectorization.

## 2026-09-10 gqmm2 ternary (`{-α,0,+α}`) — no-go for packed floor

Maple-Preview is natively ternary (`codes∈{0,1,2}`, `bias=-α`, row `α`). Opt-in kernel
`gqmm2_rows_ternary`: `y = α·Σ(q−1)·x`, no x pre-divide, α once from scales group0,
biases unused. Parity vs affine on ternary weights: rel_l2≈3e-3 (float schedule). Separate
metallib via `ensureTernaryCompiled` (not co-compiled with stock).

| Probe (interleaved) | stock | `MOMIJ_GQMM2_TERNARY=1` |
|---|---|---|
| 24L commit→1w GPU | ~5.2–5.3 ms | **~5.6 ms worse** |
| e2e p128/g128 n=3 | **~181** | **~173 regress** |

Default stays stock. Env left opt-in for further experiments. Lesson: on M1 Max the
MLX-tuned masked·pre-divide affine stream already wins; rewriting to signed trit mul
adds work / different issue mix and does not raise the packed floor. Next MoE levers:
M-row batched verify, or layout / fewer intermediate bytes — not another trit ALU rewrite.

## 2026-09-10 gqmm2 fold-α (stock qd2, hoisted row α) — default on

Keep stock `ld16_b2`/`qd2` FMA stream; only Maple identities: load row `α` once from
`scales[group0]`, pass `bias=-α`, do not advance scales/biases along K. Kernel
`gqmm2_rows_fold` in a **separate** metallib (ternary wins if both set).
Parity vs affine on row-constant-α weights: rel_l2 < 1e-3.

| Probe (interleaved) | per-group affine | fold |
|---|---|---|
| gqmm2 micro kernel/s | 2611 / 3239 | 2703 / 2672 (noise) |
| 24L commit→1w GPU | 4.97 / 5.06 ms | 4.99 / 4.92 ms (tie) |
| e2e p128/g128 n=3 | 179.1 / 180.1 | 177.0 / 182.4 (tie, mean ~179.6 vs ~179.7) |

No measured packed/e2e win, but also no loss, and the kernel does less work (no
per-group scale/bias walk). **Default is fold**; `MOMIJ_GQMM2_FOLD=0` restores affine.

## 2026-09-10 gqmm2 defer-α (α after simd_sum) — no packed win

Variant B: keep `ld16_b2` + masked 16-way accum; drop per-block `α*accum` /
`sum*bias`; `y = α · (simd_sum(accum) − simd_sum(xsum))`. Separate metallib
`gqmm2_rows_defer_a`, env `MOMIJ_GQMM2_DEFER_A=1` (takes priority over fold).
Parity vs fold on ternary weights: rel_l2 < 1e-3.

| Probe (interleaved) | fold (default) | `MOMIJ_GQMM2_DEFER_A=1` |
|---|---|---|
| 24L commit→1w GPU | 4.45 / 4.49 ms | 4.42 / 4.68 ms (tie/noise) |
| e2e p128/g128 n=3 | 211.4 / 202.0 | 203.7 / 204.5 (mean ~206.7 vs ~204.1) |

No win. Inner FMA stream is unchanged but moving α out of the K-loop did not
raise the packed floor; e2e mean is inside fold's own spread. **Default stays
fold.** Env left opt-in; do not promote on a tie.

## 2026-09-10 gqmm2 fold-A (`α·(accum−sum)` epilogue) — no stack win

Variant A: same as fold (`ld16_b2` + masked 16-way) but epilogue is
`α·(accum−sum)` instead of `α·accum + (−α)·sum`. Separate metallib
`gqmm2_rows_fold_a`, env `MOMIJ_GQMM2_FOLD_A=1`. Parity vs fold: rel_l2 < 1e-3.
Interleaved vs fold default and defer-α (B) to test whether noise-level
rewrites accumulate.

| Probe (interleaved) | fold | A (`FOLD_A=1`) | B (`DEFER_A=1`) |
|---|---|---|---|
| 24L commit→1w GPU | 4.41 / 4.36 ms | 4.52 / 5.57 ms | 4.75 / 5.56 ms |
| e2e p128/g128 n=3 | 204.1 / 205.1 | 199.8 / 205.5 | 205.0 / 205.1 |

Means: fold ~204.6, A ~202.7, B ~205.0 tok/s. A/B do not stack into a packed
or e2e win; pair-1 packed is fold < A < B (A/B slightly slower). **Default
stays fold.** Env left opt-in. Algebraic epilogue tweaks are not an additive
lever on this issue-bound `qd2` stream.

## 2026-09-10 True M-row gqmm2 (kernel-only, warm)

Qwisp `gqmm2Rows` already encoded `tid.z = m*Ktop+ki` / `x` row =
`lhsPer ? mk : mk/ktop`. Momij kernel had the same `mk/ktop` but dispatch
was `depth: Ktop` only. Wired `gqmm2(..., M:)` → `depth: M*Ktop`, split-K
off at `M>1`. Parity: M=2 batched vs two sequential M=1, rel_l2 < 1e-7.

GPU `timeCB` after 8× warmup at maxM, then interleaved `M=1,2,4,8,8,4,2,1`
(30 iters each, averaged per M). omlx stopped. Maple-like: Ktop=8, E=256.

| shape | mode | M | gpu_ms | ms/tok | vsM1 |
|---|---|---|---|---|---|
| up K=2048 N=1024 | shared | 1 | 0.074 | 0.074 | 1.00 |
| | shared | 2 | 0.145 | 0.072 | 0.98 |
| | shared | 4 | 0.240 | 0.060 | 0.81 |
| | shared | 8 | 0.367 | 0.046 | 0.62 |
| | disjoint | 1 | 0.068 | 0.068 | 1.00 |
| | disjoint | 2 | 0.088 | 0.044 | 0.65 |
| | disjoint | 4 | 0.125 | 0.031 | 0.46 |
| | disjoint | 8 | 0.226 | 0.028 | 0.42 |
| down K=512 N=2048 lhsPer | shared | 1 | 0.072 | 0.072 | 1.00 |
| | shared | 2 | 0.138 | 0.069 | 0.96 |
| | shared | 4 | 0.185 | 0.046 | 0.64 |
| | shared | 8 | 0.340 | 0.043 | 0.59 |
| | disjoint | 1 | 0.095 | 0.095 | 1.00 |
| | disjoint | 2 | 0.197 | 0.099 | 1.04 |
| | disjoint | 4 | 0.310 | 0.078 | 0.81 |
| | disjoint | 8 | 0.578 | 0.072 | 0.76 |

First sweep without warmup was invalid (cold M=1). Honest read:

- **M=2 ≈ no win** (up shared 0.98, down shared 0.96). Speculative decode
  at draft=2 will not move the sequential ~200–210 floor via this kernel.
- **M≥4 is a real packing lever.** Up shared M=8 is 0.62× ms/tok; down
  shared 0.59×. Occupancy/weight reuse, not ALU rewrite.
- Up disjoint beating shared at the same `M*Ktop` TG count is likely
  less contention (64 unique experts vs 8) — not a reason to prefer
  disjoint routing in production.
- This is **one GEMM**, not e2e. Peak 250 still needs attn + fused-expert
  M-row and a path that actually runs M≥4. Do not default decode to M-row
  until that stack exists.

## 2026-09-10 True M-row fused expert (warm)

Wired `encodeFusedExpert(..., M:)`: up/down `gqmm2` depth `M·Ktop`, SwiGLU
over stacked `M·Ktop` rows, `maple_score_reduce` now `y[m,h]` with `gid.y=m`.
`gqmm2_up_swiglu` depth likewise (opt-in fuse path). Decode still `M=1`.
Parity: M=2 batched vs two sequential M=1, rel_l2 < 1e-3.

Same warm interleaved protocol as gqmm2 (8× warmup at maxM, then
`M=1,2,4,8,8,4,2,1`). Maple-like H=2048 I=512 Ktop=8 E=256. omlx stopped.

| mode | M | gpu_ms | ms/tok | vsM1 |
|---|---|---|---|---|
| shared | 1 | 0.147 | 0.147 | 1.00 |
| shared | 2 | 0.200 | 0.100 | **0.68** |
| shared | 4 | 0.395 | 0.099 | 0.67 |
| shared | 8 | 0.600 | 0.075 | 0.51 |
| disjoint | 1 | 0.158 | 0.158 | 1.00 |
| disjoint | 2 | 0.220 | 0.110 | 0.69 |
| disjoint | 4 | 0.393 | 0.098 | 0.62 |
| disjoint | 8 | 0.686 | 0.086 | 0.54 |

Same session's gqmm2 up shared M=2 was still ~0.96. Fused-expert M=2 at
0.68 is launch/epilogue amortization across up+SwiGLU+down+reduce, not a
contradiction of the kernel-only issue wall. M=4 is a plateau vs M=2;
M=8 reaches 0.51×. Draft=2 can move the **expert block**; peak 250 still
needs attn + router M-row and a path that runs M≥2 end-to-end. Do not
default decode to M-row yet.

## 2026-09-10 True M-row attn probes (warm)

SDPA kernel already had `tid.y = q_seq` / `o_offset = head·M + seq`; dispatch
was `height: 1`. Wired `encodeSdpa(..., M:)` → `height: M`, queries
`[numHeads, M, D]`, shared KV. `encodeAttnBlock` still M=1 (qk-norm / writeKV
are one-token). QKV/O use existing dense `gqmm2` (Ktop=1, same weights).
Parity: SDPA M=2 vs two sequential M=1, rel_l2 < 1e-3.

Warm interleaved `M=1,2,4,8,8,4,2,1` (20 iters). Maple 16h/4kv/D=128. omlx stopped.

| probe | M=1 gpu_ms | M=2 vsM1 | M=4 | M=8 |
|---|---|---|---|---|
| qkv K=2048 N=3072 Ktop=1 | 0.174 | **0.40** | 0.29 | 0.24 |
| o-proj K=2048 N=2048 Ktop=1 | 0.103 | 0.92 | 0.57 | 0.43 |
| sdpa N=128 | 0.192 | 0.17 | 0.24 | 0.11 |
| sdpa N=512 | 0.259 | 0.24 | 0.26 | 0.17 |

Honest read:

- **QKV dense M-row is a real packing lever** (Ktop=1, full weight reuse).
  M=2 already 0.40× ms/tok. Stronger than MoE gqmm2 at M=2.
- **O-proj matches the MoE-gqmm2 pattern**: M=2 ≈ noise (0.92), win from M≥4.
- **SDPA M=1 micro is occupancy/launch bound** (16 TGs). vsM1 looks spectacular
  because M>1 fills the grid; do not read 0.17× as 6× math. Inside a packed
  24L CB the sequential SDPA tax is already small (`--profile-floor`).
- Full attn M-row still needs `qk_norm_rope` + `writeKV` (and RMS) for M
  tokens with per-row RoPE / cache slots. Do not default decode to M-row.

## 2026-09-10 True M-row attn block (qk-norm / writeKV / causal SDPA)

`encodeAttnBlock(..., M:)`: QKV/O `gqmm2` M, `qk_norm_rope` depth M with
RoPE `pos+m`, `writeKV` slots `writePos+m`, SDPA token-major `[M, heads, D]`
and `N = seqLen+m` when `causalM`. `rotateFirst` stays M=1-only. Decode still
`M=1`. Parity: M=2 vs two sequential M=1 (shared prefix cache), rel_l2 < 1e-3.

Warm interleaved `M=1,2,4,8,8,4,2,1` (20 iters). Maple 16h/4kv, writePos=32
so SDPA N=33. omlx stopped.

| probe | M=1 gpu_ms | M=2 vsM1 | M=4 | M=8 |
|---|---|---|---|---|
| attn-block | 0.226 | **0.90** | 0.39 | 0.39 |
| sdpa N=128 (shared N, this session) | 0.193 | 0.91 | 0.38 | 0.21 |

Earlier SDPA-only 0.17× at M=2 was M=1 launch tax. Honest packing: **attn
block is like O-proj — M=2 ≈ no win, M≥4 is 0.39×**. Draft=2 will not move
sequential attn; QKV still packs but o-proj + epilogue dominate the block.
Do not default decode to M-row. Next structural gap is router / e2e M≥2.

## 2026-09-10 Router M-row + layer config sweep

Wired `M` through `maple_rms_norm` (1 TG/token), `maple_batched_gemv` /
`maple_gate_gemv` grid `(E,M)`, `maple_route_top8` (1 TG/token),
`maple_resid_add` `[M,H]`. `encodeMoEBlock` / `encodeLayerBlock` take `M`
(default 1). Decode still M=1. Parity: MoE block M=2 vs two sequential M=1,
rel_l2 < 1e-3.

Warm MoE-block (H=2048 I=512 E=256 K=8, 20 iters) and layer config sweep
(`seq` = M×M=1, `full` = attn+moe M-row, `moe` = attn seq + moe M-row; 8 iters).
omlx stopped.

| probe | M=1 | M=2 vs | M=4 vs | M=8 vs |
|---|---|---|---|---|
| moe-block ms/tok vsM1 | 0.395 | **0.51** | 0.43 | 0.28 |

| layer cfg | M=2 vsSeq | M=4 vsSeq |
|---|---|---|
| seq | 1.00 | 1.00 |
| full | **0.81** | 0.57 |
| moe (attn seq + moe M-row) | 0.89 | **0.38** |

Honest read:

- Router gap is closed; MoE block packs at M=2 (0.51×), matching fused-expert.
- **Fastest M=2 layer config on this synthetic Maple-like probe: `full`**
  (~0.81× vs sequential). Hybrid moe is close behind.
- **Fastest M=4: `moe` hybrid** (0.38×) — attn M-row still weaker than MoE
  packing, so sequential attn + batched MoE wins. Full is second (0.57×).
- M=1 `full` outlier (~4× seq) is noise / first-touch; ignore for ranking.
- `moe` hybrid pays an extra CB + host pack; numbers are GPU-ms sum.
  Production e2e still needs decode to call `encodeLayerBlock(..., M:)`
  (or hybrid) and size scratch to M. Do not default yet.

## 2026-09-10 e2e: True M-row wired into `stepChainFeeds`

Env: `MOMIJ_MROW=1` sizes layer scratch to `SPEC_MAX_M` and packs draft verify
as one M-row forward (embed rows → `encodeStep(M:)` ×24L → FlashHead per row).
Default remains M=1. SWA rotate mid-batch falls back to sequential.

M1 Max, omlx stop, release, FlashHead fuse. Measured 2026-09-10:

| Path | gen tok/s | notes |
|---|---|---|
| greedy p128/g128 n=3 | **~172–179** | baseline |
| suffix-spec motif (accept≈0–0.25) | ~169 / batch ~105–122 | drafts miss → gate/restore tax |
| chain1cb K=8, M×M=1 (`SPEC_BATCH`) | **~180–184** | match=true, ~1.04–1.06× seq |
| **chain1cb K=8, True M-row (`MROW=1`)** | **~350–380** | **match=true, ~2.1–2.2× seq** |

Verdict (e2e tok/s, not synthetic µs):

1. **True M-row wins the forced full-accept chain verify** (~2.2× vs sequential
   steps; ~2× vs packed M×M=1 in one CB). Lossless (`match=true`).
2. Motif SuffixSpec still has near-zero draft accept, so product e2e tok/s stays
   below greedy until drafts land. M-row cannot invent accept rate.
3. Do **not** default `MOMIJ_MROW=1` for cold SuffixSpec; keep opt-in for
   high-accept / oracle-draft / chain-verify paths.

## 2026-09-10 SuffixSpec PLD drafts — raise accept%

Fixes: full-history PLD search (was abutting-window only), contiguous n-gram
(no scrambled successor bag), prefer prompt-span matches, adaptive `draftK`,
soft gate + periodic re-probe. Batch/M-row verify only when `meanAccept≥1.5`
(forced `MOMIJ_SPEC_BATCH=1` still available).

M1 Max, omlx stop, release. Code-echo prompt (`fib` rewrite):

| Config | accept/attempt | accept/gen | spec tok/s | greedy | lossless |
|---|---|---|---|---|---|
| seq verify | **0.96** | 0.26 | **~196** | ~193 | true |
| MROW + auto batch | **0.89** | **0.44** | ~178 | ~191 | true |
| MROW + `SPEC_BATCH=1` | 0.88 | 0.39 | ~88 | ~190 | true |

Notes:

1. **Accept is workload-dependent.** Copy-doc English ≈0.07; code echo ≈0.9.
   PLD helps when the model reuses prompt spans (code/agentic), not open chat.
2. chain1cb M-row still **~420–450 tok/s** (oracle draft) — upper bound when
   drafts are perfect; not product e2e on arbitrary prompts.
3. Forced batch with mid accept regresses (restore/replay tax). Default stays
   sequential early-exit until mean accept is hot.
4. Next: SuffixDecoding-style tree / online output index for agentic; keep
   gate when accept stays cold.

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
