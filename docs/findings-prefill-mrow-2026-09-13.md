# Findings: prefill M-row ring work + throughput ladder — 2026-09-13

Branch: `agent/prefill-mrow-speedup` (main + wrap-safe M-row prefill).

## What was done

The SWA ring forced sliding layers to M=1 once offset passed
`slidingWindow`, so prefill collapsed to sequential (~220 tok/s) after
the first 512 tokens of any long prompt. Attention over the ring is
permutation-invariant (RoPE is baked into K at write time with absolute
positions; eviction stays FIFO), so M-row chunks can wrap the ring if
writes and reads agree on the cyclic slot order:

- write kernels wrap slots cyclically (`ring` constant)
- `maple_sdpa_d128` reads each row's window cyclically: `nUse` slots
  ending at slot `(endSlot + m) % ring`; `ring == 0` reduces to the old
  linear pointer walk
- chunks may not straddle the FIRST wrap (write-before-SDPA order would
  break causality there); wrapped chunks run full M-row
- prefill chunk size steps down to the largest feasible M

## Measured (M1 Max, release, specMaxM=64, serve path)

| prompt tok | before | after | |
|---:|---:|---:|---|
| 418 | 577 | 681-692 | first request, incl. warmup |
| 825 | 386 | 756-775 | |
| 3,234 | 237 | 706-721 | **3.0x** |

Per-chunk wall at M=64: ~85 ms steady (1.33 ms/tok). First chunk ~114 ms
(warmup). `MOMIJ_LAYERS_PER_CB` 8/12/24 changes prefill little (706-721).
Decode bench unchanged: 189.8 tok/s (the SDPA rewrite does not regress
decode). Full suite: 106 tests, 0 failures (incl. new wrapped parity
tests: chunk sizes stay M-row post-wrap, per-layer hidden vs MLX,
greedy token parity vs MLX, wrapped M-row chain evals).

## Ladder to 1,000 tok/s prefill

Layer microbench (M-row configs, per layer): at M=4, attn 0.076 ms/tok
vs MoE 0.105 — **MoE (gqmm2 fused-expert) is the slower scaler**. The
config sweep only measures M=1/2/4 today; extend it to 16/32/64, then:

1. gqmm2 at large M: try `gqmm2_rows_w16` (4 simdgroups, 16 rows/TG)
   and splitK paths for the fused-expert dispatch (per-(row,expert) TGs
   are tiny at M=64: depth = M*Ktop = 512 TGs/kernel).
2. Multi-chunk pipelining: prefillPrompt commits+waits per chunk; GPU
   embed (`encodeEmbedToken`) + per-chunk hBuf slots would hide the
   commit→wait bubble.
3. specMaxM > 64 showed no gain (706 vs 706-721 at 128): the kernel, not
   the chunk width, is the limit now.

Target: ≥1,000 tok/s needs ~64 ms per 64-token chunk (from 85 ms).
## Raw-Metal lm_head argmax (2026-09-13, same branch)

Decomposed the decode head phase (0.93 ms): the MLX quantizedMM+argMax
eval itself is ~0.78 ms — item() sync is ~0.001 ms — so the head is
bandwidth-bound (175 MB of 4-bit head weights+groups per token, MLX
effective ~224 GB/s), not sync-bound. Two naive raw-Metal kernels
(thread-per-row, simd-per-row) both hit only ~58 GB/s — the x-vector
scalar loads (2.4 G loads/token) and 18,992 tiny threadgroups were the
limiters.

Final kernel (`maple_lm_head_argmax` + `maple_lm_head_reduce`):
- threadgroup owns a contiguous 256-row slab (~594 TGs), x staged once
  into threadgroup memory, each SIMD group walks 32 rows with coalesced
  128-byte word loads, banned ids score -inf, per-TG partial folded by
  the reduce kernel
- encoded into the layer CB tail: zero extra sync, zero MLX roundtrip;
  also replaces the chain path's per-row 608 KB host copies

Measured (same build, head on vs off): head ~0.71 ms vs MLX 0.925 ms;
decode bench 187.4 -> 193.9-196.0 tok/s (layersPerCB 4 -> 8); suffix
spec 240 -> 268.6 tok/s; chain-verify K=8 385.9 -> 459.8 tok/s
(match=true, 2.42x). E2E 4k-token prompt: 4.9-5.2 s. All parity tests
green with the head enabled by default (`MOMIJ_METAL_HEAD=0` disables;
`MOMIJ_LAYERS_PER_CB=8` pairs well with it).

Remaining decode gap to 200: ~0.5 ms/token = head bandwidth (~250-260
GB/s measured) + layer floor. Next candidates: simdgroup-matrix loads
for the head, and prefill MoE tuning (w16/splitK) + chunk pipelining
for the 1,000 tok/s prefill target.

## Round 5: specMaxM sweep + kernel variants (2026-09-13)

Serve-path prefill across specMaxM (cyclic-ring M-row, same build):

| prompt tok | M=32 | M=64 | M=128 |
|---:|---:|---:|---:|
| 825 | 691 | 721 | 734 |
| 3,234 | 688 | 710 | 728 |
| 6,457 | **603** | 489 | 479 |

- M=32 is the sweet spot at 8k tokens; M=64/128 get WORSE there (the
  fused-expert dispatch depth = M*Ktop makes per-chunk cost super-linear
  as the KV/attention term grows).
- Kernel variants `MOMIJ_GQMM2_W16=1` / `MOMIJ_GQMM2_TG2D=1`: no
  meaningful change (686-692 vs 687-692 baseline at M=32).
- The extended micro-sweep (M=1..64) runs for minutes at a time; not a
  regression, just expensive diagnostics.

Honest ceiling with current fused-expert kernels: ~690-775 tok/s
prefill. Reaching 1,000 needs a faster fused-expert kernel at M>16
(e.g., multi-row threadgroups with expert-union batching), which is
deeper kernel work than the remaining round budget.

Decode side delivered this round: raw-Metal exact head 196 tok/s
sequential (250-260 GB/s head stream, +12% over MLX), 268.6 tok/s spec,
459.8 tok/s chain-verify (2.42x, match=true), E2E 4.9-5.2 s at 4k.

## Bandwidth feasibility analysis (2026-09-13)

The MoE per-token expert traffic is the wall: 8 experts x ~2.25 MB
(up_gate 2*I*K + down K*I at 2-bit) x 24 layers = ~430 MB/token of
required weight streaming. Measured effective bandwidth for these
kernels is ~230-260 GB/s (the 24L-1CB class), which puts the hard
per-token floor at ~1.1-1.4 ms -> ~700-900 tok/s prefill even with
perfect M-row amortization and expert-union reuse.

Measured ceilings agree: prefill 690-775 tok/s across chunk widths and
kernel variants. Reaching 1,000 tok/s would need >350 GB/s sustained
expert streaming (simdgroup-matrix fused-expert kernel) or a change in
the traffic itself (e.g., quantized expert cache in DRAM-friendly
layout). Documented as the remaining headroom; decode side is at
196 sequential / 268 spec / 459.8 chain-verify (lossless, 2.42x).

## Round: acceptance diagnostics + pipelining verdict (2026-09-13, spec-throughput)

- Acceptance on realistic (non-synthetic) agentic text, fresh engine:
  **1.16 accept/attempt, 0.55 accept/gen** (142/256 tokens accepted
  free). The existing tree/recycle/PLD drafters are healthy on agentic
  content; the chain batches engage (gate 1.0).
- Chain draft width (K=8/16/24, specMaxM=64): accept/gen flat at
  0.59-0.60 — acceptance is drafter+content bound, not width bound.
- Prefill chunk decomposition (M=32): embed 0.01 ms, commitWait
  44.5-46.8 ms — the chunk is >99.9% GPU execution; multi-chunk CPU
  pipelining has ~1% headroom, ruled out by measurement.
- Pure-bandwidth probe (MLX sum): 64 MB 120 / 128 138 / 175 232 (warm,
  105 cold) / 256 209-224 / 512 271-317 / 1 GB **335 GB/s**. The
  machine reaches 350-class only at ~1 GB transfers; at the head's
  175 MB the warm ceiling is ~232-260 GB/s, and the v3 head kernel
  (246-260 GB/s) is at or above it. The 400->260 gap is per-call fixed
  cost + page warm-up + CPU-shared unified memory, not a kernel defect.

Final position with existing assets: prefill 683-734 tok/s, decode
196 sequential / 268.6 spec / 459.8 chain-verify (all lossless),
E2E 4.9-5.2 s at 4k. Remaining headroom beyond this requires either a
trained drafter (deferred) or >300 GB/s sustained streaming kernels.

## Stage 1: native OpenAI tools in momij (compositor integration, 2026-09-13)

momij now accepts OpenAI tools on /v1/chat/completions and applies the
Maple markup contract internally (ToolMarkup port of maple_tool_markup.py
+ pseudo_tool.py, byte-parity jsonDumps). Direct request verified:
tools -> tool_calls ls, finish tool_calls; full suite 107 tests green.

Hop cost measured (same tools request, median of 12, both correct):
- direct momij (native tools): 353.2 ms
- evprtr mediated:             385.7 ms
- overhead: 32.6 ms per request (9%)

The compositor is no longer required for the single-model single-runtime
path. Stage 2 candidates: verify/repair loops, buffered approvals, and
native streaming (replacing the fake SSE shim), each measured the same
way before adoption.
