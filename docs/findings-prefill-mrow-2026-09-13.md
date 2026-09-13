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