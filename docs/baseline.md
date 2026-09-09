# Baseline (M1 Max, AC power)

Measured 2026-09-09 with `mlx-lm-deepgrove` `benchmark`, omlx stopped.

| condition | decode tok/s | prefill tok/s | peak GB |
|---|---|---|---|
| exact · p128/g256 · n=5 | **182.4** | 656 | 6.65 |
| flash-head · same | **181.2** | 645 | 7.14 |
| flash · p512/g512 | 156.1 | 954 | 7.14 |
| exact · p512/g512 | 138.8 | 685 | 6.81 |

Notes:

- Bandwidth ceiling ~970 tok/s → ~19% achieved.
- FlashHead ≈ no-op on this machine → bottleneck is MoE/dispatch, not lm_head.
- With omlx resident, decode fell to ~166 tok/s. Always `omlx stop` before benches.

Reproduce:

```bash
omlx stop
cd ~/repos/mlx-lm-deepgrove
.venv/bin/python -m mlx_lm.benchmark \
  --model ~/models/deepgrove/maple-preview-2bit-mlx \
  --trust-remote-code --prompt-tokens 128 --generation-tokens 256 \
  --batch-size 1 --num-trials 5
```

Or via momij oracle:

```bash
cd ~/repos/momij/swift
swift run -c release momij -- bench --backend oracle -p 128 -g 256 -n 5
```
