# momij bench notes

## 2026-09-09 Seedless Metal microbench (M1 Max, release)

```
seedless qmv2 (H=2048→N=512) kernel/s ≈ 1180–1680
seedless fused-expert (E=4,K=8) kernel/s ≈ 0.7   # naive scaffold; not production
```

Interpretation: 2-bit qmv kernel compiles and runs at useful throughput.
Fused-expert kernel is a correctness-first nested loop — replace with tiled
gather+swiglu+down before claiming end-to-end tok/s gains.

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
