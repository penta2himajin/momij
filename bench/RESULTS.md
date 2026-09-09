# momij bench notes

## 2026-09-09 Phase profile (M1 Max, AC, omlx stop, p64/g32 n=1)

Forced `mx.eval` / Metal `waitUntilCompleted` at phase boundaries (`MOMIJ_PROFILE_MOE=1`).
Natural e2e is a separate run without per-MoE sync.

| Path | Natural decode tok/s | Sync phase (ms / decode MoE or attn call) |
|---|---|---|
| oracle (mlx-lm-deepgrove) | **182** | router **0.45** / switch **0.77** / agg **0.39** → moe **1.61**; attn **0.86** |
| momij MLX | **97** | moe (eval each) **1.37** → e2e collapses to ~30 |
| hybrid Seedless MoE | **~24–32** | router **0.92** / metal **0.62** / host in+out **~0.02** |
| Seedless micro (real weights) | — | fused-expert **~1760 steps/s ≈ 0.57 ms** |

Interpretation (measured, not guessed):

1. Host activation copy is negligible (~0.02 ms).
2. Hybrid loses to per-layer dual-runtime sync (MLX router eval + Metal wait), not to the ternary kernel.
3. Under sync, momij Metal experts (~0.62 ms) beat oracle switch (~0.77 ms); momij router (~0.92 ms) is ~2× oracle fused router (~0.45 ms).
4. Oracle/MLX natural speed comes from amortizing sync across a fused graph; forcing per-MoE eval drops MLX to ~30 tok/s (same band as hybrid).
5. momij MLX is still ~1.9× behind oracle at natural e2e — close that gap before expecting Seedless e2e wins.

## 2026-09-09 Seedless Metal microbench (M1 Max, release)

```
seedless gqmm2 (H=2048→N=512,Ktop=1) kernel/s ≈ 2800
seedless fused-expert (E=256,K=8) steps/s ≈ 1700
seedless fused-expert real-weights steps/s ≈ 1760
```

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
