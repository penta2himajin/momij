# Findings: seedless long-prompt + xgrammar cost — 2026-09-12

Re-measure after `a3fe719` / `8b47021` / P1 xgrammar. Backends side-by-side
(`mlx :8744`, `seedless :8745`), same release binary tip `5591252`.

## seedless long-prompt parity

| Case | prompt_tok | mlx | seedless |
|---|---:|---|---|
| short `OK` | 13 | OK (no im_start loop) | responds (still emits `<think>` in content) |
| med pad | 625 | OK | OK-ish (no im_start) |
| long pad `2+2` | 2834 | OK (prose, no im_start) | **FAIL quality**: `Available. Available. …` repeat |

Notes:

- mlx SWA fix holds: long prompts no longer dump `<|im_start|>`.
- seedless already keeps absolute `ropePos = offset` in Metal; crash path improved.
- **Quality parity is not achieved** at ~2.8k prompt tokens (repetition collapse).
- seedless serve still surfaces `<think>` text more than mlx (patch / strip path differs in practice).

Verdict: **not fixed** for agentic default. Keep recommending `--backend mlx` for Pi-length prompts.

## xgrammar speed / correctness

| Case | mlx wall | seedless wall | notes |
|---|---:|---:|---|
| FiniteString `ZQX` grammar | 0.01s | ~0s | fine |
| JSON const `"ZQX"` | 0.36s | 0.38s | fine |
| JSON object (14–64 tok) | **32.3s** | 0.87s | mlx slow; seedless “fast” but wrong |
| JSON object on long prompt | 4.6s | 0.9s | seedless fills `"name":"!!!!…"` |

Root cause (seedless + any `allowedNext`):

- `SeedlessBackend` uses **deterministic lowest-id frontier walk** when `allowedNext` is set (no logit argmax).
- That is OK for single-literal FiniteStringGuide; **wrong for open JSON strings** (picks `!` etc.).

mlx uses score-argmax among allowed IDs (correct) but pays **full-vocab bitmask scan + matcher replay per step** → tens of seconds even for tiny objects.

Verdict:

- **Correctness**: mlx JSON OK; seedless JSON under xgrammar **not trustworthy**.
- **Speed**: **not fixed**; short JSON object ≈ 32s on mlx is unacceptable for interactive use.

## Next (if prioritizing)

1. seedless long decode: profile SWA rotate / prefill mask parity vs mlx (quality).
2. xgrammar: stateful matcher + sparse allowed-ID extract; wire **logit-argmax** into seedless constrained path (or refuse JSON on seedless until then).
