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

## After (same day, this workstream)

Fixes landed:

1. **Seedless SWA ring write** — stop in-place `maple_shift_kv` (parallel overwrite race) and `writePos = maxLen-1` every wrapped step. `kvWritePos = offset % maxLen`, `ropePos` stays absolute. Attention is over the last window of slots (RoPE baked at write).
2. **Ban `<think>` / `<|im_start|>`** when `MOMIJ_ENABLE_THINKING` is off; strip unclosed think instead of leaking the body.
3. **xgrammar** — stateful matcher (`accept` as prefix grows); **one host copy of logits** then argmax among allowed (mlx previously called GPU `.item()` per allowed id → ~32s).
4. **seedless constrained** — real forward + logit argmax (lowest-id walk only remains as a unit-test helper).

Re-measure (release, `:8744` mlx / `:8745` seedless, temp=0):

| Case | mlx | seedless |
|---|---|---|
| short `OK` | `OK` (2 tok, stop) | `OwOwl` (5 tok, stop) — no think/im_start |
| med ~643 | `OK` | repetition (“It is a coding agent…”) — **no** `Available.` / im_start |
| long ~2889 `2+2` | `4.` | `<tool_call>` echo of the user line — **no** `Available.` / im_start |
| JSON object | **0.58s**, `{"name":": ","ok":true}` | **0.50s**, same shape, **not** `!!!!` |
| ZQX grammar | 0.01s `ZQX` | 0.06s `ZQX` |

### Verdict

- **B (xgrammar): PASS.** mlx short JSON **32s → 0.58s**; seedless JSON is valid and logit-based.
- **A (seedless long quality): PARTIAL.** Collapse modes from the remeasure (`Available.` / im_start / empty-after-strip) are gone. mlx-parity answers (`OK` / `4.`) are **not** there yet — seedless sequential Metal prefill still diverges inside the SWA window (~280 tok already garbled with think-on). Next: batched SWA prefill vs mlx, or keep `--backend mlx` for Pi-length until that lands.
