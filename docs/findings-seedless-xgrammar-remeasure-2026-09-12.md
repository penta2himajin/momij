# Findings: seedless long-prompt + xgrammar cost — 2026-09-12/13

## Latest verify (2026-09-13, tip `09fa916`)

Release binary, `mlx :8744` / `seedless :8745`, temp=0.

| Case | prompt_tok | mlx | seedless |
|---|---:|---|---|
| short `OK` | 13 | `OK` (0.13s, 2 tok) | `OK` (0.13s, 2 tok) |
| med pad | 625 | `OK` (1.0s) | `OK` (1.5s) |
| long `2+2` | 2834 | `4` (4.8s) | `4` (13.2s) |
| grammar `ZQX` | 14 | `ZQX` (0.01s) | `ZQX` (0.04s) |
| JSON object | 12 | `{"name":": ","ok":true}` (**0.56s**) | same (**0.48s**) |

### Verdict (current)

- **A seedless long quality: PASS** on these prompts (no `Available.` / im_start collapse; answers match mlx).
- **B xgrammar: PASS** — correctness OK on both backends; mlx short JSON **~32s → ~0.5s**.
- **Remaining**: seedless **wall time** on long prompts (~13s vs mlx ~5s for 2 completion tokens). Agentic default can use seedless for quality; speed still favors mlx for long context until Metal prefill/decode is closed.

Fixes that landed after the first remeasure (`d372cf6`):

1. `2fba4e2` — SWA ring write; logit-based constrained decode; xgrammar one host logits copy
2. `f4822e5` — greedy exact `lm_head` (not FlashHead probes)
3. `09fa916` — seedless owns RoPE `inv_freq` (was clobbered at layer init; pos>0 broken)

---

## Historical (2026-09-12 evening, tip `5591252`) — superseded

| Case | mlx | seedless |
|---|---|---|
| long ~2834 | OK | **FAIL** `Available.` loop |
| JSON object | **32s** OK | ~0.9s but `"!!!!…"` |

See git history of this file for intermediate PARTIAL notes after `2fba4e2` / `f4822e5`.
