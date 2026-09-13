# Verify pipeline port + fire-rate measurement (2026-09-13)

Branch `agent/verify-pipeline` (cut from main `8a07e36`).

Goal: reproduce evprtr's verify/behavior-correction stack natively in
momij, then measure how much of it the direct path actually needs.

## What was ported (`swift/Sources/MomijCore/ResponseVerify.swift`)

Parity port of `compositor/verify/` (policy_id/kind/detail match the
Python originals byte-for-byte; expected values frozen from the evprtr
venv oracle — 25 parity tests):

- `generic.repetition` — word_run 8 / ngram 3x5 / char motif, plus
  `truncate_before_repetition` (rstrip parity).
- `maple_preview.thin_content`, `.empty_content_long_reasoning`,
  `.pseudo_tool_markup` (+ `strip_pseudo_tool_markup`),
  `.degenerate_tool_args` (write/edit arg checks with all reasons).
- `verify()` = `VerifyBundle.maple_preview` bundle order;
  `Outcome.first` reproduces the short-circuit; every detector's verdict
  is recorded.
- `MissingMutationDetector` intentionally not ported — retired upstream
  from the default bundle ("do not re-wire without explicit policy").
- `FreshConstrainedRepair` intentionally not ported yet (see below).

Server wiring (`f38fedf`): both stream and non-stream paths run the
bundle after decode and record a `verify` trace event (hits + sanitized
flag). The think region maps to the `reasoning_content` channel via
`thinkInterior(of:)`. `MOMIJ_VERIFY_SANITIZE=1` opts into evprtr
sanitize semantics (repetition truncate; degenerate tool_args drop +
finish=stop); **default is measurement-only**.

## Measurement setup

Same request set (plain / list / longgen 512tok / tools-short /
tools-4k, n=3 each; plus a write-tool task n=3 and one longgen pair)
through (a) momij direct (native detectors → momij trace) and
(b) evprtr mediated (`EVPRTR_TRACE_DIR=/tmp/evprtr-verify-measure`),
both over the same seedless momij upstream. 38 momij-side requests,
19 evprtr-side.

## Fire-rate results

| detector | momij-side | evprtr-side | reading |
|---|---|---|---|
| pseudo_tool_markup | 18 (all tools-attached, raw text) | 0 | structurally subsumed by momij's native markup→tool_calls contract |
| generic.repetition (char_motif) | 8 | 4 | 1:1 same verdicts on same upstream text (parity ✓) |
| degenerate_tool_args | 0 | 0 | write args were healthy; clean conversion |
| thin_content / empty_long_reasoning | 0 | 0 | not observed in this set |
| repairs (FreshConstrainedRepair) | n/a | 0 triggers | never engaged |

## The decisive finding: shared whitespace-motif false positive

All repetition fires (`8`/`4`) are motif `"  "` — markdown hard line
breaks (`"  \n"` × 8+) in healthy long answers, not model collapse.
evprtr's production behavior truncates on this: the same longgen
request returned **1899 chars direct (momij) vs 1395 chars mediated
(evprtr)** — evprtr cut a healthy markdown answer at the motif onset.
momij's detector flags identically (parity) but the default path does
not sanitize, so nothing is cut.

## Decision: how much of the verify pipeline momij needs

1. **Detector bundle: taken** (this branch) — zero-cost observability,
   byte-parity verdicts, recorded in traces for future tuning.
2. **Sanitize: stays opt-in** (`MOMIJ_VERIFY_SANITIZE=1`). The measured
   evprtr behavior (truncate healthy markdown) argues against
   enabling it without the motif guard.
3. **PseudoToolCallInContent: observability only** — functionally
   subsumed by the native tools contract (markup never reaches the
   harness in either path).
4. **FreshConstrainedRepair: not ported** — 0 triggers in measurement;
   momij's targeted degenerate-call repair covers the engaged case.
   Revisit when real degenerate/mutation hits appear in traces.
5. **Whitespace-motif guard** is an evprtr-side improvement to
   coordinate (fix in both, or fix upstream first and re-measure).

## Verification

- 25 parity tests green (fixtures frozen from the evprtr oracle).
- Fast suite (verify + toolmarkup + tracestore): 35/35.
- Live smoke: verify verdicts recorded for stream + non-stream.
- Paired measurement: verdict parity confirmed on identical upstream
  outputs; sanitize divergence quantified (1899 vs 1395 chars).

## Follow-up: whitespace-motif guard applied and re-measured

The false positive was confirmed down to the byte level: the model's
healthy Python code block used a 20-space comment indent, which matched
char motif `"  "` × 10 at onset 1395 — evprtr truncated exactly there
(cut away: the rest of the `factorial` example including the print).

Fix (evprtr `3f515a3`, mirrored in momij `640ad3b`): a char motif that
contains no non-whitespace character is idiomatic formatting
(markdown hard breaks, code indentation) and is ignored. Real motifs
are unaffected; word_run/ngram_run are token-based and unchanged.

Re-measurement with both implementations guarded (same request set):

- momij-direct: 32 requests — char_motif fires **8 -> 0**; only the
  by-design pseudo_tool_markup observability records remain (12).
- evprtr-mediated: 16 requests — verify fires **NONE** (was 4).
- Decisive pair (longgen 512tok): direct **1899 == mediated 1899** —
  evprtr no longer cuts healthy content; verdict parity preserved.