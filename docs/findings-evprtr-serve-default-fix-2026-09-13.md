# Findings: evprtr serve-default regression + fixes — 2026-09-13

## Symptom (Pi → evprtr → momij)

- Pi smoke that passed on 2026-09-12 (tip `a3fe719`, `--backend mlx`) broke after
  `ffe44eb` (09-13 01:02, "default FlashHead SuffixSpec on") + rebuild.
- Two independent failures, both rooted in `ffe44eb` default changes:

1. **mlx backend hangs on long prompts.** `Server.swift` now defaults
   `useSuffixSpec` on; the MLX SuffixSpec verify regenerates from the full
   prefix per draft step (O(prompt × drafts)). Measured: 1,668 tok → 88.1 s
   (vs 4.3 s plain greedy); ~4.8k tok → >240 s (vs 7.7 s). Pi's ~4.7–4.9k
   prompts never returned.
2. **seedless (serve default backend) broke markup adherence.** With FlashHead
   verify as default, the evprtr tool-markup prompt collapsed into a degenerate
   repetition (finish=length, no tool_calls) while mlx and exact-head seedless
   returned correct `ls` / `grep` tool calls.

## Engine-level reproduction (new tests)

`Tests/MomijCoreTests/RealTextParityTests.swift` — real-text prompt, tokenizer:

- `testExactHeadSuffixSpecMatchesExactGreedy` / `testServeDefaultSuffixSpec…` /
  `testForcedBatchSuffixSpec…`: SuffixSpec output must equal exact sequential
  greedy; forced batching exercises snapshot/restore+replay on rejects.
- `testChainEvalsMatchSequential`: M-row chain evals == sequential evals.

Result: the M-row chain and restore/replay are **lossless on real text**
(both at `ffe44eb` and after fixes). Captured defect: the batch branch could
**overshoot `max_tokens`** (requested 64 → emitted 65; chain chunk + bonus
appended past the budget).

## Fixes

1. `SeedlessDecodeEngine.generateSuffixSpecFromPrefill`: clamp chunk + bonus
   appends to the remaining `max_tokens` budget.
2. `SeedlessServeDefaults.exactHeadFromEnv`: exact `lm_head` is the **default**
   verify head (lossless). FlashHead probes are opt-in via
   `MOMIJ_EXACT_HEAD=0` — they are an approximation and measurably drift off
   exact greedy on agentic markup prompts.
3. `MapleMLXBackend.generate`: ignore `useSuffixSpec`. The MLX spec path is a
   naive full-prefix-regen wiring; serve must not hang. MLX stays the
   parity/fallback path (plain greedy ~5.4–7.6 s at 2.6–4.8k prompt).
4. eos handling: a speculative chunk could carry eos mid-sequence, leaking
   `<|im_end|>` into `message.content`. The batch branch now keeps eos as the
   chunk tail and drops the post-eos bonus; `OpenAIChatCompat.contentTokenIds`
   cuts at the first eos (was: trailing only).

## Post-fix measurements (M1 Max, release, omlx stopped)

- `momij bench --backend seedless -p 128 -g 256 -n 3`:
  prefill **193.9 tok/s**, decode **190.7 tok/s** (exact head, sequential).
- `momij bench --backend seedless --suffix-spec -p 128 -g 256 -n 3`:
  greedy 165.3 / spec **171.2 tok/s**, `lossless=true`;
  chain-verify K=8 head=exact: sequential 181.4 vs **365.6 tok/s**,
  `match=true`, 2.02x.
- E2E Pi-like (3,985 prompt tok): correct `ls`+`grep` tool_calls in **19.1 s**
  wall (was: no response). Short tools request: 0.7 s. Plain: OK.

## Pre-existing, unrelated (observed while running the full suite)

- Full `swift test` run shows order-dependent failures at `ffe44eb` (identical
  with and without this fix): `testAttnBlockMrowMatchesSequential` rel_l2=1.034,
  `testBannedPrefillArgmaxMatchesMLX` argmax flip, and a `signal 5` crash in
  `testFlashHeadVsExactOnDummyAndChat` (Index out of range). Same tests pass in
  isolation. Suspect: fresh engines read recycled/uninitialized Metal buffer
  regions (process-state dependent), i.e. cross-test contamination. Not caused
  by this fix; tracked separately.