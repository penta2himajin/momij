# Findings: long-prompt degeneration (P0-E) — 2026-09-12

## Symptom

mlx `serve` collapsed for `prompt_tokens ≳ 512–600` (raw `<|im_start|>`, `</think>` loops).
Matched Maple `sliding_window=512`, not chat length alone.

## Root cause (measured)

1. **SWA `KVCache.offset` clamped to `maxSize` after rotate** — RoPE reused pos≈512 on every decode step past the window. Oracle `RotatingKVCache` keeps absolute `offset` and a separate write head.
2. **Prefill `L > sliding_window` used full causal** — SWA layers attended past the trained window. Oracle uses `create_causal_mask(..., window_size=)`.
3. (Secondary) Stock Maple `chat_template.jinja` always opens `<|im_start|>assistant\n<think>\n`. Serve now strips that unless `MOMIJ_ENABLE_THINKING=1`.

## Fix

- `KVCache` rotating path ≈ mlx-lm `RotatingKVCache` (`updateConcat` / `updateInPlace`, absolute `offset`).
- `AttentionMasks.slidingCausal` + apply when `useRope && L > window` (mask dtype = query dtype).
- `ChatTemplatePatch` for non-thinking generation prompt.

## Re-measure (mlx release, temp=0)

| prompt_tokens | before | after |
|---|---|---|
| ≤409 | OK | OK |
| 539–2099 | fail (im_start / loops) | **OK** (`Reply with exactly: OK` → `OK`) |
| ~2400 (pi-ish) | fail | **OK** (`2+2` → `4`) |

## Still open

- Full Pi + compositor smoke (markup / tools).
- seedless long-prompt parity (Metal path has its own SWA; re-check).
- xgrammar perf on long prompts.
