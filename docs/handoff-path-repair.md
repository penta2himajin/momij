# Handoff: path-repair wiring (2026-09-14)

Continues from `96b169c` on `main`. Items 1–2 of the original plan are DONE
(`4b67fbf`), the engine long-context blocker found during live verification is
FIXED (`38c9811`). The next session picks up from "What's left" below.

## Context (why this exists)

The DSH-subagent verification loop exposed Maple-Preview's path
hallucination: the model emits tool calls with invented absolute paths
(`/mijij-subagent-test/...`, even a typo'd repo name) or paths with escape
debris. The repair pipeline catches these, but the "field re-ask" stage
re-asks the MODEL — and the model repeats its hallucination. The
deterministic fix: the task text names the exact relative path; extract it
programmatically.

## What's done and committed (all on main, tests green)

- **Item 1 — pathFromTask wiring** (`4b67fbf`): path-kind pending fixes
  (`path_outside_workspace`, aliased path fields, missing required path) are
  answered by `ArgRepair.taskPathMerge` (extraction from the task text)
  BEFORE any model re-ask; adoption gated by the shared `MomijHTTP.callsClean`;
  traced as `repair mode=field source=task`. Both paths. LIVE-FIRED 3× in the
  probe run.
- **Item 2 — ProtocolAbsorb** (`4b67fbf`): `MOMIJ_PROTOCOL_TOOLS` (default
  `submit,ask_user_question`); a parsed call targeting one of them drops ALL
  calls, finish=stop, description-or-placeholder as content; traced
  `protocol_absorb`. Both paths. (Not yet observed firing live — the probe
  run died before the model's ending fumble.)
- **Engine long-context fix** (`38c9811`): `MOMIJ_FULL_MAX_LEN` sizes the
  full-attn KV cache for the serve command; Server pre-flight rejects
  oversized prompts with an OpenAI-style 400; `maxPromptTokens` is a protocol
  requirement (extension-only declarations do NOT dispatch through
  `any LLMBackend` existentials — the exact live hang); capacity guard shared
  by ALL generate entry points (`generateSuffixSpec` had none — an oversized
  prompt prefilled for minutes with KV buffer-overrun risk).
  Live-verified: 17,023-token prompt → 200 (56 s); 78,223-token prompt →
  instant 400. Measured RSS: 2.99 GiB @16k → 3.25 GiB @32k.
- **testPathFromTask was a free function outside the class** in the WIP
  commit — XCTest never ran it; now in-class and executing.
- Everything from the previous handoff (PathPolicy, 3-stage repair,
  LoopBreaker, terminator recovery, lenient JSON, trace diagnostics,
  workspace wiring) — all still in place, suite 201/201.

## What's left (the next session's job)

1. **Extraction picked the wrong text (NEW, measured live)**: the repaired
   path was `github/PULL_REQUEST_TEMPLATE.md` — `originalUserText` selected
   the DSH-injected long context (which quotes `.github/PULL_REQUEST_TEMPLATE.md`
   from AGENTS.md) instead of the task text. Two sub-fixes:
   - Add `task_head` (first ~200 chars of the selected text) to the
     `repair mode=field source=task` trace event so the selection is
     observable; re-probe to confirm WHICH message wins.
   - Then decide the fix: prefer the task-named candidate (e.g. scan ALL
     candidate user texts, or bias toward the message containing the
     pending-fix tool's context). Measurement first — do not spec-fix.
2. **pathFromTask leading-dot trim**: token trimming strips a lone leading
   `.` (`.github/...` → `github/...`, observed in the adopted path). Strip
   only `./` pairs; keep hidden-file semantics. Red test included in the
   fix: `pathFromTask("per .github/PULL_REQUEST_TEMPLATE.md")` should keep
   the dot.
3. **protocol_absorb live confirmation**: needs a run where Maple emits its
   ending fumble; watch `.momij-traces` for `protocol_absorb`.
4. Optional (deferred): Needle 2 sidecar; momijctl docs note.

## Live verification setup (UPDATED)

The probe subagent must be CONFINED — Maple gravitates to writing
`.github/PULL_REQUEST_TEMPLATE.md`-style paths and the first unconfined run
wrote debris into the repo (`github/`, since removed):

```bash
mkdir -p /tmp/momij-probe/ws
ln -sfn ~/repos/momij/.momij-traces /tmp/momij-probe/ws/.momij-traces
MOMIJ_WORKSPACE=/tmp/momij-probe/ws MOMIJ_FULL_MAX_LEN=32768 \
  ./bin/momijctl restart
# one-shot subagent via DSH workflow: provider=momij model=maple-preview,
# task with RELATIVE paths only; audit via .momij-traces tool_args +
# git status before/after.
```

The daemon serves on 8755; DSH provider `momij` is wired in
`~/.dsh/settings.yaml` (contextWindow advertised 128000 — KV capacity is
now sized by MOMIJ_FULL_MAX_LEN instead of a hard 16384).

## Known model-quality limits (not fixable by more plumbing)

- Path CONTENT hallucination (`.github/PULL_REQUEST_TEMPLATE.md` picked from
  in-context AGENTS.md text instead of the task's path) — extraction is the
  intended fix but its TEXT SELECTION is currently wrong (item 1 above).
- Ending protocol: Maple tries submit/ask_user_question with wrong args
  after long tool-call runs — the interception (item 2) absorbs it.
- rope_theta=10000 with no scaling (config.json): serving past ~32k is
  untested for quality; capacity and correctness are separate questions.