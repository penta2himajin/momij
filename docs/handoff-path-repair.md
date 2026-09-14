# Handoff: path-repair wiring (2026-09-14)

WIP commit `97ef787` — continues from `96b169c` (outside-workspace drop
checks) on `main`. The next session picks up from here.

## Context (why this exists)

The DSH-subagent verification loop exposed Maple-Preview's path
hallucination: the model emits tool calls with invented absolute paths
(`/mijij-subagent-test/...`, even a typo'd repo name) or paths with
escape debris (`\\\\/Users`). The repair pipeline catches these, but the
"field re-ask" stage re-asks the MODEL for the path — and the model
repeats its hallucination. The deterministic fix: **the task text names
the exact relative path; extract it programmatically instead of asking
the model.**

## What's done and committed (all on main, tests green)

- **PathPolicy** (`swift/Sources/MomijCore/PathPolicy.swift`): workspace-
  relative path mode behind `MOMIJ_WORKSPACE` — resolve(relative ->
  absolute), escape normalization, root-prefix stripping from the
  conversation, isOutsideWorkspace, suffixLine instruction. 10 tests.
- **PathFromTask** (`ArgRepair.pathFromTask`, this commit): deterministic
  path extraction from the task text. 11 ArgRepair tests green incl. the
  live task shape ("Write a Python 3 script at
  scratch/momij-subagent-test/count_verify_fires.py ..."). NOT YET WIRED
  into the Server (see next).
- **3-stage repair** (field re-ask -> whole-reply nudge -> drop) on both
  non-stream and stream paths, with schema-required repair
  (missingRequiredFields), path<->file_path alias (repairFields),
  outside-workspace as a repairable violation, and the broken-call drop
  (all live-verified firing).
- **LoopBreaker**: trailing-run detection of identical calls
  (escape-escalation variants normalize equal) + steering message.
  Live-verified firing on a 12-turn chmod retry loop.
- **Terminator-completion recovery** for unterminated tool-call blocks;
  **lenient JSON parse** (literal newlines/tabs inside string values —
  the multiline `python3 -c` failure class).
- **Trace diagnostics**: raw_head / content_head / tool_args on present
  events (these found every failure class above).
- **Workspace mode wiring in Server** (`Sources/momij/Server.swift`):
  path-convention line in the tools instruction, root-prefix strip on the
  model-visible history, path_resolve on outgoing calls, outside-root
  repair in the pendingFix scan, workspace note in repair prompts.

## What's left (the next session's job)

1. **Wire pathFromTask into the Server pendingFix flow** (extraction
   first): in `MomijHTTP`'s repair stage (non-stream ~line 335, stream
   ~line 790 in `Sources/momij/Server.swift`), when the pending fix is a
   path field (`kind == "path_outside_workspace"` or a path alias from
   `degenerate_tool_args`), FIRST try `ArgRepair.pathFromTask(task)` on
   `ArgRepair.originalUserText(effective.messages)`; if it returns a
   path, merge it into the broken call's args (skip the model re-ask
   entirely), run the clean check, and adopt. Only fall back to the
   model re-ask when the task names no path. Trace the event as
   `repair mode=field source=task` so the fire rate is measurable.
2. **Protocol-tool interception** (the ending fumble): env
   `MOMIJ_PROTOCOL_TOOLS` (default `submit,ask_user_question`). When a
   parsed call targets one of these, drop all calls and set finish=stop
   with the call's `description` (or a placeholder) as the final content
   — the model's "I'm done" intent becomes a clean ending instead of a
   rejection loop. Both paths; trace as `protocol_absorb`.
3. **Optional (deferred)**: real Needle 2 sidecar. cactus-needle is
   locally available (`~/models/Cactus-Compute/needle2`,
   `import needle` works in the evprtr venv). A small Python HTTP
   sidecar wrapping `NeedleToolRuntime` would cover the remaining
   semantic cases; the deterministic extraction covers the observed
   ones.
4. **momijctl**: consider a `MOMIJ_WORKSPACE` passthrough note in
   `bin/momijctl` docs (the env propagates automatically through nohup;
   nothing to change — just start the daemon with it set).

## Live verification setup

```bash
MOMIJ_WORKSPACE=/Users/penta2himajin/repos ./bin/momijctl restart
# then run the workflow task with the RELATIVE path in the prompt and
# check .momij-traces for repair/path_resolve/loop_break events
```

The daemon on port 8755 serves `maple-preview`; DSH provider `momij`
(points at the same port) is wired in `~/.dsh/settings.yaml`.

## Known model-quality limits (not fixable by more plumbing)

- Path CONTENT hallucination (`mijij` vs `momij` typo) — the repairs fix
  the FORM; content comes from the task extraction (this commit).
- Ending protocol: Maple tries submit/ask_user_question with wrong args
  after long tool-call runs — the interception (item 2) absorbs it.
- The task prompt should carry the EXACT relative path; the model copies
  it imperfectly — the extraction (item 1) is the reliable path.