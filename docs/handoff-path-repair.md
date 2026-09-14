# Handoff: path-repair wiring (2026-09-14)

All three workstream items are DONE and LIVE-VERIFIED on `main`:
extraction-first path repair (`4b67fbf`, `921c4ca`, `f0df043`), protocol-tool
absorb (`4b67fbf`), and the seedless long-context capacity fix (`38c9811`).
The next session picks up from "What's left" below.

## Context (why this exists)

The DSH-subagent verification loop exposed Maple-Preview's path
hallucination: the model emits tool calls with invented absolute paths
(`/mijij-subagent-test/...`, even a typo'd repo name) or paths with escape
debris. The repair pipeline catches these, but the "field re-ask" stage
re-asks the MODEL — and the model repeats its hallucination. The
deterministic fix: the task text names the exact relative path; extract it
programmatically.

## What's done and committed (all on main, tests green)

- **Item 1 — extraction-first path repair** (`4b67fbf`, `921c4ca`, `f0df043`):
  path-kind pending fixes (`path_outside_workspace`, aliased path fields,
  missing required path) are answered by `ArgRepair.taskPathMerge` before any
  model re-ask; adoption gated by `MomijHTTP.callsClean`; traced as
  `repair mode=field source=task` with `task_head`. Both paths.
  LIVE-VERIFIED: probe 9 repaired a hallucinated absolute path into
  `/tmp/momij-probe/ws/scratch/momij-subagent-test/count_verify_fires.py`
  (the task-named path), `path_resolve resolved=1`, and the harness wrote the
  artifact there.
- **Item 2 — ProtocolAbsorb** (`4b67fbf`): `MOMIJ_PROTOCOL_TOOLS` (default
  `submit,ask_user_question`); a parsed call targeting one drops ALL calls,
  finish=stop, description-or-placeholder as content; traced `protocol_absorb`.
  LIVE-VERIFIED (probe 9): the model's ending
  `<tool_call>{"name": "submit", "arguments": {}}` was absorbed into a clean
  stop; the child's final answer became the placeholder "Task complete."
- **Item 3 — task-extraction input correctness** (`921c4ca`, `f0df043`): DSH
  sends harness scaffolding as SEPARATE user turns, each longer than the task,
  so the old "longest early" heuristic selected them. Three shapes measured
  live: `<system-reminder>` (adopted `.github/PULL_REQUEST_TEMPLATE.md`),
  `<tool_result>` (role "tool" folded to user by `rewriteMessages`), and
  `Current runtime context.` (DSH runtime/policy preamble → no path at all).
  Extraction now skips those (`ArgRepair.nonTaskTurnPrefixes`) and reads the
  ORIGINAL conversation (role "tool" intact) rather than the rewritten history.
  Silent field-repair failures are now traced
  (`no_task_path` / `unclean_after_task_merge` / `unclean_after_model_merge` /
  `model_reply_unparsable`), the parse event records the original role layout,
  and path-fix skips record candidate `user_heads`.
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

## What's left (the next session's job)

1. **Model behaviour, not plumbing** (measured, not fixable by repair):
   Maple's act/plan output flips with small prompt changes. It prose-ends some
   runs (plan text + an unterminated tool call), and its bash calls use
   absolute repo paths / the repo as workdir (`cd /Users/.../repos/momij &&
   python3 scratch/...`). `protocol_absorb` now ends those turns cleanly, but
   the task itself may not be completed. If probe work continues, pin the
   workflow meta name AND task text (they are part of the delivered prompt)
   and expect this variance.
2. **bash escapes the workspace confinement** (measured live): `MOMIJ_WORKSPACE`
   rewrites path FIELDS (`path`/`file_path`) only; a `bash` command string with
   `mkdir -p scratch/...` executes in the harness workspace and created
   `scratch/` inside the repo (removed). Path-field writes stayed confined.
   Options if stricter isolation is needed: withhold bash from the probe, or
   run the probe under a session whose workspace IS the probe dir.
3. Optional (deferred): Needle 2 sidecar; momijctl docs note.

## Live verification setup (UPDATED, confinement-hardened)

```bash
mkdir -p /tmp/momij-probe/ws
mkdir -p /tmp/momij-probe/ws/.momij-traces
cp ~/repos/momij/.momij-traces/tr-*.json /tmp/momij-probe/ws/.momij-traces/
MOMIJ_WORKSPACE=/tmp/momij-probe/ws MOMIJ_FULL_MAX_LEN=32768 \
  ./bin/momijctl restart
# one-shot subagent via DSH workflow: provider=momij model=maple-preview,
# task with RELATIVE paths only.
# Audit: .momij-traces (repair/path_resolve/protocol_absorb + parse roles +
# task_head/user_heads), git status, and the probe ws contents.
```

Use REAL trace COPIES, not a symlink: a symlinked `.momij-traces` let a model
write land back inside the repo (observed).

## Known model-quality limits (not fixable by more plumbing)

- Path CONTENT hallucination is repaired from the task text (item 1/3), but
  the model's own choice of path/command remains unreliable.
- Act/plan flakiness: measured deterministic at the engine level — three
  identical short requests produced byte-identical content/usage, and an
  identical task+meta reproduced the acting behaviour. The flip therefore
  comes from prompt content differing between runs (workflow meta name, task
  escaping, DSH-injected context), not from momij nondeterminism. An earlier
  "M-row prefill boundary" hypothesis was retracted after this measurement.
- Ending protocol: submit/ask_user_question after long runs — absorbed.
- rope_theta=10000 with no scaling (config.json): serving past ~32k is
  untested for quality; capacity and correctness are separate questions.
