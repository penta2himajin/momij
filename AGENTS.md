# <Project Name>

## Overview

<!-- One to three paragraphs describing the project's purpose, target domain, and distinguishing characteristics. If detailed specs live under docs/, reference them with @docs/<file>.md. -->

## Project Structure

<!-- Directory layout with the role of each. Make explicit the boundary between source code, documentation, and generated artifacts. -->

```
src/         # ...
docs/        # ...
tests/       # ...
```

## Development Setup

<!-- Required toolchain pins, bootstrap commands, external dependencies (DB, MCP servers). -->

```bash
# example
cargo install ...

# Pre-push hook (format / lint / clippy).
cp git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

## Build & Test

<!-- Canonical verification commands. Must be runnable without prior setup so agents can self-verify. -->

```bash
cargo build --workspace
cargo test  --workspace
```

## Development Principles

<!-- Project-specific additions only. Do not restate the common rules below. Examples:
- "All features touching target-adjacent columns must be registered in LEAK_FEATURES."
- "Public API changes require an ADR in docs/decisions/." -->

## Architectural Boundaries

<!-- Structural invariants that, if violated, break the design. Examples:
- "core crate stays domain-agnostic."
- "Generated code under gen/ is never hand-edited."
- "Layer X must not depend on layer Y." -->

## Prohibitions

<!-- Numbered list of "do not" rules, written so each is verifiable. Do not duplicate the common prohibitions below. -->

1. ...
2. ...

## Git Conventions

<!-- Differences from the common rules below. Examples: scoped Conventional Commits like `feat(phase1d):`, mandatory issue links in PR bodies. -->

## Session Handoff

Long-running workstreams use GitHub issues for cross-session continuity. See `docs/handoff-protocol.md` for the full protocol.

- Label: `session-handoff`
- One issue per workstream (not per session)
- On session start, read the relevant handoff issue and confirm the **Next action** with the user before executing.

## Internationalisation

If this project ships a Japanese-facing entry point, follow `docs/i18n-policy.md`:

- Translations are suffix files (`README.ja.md` next to `README.md`); no language directories.
- Only `README.md` and the user-facing introduction tier of `docs/` are in scope. Engineering docs and ADRs stay English-only.
- Each translated file carries a `> Source: <name>.md @ <sha>` header. PRs are never blocked on translation parity.

---

<!-- Common rules below this line apply to every project. -->

## Common Development Rules

### TDD (Red → Green → Refactor)

All implementation work proceeds in this cycle:

1. **Red**: write a failing test that captures the intended behaviour.
2. **Green**: write the minimum code that makes the test pass.
3. **Refactor**: tidy up while keeping tests green.

When a test fails, fix the production code — do not delete, skip, or weaken the test.

### Measure, Don't Conjecture

Base decisions on observed data, not assumptions. Before optimising, claiming a bottleneck, or asserting that something is slow or broken, measure it — profile, benchmark, log, or reproduce. When you report a cause, cite the measurement that supports it.

### Git Conventions

- **Conventional Commits**: `feat:` `fix:` `docs:` `refactor:` `test:` `ci:` `chore:`. Project-specific prefixes (e.g. `data:`, `experiments:`) live in the project's `AGENTS.md`.
- **Branch naming**: use a short prefix for the agent or author followed by a topic, e.g. `claude/<topic>`, `codex/<topic>`, or `human/<topic>`.
- **Trailer**: when an AI agent authors the commit, append a trailer crediting the agent. Do not embed model name or session info in the trailer; put those in the commit body if needed.
- **Pre-push hook**: install via `cp git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push` (or `git config core.hooksPath git-hooks`). The hook runs format / lint / clippy before every push. Tests are intentionally omitted — TDD keeps them green at commit time.

### Pull Requests

- **Always ready for review.** Open PRs in the "ready" state, never as drafts. Draft PRs do not fire review-requested events and slow the loop.
- **Auto-subscribe after creating a PR.** Immediately after the PR is created, subscribe to its activity without asking the user. Rationale: the user explicitly opted into the "agent opens and watches its own PRs" workflow at the template level, so the per-PR confirmation is noise. Unsubscribe only when the user says to stop, when the PR merges, or when it is closed unmerged.
- **One PR per workstream**, matching the handoff issue. Reference the issue with `Closes #N` per `.github/PULL_REQUEST_TEMPLATE.md`.

### Stream Idle Timeout Mitigation

Cloud agent sessions occasionally fail with `Stream idle timeout - partial response received` on long output. To reduce risk:

1. **Stage long writes.** For long documents or source files, write the skeleton (headings, function signatures, trait stubs) first, then fill each section in follow-up edits. Avoid single blocks larger than ~200 lines.
2. **Watch out after large reads.** Reading a big file (e.g. `Cargo.lock`, large generated modules) and then immediately producing long output is a common trigger. Split into separate turns or excerpt only the relevant portion.
3. **Recover carefully.** A timeout can still leave the file write completed. Run `git status` before retrying so the same content is not written twice.

### Common Prohibitions

1. Do not delete, skip, or comment out existing tests.
2. Do not modify CI configuration without explicit instruction.
3. Do not weaken production code merely to make tests pass.
4. Do not commit credentials, API keys, signed URLs, or anything in `.env*`.
