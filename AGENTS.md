# momij

## Overview

Maple-Preview (deepgrove 20B-A1B ternary MoE) high-speed inference engine for
Apple Silicon. Primary product: OpenAI-compatible server intended to replace
oMLX when serving Maple. See @docs/baseline.md and @docs/research-catalog.md.

## Project Structure

```
swift/Sources/MomijCore/   # engine, Seedless Metal, SuffixSpec
swift/Sources/momij/       # CLI + OpenAI server
python/momij_oracle/       # mlx-lm-deepgrove JSONL worker (baseline oracle)
docs/                      # baseline + research catalog
bench/                     # recorded results
```

## Development Setup

```bash
git config core.hooksPath git-hooks
cd swift && swift build
# Oracle path needs:
#   ~/repos/mlx-lm-deepgrove/.venv
```

## Build & Test

```bash
cd swift
swift build
swift test
swift run momij -- seedless-bench
```

## Development Principles

- Measurement first: reproduce `docs/baseline.md` before claiming speedups.
- Stop omlx before GPU benches.
- DeepGrove custom lib: measure delta before duplicating.

## Architectural Boundaries

- Production decode target = Swift raw Metal (Seedless), not Python op-graph.
- Oracle (Python) is for baseline parity and fallback, not the long-term hot path.
- Do not port Qwisp GDN / 4-bit Qwen kernels; Maple is 2-bit ternary + SWA/NoPE.

## Prohibitions

1. Do not weaken baseline harness numbers.
2. Do not modify CI without explicit instruction.
3. Do not commit credentials or `.env*`.

## Git Conventions

- Conventional Commits; branch prefix `cursor/<topic>` or `human/<topic>`.
- Agent commits append an agent trailer (no model name in trailer).

## Session Handoff

See `docs/handoff-protocol.md`. Label: `session-handoff`.

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

- **Conventional Commits**: `feat:` `fix:` `docs:` `refactor:` `test:` `ci:` `chore:`.
- **Branch naming**: short prefix + topic, e.g. `cursor/<topic>`, `codex/<topic>`, or `human/<topic>`.
- **Trailer**: when an AI agent authors the commit, append a trailer crediting the agent.
- **Pre-push hook**: `cp git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push`.

### Pull Requests

- Always ready for review (never draft).
- Auto-subscribe after creating a PR.
- One PR per workstream; `Closes #N` per `.github/PULL_REQUEST_TEMPLATE.md`.

### Common Prohibitions

1. Do not delete, skip, or comment out existing tests.
2. Do not modify CI configuration without explicit instruction.
3. Do not weaken production code merely to make tests pass.
4. Do not commit credentials, API keys, signed URLs, or anything in `.env*`.
