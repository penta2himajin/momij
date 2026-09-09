# `git-hooks/`

Sharable git hooks for consumer repos.

## What's here

| File | Purpose |
|---|---|
| `pre-push` | Run format / lint / clippy locally before `git push`. Detects toolchains (Cargo, npm, ruff) by file presence and skips anything the repo doesn't have. |

## Install (per repo)

Point `core.hooksPath` at the checked-in directory so every contributor — and every Claude session — gets the same hooks without manual symlinks:

```bash
git config core.hooksPath git-hooks
```

If the consumer repo prefers `.git/hooks/` symlinks:

```bash
ln -s ../../git-hooks/pre-push .git/hooks/pre-push
```

## Rationale

- **Tests are not in the hook by default.** Under the TDD rule in `claude-md/user-level.md`, the suite is green at every commit, so re-running it on push is redundant. The `pre-push` template ships the `cargo test` / `pytest` / `npm test` lines commented out — uncomment if the project wants the extra safety net at the cost of slower pushes.
- **Format and lint are not redundant.** Auto-formatters and clippy/eslint/ruff catch issues that aren't captured by tests — style drift, dead code, unused imports, denied lints — and are cheap to run.
- **Skip-when-absent.** Each block guards on a marker file (`Cargo.toml`, `package.json`, `pyproject.toml`). A polyglot repo runs all that apply; a single-language repo runs only its own.

## Bypassing

`git push --no-verify` skips the hook. Do not use this to paper over a failure — fix the underlying issue. The hook exists so CI doesn't have to be the first place a `cargo fmt` violation is noticed.
