# Findings: momij structured_outputs (P1) — 2026-09-12

Branch: `cursor/omlx-dropin-api`.

## Implemented

Constrained decode matching oMLX's insurance surface (evprtr `EVPRTR_TOOLS_GRAMMAR=1`):

| Field | Backend | Result (live) |
|---|---|---|
| `structured_outputs.grammar` = `root ::= "ZQX"` | FiniteStringGuide | → `ZQX` |
| `guided_grammar` (same) | FiniteStringGuide | → `ZQX` |
| `structured_outputs.choice` | FiniteStringGuide | → `red` (first-alt on ties) |
| `structured_outputs.json` const / object | XGrammarTokenGuide | valid JSON (e.g. `"ZQX"` / `{"name":": ","ok":true}`) |
| complex EBNF / non-literal `regex` | XGrammarTokenGuide | wired |
| Top-level `grammar` | — | **ignored** (oMLX parity) |

Dependency: `https://github.com/mattt/swift-xgrammar.git` from `0.1.0`.

- Finite languages: `FiniteStringGuide` + token frontier.
- JSON / complex: `XGrammarTokenGuide` — tokenizer.json → byteLevel vocab; per step matcher replay + full-vocab bitmask (correct but slow on long prompts).

## Live smoke

`--backend mlx` and `--backend seedless` both produce `ZQX` / choice / JSON for short prompts (release binary).

## Not done / known limits

- Stateful xgrammar matcher + sparse allowed-ID enumeration (per-step full vocab scan is too slow for Pi-length prompts).
- Seedless serve still crashes under agentic load (`Command encoder released without endEncoding`); see evprtr `docs/findings/momij-upstream-e-2026-09-12.md`.
- P0-E Pi smoke still blocked on long-prompt degeneration (mlx) / seedless instability — not on missing grammar wiring.
