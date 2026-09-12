# Findings: momij structured_outputs (P1) — 2026-09-12

Branch: `cursor/omlx-dropin-api` (includes `399b4a4` + follow-on fixes).

## Implemented

Constrained decode matching oMLX's insurance surface (evprtr `EVPRTR_TOOLS_GRAMMAR=1`):

| Field | Backend | Result (live) |
|---|---|---|
| `structured_outputs.grammar` = `root ::= "ZQX"` | FiniteStringGuide | → `ZQX` |
| `guided_grammar` (same) | FiniteStringGuide | → `ZQX` |
| `structured_outputs.choice` | FiniteStringGuide | → `red` (first-alt on ties) |
| `structured_outputs.json` const / object | XGrammarTokenGuide | valid JSON |
| complex EBNF / non-literal `regex` | XGrammarTokenGuide | wired |
| Top-level `grammar` | — | **ignored** (oMLX parity) |

Dependency: `https://github.com/mattt/swift-xgrammar.git` from `0.1.0`.

## Related fixes (same day)

- `8b47021` — seedless `endEncoding` on encode errors; context 16k
- `a3fe719` — SWA absolute RoPE + sliding mask; default serve strips forced `<think>`  
  → unblocked Pi-length prompts for **mlx** markup primary (P0-E PASS)

## Known limits

- Stateful xgrammar matcher + sparse allowed-ID enumeration (full-vocab scan is slow)
- seedless long-prompt parity not re-verified after SWA work (Metal path separate)
- Markup primary strips OpenAI `tools` before `attach_tools_structured_outputs` — grammar insurance needs order fix on evprtr if used with markup
