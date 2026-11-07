//! oMLX-compatible constrained-decoding request fields.
//!
//! Port of `StructuredOutputs.swift` (minus `ConstrainedPick.hostScores`, which
//! reads MLX arrays). Finite languages (`choice`, literal-union EBNF/regex) are
//! recognised here; open JSON Schema / regex / EBNF are carried through so the
//! backend can hand them to a real grammar engine (llama.cpp's GBNF).

use std::collections::HashSet;
use std::fmt;

use serde_json::Value;

use crate::TokenId;
use crate::json::trim_ws_nl;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constraint {
    /// One of the exact strings (then EOS).
    Choice(Vec<String>),
    /// JSON Schema string.
    JsonSchema(String),
    /// Full EBNF (non-literal).
    Ebnf(String),
    Regex(String),
    /// Opaque / not enforced.
    Unsupported(String),
}

/// Extract the constraint from a chat.completions JSON object.
/// Prefer `structured_outputs.*`; `guided_grammar` is an alias for grammar.
pub fn parse_constraint(root: &Value) -> Option<Constraint> {
    if let Some(so) = root.get("structured_outputs").and_then(Value::as_object) {
        if let Some(choices) = so.get("choice").and_then(Value::as_array) {
            let strs: Vec<String> = choices
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            if !strs.is_empty() {
                return Some(Constraint::Choice(strs));
            }
        }
        if let Some(g) = so.get("grammar").and_then(Value::as_str) {
            return Some(parse_literal_union_ebnf(g).unwrap_or_else(|| Constraint::Ebnf(g.into())));
        }
        if let Some(r) = so.get("regex").and_then(Value::as_str) {
            return Some(
                parse_literal_union_regex(r).unwrap_or_else(|| Constraint::Regex(r.into())),
            );
        }
        if let Some(json) = so.get("json") {
            return Some(match json {
                Value::String(s) => Constraint::JsonSchema(s.clone()),
                Value::Object(_) | Value::Array(_) => match serde_json::to_string(json) {
                    Ok(s) => Constraint::JsonSchema(s),
                    Err(_) => {
                        Constraint::Unsupported("structured_outputs.json (unserializable)".into())
                    }
                },
                _ => Constraint::Unsupported("structured_outputs.json (unserializable)".into()),
            });
        }
    }
    if let Some(g) = root.get("guided_grammar").and_then(Value::as_str) {
        return Some(parse_literal_union_ebnf(g).unwrap_or_else(|| Constraint::Ebnf(g.into())));
    }
    // Top-level `grammar` is ignored (oMLX parity).
    None
}

/// `root ::= "a" | "b"` (optional whitespace). `None` if not this shape.
pub fn parse_literal_union_ebnf(src: &str) -> Option<Constraint> {
    let mut s = trim_ws_nl(src);
    if s.starts_with("root") {
        s = s.find('=').map_or("", |i| &s[i..]);
        if let Some(rest) = s.strip_prefix("::=") {
            s = rest;
        } else if let Some(rest) = s.strip_prefix('=') {
            s = rest;
        }
        s = trim_ws_nl(s);
    }
    parse_quoted_alternation(s)
}

/// `^(?:a|b)$` or `a|b` of quoted/plain literals without regex metachars.
pub fn parse_literal_union_regex(src: &str) -> Option<Constraint> {
    let mut s = trim_ws_nl(src);
    if let Some(rest) = s.strip_prefix('^') {
        s = rest;
    }
    if let Some(rest) = s.strip_suffix('$') {
        s = rest;
    }
    if s.starts_with("(?:") && s.ends_with(')') {
        s = &s[3..s.len() - 1];
    }
    parse_quoted_alternation(s)
}

fn parse_quoted_alternation(src: &str) -> Option<Constraint> {
    let parts = split_top_level_alternation(src);
    if parts.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(parts.len());
    for p in &parts {
        let lit = unwrap_quoted_literal(trim_ws_nl(p))?;
        // Reject regex/ebnf metachar leftovers inside tokens.
        if lit.chars().any(|c| "[](){}*+?.\\".contains(c)) {
            return None;
        }
        out.push(lit);
    }
    Some(Constraint::Choice(out))
}

fn unwrap_quoted_literal(s: &str) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() >= 2 {
        let (a, b) = (chars[0], chars[chars.len() - 1]);
        if (a == '"' && b == '"') || (a == '\'' && b == '\'') {
            return Some(chars[1..chars.len() - 1].iter().collect());
        }
    }
    // Bare token (ZQX, red, ...): letters, digits and `_` only.
    if s.is_empty() || !s.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(s.to_owned())
}

fn split_top_level_alternation(src: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut depth = 0usize;
    let mut in_quote: Option<char> = None;
    for ch in src.chars() {
        if let Some(q) = in_quote {
            cur.push(ch);
            if ch == q {
                in_quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => {
                in_quote = Some(ch);
                cur.push(ch);
            }
            '(' => {
                depth += 1;
                cur.push(ch);
            }
            ')' => {
                depth = depth.saturating_sub(1);
                cur.push(ch);
            }
            '|' if depth == 0 => parts.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuideError {
    EncodeEmpty(String),
    DeadEnd,
    NoAllowedToken,
    /// A tokenizer failure carried through the `encode` callback.
    Encode(String),
}

impl fmt::Display for GuideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EncodeEmpty(s) => write!(f, "encode produced no tokens for {s:?}"),
            Self::DeadEnd => f.write_str("constraint dead-end"),
            Self::NoAllowedToken => f.write_str("no allowed next token under constraint"),
            Self::Encode(s) => write!(f, "encode failed: {s}"),
        }
    }
}

impl std::error::Error for GuideError {}

/// Greedy constrained decode over a finite set of UTF-8 strings.
///
/// Builds token sequences with `encode` (no special tokens), then at each step
/// only tokens that continue at least one remaining alternative are allowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteStringGuide {
    pub alternatives: Vec<Vec<TokenId>>,
    pub eos_token_id: TokenId,
}

impl FiniteStringGuide {
    pub fn new<S: AsRef<str>>(
        strings: &[S],
        mut encode: impl FnMut(&str) -> Result<Vec<TokenId>, GuideError>,
        eos_token_id: TokenId,
    ) -> Result<Self, GuideError> {
        let mut alternatives = Vec::with_capacity(strings.len());
        for s in strings {
            let s = s.as_ref();
            let ids = encode(s)?;
            if ids.is_empty() && !s.is_empty() {
                return Err(GuideError::EncodeEmpty(s.to_owned()));
            }
            alternatives.push(ids);
        }
        Ok(Self {
            alternatives,
            eos_token_id,
        })
    }

    /// Alternatives still compatible with `prefix` (generated so far).
    pub fn surviving(&self, prefix: &[TokenId]) -> Vec<&[TokenId]> {
        self.alternatives
            .iter()
            .filter(|alt| alt.len() >= prefix.len() && alt[..prefix.len()] == *prefix)
            .map(Vec::as_slice)
            .collect()
    }

    /// Allowed next token ids given generated `prefix`. Includes EOS when a
    /// surviving alternative is exactly complete.
    pub fn allowed_next(&self, prefix: &[TokenId]) -> HashSet<TokenId> {
        self.surviving(prefix)
            .into_iter()
            .map(|alt| {
                if alt.len() == prefix.len() {
                    self.eos_token_id
                } else {
                    alt[prefix.len()]
                }
            })
            .collect()
    }

    /// Argmax among `scores` restricted to `allowed`; `ids[i]` pairs with `scores[i]`.
    pub fn argmax_allowed(
        ids: &[TokenId],
        scores: &[f32],
        allowed: &HashSet<TokenId>,
    ) -> Option<TokenId> {
        let mut best_id = None;
        let mut best = -f32::MAX;
        for (i, id) in ids.iter().enumerate() {
            if allowed.contains(id) && scores[i] > best {
                best = scores[i];
                best_id = Some(*id);
            }
        }
        best_id
    }
}

/// Host-side argmax among a grammar-allowed set.
pub mod constrained_pick {
    use std::collections::HashSet;

    use crate::TokenId;

    /// Highest-scoring allowed id. Ties break toward the lowest id so the
    /// result is deterministic (the Swift original iterated an unordered set).
    pub fn argmax(allowed: &HashSet<TokenId>, scores: &[f32]) -> Option<TokenId> {
        let mut best: Option<(TokenId, f32)> = None;
        for &id in allowed {
            let Some(&s) = usize::try_from(id).ok().and_then(|i| scores.get(i)) else {
                continue;
            };
            let better = match best {
                None => s > -f32::MAX,
                Some((bid, bs)) => s > bs || (s == bs && id < bid),
            };
            if better {
                best = Some((id, s));
            }
        }
        best.map(|(id, _)| id)
    }

    pub fn apply_banned(scores: &mut [f32], banned: &[TokenId]) {
        for &id in banned {
            if let Some(slot) = usize::try_from(id).ok().and_then(|i| scores.get_mut(i)) {
                *slot = f32::NEG_INFINITY;
            }
        }
    }

    /// Index of the first maximum.
    pub fn argmax_all(scores: &[f32]) -> TokenId {
        let mut best_id = 0usize;
        let mut best = -f32::MAX;
        for (i, &s) in scores.iter().enumerate() {
            if s > best {
                best = s;
                best_id = i;
            }
        }
        best_id as TokenId
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ascii(s: &str) -> Result<Vec<TokenId>, GuideError> {
        Ok(s.chars().map(|c| c as TokenId).collect())
    }

    #[test]
    fn parse_literal_ebnf() {
        assert_eq!(
            parse_literal_union_ebnf(r#"root ::= "ZQX""#),
            Some(Constraint::Choice(vec!["ZQX".into()]))
        );
        assert_eq!(
            parse_literal_union_ebnf(r#"root ::= "red" | "green" | "blue""#),
            Some(Constraint::Choice(vec![
                "red".into(),
                "green".into(),
                "blue".into()
            ]))
        );
    }

    #[test]
    fn non_literal_grammar_is_not_a_choice() {
        assert_eq!(parse_literal_union_ebnf(r#"root ::= [a-z]+"#), None);
        let c = parse_constraint(&json!({"guided_grammar": "root ::= [a-z]+"}));
        assert_eq!(c, Some(Constraint::Ebnf("root ::= [a-z]+".into())));
    }

    #[test]
    fn parse_literal_regex() {
        assert_eq!(
            parse_literal_union_regex("^(?:red|green)$"),
            Some(Constraint::Choice(vec!["red".into(), "green".into()]))
        );
        assert_eq!(parse_literal_union_regex(r"^\d+$"), None);
        let c = parse_constraint(&json!({"structured_outputs": {"regex": r"^\d+$"}}));
        assert_eq!(c, Some(Constraint::Regex(r"^\d+$".into())));
    }

    #[test]
    fn structured_outputs_json_object_becomes_schema_string() {
        let c = parse_constraint(&json!({"structured_outputs": {"json": {"type": "object"}}}));
        assert_eq!(
            c,
            Some(Constraint::JsonSchema(r#"{"type":"object"}"#.into()))
        );
        let c = parse_constraint(&json!({"structured_outputs": {"json": 5}}));
        assert!(matches!(c, Some(Constraint::Unsupported(_))));
    }

    #[test]
    fn empty_choice_list_falls_through_to_other_keys() {
        let c = parse_constraint(
            &json!({"structured_outputs": {"choice": [""], "grammar": "root ::= \"a\""}}),
        );
        assert_eq!(c, Some(Constraint::Choice(vec!["a".into()])));
    }

    #[test]
    fn finite_string_guide_frontier() {
        let guide = FiniteStringGuide::new(&["ab", "ac"], ascii, 999).unwrap();
        let a = 'a' as TokenId;
        let (b, c) = ('b' as TokenId, 'c' as TokenId);
        assert_eq!(guide.allowed_next(&[]), HashSet::from([a]));
        assert_eq!(guide.allowed_next(&[a]), HashSet::from([b, c]));
        assert_eq!(guide.allowed_next(&[a, b]), HashSet::from([999]));
    }

    #[test]
    fn finite_string_guide_rejects_unencodable_string() {
        let err = FiniteStringGuide::new(&["x"], |_| Ok(vec![]), 1).unwrap_err();
        assert_eq!(err, GuideError::EncodeEmpty("x".into()));
    }

    /// Port of `testSeedlessConstrainedWalkEmitsLiteral`: walking the frontier
    /// (lowest id first, EOS last) emits the literal then EOS.
    #[test]
    fn constrained_walk_emits_literal() {
        let guide = FiniteStringGuide::new(&["ZQX"], ascii, 7).unwrap();
        let mut out: Vec<TokenId> = Vec::new();
        while out.len() < 16 {
            let allow = guide.allowed_next(&out);
            let mut cont: Vec<TokenId> = allow.iter().copied().filter(|&t| t != 7).collect();
            cont.sort_unstable();
            let pick = match cont.first() {
                Some(&t) => t,
                None if allow.contains(&7) => 7,
                None => break,
            };
            out.push(pick);
            if pick == 7 {
                break;
            }
        }
        let mut want: Vec<TokenId> = "ZQX".chars().map(|c| c as TokenId).collect();
        want.push(7);
        assert_eq!(out, want);
    }

    /// Open JSON must not pick the lowest allowed id (`!` is often vocab 0-ish).
    #[test]
    fn constrained_pick_uses_scores_not_lowest_id() {
        let mut scores = vec![0.0f32; 16];
        scores[1] = 1.0;
        scores[5] = 9.0;
        scores[7] = 3.0;
        let allowed = HashSet::from([1, 5, 7]);
        assert_eq!(constrained_pick::argmax(&allowed, &scores), Some(5));
        assert_ne!(allowed.iter().min().copied(), Some(5));
    }

    #[test]
    fn constrained_pick_ties_and_bounds() {
        let scores = vec![0.0f32, 2.0, 2.0];
        assert_eq!(
            constrained_pick::argmax(&HashSet::from([1, 2]), &scores),
            Some(1)
        );
        assert_eq!(
            constrained_pick::argmax(&HashSet::from([99, -1]), &scores),
            None
        );
        let mut s = vec![1.0f32, 2.0, 3.0];
        constrained_pick::apply_banned(&mut s, &[2, 50, -3]);
        assert_eq!(s[2], f32::NEG_INFINITY);
        assert_eq!(constrained_pick::argmax_all(&s), 1);
    }

    #[test]
    fn argmax_allowed_pairs_ids_with_scores() {
        let ids = [10, 20, 30];
        let scores = [0.1f32, 0.9, 0.5];
        let allowed = HashSet::from([10, 30]);
        assert_eq!(
            FiniteStringGuide::argmax_allowed(&ids, &scores, &allowed),
            Some(30)
        );
    }
}
