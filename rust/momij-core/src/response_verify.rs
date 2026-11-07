//! Verify/repair policies ported from evprtr `compositor/verify/`.
//! Port of `ResponseVerify.swift`.
//!
//! `policy_id` / `kind` / `detail` values match the Python originals
//! byte-for-byte so trace records can be compared across both implementations
//! (evprtr as oracle, momij as native runtime). Offsets are **code points**,
//! like Python (the Swift original used UTF-16 units; identical for ASCII).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::JsonMap;
use crate::json::{trim_blanks, trim_ws_nl};
use crate::openai_compat::ToolCallSpec;

pub const GENERIC_REPETITION_ID: &str = "generic.repetition";
pub const THIN_CONTENT_ID: &str = "maple_preview.thin_content";
pub const EMPTY_LONG_REASONING_ID: &str = "maple_preview.empty_content_long_reasoning";
pub const PSEUDO_TOOL_MARKUP_ID: &str = "maple_preview.pseudo_tool_markup";
pub const DEGENERATE_TOOL_ARGS_ID: &str = "maple_preview.degenerate_tool_args";

// Markup markers, assembled at compile time so source text never embeds a
// contiguous marker (same trick `tool_markup` callers use for the call tags).
const TOOL_CALL_OPEN: &str = concat!("<", "tool_call");
pub const THINK_OPEN: &str = concat!("<", "think", ">");
pub const THINK_CLOSE: &str = concat!("<", "/", "think", ">");

/// A verify hit (evprtr `Diagnosis`).
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub policy_id: String,
    pub field: String,
    pub kind: String,
    pub onset: usize,
    pub detail: JsonMap,
}

impl Hit {
    pub fn event_detail(&self) -> Value {
        json!({
            "policy_id": self.policy_id,
            "field": self.field,
            "kind": self.kind,
            "onset": self.onset,
            "detail": self.detail,
        })
    }
}

/// Raw repetition finding (evprtr `RepetitionHit`).
#[derive(Debug, Clone, PartialEq)]
pub struct RepetitionHit {
    pub kind: String,
    pub onset: usize,
    pub detail: JsonMap,
}

/// Assistant-message view the detectors operate on (evprtr passes the OpenAI
/// message dict; momij passes decoded channels explicitly).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Message {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallSpec>,
}

impl Message {
    pub fn content(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            ..Self::default()
        }
    }

    pub fn with_tool_calls(tool_calls: Vec<ToolCallSpec>) -> Self {
        Self {
            tool_calls,
            ..Self::default()
        }
    }
}

/// Per-request verdicts: every fired detector (bundle order). `first()` is the
/// evprtr-equivalent first hit (their pipeline short-circuits).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    pub hits: Vec<Hit>,
}

impl Outcome {
    pub fn first(&self) -> Option<&Hit> {
        self.hits.first()
    }
}

fn py_prefix(chars: &[char], n: usize) -> String {
    chars.iter().take(n).collect()
}

fn map(value: Value) -> JsonMap {
    match value {
        Value::Object(m) => m,
        _ => JsonMap::new(),
    }
}

// MARK: - repetition.py port

/// `(start, end)` char ranges of whitespace-delimited tokens (`\S+`).
fn token_ranges(chars: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in chars.iter().enumerate() {
        match (c.is_whitespace(), start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, chars.len()));
    }
    out
}

/// First match of `(.{2,24}?)\1{7,}` (the dot does not match `\n`, like
/// Python): `(start, unit_len, span)`. `regex` has no back-references, so the
/// scan is explicit: leftmost start, shortest unit first.
fn first_char_motif(chars: &[char]) -> Option<(usize, usize, usize)> {
    let n = chars.len();
    for i in 0..n {
        if chars[i] == '\n' {
            continue;
        }
        for len in 2..=24 {
            if i + len > n || chars[i + len - 1] == '\n' {
                break;
            }
            let unit = &chars[i..i + len];
            let mut repeats = 0;
            let mut pos = i + len;
            while pos + len <= n && &chars[pos..pos + len] == unit {
                repeats += 1;
                pos += len;
            }
            if repeats >= 7 {
                return Some((i, len, len * (1 + repeats)));
            }
        }
    }
    None
}

/// Detect runaway token/phrase repetition. Port of
/// `compositor/verify/repetition.py:find_repetition` (same thresholds:
/// word_run 8, ngram 3x5, char motif 2..24 chars repeated 8+ times).
pub fn find_repetition(text: &str) -> Option<RepetitionHit> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 32 {
        return None;
    }
    let mut hits: Vec<RepetitionHit> = Vec::new();
    let ranges = token_ranges(&chars);
    let tok = |i: usize| &chars[ranges[i].0..ranges[i].1];

    // Word run: >= 8 identical consecutive tokens.
    if ranges.len() >= 8 {
        let mut run = 1;
        for i in 1..ranges.len() {
            if tok(i) == tok(i - 1) {
                run += 1;
                if run >= 8 {
                    hits.push(RepetitionHit {
                        kind: "word_run".into(),
                        onset: ranges[i + 1 - run].0,
                        detail: map(json!({"token": py_prefix(tok(i), 80), "run": run})),
                    });
                    break;
                }
            } else {
                run = 1;
            }
        }
    }

    // N-gram run: >= 5 consecutive identical 3-grams (space-joined tokens).
    if ranges.len() >= 3 * 5 {
        let same_gram = |a: usize, b: usize| (0..3).all(|d| tok(a + d) == tok(b + d));
        let mut run = 1;
        for i in 1..=(ranges.len() - 3) {
            if same_gram(i, i - 1) {
                run += 1;
                if run >= 5 {
                    let gram = (0..3)
                        .map(|d| tok(i + d).iter().collect::<String>())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let gram_chars: Vec<char> = gram.chars().collect();
                    hits.push(RepetitionHit {
                        kind: "ngram_run".into(),
                        onset: ranges[i + 1 - run].0,
                        detail: map(
                            json!({"ngram": py_prefix(&gram_chars, 120), "run": run, "n": 3}),
                        ),
                    });
                    break;
                }
            } else {
                run = 1;
            }
        }
    }

    // Char motif: a 2..24 char unit repeated >= 8 times contiguously. Only the
    // FIRST regex match counts (evprtr `re.search`). Guard (evprtr parity,
    // 2026-09-13 fix): whitespace-only motifs (markdown hard breaks, code
    // indentation runs) are idiomatic formatting, not model collapse.
    if let Some((start, len, span)) = first_char_motif(&chars) {
        let motif = &chars[start..start + len];
        let as_string: String = motif.iter().collect();
        if !trim_ws_nl(&as_string).is_empty() {
            hits.push(RepetitionHit {
                kind: "char_motif".into(),
                onset: start,
                detail: map(json!({"motif": py_prefix(motif, 80), "span": span})),
            });
        }
    }

    // Earliest onset wins; ties keep bundle order (word_run, ngram_run, char_motif).
    let mut iter = hits.into_iter();
    let mut best = iter.next()?;
    for h in iter {
        if h.onset < best.onset {
            best = h;
        }
    }
    Some(best)
}

/// `text[: onset].rstrip()` (evprtr `truncate_before_repetition`).
pub fn truncate_before_repetition(text: &str) -> String {
    let Some(hit) = find_repetition(text) else {
        return text.to_owned();
    };
    let cut: String = text.chars().take(hit.onset).collect();
    cut.trim_end().to_owned()
}

// MARK: - detectors_maple_preview.py port

/// Preamble-only answers (ends with ':' and never delivers bullets).
pub fn thin_content(message: &Message) -> Option<Hit> {
    let text = trim_ws_nl(message.content.as_deref()?);
    if text.is_empty() {
        return None;
    }
    let has_list_marker = ["\n-", "\n*", "\n1.", "\n1)"]
        .iter()
        .any(|m| text.contains(m));
    if text.ends_with(':') && !has_list_marker {
        return Some(Hit {
            policy_id: THIN_CONTENT_ID.into(),
            field: "content".into(),
            kind: "thin_content".into(),
            onset: 0,
            detail: map(json!({"len": text.chars().count()})),
        });
    }
    None
}

/// Empty content with a long reasoning blob (common Maple collapse shape).
pub fn empty_content_long_reasoning(message: &Message, min_reasoning_len: usize) -> Option<Hit> {
    if !trim_ws_nl(message.content.as_deref().unwrap_or("")).is_empty() {
        return None;
    }
    let reasoning = message.reasoning_content.as_deref()?;
    let len = reasoning.chars().count();
    if len <= min_reasoning_len {
        return None;
    }
    if let Some(hit) = find_repetition(reasoning) {
        let mut detail = hit.detail;
        detail.insert("reasoning_len".into(), json!(len));
        return Some(Hit {
            policy_id: EMPTY_LONG_REASONING_ID.into(),
            field: "reasoning_content".into(),
            kind: hit.kind,
            onset: hit.onset,
            detail,
        });
    }
    Some(Hit {
        policy_id: EMPTY_LONG_REASONING_ID.into(),
        field: "reasoning_content".into(),
        kind: "empty_content_long_reasoning".into(),
        onset: len.min(400),
        detail: map(json!({"reasoning_len": len})),
    })
}

/// `_PSEUDO_TOOL_BLOCK` — full block or a bare open/close tag; lazy body,
/// DOTALL via bracketed classes, case-insensitive.
static PSEUDO_TOOL_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i){TOOL_CALL_OPEN}\b[^>]*>[\s\S]*?</tool_call\b[^>]*>|</?tool_call\b[^>]*>"
    ))
    .expect("static pseudo-tool regex")
});

/// Tool markup leaked into assistant content instead of structured calls.
pub fn pseudo_tool_call_in_content(message: &Message) -> Option<Hit> {
    let content = message.content.as_deref()?;
    let lowered = content.to_lowercase();
    if !lowered.contains(TOOL_CALL_OPEN) {
        return None;
    }
    let onset = match PSEUDO_TOOL_BLOCK.find(content) {
        Some(m) => content[..m.start()].chars().count(),
        None => lowered
            .find(TOOL_CALL_OPEN)
            .map_or(0, |b| lowered[..b].chars().count()),
    };
    Some(Hit {
        policy_id: PSEUDO_TOOL_MARKUP_ID.into(),
        field: "content".into(),
        kind: "pseudo_tool_markup".into(),
        onset,
        detail: map(json!({"has_structured_tool_calls": !message.tool_calls.is_empty()})),
    })
}

/// `strip_pseudo_tool_markup`: remove markup blocks/tags, trim.
pub fn strip_pseudo_tool_markup(text: &str) -> String {
    trim_ws_nl(&PSEUDO_TOOL_BLOCK.replace_all(text, "")).to_owned()
}

// MARK: - detectors_tools.py port

const CODEISH_SUFFIXES: [&str; 10] = [
    ".toml", ".rs", ".py", ".ts", ".js", ".json", ".md", ".txt", ".yml", ".yaml",
];

/// Parse `function.arguments` (JSON object string) like evprtr `_fn_args`.
fn fn_args(call: &ToolCallSpec) -> (&str, JsonMap) {
    let args = if call.arguments.is_empty() {
        JsonMap::new()
    } else {
        match serde_json::from_str::<Value>(&call.arguments) {
            Ok(Value::Object(m)) => m,
            _ => JsonMap::new(),
        }
    };
    (call.name.as_str(), args)
}

fn looks_like_path(path: &str) -> bool {
    if path.is_empty() || path == "." || path == ".." {
        return false;
    }
    if path.contains('/') || path.contains('\\') {
        return true;
    }
    let lower = path.to_lowercase();
    CODEISH_SUFFIXES.iter().any(|s| lower.ends_with(s))
}

fn string_arg<'a>(args: &'a JsonMap, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn path_arg(args: &JsonMap) -> &str {
    trim_blanks(
        string_arg(args, "path")
            .or_else(|| string_arg(args, "file_path"))
            .unwrap_or(""),
    )
}

/// Reject structured tool_calls whose arguments cannot be a real edit.
pub fn degenerate_tool_args(message: &Message) -> Option<Hit> {
    for (idx, call) in message.tool_calls.iter().enumerate() {
        let (name, args) = fn_args(call);
        let problem = match name {
            "write" => write_problem(&args).map(|p| (p, "write")),
            "edit" => edit_problem(&args).map(|p| (p, "edit")),
            _ => None,
        };
        if let Some((mut detail, tool)) = problem {
            detail.insert("tool".into(), json!(tool));
            return Some(Hit {
                policy_id: DEGENERATE_TOOL_ARGS_ID.into(),
                field: "tool_calls".into(),
                kind: "degenerate_tool_args".into(),
                onset: idx,
                detail,
            });
        }
    }
    None
}

fn write_problem(args: &JsonMap) -> Option<JsonMap> {
    let path = path_arg(args);
    let content = string_arg(args, "content").unwrap_or("");
    let content_len = content.chars().count();
    let problem = |reason: &str| {
        Some(map(
            json!({"reason": reason, "path": path, "content_len": content_len}),
        ))
    };
    if !looks_like_path(path) {
        return problem("path_not_filelike");
    }
    if trim_blanks(content) == path {
        return problem("content_equals_path");
    }
    let lower_path = path.to_lowercase().replace('\\', "/");
    if lower_path.ends_with("cargo.toml")
        && !content.contains("[package]")
        && !content.contains("axum")
        && content_len < 80
    {
        return problem("cargo_toml_too_thin");
    }
    if [".rs", ".toml", ".py", ".ts"]
        .iter()
        .any(|s| lower_path.ends_with(s))
        && trim_blanks(content).chars().count() < 12
    {
        return problem("source_content_too_short");
    }
    None
}

fn edit_problem(args: &JsonMap) -> Option<JsonMap> {
    let path = path_arg(args);
    let reason = |r: &str| Some(map(json!({"reason": r, "path": path})));
    if !path.is_empty() && !looks_like_path(path) {
        return reason("path_not_filelike");
    }
    // `edits` must be a non-empty array of objects.
    let edits: Option<Vec<&JsonMap>> = args
        .get("edits")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(Value::as_object).collect::<Option<Vec<_>>>())
        .unwrap_or(None);
    let Some(edits) = edits.filter(|e| !e.is_empty()) else {
        return reason("empty_edits");
    };
    const INVENTED: [&str; 5] = [
        "non-empty oldtext",
        "oldtext and newtext",
        "mutation tool",
        "use the edit",
        "do not only",
    ];
    for edit in edits {
        let old = string_arg(edit, "oldText")
            .or_else(|| string_arg(edit, "old_string"))
            .unwrap_or("");
        let new = string_arg(edit, "newText")
            .or_else(|| string_arg(edit, "new_string"))
            .unwrap_or("");
        if old.is_empty() && new.is_empty() {
            return reason("empty_edit_texts");
        }
        if !old.is_empty() && old == new {
            return Some(map(
                json!({"reason": "noop_edit", "path": path, "old_len": old.chars().count()}),
            ));
        }
        let low = old.to_lowercase();
        if INVENTED.iter().any(|t| low.contains(t)) {
            return reason("invented_old_text");
        }
        if !old.is_empty()
            && old.chars().count() < 80
            && low.replace(' ', "").contains("tokio{version")
        {
            return reason("invented_old_text");
        }
    }
    None
}

// MARK: - pipeline.py port (VerifyBundle.maple_preview)

/// Bundle order matches `VerifyBundle.maple_preview`: repetition -> degenerate
/// tool args -> pseudo markup -> thin -> empty+long. `Outcome::first`
/// reproduces their short-circuit semantics.
pub fn verify(message: &Message) -> Outcome {
    let mut hits = Vec::new();
    for (field, text) in message_fields(message) {
        if let Some(hit) = find_repetition(text) {
            hits.push(Hit {
                policy_id: GENERIC_REPETITION_ID.into(),
                field: field.into(),
                kind: hit.kind,
                onset: hit.onset,
                detail: hit.detail,
            });
        }
    }
    hits.extend(degenerate_tool_args(message));
    hits.extend(pseudo_tool_call_in_content(message));
    hits.extend(thin_content(message));
    hits.extend(empty_content_long_reasoning(message, 800));
    Outcome { hits }
}

/// evprtr `message_fields`: text-bearing channels, content first.
fn message_fields(message: &Message) -> Vec<(&'static str, &str)> {
    let mut out = Vec::new();
    if let Some(c) = message
        .content
        .as_deref()
        .filter(|c| !trim_ws_nl(c).is_empty())
    {
        out.push(("content", c));
    }
    if let Some(r) = message
        .reasoning_content
        .as_deref()
        .filter(|r| !trim_ws_nl(r).is_empty())
    {
        out.push(("reasoning_content", r));
    }
    out
}

/// The interior of the first think block, if any — momij's stand-in for
/// `reasoning_content` when the backend returns the think region as raw text.
pub fn think_interior(raw: &str) -> Option<String> {
    let open = raw.find(THINK_OPEN)? + THINK_OPEN.len();
    Some(match raw[open..].find(THINK_CLOSE) {
        Some(rel) => raw[open..open + rel].to_owned(),
        None => raw[open..].to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_markup::{CALL_CLOSE, CALL_OPEN};

    fn block(body: &str) -> String {
        format!("{CALL_OPEN}\n{body}\n{CALL_CLOSE}")
    }

    fn call(name: &str, args: &str) -> ToolCallSpec {
        ToolCallSpec::new("c1", name, args)
    }

    fn degenerate(name: &str, args: &str) -> Option<Hit> {
        degenerate_tool_args(&Message::with_tool_calls(vec![call(name, args)]))
    }

    fn reason(hit: &Option<Hit>) -> Option<&str> {
        hit.as_ref()
            .and_then(|h| h.detail.get("reason"))
            .and_then(Value::as_str)
    }

    // MARK: repetition.py

    #[test]
    fn find_repetition_healthy() {
        let text = "Sure. Here are the main entry points in this repository and what they do.";
        assert!(find_repetition(text).is_none());
        assert!(find_repetition("tiny").is_none()); // < 32 chars
    }

    #[test]
    fn find_repetition_char_motif() {
        let text = format!("processing complete: {} end", "tr".repeat(20));
        let hit = find_repetition(&text).unwrap();
        assert_eq!(hit.kind, "char_motif");
        assert_eq!(hit.onset, 21);
        assert_eq!(hit.detail["motif"], "tr");
        assert_eq!(hit.detail["span"], 40);
        assert_eq!(truncate_before_repetition(&text), "processing complete:");
    }

    #[test]
    fn find_repetition_char_motif_word_run_case() {
        // "agent" x10 with single spaces: the contiguous " agent" unit is
        // itself a char motif and its onset beats the word-run onset.
        let text = format!(
            "let me think about this {}and continue.",
            "agent ".repeat(10)
        );
        let hit = find_repetition(&text).unwrap();
        assert_eq!(hit.kind, "char_motif");
        assert_eq!(hit.onset, 23);
        assert_eq!(hit.detail["motif"], " agent");
        assert_eq!(hit.detail["span"], 60);
        assert_eq!(truncate_before_repetition(&text), "let me think about this");
    }

    #[test]
    fn find_repetition_word_run_separators() {
        // 8 identical tokens with alternating separator widths defeat the
        // contiguous char motif, so word_run wins. (Python oracle: onset 27.)
        let text = "start of the analysis here beta  beta beta  beta beta  beta beta  beta and the conclusion follows now";
        let hit = find_repetition(text).unwrap();
        assert_eq!(hit.kind, "word_run");
        assert_eq!(hit.onset, 27);
        assert_eq!(hit.detail["token"], "beta");
        assert_eq!(hit.detail["run"], 8);
        assert_eq!(
            truncate_before_repetition(text),
            "start of the analysis here"
        );
    }

    #[test]
    fn find_repetition_ignores_whitespace_only_motif() {
        // Live-measured false positive (both implementations): a healthy
        // Python code block fired char_motif "  " x10 on the 20-space comment
        // indent; evprtr truncated the answer 1899 -> 1395 chars.
        let text = format!(
            "Recursion is a function calling itself.\n\n```python\ndef factorial(n):\n    if n == 0:{}# base case\n        return 1\n    else:\n{}# recursive case\n        return n * factorial(n - 1)\n```\nThe base case stops the recursion.",
            " ".repeat(15),
            " ".repeat(20)
        );
        assert!(
            find_repetition(&text).is_none(),
            "healthy code indentation must not fire"
        );
    }

    #[test]
    fn find_repetition_non_whitespace_motif_still_fires() {
        let text = format!("processing complete: {} end", "tr".repeat(20));
        let hit = find_repetition(&text).unwrap();
        assert_eq!(hit.kind, "char_motif");
        assert_eq!(hit.detail["motif"], "tr");
    }

    #[test]
    fn find_repetition_ngram_run() {
        // 7 identical tokens (5 identical 3-grams, total words >= 15):
        // ngram_run fires; word_run does not (needs 8).
        let text = "start of the analysis here alpha  alpha alpha  alpha alpha  alpha alpha and the conclusion follows now";
        let hit = find_repetition(text).unwrap();
        assert_eq!(hit.kind, "ngram_run");
        assert_eq!(hit.onset, 27);
        assert_eq!(hit.detail["ngram"], "alpha alpha alpha");
        assert_eq!(hit.detail["run"], 5);
        assert_eq!(hit.detail["n"], 3);
        assert_eq!(
            truncate_before_repetition(text),
            "start of the analysis here"
        );
    }

    #[test]
    fn find_repetition_no_false_positive_on_reasoning_blob() {
        // Period-4 phrase repetition is below every threshold (evprtr: null).
        let text = format!(
            "Step by step analysis. {}concludes.",
            "therefore the result ".repeat(6)
        );
        assert!(find_repetition(&text).is_none());
    }

    #[test]
    fn repetition_short_guard() {
        assert!(find_repetition("short text under limit").is_none());
    }

    #[test]
    fn char_motif_does_not_span_newlines() {
        // The dot must not match "\n" (Python re): "ab\n" x10 is not one motif.
        let text = format!("{}the end of the text goes here", "ab\n".repeat(10));
        let hit = find_repetition(&text);
        assert!(
            hit.as_ref().is_none_or(|h| h.kind != "char_motif"),
            "{hit:?}"
        );
    }

    #[test]
    fn offsets_are_code_points() {
        let text = format!("日本語の前置きです。{}おわり。", "くり".repeat(12));
        let hit = find_repetition(&text).unwrap();
        assert_eq!(hit.kind, "char_motif");
        assert_eq!(hit.onset, "日本語の前置きです。".chars().count());
        assert_eq!(truncate_before_repetition(&text), "日本語の前置きです。");
    }

    #[test]
    fn long_repeated_text_is_scanned_quickly() {
        let text = "a".repeat(200_000);
        let hit = find_repetition(&text).unwrap();
        assert_eq!(hit.kind, "char_motif");
        assert_eq!(hit.onset, 0);
    }

    // MARK: thin content

    #[test]
    fn thin_content_colon() {
        let hit = thin_content(&Message::content("Here are the files in the directory:")).unwrap();
        assert_eq!(hit.policy_id, THIN_CONTENT_ID);
        assert_eq!(hit.kind, "thin_content");
        assert_eq!(hit.field, "content");
        assert_eq!(hit.onset, 0);
        assert_eq!(hit.detail["len"], 36);
    }

    #[test]
    fn thin_content_bullets_pass() {
        assert!(
            thin_content(&Message::content(
                "Here are the files in the directory:\n- a.txt\n- b.txt"
            ))
            .is_none()
        );
        // Healthy sentence: not colon-terminated.
        assert!(
            thin_content(&Message::content(
                "The command completed successfully without errors."
            ))
            .is_none()
        );
    }

    // MARK: pseudo tool markup

    #[test]
    fn pseudo_tool_markup_in_content() {
        let content = format!(
            "I will list files.\n\n{}",
            block("{\"name\": \"ls\", \"arguments\": {\"path\": \"/tmp\"}}")
        );
        let hit = pseudo_tool_call_in_content(&Message::content(content.clone())).unwrap();
        assert_eq!(hit.policy_id, PSEUDO_TOOL_MARKUP_ID);
        assert_eq!(hit.kind, "pseudo_tool_markup");
        assert_eq!(hit.field, "content");
        assert_eq!(hit.onset, 20);
        assert_eq!(hit.detail["has_structured_tool_calls"], false);
        assert_eq!(strip_pseudo_tool_markup(&content), "I will list files.");
    }

    #[test]
    fn pseudo_tool_markup_unterminated() {
        let content = format!("oops {CALL_OPEN}\n{{\"name\": \"ls\"");
        let hit = pseudo_tool_call_in_content(&Message::content(content.clone())).unwrap();
        assert_eq!(hit.onset, 5);
        // Only the bare tag is removed; the JSON body stays (Python parity).
        assert_eq!(
            strip_pseudo_tool_markup(&content),
            "oops \n{\"name\": \"ls\""
        );
    }

    #[test]
    fn pseudo_tool_markup_clean_content() {
        assert!(
            pseudo_tool_call_in_content(&Message::content("plain answer without markup")).is_none()
        );
    }

    #[test]
    fn pseudo_tool_markup_is_case_insensitive_and_reports_structured_calls() {
        let content = "x <TOOL_CALL>{}</TOOL_CALL>";
        let msg = Message {
            content: Some(content.into()),
            tool_calls: vec![call("ls", "{}")],
            ..Message::default()
        };
        let hit = pseudo_tool_call_in_content(&msg).unwrap();
        assert_eq!(hit.onset, 2);
        assert_eq!(hit.detail["has_structured_tool_calls"], true);
        assert_eq!(strip_pseudo_tool_markup(content), "x");
    }

    // MARK: empty content + long reasoning

    #[test]
    fn empty_content_long_reasoning_hit() {
        let reasoning = "reasoning about the task. ".repeat(40);
        let msg = Message {
            content: Some(String::new()),
            reasoning_content: Some(reasoning),
            ..Message::default()
        };
        let hit = empty_content_long_reasoning(&msg, 800).unwrap();
        assert_eq!(hit.policy_id, EMPTY_LONG_REASONING_ID);
        assert_eq!(hit.kind, "empty_content_long_reasoning");
        assert_eq!(hit.onset, 400);
        assert_eq!(hit.detail["reasoning_len"], 1040);
        // Short reasoning: no hit.
        let short = Message {
            content: Some(String::new()),
            reasoning_content: Some("short".into()),
            ..Message::default()
        };
        assert!(empty_content_long_reasoning(&short, 800).is_none());
        // Non-empty content: no hit.
        let with_content = Message {
            content: Some("answer".into()),
            reasoning_content: Some("x".repeat(2000)),
            ..Message::default()
        };
        assert!(empty_content_long_reasoning(&with_content, 800).is_none());
    }

    #[test]
    fn empty_content_long_reasoning_reports_repetition_kind() {
        let reasoning = format!("start {}", "tr".repeat(500));
        let msg = Message {
            reasoning_content: Some(reasoning),
            ..Message::default()
        };
        let hit = empty_content_long_reasoning(&msg, 800).unwrap();
        assert_eq!(hit.kind, "char_motif");
        assert_eq!(hit.field, "reasoning_content");
        assert!(hit.detail.contains_key("reasoning_len"));
    }

    // MARK: degenerate tool args

    #[test]
    fn degenerate_write_bad_path() {
        let hit = degenerate("write", "{\"path\": \"Tower\", \"content\": \"0.5\"}").unwrap();
        assert_eq!(hit.policy_id, DEGENERATE_TOOL_ARGS_ID);
        assert_eq!(hit.kind, "degenerate_tool_args");
        assert_eq!(hit.field, "tool_calls");
        assert_eq!(hit.onset, 0);
        assert_eq!(hit.detail["tool"], "write");
        assert_eq!(hit.detail["reason"], "path_not_filelike");
        assert_eq!(hit.detail["path"], "Tower");
        assert_eq!(hit.detail["content_len"], 3);
    }

    #[test]
    fn degenerate_write_content_equals_path() {
        let hit = degenerate(
            "write",
            "{\"path\": \"README.md\", \"content\": \"README.md\"}",
        )
        .unwrap();
        assert_eq!(hit.detail["reason"], "content_equals_path");
        assert_eq!(hit.detail["content_len"], 9);
    }

    #[test]
    fn degenerate_write_thin_source_edge() {
        // 12-char .rs content is NOT flagged (evprtr parity: < 12 required).
        assert!(
            degenerate(
                "write",
                "{\"path\": \"main.rs\", \"content\": \"fn main() {}\"}"
            )
            .is_none()
        );
        let hit = degenerate(
            "write",
            "{\"path\": \"main.rs\", \"content\": \"fn main(){}\"}",
        );
        assert_eq!(reason(&hit), Some("source_content_too_short"));
    }

    #[test]
    fn degenerate_write_healthy() {
        let args =
            "{\"path\": \"src/main.rs\", \"content\": \"fn main() { println!(\\\"ok\\\"); }\"}";
        assert!(degenerate("write", args).is_none());
    }

    #[test]
    fn degenerate_write_thin_cargo_toml() {
        let hit = degenerate(
            "write",
            "{\"file_path\": \"Cargo.toml\", \"content\": \"x = 1\"}",
        );
        assert_eq!(reason(&hit), Some("cargo_toml_too_thin"));
        let ok = degenerate(
            "write",
            "{\"path\": \"Cargo.toml\", \"content\": \"[package]\\nname = \\\"x\\\"\"}",
        );
        assert!(ok.is_none());
    }

    #[test]
    fn degenerate_edit_noop() {
        let hit = degenerate(
            "edit",
            "{\"path\": \"a.txt\", \"edits\": [{\"oldText\": \"same\", \"newText\": \"same\"}]}",
        )
        .unwrap();
        assert_eq!(hit.detail["reason"], "noop_edit");
        assert_eq!(hit.detail["path"], "a.txt");
        assert_eq!(hit.detail["old_len"], 4);
    }

    #[test]
    fn degenerate_edit_empty() {
        let hit = degenerate("edit", "{\"path\": \"a.txt\", \"edits\": []}");
        assert_eq!(reason(&hit), Some("empty_edits"));
        // Non-object entries invalidate the whole list.
        let hit = degenerate("edit", "{\"path\": \"a.txt\", \"edits\": [1]}");
        assert_eq!(reason(&hit), Some("empty_edits"));
    }

    #[test]
    fn degenerate_edit_empty_texts_and_bad_path() {
        let hit = degenerate(
            "edit",
            "{\"path\": \"a.txt\", \"edits\": [{\"oldText\": \"\", \"newText\": \"\"}]}",
        );
        assert_eq!(reason(&hit), Some("empty_edit_texts"));
        let hit = degenerate("edit", "{\"path\": \"Tower\", \"edits\": []}");
        assert_eq!(reason(&hit), Some("path_not_filelike"));
    }

    #[test]
    fn degenerate_edit_invented_old_text() {
        let hit = degenerate(
            "edit",
            "{\"path\": \"Cargo.toml\", \"edits\": [{\"oldText\": \"non-empty oldtext here\", \"newText\": \"x\"}]}",
        );
        assert_eq!(reason(&hit), Some("invented_old_text"));
        let hit = degenerate(
            "edit",
            "{\"path\": \"Cargo.toml\", \"edits\": [{\"old_string\": \"tokio { version\", \"new_string\": \"y\"}]}",
        );
        assert_eq!(reason(&hit), Some("invented_old_text"));
    }

    #[test]
    fn degenerate_edit_healthy() {
        let hit = degenerate(
            "edit",
            "{\"path\": \"src/main.rs\", \"edits\": [{\"oldText\": \"let x = 1;\", \"newText\": \"let x = 2;\"}]}",
        );
        assert!(hit.is_none());
    }

    #[test]
    fn degenerate_unparsable_arguments() {
        // Unparsable arguments -> empty args -> write path check fails.
        assert_eq!(
            reason(&degenerate("write", "not json")),
            Some("path_not_filelike")
        );
        // Other tools are never flagged.
        assert!(degenerate("bash", "not json").is_none());
    }

    #[test]
    fn degenerate_reports_the_index_of_the_offending_call() {
        let msg = Message::with_tool_calls(vec![
            call("ls", "{}"),
            call("write", "{\"path\": \"Tower\", \"content\": \"0.5\"}"),
        ]);
        assert_eq!(degenerate_tool_args(&msg).unwrap().onset, 1);
    }

    // MARK: bundle order (pipeline parity)

    #[test]
    fn verify_bundle_order_and_first() {
        // Repetition in content + markup leak: repetition first (bundle order).
        let messy = format!("{} more text here for length", "tr".repeat(20));
        let content = format!("{messy}\n{}", block("{}"));
        let outcome = verify(&Message::content(content));
        assert_eq!(
            outcome.hits.first().unwrap().policy_id,
            GENERIC_REPETITION_ID
        );
        assert!(
            outcome
                .hits
                .iter()
                .any(|h| h.policy_id == PSEUDO_TOOL_MARKUP_ID)
        );
        assert_eq!(outcome.first(), outcome.hits.first());
    }

    #[test]
    fn verify_healthy_message() {
        let outcome = verify(&Message::content(
            "A perfectly normal, complete, and useful answer to the user's question.",
        ));
        assert!(outcome.hits.is_empty());
        assert!(outcome.first().is_none());
    }

    #[test]
    fn event_detail_shape() {
        let hit = thin_content(&Message::content("Here are the files in the directory:")).unwrap();
        let ev = hit.event_detail();
        assert_eq!(ev["policy_id"], THIN_CONTENT_ID);
        assert_eq!(ev["onset"], 0);
        assert_eq!(ev["detail"]["len"], 36);
    }

    // MARK: think interior

    #[test]
    fn think_interior_cases() {
        let raw = format!("prose{THINK_OPEN}deep reasoning here{THINK_CLOSE}answer");
        assert_eq!(think_interior(&raw).as_deref(), Some("deep reasoning here"));
        let unclosed = format!("x{THINK_OPEN}dangling");
        assert_eq!(think_interior(&unclosed).as_deref(), Some("dangling"));
        assert_eq!(think_interior("no block"), None);
    }
}
