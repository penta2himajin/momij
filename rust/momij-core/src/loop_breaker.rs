//! Tool-call loop breaker: detect a trailing run of identical tool calls (same
//! tool, arguments equal after escape-normalization) and produce a steering
//! message that breaks the retry loop. Port of `LoopBreaker.swift`.
//!
//! Motivated by a live DSH-subagent session (2026-09-13): the model retried
//! `chmod +x` 12+ times, doubling backslash escapes after each failure — the
//! error echoed the mangled path and the model "fixed" it by escaping harder.
//! Normalization makes those variants compare equal so the loop is visible
//! before it burns the budget.

use crate::json::trim_blanks;
use crate::openai_compat::ChatMessage;

/// Collapse escape-escalation variants to one comparable form: remove
/// backslashes and quotes, collapse whitespace.
pub fn normalized_args(arguments: &str) -> String {
    let mut out = String::with_capacity(arguments.len());
    let mut last_was_space = false;
    for ch in arguments.chars() {
        if matches!(ch, '\\' | '"' | '\'') {
            continue;
        }
        if matches!(ch, ' ' | '\t' | '\n') {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    trim_blanks(&out).to_owned()
}

/// Trailing run of assistant tool calls with the same tool name and equal
/// normalized arguments. `Some((tool, repeat count))` when the run reaches
/// `min_repeats`.
pub fn detect(messages: &[ChatMessage], min_repeats: usize) -> Option<(String, usize)> {
    let seq: Vec<(&str, String)> = messages
        .iter()
        .filter(|m| m.role == "assistant")
        .flat_map(|m| m.tool_calls.iter())
        .map(|c| (c.name.as_str(), normalized_args(&c.arguments)))
        .collect();
    let last = seq.last()?;
    let run = seq
        .iter()
        .rev()
        .take_while(|e| e.0 == last.0 && e.1 == last.1)
        .count();
    (run >= min_repeats && !last.1.is_empty()).then(|| (last.0.to_owned(), run))
}

/// Default threshold used by the serve path.
pub const DEFAULT_MIN_REPEATS: usize = 3;

/// The per-request steering message that breaks the loop. English only
/// (Maple); states the concrete lesson (plain forward slashes) and the
/// instruction to move on.
pub fn break_nudge(tool: &str, count: usize) -> String {
    format!(
        "You have now called the `{tool}` tool {count} times in a row with effectively the same \
         arguments (only the escaping changed). Repeating it will not change the result. If a path \
         is involved, use plain forward slashes with no backslashes and no extra quotes. Do NOT \
         retry the same call. Continue with the next step of the task."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_compat::ToolCallSpec;

    #[test]
    fn normalized_args_collapses_escalation() {
        let variants = [
            r#"chmod +x /Users/penta2himajin/repos/scratch/momij-subagent-test/count_verify_fires.py"#,
            r#"chmod +x \"\\/Users\\/penta2himajin\\/repos\\/scratch\\/momij-subagent-test\\/count_verify_fires.py\""#,
            r#"chmod +x \"\\\\\\/Users\\\\\\/penta2himajin\\/repos\\/scratch"#,
            r#"chmod +x \\"\\\\\\\\\\/Users\\\\\\\\\\/penta2himajin\\/repos"#,
        ];
        for v in variants {
            let n = normalized_args(v);
            assert!(!n.contains('\\'), "{n}");
            assert!(!n.contains('"'), "{n}");
            assert!(!n.contains('\''), "{n}");
        }
        assert_eq!(normalized_args(variants[0]), normalized_args(variants[0]));
        assert_eq!(normalized_args("a  \n b"), "a b");
    }

    #[test]
    fn normalized_args_keeps_plain_content() {
        assert_eq!(normalized_args(r#"echo "hello world""#), "echo hello world");
    }

    fn assistant(calls: &[(&str, &str)]) -> ChatMessage {
        ChatMessage::new("assistant", "").with_tool_calls(
            calls
                .iter()
                .enumerate()
                .map(|(i, (n, a))| ToolCallSpec::new(format!("c{i}"), *n, *a))
                .collect(),
        )
    }

    #[test]
    fn detect_fires_on_trailing_run() {
        let same = r#"{"command": "chmod +x \/Users\/x\/f.py"}"#;
        let escalated = r#"{"command": "chmod +x \"\\\\/Users\\\\/x\/f.py\""}"#;
        let messages = vec![
            ChatMessage::new("user", "do it"),
            assistant(&[("chmod", same)]),
            ChatMessage::new("tool", "chmod: error"),
            assistant(&[("chmod", escalated)]),
            ChatMessage::new("tool", "chmod: error again"),
            assistant(&[("chmod", escalated)]),
            ChatMessage::new("tool", "chmod: error"),
            assistant(&[("chmod", escalated)]),
        ];
        assert_eq!(
            detect(&messages, DEFAULT_MIN_REPEATS),
            Some(("chmod".to_owned(), 4))
        );
    }

    #[test]
    fn detect_below_threshold_returns_none() {
        let same = r#"{"command": "chmod +x \/Users\/x\/f.py"}"#;
        let messages = vec![
            ChatMessage::new("user", "do it"),
            assistant(&[("chmod", same)]),
            ChatMessage::new("tool", "ok"),
            assistant(&[("chmod", same)]),
        ];
        assert_eq!(detect(&messages, DEFAULT_MIN_REPEATS), None);
    }

    #[test]
    fn detect_different_args_do_not_fire() {
        // Legitimate repetition (same tool, different files) must not fire.
        let messages = vec![
            assistant(&[("read", r#"{"path": "a.md"}"#)]),
            assistant(&[("read", r#"{"path": "b.md"}"#)]),
            assistant(&[("read", r#"{"path": "c.md"}"#)]),
        ];
        assert_eq!(detect(&messages, DEFAULT_MIN_REPEATS), None);
    }

    #[test]
    fn detect_breaks_run_on_different_call() {
        let same = r#"{"command": "chmod +x \/Users\/x"}"#;
        let messages = vec![
            assistant(&[("chmod", same)]),
            assistant(&[("chmod", same)]),
            assistant(&[("ls", r#"{"path": "."}"#)]),
            assistant(&[("chmod", same)]),
        ];
        assert_eq!(detect(&messages, DEFAULT_MIN_REPEATS), None);
    }

    #[test]
    fn detect_ignores_empty_args_runs() {
        let messages = vec![assistant(&[("ls", "")]); 5];
        assert_eq!(detect(&messages, DEFAULT_MIN_REPEATS), None);
    }

    #[test]
    fn break_nudge_shape() {
        let nudge = break_nudge("chmod", 5);
        assert!(nudge.contains("`chmod`"), "{nudge}");
        assert!(nudge.contains("5 times"), "{nudge}");
        assert!(nudge.contains("plain forward slashes"), "{nudge}");
        assert!(nudge.contains("Do NOT retry"), "{nudge}");
    }
}
