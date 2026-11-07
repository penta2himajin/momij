//! Maple HF/DeepGrove-style tools / tool-call markup contract, ported from the
//! compositor (evprtr `compositor/tools/maple_tool_markup.py` and
//! `pseudo_tool.py`). The harness speaks OpenAI tools / tool_calls; momij
//! rewrites the Maple-bound request into markup form, then parses tool-call
//! blocks back to OpenAI tool_calls. Port of `ToolMarkup.swift`.

use serde_json::Value;

use crate::json::{json_dump_string, json_dumps, json_parse_value, random_id, trim_ws_nl};
use crate::openai_compat::{ChatMessage, ToolCallSpec};

const TOOLS_OPEN: &str = "<tools>";
const TOOLS_CLOSE: &str = "</tools>";
pub const CALL_OPEN: &str = "<tool_call>";
pub const CALL_CLOSE: &str = "</tool_call>";

fn preamble() -> String {
    format!(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within {TOOLS_OPEN} XML tags:\n{TOOLS_OPEN}\n"
    )
}

fn epilogue() -> String {
    format!(
        "\n{TOOLS_CLOSE}\n\n\
         For each function call, return a json object with function name and arguments within {CALL_OPEN} XML tags:\n\
         {CALL_OPEN}\n\
         {{\"name\": <function-name>, \"arguments\": <args-json-object>}}{CALL_CLOSE}\n\
         If you can answer without a tool, reply with plain text only (no {CALL_OPEN} tags).\n"
    )
}

/// The tools instruction block appended to the system message.
pub fn system_suffix(tool_lines: &[String]) -> String {
    format!("{}{}\n{}", preamble(), tool_lines.join("\n"), epilogue())
}

/// Map OpenAI tool history into text: assistant tool_calls become tool-call
/// blocks, tool role messages become wrapped user messages.
pub fn rewrite_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        if msg.role == "assistant" && !msg.tool_calls.is_empty() {
            let mut parts: Vec<String> = Vec::new();
            let trimmed = trim_ws_nl(&msg.content);
            if !trimmed.is_empty() {
                parts.push(trimmed.to_owned());
            }
            for call in &msg.tool_calls {
                let args = json_parse_value(&call.arguments).unwrap_or(Value::Null);
                // NOTE: the Swift original closed this object with "{" (a typo
                // present since the first commit), emitting malformed JSON
                // into the model-visible history. The documented contract
                // (python `json.dumps`) closes it with "}".
                let payload = format!(
                    "{{\"name\": {}, \"arguments\": {}}}",
                    json_dump_string(&call.name),
                    json_dumps(&args)
                );
                parts.push(format!("{CALL_OPEN}\n{payload}\n{CALL_CLOSE}"));
            }
            out.push(ChatMessage::new(msg.role.clone(), parts.join("\n")));
            continue;
        }
        if msg.role == "tool" {
            out.push(ChatMessage::new(
                "user",
                format!("<tool_result>\n{}\n</tool_result>", msg.content),
            ));
            continue;
        }
        out.push(msg.clone());
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCalls {
    pub cleaned_content: String,
    pub calls: Vec<ToolCallSpec>,
}

/// Parse tool-call blocks out of the completion text: returns the prose with
/// blocks removed and one `ToolCallSpec` per block.
pub fn parse_pseudo_tool_calls(content: &str) -> ParsedToolCalls {
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = content;
    while let Some(open) = rest.find(CALL_OPEN) {
        cleaned.push_str(&rest[..open]);
        rest = &rest[open + CALL_OPEN.len()..];
        let Some(close) = rest.find(CALL_CLOSE) else {
            // Unterminated block (model stopped mid-call). First try
            // terminator completion: if the body already contains a complete
            // JSON call object, recover it as a real call — only the tag is
            // missing. NEVER invent missing JSON. Otherwise strip the bare tag
            // and keep the body as prose (evprtr strip parity).
            let body = rest;
            let recovered = json_parse_value(&first_complete_json_object(body))
                .as_ref()
                .and_then(Value::as_object)
                .and_then(|obj| {
                    let name = obj.get("name")?.as_str().filter(|n| !n.is_empty())?;
                    let args = json_dumps(obj.get("arguments").unwrap_or(&Value::Null));
                    Some(ToolCallSpec::new(new_call_id(), name, args))
                });
            match recovered {
                Some(call) => calls.push(call),
                None => cleaned.push_str(body),
            }
            rest = "";
            break;
        };
        let inner = trim_ws_nl(&rest[..close]);
        if let Some(Value::Object(obj)) = json_parse_value(inner) {
            let name = obj.get("name").and_then(Value::as_str).unwrap_or("");
            let args = json_dumps(obj.get("arguments").unwrap_or(&Value::Null));
            calls.push(ToolCallSpec::new(new_call_id(), name, args));
        }
        rest = &rest[close + CALL_CLOSE.len()..];
    }
    cleaned.push_str(rest);
    // Stray close tags (close without a matching open) are bare-tag garbage —
    // strip them like evprtr's block regex does.
    let cleaned = cleaned.replace(CALL_CLOSE, "");
    ParsedToolCalls {
        cleaned_content: trim_ws_nl(&cleaned).to_owned(),
        calls,
    }
}

/// First balanced `{...}` object from text, string- and escape-aware (braces
/// inside JSON strings do not count). Returns `""` when none.
pub fn first_complete_json_object(text: &str) -> String {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut start: Option<usize> = None;
    for (offset, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_string {
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(offset);
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if let (0, Some(s)) = (depth, start) {
                    return text[s..offset + 1].to_owned();
                }
            }
            _ => {}
        }
    }
    String::new()
}

pub fn new_call_id() -> String {
    random_id("call_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const LS_TOOL_LINE: &str = r#"{"type": "function", "function": {"name": "ls", "description": "List directory contents"}}"#;

    fn call_block(name: &str, args: &str) -> String {
        format!(
            "{CALL_OPEN}\n{{\"name\": {}, \"arguments\": {args}}}\n{CALL_CLOSE}",
            json_dump_string(name)
        )
    }

    #[test]
    fn system_suffix_bytes() {
        let suffix = system_suffix(&[LS_TOOL_LINE.to_owned()]);
        assert!(
            suffix.starts_with("# Tools\n\nYou may call one or more functions"),
            "{suffix}"
        );
        assert!(
            suffix.contains(&format!("within {CALL_OPEN} XML tags:")),
            "{suffix}"
        );
        assert!(suffix.contains("List directory contents"), "{suffix}");
        assert!(suffix.ends_with("tags).\n"), "{suffix}");
        assert!(suffix.contains("</tools>"), "{suffix}");
        assert!(suffix.contains("<function-name>"), "{suffix}");
        // The example payload line directly precedes the close tag.
        assert!(
            suffix.contains(&format!("<args-json-object>}}{CALL_CLOSE}\n")),
            "{suffix}"
        );
    }

    #[test]
    fn parse_single_tool_call() {
        let content = format!("I will list.\n\n{}", call_block("ls", "{}"));
        let parsed = parse_pseudo_tool_calls(&content);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "ls");
        assert_eq!(parsed.calls[0].arguments, "{}");
        assert_eq!(parsed.cleaned_content, "I will list.");
        assert!(parsed.calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let one = call_block("ls", "{}");
        let two = call_block("grep", "{\"pattern\": \"Overview\"}");
        let parsed = parse_pseudo_tool_calls(&format!("go.\n\n{one}\n\n{two}"));
        let names: Vec<&str> = parsed.calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["ls", "grep"]);
        assert_eq!(parsed.cleaned_content, "go.");
        assert_eq!(parsed.calls[1].arguments, r#"{"pattern": "Overview"}"#);
    }

    #[test]
    fn parse_robustness() {
        // Unterminated block with truncated JSON: no recovery possible — bare
        // open tag stripped, body kept as prose, no phantom call.
        let p1 = parse_pseudo_tool_calls(&format!("oops {CALL_OPEN}\n{{\"name\""));
        assert!(p1.calls.is_empty());
        assert!(
            !p1.cleaned_content.contains(CALL_OPEN),
            "{}",
            p1.cleaned_content
        );
        assert!(
            p1.cleaned_content.contains("{\"name\""),
            "{}",
            p1.cleaned_content
        );
        // Garbage inside: no phantom call, prose kept.
        let p2 = parse_pseudo_tool_calls(&format!("not json {CALL_OPEN} ??? {CALL_CLOSE} done"));
        assert!(p2.calls.is_empty());
        assert_eq!(p2.cleaned_content, "not json  done");
        // Empty content.
        let p3 = parse_pseudo_tool_calls("");
        assert!(p3.calls.is_empty());
        assert_eq!(p3.cleaned_content, "");
    }

    #[test]
    fn unterminated_block_recovery() {
        // JSON complete, only the close tag missing (EOS cut right after the
        // object): recover the real call, consume the body.
        let body = r#"{"name": "bash", "arguments": {"command": "ls -la"}}"#;
        let p = parse_pseudo_tool_calls(&format!("listing:\n{CALL_OPEN}\n{body}"));
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "bash");
        assert!(p.calls[0].arguments.contains("ls -la"));
        assert!(
            !p.cleaned_content.contains(CALL_OPEN),
            "{}",
            p.cleaned_content
        );
        assert!(
            !p.cleaned_content.contains("ls -la"),
            "{}",
            p.cleaned_content
        );
    }

    #[test]
    fn unterminated_block_recovers_first_of_concatenated_objects() {
        // Live shape (2026-09-13 subagent session): a complete call object
        // followed by a stray second object. Recover the FIRST complete one.
        let body = "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la x\"}}{\n  \"description\": \"List test file\"}\n}";
        let p = parse_pseudo_tool_calls(&format!("{CALL_OPEN}\n{body}"));
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "bash");
        assert!(p.calls[0].arguments.contains("ls -la x"));
        assert!(
            !p.cleaned_content.contains("description"),
            "{}",
            p.cleaned_content
        );
    }

    #[test]
    fn unterminated_block_string_aware_scanner() {
        // A brace inside a string must not count for balancing; the object
        // never completes (string left open) -> prose fallback.
        let body = "{\"name\": \"bash\", \"command\": \"echo \\\"}\"";
        let p = parse_pseudo_tool_calls(&format!("note {CALL_OPEN}{body}"));
        assert!(p.calls.is_empty());
        assert!(
            !p.cleaned_content.contains(CALL_OPEN),
            "{}",
            p.cleaned_content
        );
    }

    #[test]
    fn stray_close_tag_stripped() {
        let p = parse_pseudo_tool_calls(&format!("text {CALL_CLOSE} more"));
        assert!(p.calls.is_empty());
        assert!(!p.cleaned_content.contains(CALL_CLOSE));
        assert_eq!(p.cleaned_content, "text  more");
    }

    #[test]
    fn first_complete_json_object_scanner() {
        assert_eq!(
            first_complete_json_object("x {\"a\": \"}\"} tail"),
            "{\"a\": \"}\"}"
        );
        assert_eq!(first_complete_json_object("{\"broken {"), "");
        assert_eq!(first_complete_json_object("no braces"), "");
        assert_eq!(
            first_complete_json_object("日本語 {\"k\": \"値\"} 後"),
            "{\"k\": \"値\"}"
        );
    }

    #[test]
    fn rewrite_messages_maps_tool_history() {
        let calls = vec![ToolCallSpec::new("c1", "ls", "{}")];
        let messages = vec![
            ChatMessage::new("system", "sys"),
            ChatMessage::new("user", "list files"),
            ChatMessage::new("assistant", "listing").with_tool_calls(calls),
            ChatMessage::new("tool", "a.txt\nb.txt"),
        ];
        let out = rewrite_messages(&messages);
        assert!(
            out[2].content.contains(CALL_OPEN) && out[2].content.contains("ls"),
            "{}",
            out[2].content
        );
        assert_eq!(out[3].role, "user");
        assert_eq!(
            out[3].content,
            "<tool_result>\na.txt\nb.txt\n</tool_result>"
        );
        assert_eq!(out[0].content, "sys");
        assert_eq!(out[1].content, "list files");
    }

    /// Regression for the Swift `+ "{"` typo: the rewritten block must be
    /// well-formed and must parse back into the same call.
    #[test]
    fn rewritten_assistant_block_is_valid_json_and_roundtrips() {
        let calls = vec![ToolCallSpec::new(
            "c1",
            "bash",
            r#"{"command":"ls -la","n":1}"#,
        )];
        let out = rewrite_messages(&[ChatMessage::new("assistant", "").with_tool_calls(calls)]);
        let expected = format!(
            "{CALL_OPEN}\n{{\"name\": \"bash\", \"arguments\": {{\"command\": \"ls -la\", \"n\": 1}}}}\n{CALL_CLOSE}"
        );
        assert_eq!(out[0].content, expected);
        let parsed = parse_pseudo_tool_calls(&out[0].content);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "bash");
        let args: Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
        assert_eq!(args, json!({"command": "ls -la", "n": 1}));
        assert_eq!(parsed.cleaned_content, "");
    }

    /// Live failure (2026-09-14): a bash tool call whose `command` contained
    /// RAW newlines (multiline `python -c`) — invalid strict JSON — used to be
    /// dropped and leaked as prose. Inner quotes are escaped (`\"`); only the
    /// newlines are raw. The Swift version of this test sat outside the
    /// XCTestCase class (never ran) and also lacked the quote escapes.
    #[test]
    fn lenient_json_with_literal_newlines() {
        let inner = "{\"name\": \"bash\", \"arguments\": {\"command\": \"python3 -c \\\"\nimport json\nprint(1)\n\\\"\", \"description\": \"run\"}}";
        let p = parse_pseudo_tool_calls(&format!("trying:\n{CALL_OPEN}\n{inner}\n{CALL_CLOSE}"));
        assert_eq!(p.calls.len(), 1, "multiline command must parse as a call");
        assert_eq!(p.calls[0].name, "bash");
        assert!(p.calls[0].arguments.contains("import json"));
        assert!(
            p.cleaned_content.contains("trying:"),
            "{}",
            p.cleaned_content
        );
        assert!(
            !p.cleaned_content.contains(CALL_OPEN),
            "{}",
            p.cleaned_content
        );
        let args: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(args["command"], "python3 -c \"\nimport json\nprint(1)\n\"");
    }

    /// The Swift test's literal input has UNESCAPED inner quotes, so the string
    /// ends at the second quote. That is not recoverable by escaping control
    /// characters; the block must be dropped without inventing a call.
    #[test]
    fn unescaped_inner_quotes_are_not_recovered() {
        let inner = "{\"name\": \"bash\", \"arguments\": {\"command\": \"python3 -c \"\nimport json\nprint(1)\n\", \"description\": \"run\"}}";
        let p = parse_pseudo_tool_calls(&format!("{CALL_OPEN}\n{inner}\n{CALL_CLOSE}"));
        assert!(p.calls.is_empty());
    }
}
