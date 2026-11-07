//! JSON helpers shared by the tool-call pipeline.
//!
//! Port of the JSON helpers in `ToolMarkup.swift`. The tool-call contract uses
//! Python `json.dumps` formatting (`", "` / `": "` separators,
//! `ensure_ascii=False`), so the serializer here is hand-rolled rather than
//! `serde_json::to_string`.

use serde_json::Value;

/// Serialize like Python's default `json.dumps(..., ensure_ascii=False)`.
///
/// Object keys keep insertion order (the Swift port iterated an unordered
/// dictionary) and are escaped like any other string.
pub fn json_dumps(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", json_dump_string(k), json_dumps(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(list) => {
            let parts: Vec<String> = list.iter().map(json_dumps).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::String(s) => json_dump_string(s),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => "null".to_string(),
    }
}

/// Serialize one string with JSON escapes (no `\/` escaping, like Python).
pub fn json_dump_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// Parse an object or array. Top-level scalars are rejected (the Swift
/// `JSONSerialization` default), so `"5"` is `None`, not `5`.
///
/// Lenient retry: the model emits literal newlines/tabs *inside* string values
/// (invalid strict JSON, observed live in multiline `python -c` commands).
/// Control characters inside string literals are escaped and parsing retried.
pub fn json_parse_value(text: &str) -> Option<Value> {
    if let Some(v) = parse_strict(text) {
        return Some(v);
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
            continue;
        }
        if in_string {
            match ch {
                '\\' => {
                    out.push(ch);
                    escaped = true;
                }
                '"' => {
                    out.push(ch);
                    in_string = false;
                }
                '\n' => out.push_str("\\n"),
                '\t' => out.push_str("\\t"),
                '\r' => out.push_str("\\r"),
                _ => out.push(ch),
            }
        } else {
            if ch == '"' {
                in_string = true;
            }
            out.push(ch);
        }
    }
    parse_strict(&out)
}

fn parse_strict(text: &str) -> Option<Value> {
    match serde_json::from_str::<Value>(text) {
        Ok(v @ (Value::Object(_) | Value::Array(_))) => Some(v),
        _ => None,
    }
}

/// Swift `trimmingCharacters(in: .whitespaces)`: blanks, but not newlines.
pub(crate) fn trim_blanks(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() && c != '\n' && c != '\r')
}

/// Swift `trimmingCharacters(in: .whitespacesAndNewlines)`.
pub(crate) fn trim_ws_nl(s: &str) -> &str {
    s.trim()
}

/// `"prefix" + 12 random bytes as lowercase hex`.
pub(crate) fn random_id(prefix: &str) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let hex: String = (0..12)
        .map(|_| format!("{:02x}", rng.random::<u8>()))
        .collect();
    format!("{prefix}{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_uses_python_spacing() {
        assert_eq!(json_dumps(&json!([1, 2, 3])), "[1, 2, 3]");
        assert_eq!(json_dumps(&Value::Null), "null");
        let d = json_dumps(&json!({"name": "ls", "arguments": {"path": "/tmp"}}));
        assert_eq!(d, r#"{"name": "ls", "arguments": {"path": "/tmp"}}"#);
    }

    #[test]
    fn dumps_keeps_insertion_order_and_escapes_keys() {
        let d = json_dumps(&json!({"z": 1, "a": 2, "q\"k": true}));
        assert_eq!(d, r#"{"z": 1, "a": 2, "q\"k": true}"#);
    }

    #[test]
    fn dump_string_escapes_control_chars() {
        let out = json_dump_string("a\nb");
        assert!(!out.contains('\n'), "{out}");
        assert_eq!(out, r#""a\nb""#);
        // Python does not escape forward slashes.
        assert_eq!(json_dump_string("/tmp/x"), r#""/tmp/x""#);
    }

    #[test]
    fn parse_rejects_top_level_scalars() {
        assert!(json_parse_value("5").is_none());
        assert!(json_parse_value("\"x\"").is_none());
        assert!(json_parse_value("{}").is_some());
        assert!(json_parse_value("[1]").is_some());
    }

    #[test]
    fn parse_is_lenient_about_raw_control_chars_in_strings() {
        let v = json_parse_value("{\"command\": \"a\nb\tc\"}").expect("lenient parse");
        assert_eq!(v["command"], "a\nb\tc");
    }

    #[test]
    fn trim_helpers_match_swift_character_sets() {
        assert_eq!(trim_blanks("  a \t"), "a");
        assert_eq!(trim_blanks("\n a \n"), "\n a \n");
        assert_eq!(trim_ws_nl("\n a \n"), "a");
    }

    #[test]
    fn random_id_shape() {
        let id = random_id("call_");
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), "call_".len() + 24);
        assert!(id["call_".len()..].chars().all(|c| c.is_ascii_hexdigit()));
    }
}
