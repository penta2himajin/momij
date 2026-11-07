//! Targeted field repair for degenerate tool-call arguments.
//!
//! Instead of regenerating the whole reply (which breaks good parts), re-ask
//! the model for ONLY the broken fields, merge them into the existing
//! arguments, and re-validate. Ported concept from evprtr's
//! FreshConstrainedRepair narrowed to field scope; the field/reason mapping
//! follows `ResponseVerify.degenerateToolArgs` detail keys.
//! Port of `ArgRepair.swift`.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::JsonMap;
use crate::json::{json_dumps, json_parse_value, trim_ws_nl};
use crate::openai_compat::ChatMessage;
use crate::path_policy::PATH_FIELDS;

/// Prefix markers of harness scaffolding turns (see [`is_non_task_user_turn`]).
pub const NON_TASK_TURN_PREFIXES: [&str; 3] = [
    "<system-reminder>",
    "<tool_result>",
    "Current runtime context",
];

const PATH_SUFFIXES: [&str; 10] = [
    ".toml", ".rs", ".py", ".ts", ".js", ".json", ".md", ".txt", ".yml", ".yaml",
];

// MARK: - Which fields to re-ask for

/// Map a degenerate hit's reasons to the argument fields to fix. `detail` is
/// the hit's detail dict (`reason` / `tool` keys).
pub fn fields_to_fix(detail: &JsonMap) -> Vec<String> {
    let fields: &[&str] = match detail.get("reason").and_then(Value::as_str) {
        Some("path_not_filelike") => &["path"],
        Some("content_equals_path" | "cargo_toml_too_thin" | "source_content_too_short") => {
            &["content"]
        }
        Some("empty_edits" | "empty_edit_texts" | "noop_edit" | "invented_old_text") => &["edits"],
        _ => &[],
    };
    fields.iter().map(|s| (*s).to_owned()).collect()
}

/// [`fields_to_fix`] adapted to the field names the call actually uses
/// (`path` <-> `file_path` aliasing — live calls use either key).
pub fn repair_fields(detail: &JsonMap, args: &JsonMap) -> Vec<String> {
    fields_to_fix(detail)
        .into_iter()
        .map(|f| {
            if args.contains_key(&f) {
                f
            } else if f == "path" && args.contains_key("file_path") {
                "file_path".to_owned()
            } else if f == "file_path" && args.contains_key("path") {
                "path".to_owned()
            } else {
                f
            }
        })
        .collect()
}

/// Required argument fields (per the tool's JSON schema) that are absent or
/// null — the harness-side "missing required property" rejection, caught here
/// so the repair can fill them BEFORE the call leaves.
pub fn missing_required_fields(tool_lines: &[String], tool: &str, args: &JsonMap) -> Vec<String> {
    let Some(schema) = tool_schema(tool_lines, tool) else {
        return Vec::new();
    };
    // `required` must be an array of strings, else there is no opinion.
    let Some(required) = schema
        .get("required")
        .and_then(Value::as_array)
        .and_then(|a| {
            a.iter()
                .map(|v| v.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
        })
    else {
        return Vec::new();
    };
    required
        .into_iter()
        .filter(|f| match args.get(f) {
            None | Some(Value::Null) => true,
            Some(Value::String(s)) => s.is_empty(),
            Some(_) => false,
        })
        .collect()
}

/// evprtr `original_user_text`: prefer the primary user task (the longest early
/// user message), skipping tiny placeholder turns AND harness-generated
/// scaffolding turns. The latter are longer than the task, so the length
/// heuristic would otherwise select them and path extraction would read paths
/// quoted from injected context or tool output instead of the task.
pub fn original_user_text(messages: &[ChatMessage]) -> String {
    const NONE: &str = "(no user text)";
    struct Cand<'a> {
        idx: usize,
        len: usize,
        text: &'a str,
    }
    let mut candidates: Vec<Cand> = Vec::new();
    for (idx, m) in messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "user")
    {
        let text = trim_ws_nl(&m.content);
        if text.is_empty() {
            continue;
        }
        if ["agent", "(no user text)", "no user text"].contains(&text.to_lowercase().as_str()) {
            continue;
        }
        if is_non_task_user_turn(text) {
            continue;
        }
        candidates.push(Cand {
            idx,
            len: text.chars().count(),
            text,
        });
    }
    if candidates.is_empty() {
        return NONE.to_owned();
    }
    let max_len = candidates.iter().map(|c| c.len).max().unwrap_or(0);
    let threshold = 80usize.max((max_len as f64 * 0.6) as usize);
    let primary: Vec<&Cand> = candidates.iter().filter(|c| c.len >= threshold).collect();
    let pool: Vec<&Cand> = if primary.is_empty() {
        candidates.iter().collect()
    } else {
        primary
    };
    // min by (idx, -len): earliest message wins, then the longest.
    pool.into_iter()
        .min_by(|a, b| a.idx.cmp(&b.idx).then(b.len.cmp(&a.len)))
        .map_or_else(|| NONE.to_owned(), |c| c.text.to_owned())
}

/// A user turn that is harness-generated scaffolding rather than task text.
/// Only the WRAPPER form (turn starts with the marker) is excluded; a task
/// message that merely quotes one stays a candidate.
pub fn is_non_task_user_turn(text: &str) -> bool {
    let t = trim_ws_nl(text);
    NON_TASK_TURN_PREFIXES.iter().any(|p| t.starts_with(p))
}

// MARK: - Schema for the constrained re-ask

/// Extract the tool's `parameters` JSON schema from the request tool lines.
pub fn tool_schema(tool_lines: &[String], tool: &str) -> Option<JsonMap> {
    tool_lines.iter().find_map(|line| {
        let obj: Value = serde_json::from_str(line).ok()?;
        let func = obj.get("function")?.as_object()?;
        if func.get("name")?.as_str()? != tool {
            return None;
        }
        func.get("parameters")?.as_object().cloned()
    })
}

/// Build a strict object schema limited to `fields`, pulling each property's
/// schema from the tool definition (fallback: string).
pub fn constrained_object_schema(
    tool_lines: &[String],
    tool: &str,
    fields: &[String],
) -> Option<JsonMap> {
    if fields.is_empty() {
        return None;
    }
    let schema = tool_schema(tool_lines, tool);
    let props = schema
        .as_ref()
        .and_then(|s| s.get("properties"))
        .and_then(Value::as_object);
    let mut out_props = JsonMap::new();
    for f in fields {
        let p = props
            .and_then(|p| p.get(f))
            .filter(|p| p.is_object())
            .cloned()
            .unwrap_or_else(|| json!({"type": "string"}));
        out_props.insert(f.clone(), p);
    }
    let Value::Object(map) = json!({
        "type": "object",
        "properties": out_props,
        "required": fields,
        "additionalProperties": false,
    }) else {
        return None;
    };
    Some(map)
}

// MARK: - Prompt

/// English-only (Maple). Asks for exactly the broken fields as a JSON object;
/// everything else in the call is preserved.
pub fn field_repair_prompt(
    task: &str,
    tool: &str,
    args_json: &str,
    fields: &[String],
    reasons: &HashMap<String, String>,
) -> String {
    let problem_lines = fields
        .iter()
        .map(|f| {
            format!(
                "- {f}: {}",
                reasons.get(f).map_or("unusable", String::as_str)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Fix the arguments of a failed tool call. Reply with ONLY a JSON object and nothing else.\n\n\
         Task (what the user asked for):\n{task}\n\n\
         The assistant tried to call the `{tool}` tool with these arguments:\n{args_json}\n\n\
         Problems found:\n{problem_lines}\n\n\
         Reply with ONLY a JSON object that has exactly these keys: {}\n\
         Use corrected values that make the `{tool}` call usable. No prose, no markdown.",
        fields.join(", ")
    )
}

// MARK: - Merge + serialize

/// Merge the generated field values into the original arguments. Only `fields`
/// are taken from `generated`; everything else is kept.
pub fn merge_fields(args: &JsonMap, generated: &JsonMap, fields: &[String]) -> JsonMap {
    let mut out = args.clone();
    for f in fields {
        if let Some(v) = generated.get(f) {
            out.insert(f.clone(), v.clone());
        }
    }
    out
}

/// Serialize merged args back to the OpenAI arguments JSON string
/// (`json_dumps` keeps the contract's python-style spacing).
pub fn args_json_string(args: &JsonMap) -> String {
    json_dumps(&Value::Object(args.clone()))
}

/// Parse `function.arguments` (JSON object string) leniently; falls back to an
/// empty map like evprtr `_fn_args`.
pub fn parse_args(arguments: &str) -> JsonMap {
    match json_parse_value(arguments) {
        Some(Value::Object(m)) => m,
        _ => JsonMap::new(),
    }
}

/// Extract the first JSON object from a model reply (the constrained
/// generation should be pure JSON; tolerate stray prose/markdown and literal
/// newlines inside strings).
pub fn first_json_object(text: &str) -> Option<JsonMap> {
    let trimmed = trim_ws_nl(text);
    if let Some(Value::Object(m)) = json_parse_value(trimmed) {
        return Some(m);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if start >= end {
        return None;
    }
    match json_parse_value(&trimmed[start..=end])? {
        Value::Object(m) => Some(m),
        _ => None,
    }
}

/// Extract the most likely target path from a task text (deterministic — the
/// task names the exact relative path; the model only needs to copy it, but
/// live sessions showed it corrupting instead). Codeish suffixes define
/// candidate tokens; the longest slash-joined candidate wins. `None` when the
/// task names no path (the caller falls back to the model re-ask).
pub fn path_from_task(task: &str) -> Option<String> {
    let is_word = |c: char| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-');
    let tokens = task.split(|c: char| !is_word(c)).filter(|t| !t.is_empty());
    let mut candidates: Vec<String> = Vec::new();
    for tok in tokens {
        // Leading "./" pairs and bare slashes are debris; a lone leading dot is
        // hidden-directory semantics and must survive
        // (".github/PULL_REQUEST_TEMPLATE.md" is not "github/...").
        let mut t = tok;
        while let Some(rest) = t.strip_prefix("./") {
            t = rest;
        }
        while let Some(rest) = t.strip_prefix('/') {
            t = rest;
        }
        t = t.trim_end_matches(['.', '/']);
        if t.is_empty() {
            continue;
        }
        let lowered = t.to_lowercase();
        if PATH_SUFFIXES.iter().any(|s| lowered.ends_with(s)) {
            candidates.push(t.to_owned());
        }
    }
    // Longest wins; ties keep the first (Swift `max(by:)` semantics).
    let longest = |it: Vec<&String>| -> Option<String> {
        let mut best: Option<&String> = None;
        for c in it {
            if best.is_none_or(|b| c.chars().count() > b.chars().count()) {
                best = Some(c);
            }
        }
        best.cloned()
    };
    let slashed: Vec<&String> = candidates.iter().filter(|c| c.contains('/')).collect();
    longest(slashed).or_else(|| longest(candidates.iter().collect()))
}

// MARK: - Extraction-first path fix (path_from_task wiring)

/// True when a pending fix targets a path field: the workspace scan
/// (`path_outside_workspace`), an aliased path field from a degenerate hit, or
/// a schema-required path property missing on the call.
pub fn is_path_fix(kind: &str, fields: &[String]) -> bool {
    kind == "path_outside_workspace" || fields.iter().any(|f| PATH_FIELDS.contains(&f.as_str()))
}

/// Extraction-first answer for a path-kind pending fix: the task text names
/// the exact relative path, so merge `path_from_task(task)` into the pending
/// path field(s) instead of re-asking the model (which repeats its
/// hallucination). `None` when the fix is not path-kind, the task names no
/// path, or the fix also covers non-path fields. The adoption decision stays
/// with the caller's clean check.
pub fn task_path_merge(
    kind: &str,
    fields: &[String],
    args: &JsonMap,
    task: &str,
) -> Option<JsonMap> {
    if !is_path_fix(kind, fields) {
        return None;
    }
    let extracted = path_from_task(task)?;
    let touched: Vec<&String> = fields
        .iter()
        .filter(|f| PATH_FIELDS.contains(&f.as_str()))
        .collect();
    if touched.is_empty() || touched.len() != fields.len() {
        return None;
    }
    let mut out = args.clone();
    for f in touched {
        out.insert(f.clone(), Value::String(extracted.clone()));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: Value) -> JsonMap {
        v.as_object().expect("object literal").clone()
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    fn user(s: &str) -> ChatMessage {
        ChatMessage::new("user", s)
    }

    #[test]
    fn fields_to_fix_mapping() {
        let f = |r: &str| fields_to_fix(&map(json!({ "reason": r })));
        assert_eq!(f("path_not_filelike"), ["path"]);
        assert_eq!(f("content_equals_path"), ["content"]);
        assert_eq!(f("cargo_toml_too_thin"), ["content"]);
        assert_eq!(f("source_content_too_short"), ["content"]);
        assert_eq!(f("empty_edits"), ["edits"]);
        assert_eq!(f("empty_edit_texts"), ["edits"]);
        assert_eq!(f("noop_edit"), ["edits"]);
        assert_eq!(f("invented_old_text"), ["edits"]);
        assert!(f("unknown_thing").is_empty());
    }

    #[test]
    fn original_user_text_prefers_longest_early_message() {
        let messages = vec![
            ChatMessage::new("system", "sys"),
            user("short steer"),
            ChatMessage::new("assistant", "ok"),
            ChatMessage::new("tool", "result"),
            user(&"primary task text. ".repeat(20)),
        ];
        let out = original_user_text(&messages);
        assert!(out.starts_with("primary task text."), "{out}");
    }

    #[test]
    fn original_user_text_skips_placeholders() {
        let messages = vec![user("agent"), user("(no user text)")];
        assert_eq!(original_user_text(&messages), "(no user text)");
    }

    #[test]
    fn original_user_text_skips_harness_reminder_blocks() {
        // DSH wraps harness context as <system-reminder> user turns. They are
        // LONGER than the task, so "longest early" would pick them.
        let reminder = format!(
            "<system-reminder>\nThe following workspace instructions {}.github/PULL_REQUEST_TEMPLATE.md\n</system-reminder>",
            "may be relevant to your work. ".repeat(30)
        );
        let task = "write scratch/momij-subagent-test/x.py "
            .repeat(6)
            .trim()
            .to_owned();
        let messages = vec![
            ChatMessage::new("system", "sys"),
            user(&reminder),
            user(&task),
        ];
        assert_eq!(original_user_text(&messages), task);
    }

    #[test]
    fn original_user_text_skips_tool_result_turns() {
        let tool_result = format!(
            "<tool_result>\n{}</tool_result>",
            "drwxr-xr-x  315 staff  10080 Sep 14 14:05 .\n".repeat(30)
        );
        let task = "Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py.";
        let messages = vec![
            user(task),
            ChatMessage::new("assistant", "listing the directory"),
            user(&tool_result),
        ];
        assert_eq!(original_user_text(&messages), task);
    }

    #[test]
    fn original_user_text_skips_runtime_context_blocks() {
        let runtime = format!(
            "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\n\
             Current DSH file policy: workspace-write. {}",
            "Any available operation enforced by the DSH file sandbox. ".repeat(15)
        );
        let task = "Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py.";
        let messages = vec![
            ChatMessage::new("system", "sys"),
            user(&runtime),
            user(task),
        ];
        assert_eq!(original_user_text(&messages), task);
    }

    #[test]
    fn non_task_user_turn_markers() {
        assert!(is_non_task_user_turn("<system-reminder>\nx"));
        assert!(is_non_task_user_turn("<tool_result>\nx"));
        assert!(is_non_task_user_turn(
            "Current runtime context. This snapshot …"
        ));
        assert!(!is_non_task_user_turn(
            "Write a script per <system-reminder> notes"
        ));
    }

    fn tool_lines() -> Vec<String> {
        strs(&[
            r#"{"type":"function","function":{"name":"write","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string"}},"required":["path","content"]}}}"#,
            r#"{"type":"function","function":{"name":"ls","description":"List directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}"#,
        ])
    }

    #[test]
    fn constrained_object_schema_uses_tool_properties() {
        let schema = constrained_object_schema(&tool_lines(), "write", &strs(&["path"])).unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"], json!(["path"]));
        assert_eq!(schema["additionalProperties"], false);
        let props = schema["properties"].as_object().unwrap();
        assert_eq!(props["path"]["type"], "string");
        assert_eq!(props["path"]["description"], "File path");
        assert!(
            !props.contains_key("content"),
            "only requested fields are included"
        );
    }

    #[test]
    fn constrained_object_schema_falls_back_to_string() {
        let schema =
            constrained_object_schema(&tool_lines(), "nonexistent", &strs(&["edits"])).unwrap();
        assert_eq!(schema["properties"]["edits"]["type"], "string");
        assert!(constrained_object_schema(&tool_lines(), "write", &[]).is_none());
    }

    #[test]
    fn field_repair_prompt_shape() {
        let reasons = HashMap::from([("path".to_owned(), "path_not_filelike".to_owned())]);
        let prompt = field_repair_prompt(
            "Create notes.md",
            "write",
            "{\"content\": \"hello\"}",
            &strs(&["path"]),
            &reasons,
        );
        assert!(prompt.contains("ONLY a JSON object"), "{prompt}");
        assert!(prompt.contains("Create notes.md"), "{prompt}");
        assert!(prompt.contains("`write`"), "{prompt}");
        assert!(prompt.contains("- path: path_not_filelike"), "{prompt}");
        assert!(prompt.ends_with("No prose, no markdown."), "{prompt}");
        let unknown = field_repair_prompt("t", "w", "{}", &strs(&["edits"]), &HashMap::new());
        assert!(unknown.contains("- edits: unusable"), "{unknown}");
    }

    #[test]
    fn merge_fields_keeps_existing() {
        let args = map(json!({"path": "", "content": "long content body"}));
        let merged = merge_fields(&args, &map(json!({"path": "notes.md"})), &strs(&["path"]));
        assert_eq!(merged["path"], "notes.md");
        assert_eq!(merged["content"], "long content body");
    }

    #[test]
    fn merge_fields_accepts_array_value() {
        let merged = merge_fields(
            &map(json!({"path": "a.txt"})),
            &map(json!({"edits": [{"oldText": "x", "newText": "y"}]})),
            &strs(&["edits"]),
        );
        assert_eq!(merged["edits"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn merge_fields_ignores_fields_outside_the_list() {
        let merged = merge_fields(
            &map(json!({"a": 1})),
            &map(json!({"a": 2, "b": 3})),
            &strs(&["b"]),
        );
        assert_eq!(merged["a"], 1);
        assert_eq!(merged["b"], 3);
    }

    #[test]
    fn args_json_string_roundtrip() {
        let json = args_json_string(&map(json!({"path": "notes.md", "content": "hi"})));
        let back = parse_args(&json);
        assert_eq!(back["path"], "notes.md");
        // python-style spacing contract (json_dumps parity)
        assert!(json.contains("\"path\": \"notes.md\""), "{json}");
        assert!(parse_args("garbage").is_empty());
    }

    #[test]
    fn missing_required_fields_cases() {
        let bash = strs(&[
            r#"{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command","description"]}}}"#,
        ]);
        // description absent -> flagged.
        assert_eq!(
            missing_required_fields(&bash, "bash", &map(json!({"command": "ls"}))),
            ["description"]
        );
        // Empty string counts as missing.
        assert_eq!(
            missing_required_fields(
                &bash,
                "bash",
                &map(json!({"command": "ls", "description": ""}))
            ),
            ["description"]
        );
        // Null counts as missing.
        assert_eq!(
            missing_required_fields(
                &bash,
                "bash",
                &map(json!({"command": "ls", "description": null}))
            ),
            ["description"]
        );
        // All present -> empty.
        assert!(
            missing_required_fields(
                &bash,
                "bash",
                &map(json!({"command": "ls", "description": "List files"}))
            )
            .is_empty()
        );
        // Unknown tool -> no schema, no opinion.
        assert!(missing_required_fields(&bash, "nonexistent", &JsonMap::new()).is_empty());
    }

    #[test]
    fn first_json_object_tolerates_prose() {
        let obj = first_json_object("Sure.\n\n{\"path\": \"a.md\"}\n\nDone.").unwrap();
        assert_eq!(obj["path"], "a.md");
        assert!(first_json_object("no object here").is_none());
        assert!(first_json_object("} backwards {").is_none());
    }

    #[test]
    fn path_from_task_cases() {
        assert_eq!(
            path_from_task("Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py (this path is RELATIVE to the workspace root; always write paths in this relative form).").as_deref(),
            Some("scratch/momij-subagent-test/count_verify_fires.py")
        );
        assert_eq!(
            path_from_task("Create a new file named notes.md using the write tool. Put two short lines of text in it.").as_deref(),
            Some("notes.md")
        );
        assert_eq!(path_from_task("Reply with exactly OK"), None);
    }

    #[test]
    fn path_from_task_keeps_leading_dot() {
        // ".github/..." must not be mangled into "github/..." (the live probe
        // adopted exactly that mangled path from harness-injected context).
        assert_eq!(
            path_from_task(
                "One PR per workstream; closes the work per .github/PULL_REQUEST_TEMPLATE.md."
            )
            .as_deref(),
            Some(".github/PULL_REQUEST_TEMPLATE.md")
        );
        // "./debris" and leading-slash debris still normalize; trailing
        // sentence punctuation still strips.
        assert_eq!(
            path_from_task("Save to ./notes.md.").as_deref(),
            Some("notes.md")
        );
        assert_eq!(
            path_from_task("Write /scratch/a.py now").as_deref(),
            Some("scratch/a.py")
        );
    }

    #[test]
    fn path_from_task_tie_keeps_first_and_handles_yaml() {
        assert_eq!(
            path_from_task("edit a.py then b.py").as_deref(),
            Some("a.py")
        );
        assert_eq!(
            path_from_task("see config/ci.YAML").as_deref(),
            Some("config/ci.YAML")
        );
        // Slashed candidates beat longer bare ones.
        assert_eq!(
            path_from_task("a/b.py and verylongname.py").as_deref(),
            Some("a/b.py")
        );
    }

    #[test]
    fn repair_fields_alias() {
        // The call used file_path; the detector speaks of path.
        let detail = map(json!({"reason": "path_not_filelike"}));
        assert_eq!(
            repair_fields(&detail, &map(json!({"file_path": "x"}))),
            ["file_path"]
        );
        assert_eq!(repair_fields(&detail, &map(json!({"path": "x"}))), ["path"]);
        assert!(repair_fields(&map(json!({"reason": "other"})), &JsonMap::new()).is_empty());
    }

    #[test]
    fn is_path_fix_cases() {
        assert!(is_path_fix("path_outside_workspace", &strs(&["path"])));
        assert!(is_path_fix("degenerate_tool_args", &strs(&["file_path"])));
        assert!(is_path_fix("missing_required_property", &strs(&["path"])));
        assert!(!is_path_fix("degenerate_tool_args", &strs(&["content"])));
        assert!(!is_path_fix(
            "missing_required_property",
            &strs(&["description"])
        ));
    }

    #[test]
    fn task_path_merge_answers_path_outside_workspace() {
        let task = "Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py (this path is RELATIVE to the workspace root; always write paths in this relative form).";
        let merged = task_path_merge(
            "path_outside_workspace",
            &strs(&["path"]),
            &map(json!({"path": "/mijij-subagent-test/count_verify_fires.py", "content": "print(1)"})),
            task,
        )
        .unwrap();
        assert_eq!(
            merged["path"],
            "scratch/momij-subagent-test/count_verify_fires.py"
        );
        assert_eq!(merged["content"], "print(1)", "other fields survive");
    }

    #[test]
    fn task_path_merge_aliases_file_path() {
        let merged = task_path_merge(
            "degenerate_tool_args",
            &strs(&["file_path"]),
            &map(json!({"file_path": "", "content": "body"})),
            "Create notes.md using the write tool. Put two short lines of text in it.",
        )
        .unwrap();
        assert_eq!(merged["file_path"], "notes.md");
        assert_eq!(merged["content"], "body");
    }

    #[test]
    fn task_path_merge_none_when_not_path_kind() {
        assert!(
            task_path_merge(
                "degenerate_tool_args",
                &strs(&["content"]),
                &map(json!({"path": "a.py", "content": "x"})),
                "Write a.py with the content x"
            )
            .is_none()
        );
    }

    #[test]
    fn task_path_merge_none_when_task_names_no_path() {
        assert!(
            task_path_merge(
                "path_outside_workspace",
                &strs(&["path"]),
                &map(json!({"path": "/elsewhere/x.py"})),
                "Reply with exactly OK"
            )
            .is_none()
        );
    }

    #[test]
    fn task_path_merge_none_when_non_path_fields_pending() {
        // A mixed fix (path + description) must fall back to the model re-ask.
        assert!(
            task_path_merge(
                "missing_required_property",
                &strs(&["path", "description"]),
                &JsonMap::new(),
                "Write notes.md with the report"
            )
            .is_none()
        );
    }
}
