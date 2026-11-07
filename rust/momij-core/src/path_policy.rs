//! Workspace-relative path policy: the model writes paths RELATIVE to a fixed
//! workspace root (so it never fumbles long absolute paths), and momij
//! resolves them to absolute before the harness executes the call. Absolute
//! paths inside the workspace are normalized back to the root, and the
//! conversation history is stripped of the root prefix so the model's world
//! stays purely relative. Port of `PathPolicy.swift`.

use serde_json::Value;

use crate::JsonMap;
use crate::json::{json_dumps, json_parse_value, trim_blanks, trim_ws_nl};

/// Path-bearing argument fields known to harness tools.
pub const PATH_FIELDS: [&str; 2] = ["path", "file_path"];

/// `MOMIJ_WORKSPACE` env: the fixed workspace root. `None` = mode off.
pub fn workspace_root() -> Option<String> {
    workspace_root_from(std::env::var("MOMIJ_WORKSPACE").ok().as_deref())
}

pub fn workspace_root_from(raw: Option<&str>) -> Option<String> {
    let s = trim_blanks(raw?);
    (!s.is_empty()).then(|| s.to_owned())
}

/// Strip escape debris (backslashes, quotes) the model tends to add to paths,
/// and collapse doubled separators. Path fields only — never free text.
pub fn normalize_escapes(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut prev_slash = false;
    for ch in path.chars().filter(|&c| c != '\\' && c != '"') {
        if ch == '/' {
            if prev_slash {
                continue;
            }
            prev_slash = true;
        } else {
            prev_slash = false;
        }
        out.push(ch);
    }
    out
}

/// Resolve a tool-call path value against the workspace root.
/// - relative -> root + "/" + value
/// - absolute (inside or outside root) -> kept, normalized
pub fn resolve(value: &str, root: &str) -> String {
    let normalized = normalize_escapes(value);
    let cleaned = trim_ws_nl(&normalized);
    if cleaned.is_empty() || cleaned.starts_with('/') {
        return cleaned.to_owned();
    }
    if root.ends_with('/') {
        format!("{root}{cleaned}")
    } else {
        format!("{root}/{cleaned}")
    }
}

/// Rewrite path fields inside an arguments JSON string. `None` when nothing
/// changed (caller keeps the original).
pub fn rewritten_args(args_json: &str, root: &str) -> Option<String> {
    let Value::Object(args) = json_parse_value(args_json)? else {
        return None;
    };
    let mut out = args.clone();
    let mut changed = false;
    for field in PATH_FIELDS {
        let Some(v) = args
            .get(field)
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
        else {
            continue;
        };
        let resolved = resolve(v, root);
        if resolved != v {
            out.insert(field.to_owned(), Value::String(resolved));
            changed = true;
        }
    }
    changed.then(|| json_dumps(&Value::Object(out)))
}

/// Strip the workspace-root prefix from text the model sees (tool results,
/// history) so its world stays relative and short.
pub fn strip_root_prefix(text: &str, root: &str) -> String {
    if root.is_empty() {
        return text.to_owned();
    }
    let with_slash = if root.ends_with('/') {
        root.to_owned()
    } else {
        format!("{root}/")
    };
    text.replace(&with_slash, "").replace(root, "")
}

/// True when the path (after escape normalization) is absolute and OUTSIDE the
/// workspace root — a workspace-discipline violation the repair pipeline
/// should re-ask for, and the sandbox would reject.
pub fn is_outside_workspace(value: &str, root: &str) -> bool {
    let normalized = normalize_escapes(value);
    let cleaned = trim_ws_nl(&normalized);
    cleaned.starts_with('/') && !cleaned.starts_with(root)
}

/// True when ANY path field in the parsed args resolves outside the workspace
/// root (workspace mode only) — a repairable violation.
pub fn args_outside_workspace(args: &JsonMap, root: &str) -> bool {
    PATH_FIELDS.iter().any(|f| {
        args.get(*f)
            .and_then(Value::as_str)
            .is_some_and(|v| !v.is_empty() && is_outside_workspace(v, root))
    })
}

/// The instruction line appended to the tools suffix in workspace mode.
pub fn suffix_line() -> &'static str {
    "Paths in tool calls are RELATIVE to the workspace root. Never write absolute paths; never escape slashes."
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const ROOT: &str = "/Users/penta2himajin/repos";

    #[test]
    fn normalize_escapes_strips_debris() {
        assert_eq!(
            normalize_escapes(r"\\\/Users\/penta2himajin\/repos\/x.py"),
            "/Users/penta2himajin/repos/x.py"
        );
        assert_eq!(
            normalize_escapes(r#""\\\\/repos/scratch/a.py""#),
            "/repos/scratch/a.py"
        );
        assert_eq!(normalize_escapes("scratch/x.py"), "scratch/x.py");
    }

    #[test]
    fn resolve_relative_joins_root() {
        assert_eq!(
            resolve("scratch/momij-subagent-test/count_verify_fires.py", ROOT),
            format!("{ROOT}/scratch/momij-subagent-test/count_verify_fires.py")
        );
        assert_eq!(resolve("notes.md", ROOT), format!("{ROOT}/notes.md"));
        assert_eq!(resolve("notes.md", "/r/"), "/r/notes.md");
    }

    #[test]
    fn resolve_absolute_inside_root_kept() {
        let abs = format!("{ROOT}/scratch/x.py");
        assert_eq!(resolve(&abs, ROOT), abs);
    }

    #[test]
    fn resolve_absolute_outside_root_kept_unchanged() {
        let abs = "/Users/penta2himajin/other/repo/x.py";
        assert_eq!(resolve(abs, ROOT), abs);
    }

    #[test]
    fn resolve_normalizes_escape_debris_to_absolute() {
        assert_eq!(
            resolve(r"\\\/Users\/penta2himajin\/repos\/scratch\/a.py", ROOT),
            format!("{ROOT}/scratch/a.py")
        );
    }

    #[test]
    fn rewritten_args_resolves_path_fields() {
        let args = json_dumps(&json!({"path": "scratch/a.py", "content": "body"}));
        let out = rewritten_args(&args, ROOT).expect("path should be rewritten");
        let parsed = json_parse_value(&out).unwrap();
        assert_eq!(parsed["path"], format!("{ROOT}/scratch/a.py"));
        assert_eq!(parsed["content"], "body", "non-path fields untouched");
    }

    #[test]
    fn rewritten_args_none_when_no_change() {
        assert_eq!(
            rewritten_args(&json_dumps(&json!({"command": "ls"})), ROOT),
            None
        );
        // Absolute inside root: after resolve it stays the same string -> None.
        assert_eq!(
            rewritten_args(&json_dumps(&json!({"path": format!("{ROOT}/a.py")})), ROOT),
            None
        );
        assert_eq!(rewritten_args("not json", ROOT), None);
    }

    #[test]
    fn strip_root_prefix_removes_root() {
        assert_eq!(
            strip_root_prefix(&format!("wrote {ROOT}/scratch/a.py and {ROOT}/b.txt"), ROOT),
            "wrote scratch/a.py and b.txt"
        );
        assert_eq!(strip_root_prefix("unchanged", ""), "unchanged");
    }

    #[test]
    fn suffix_line_content() {
        let line = suffix_line();
        assert!(line.contains("RELATIVE to the workspace root"), "{line}");
        assert!(line.contains("Never write absolute paths"), "{line}");
    }

    #[test]
    fn is_outside_workspace_cases() {
        assert!(is_outside_workspace("/mijij-subagent-test/x.py", ROOT));
        assert!(!is_outside_workspace(&format!("{ROOT}/scratch/x.py"), ROOT));
        // Relative is always fine.
        assert!(!is_outside_workspace("scratch/x.py", ROOT));
        // Escape debris normalized before the check.
        assert!(is_outside_workspace(r"\\\/mnt\\\/x.py", ROOT));
    }

    #[test]
    fn args_outside_workspace_checks_all_path_fields() {
        let args = json!({"file_path": "/etc/passwd", "content": "x"});
        assert!(args_outside_workspace(args.as_object().unwrap(), ROOT));
        let ok = json!({"path": "a.py"});
        assert!(!args_outside_workspace(ok.as_object().unwrap(), ROOT));
    }

    /// Env-independent port of `testWorkspaceRootFromEnv`.
    #[test]
    fn workspace_root_from_values() {
        assert_eq!(workspace_root_from(None), None);
        assert_eq!(workspace_root_from(Some("  ")), None);
        assert_eq!(workspace_root_from(Some(" /w ")), Some("/w".to_owned()));
    }
}
