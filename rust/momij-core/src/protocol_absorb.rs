//! Protocol-tool interception (the ending fumble absorb): after long tool-call
//! runs Maple signals "I'm done" by calling a protocol tool (`submit`,
//! `ask_user_question`) — often with unusable arguments, which the harness
//! rejects, looping the turn. momij absorbs the intent: drop every parsed call
//! and end the turn with a clean stop + the call's description (or a
//! placeholder) as the final content. Port of `ProtocolAbsorb.swift`.

use std::collections::HashSet;

use serde_json::Value;

use crate::arg_repair::parse_args;
use crate::json::{trim_blanks, trim_ws_nl};
use crate::openai_compat::ToolCallSpec;

/// The tool names whose calls get absorbed. `MOMIJ_PROTOCOL_TOOLS` overrides
/// the default (comma-separated); an empty value disables interception.
/// `None` = variable unset = default set.
pub fn tool_names(env: Option<&str>) -> HashSet<String> {
    match env {
        None => ["submit", "ask_user_question"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        Some(raw) => raw
            .split(',')
            .map(trim_blanks)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect(),
    }
}

/// [`tool_names`] reading `MOMIJ_PROTOCOL_TOOLS` from the process environment.
pub fn tool_names_from_env() -> HashSet<String> {
    tool_names(std::env::var("MOMIJ_PROTOCOL_TOOLS").ok().as_deref())
}

/// The first parsed call targeting a protocol tool (`None` = no absorb).
/// Empty-named calls can never match.
pub fn absorb_call<'a>(
    calls: &'a [ToolCallSpec],
    names: &HashSet<String>,
) -> Option<&'a ToolCallSpec> {
    calls
        .iter()
        .find(|c| !c.name.is_empty() && names.contains(&c.name))
}

/// Final content for an absorbed call: its `description` argument (when
/// usable) or the placeholder — the model's "I'm done" intent becomes the
/// final assistant message.
pub fn final_content(call: &ToolCallSpec) -> String {
    let args = parse_args(&call.arguments);
    if let Some(Value::String(d)) = args.get("description") {
        let trimmed = trim_ws_nl(d);
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    "Task complete.".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &str) -> ToolCallSpec {
        ToolCallSpec::new("call-1", name, args)
    }

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn tool_names_default() {
        assert_eq!(tool_names(None), set(&["submit", "ask_user_question"]));
    }

    #[test]
    fn tool_names_override_parses_comma_list() {
        assert_eq!(
            tool_names(Some(" submit , report_done ")),
            set(&["submit", "report_done"])
        );
    }

    #[test]
    fn tool_names_empty_disables() {
        assert_eq!(tool_names(Some("  ")), set(&[]));
        assert_eq!(tool_names(Some(",")), set(&[]));
    }

    #[test]
    fn absorb_call_fires_on_protocol_tool() {
        let names = tool_names(None);
        let calls = [call("write", r#"{"path": "a.md"}"#), call("submit", "{}")];
        assert_eq!(
            absorb_call(&calls, &names).map(|c| c.name.as_str()),
            Some("submit")
        );
        let calls = [call("ask_user_question", "{}")];
        assert_eq!(
            absorb_call(&calls, &names).map(|c| c.name.as_str()),
            Some("ask_user_question")
        );
    }

    #[test]
    fn absorb_call_none_when_no_protocol_tool() {
        let names = tool_names(None);
        assert!(absorb_call(&[call("write", "{}"), call("bash", "{}")], &names).is_none());
        // Empty names can never match a protocol tool.
        assert!(absorb_call(&[call("", "{}")], &names).is_none());
        assert!(absorb_call(&[call("", "{}")], &set(&[""])).is_none());
    }

    #[test]
    fn absorb_disabled_by_empty_env() {
        assert!(absorb_call(&[call("submit", "{}")], &tool_names(Some(""))).is_none());
    }

    #[test]
    fn final_content_prefers_description() {
        assert_eq!(
            final_content(&call(
                "submit",
                r#"{"description": "Wrote the script and verified its output."}"#
            )),
            "Wrote the script and verified its output."
        );
    }

    #[test]
    fn final_content_placeholder_when_description_unusable() {
        assert_eq!(final_content(&call("submit", "{}")), "Task complete.");
        assert_eq!(
            final_content(&call("ask_user_question", r#"{"description": "   "}"#)),
            "Task complete."
        );
        assert_eq!(
            final_content(&call("submit", r#"{"description": 5}"#)),
            "Task complete."
        );
        assert_eq!(final_content(&call("submit", "garbage")), "Task complete.");
    }
}
