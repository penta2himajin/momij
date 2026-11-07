//! Maple's shipped `chat_template.jinja` always opens assistant with
//! `<think>\n`. That forces a reasoning channel and, on long prompts, collapses
//! into loops / raw `<|im_start|>` text. Serve defaults to non-thinking
//! generation; set `MOMIJ_ENABLE_THINKING=1` to keep the stock template.
//! Port of `ChatTemplatePatch.swift`.

use std::fs;
use std::path::Path;

use crate::TokenId;
use crate::json::trim_ws_nl;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// `<think>` and `<|im_start|>` token ids in Maple's vocabulary.
pub const THINK_TOKEN_ID: TokenId = 151_667;
pub const IM_START_TOKEN_ID: TokenId = 151_644;

pub fn enable_thinking() -> bool {
    enable_thinking_from(std::env::var("MOMIJ_ENABLE_THINKING").ok().as_deref())
}

pub fn enable_thinking_from(raw: Option<&str>) -> bool {
    raw.is_some_and(|s| s == "1" || s.eq_ignore_ascii_case("true"))
}

/// Replace the generation-prompt suffix that injects `<think>`. Maple's jinja
/// stores escaped `\n` inside the string literal (backslash + n).
pub fn without_forced_think(template: &str) -> String {
    const FROM: &str = r"{{- '<|im_start|>assistant\n<think>\n' }}";
    const TO: &str = r"{{- '<|im_start|>assistant\n' }}";
    template.replace(FROM, TO)
}

pub fn load_model_template(model_dir: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(model_dir.as_ref().join("chat_template.jinja")).ok()
}

/// Token ids never sampled in assistant output when thinking is disabled.
pub fn banned_assistant_token_ids_for(thinking_enabled: bool) -> Vec<TokenId> {
    if thinking_enabled {
        Vec::new()
    } else {
        vec![THINK_TOKEN_ID, IM_START_TOKEN_ID]
    }
}

pub fn banned_assistant_token_ids() -> Vec<TokenId> {
    banned_assistant_token_ids_for(enable_thinking())
}

/// Patched template string, or `None` to use the model's own template.
pub fn serve_template(model_dir: impl AsRef<Path>) -> Option<String> {
    if enable_thinking() {
        return None;
    }
    let raw = load_model_template(model_dir)?;
    let patched = without_forced_think(&raw);
    (patched != raw).then_some(patched)
}

/// If thinking is disabled and the model still emitted a think block, keep the
/// visible answer only (client content channel). An unclosed `<think>` is
/// dropped rather than leaked as assistant text.
pub fn strip_think_for_content(text: &str, thinking_enabled: bool) -> String {
    if thinking_enabled {
        return text.to_owned();
    }
    let mut s = text.to_owned();
    while let Some(open) = s.find(THINK_OPEN) {
        let after_open = open + THINK_OPEN.len();
        match s[after_open..].find(THINK_CLOSE) {
            Some(rel) => s.replace_range(open..after_open + rel + THINK_CLOSE.len(), ""),
            None => {
                s.truncate(open);
                break;
            }
        }
    }
    trim_ws_nl(&s.replace(THINK_CLOSE, "")).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_forced_think_removes_think_opener() {
        // Maple jinja uses escaped \n inside the quoted string.
        let file_style = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<think>\\n' }}\n{%- endif %}\n";
        let patched = without_forced_think(file_style);
        assert!(!patched.contains("<think>"));
        assert!(patched.contains("<|im_start|>assistant\\n' }}"));
    }

    #[test]
    fn without_forced_think_leaves_other_templates_alone() {
        let t = "{{- 'hello' }}";
        assert_eq!(without_forced_think(t), t);
    }

    #[test]
    fn strip_think_for_content_keeps_answer() {
        let text = "<think>\nplan\n</think>\nhello";
        assert_eq!(strip_think_for_content(text, false), "hello");
        assert_eq!(strip_think_for_content(text, true), text);
    }

    #[test]
    fn banned_assistant_tokens_when_thinking_off() {
        let banned = banned_assistant_token_ids_for(false);
        assert!(banned.contains(&151_667));
        assert!(banned.contains(&151_644));
        assert!(banned_assistant_token_ids_for(true).is_empty());
    }

    #[test]
    fn strip_think_drops_unclosed_block() {
        assert_eq!(
            strip_think_for_content("<think>\nAvailable. Available.", false),
            ""
        );
        assert_eq!(strip_think_for_content("ok <think>secret", false), "ok");
    }

    #[test]
    fn strip_think_handles_multiple_blocks_and_stray_close() {
        assert_eq!(
            strip_think_for_content("a<think>x</think>b<think>y</think>c", false),
            "abc"
        );
        assert_eq!(strip_think_for_content("answer</think>", false), "answer");
    }

    #[test]
    fn enable_thinking_parsing() {
        assert!(!enable_thinking_from(None));
        assert!(enable_thinking_from(Some("1")));
        assert!(enable_thinking_from(Some("TRUE")));
        assert!(!enable_thinking_from(Some("0")));
        assert!(!enable_thinking_from(Some("yes")));
    }

    /// Port of `testServeTemplatePatchesMapleJinja`: a model dir holding the
    /// stock Maple template gets a patched, think-free generation prompt.
    #[test]
    fn serve_template_patches_stock_template() {
        let tmp = tempfile::tempdir().unwrap();
        let stock = "{{- '<|im_start|>assistant\\n<think>\\n' }}";
        fs::write(tmp.path().join("chat_template.jinja"), stock).unwrap();
        let raw = load_model_template(tmp.path()).unwrap();
        assert!(
            raw.contains("<think>"),
            "stock Maple template must force think"
        );
        let patched = without_forced_think(&raw);
        assert!(!patched.contains("<|im_start|>assistant\\n<think>\\n"));
        assert!(patched.contains("<|im_start|>assistant\\n' }}"));
        assert!(load_model_template(tmp.path().join("missing")).is_none());
    }
}
