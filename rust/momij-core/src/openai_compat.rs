//! OpenAI-compatible chat request/response helpers for the momij serve path.
//! Unknown JSON fields are ignored (evprtr / Pi may attach tools,
//! structured_outputs, ...). Port of `OpenAIChatCompat.swift`.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::TokenId;
use crate::structured_outputs::{self, Constraint};

/// ChatML `<|im_end|>` (the Swift original's default EOS).
pub const DEFAULT_EOS_TOKEN_IDS: [TokenId; 1] = [151_645];

/// One entry of an assistant `tool_calls` array (OpenAI contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallSpec {
    pub id: String,
    pub name: String,
    /// The arguments as a JSON object string (as the harness sent it).
    pub arguments: String,
}

impl ToolCallSpec {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// assistant messages: parsed `tool_calls` entries (OpenAI contract).
    pub tool_calls: Vec<ToolCallSpec>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    pub fn with_tool_calls(mut self, calls: Vec<ToolCallSpec>) -> Self {
        self.tool_calls = calls;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    /// OpenAI `tools` entries, serialized as one JSON line per tool.
    pub tools_lines: Vec<String>,
    pub max_tokens: i64,
    pub temperature: f64,
    pub top_p: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub repetition_penalty: f64,
    pub stream: bool,
    /// `stream_options.include_usage` — clients request usage in the SSE tail.
    pub stream_options_include_usage: bool,
    pub n: i64,
    pub seed: Option<u64>,
    /// From `structured_outputs` / `guided_grammar` (None = unconstrained).
    pub structured_constraint: Option<Constraint>,
}

impl ChatCompletionRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            model: None,
            messages,
            tools_lines: Vec::new(),
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            stream: false,
            stream_options_include_usage: false,
            n: 1,
            seed: None,
            structured_constraint: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    InvalidJson,
    MissingMessages,
    EmptyMessages,
    InvalidMessage(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson => f.write_str("invalid JSON body"),
            Self::MissingMessages => f.write_str("missing messages"),
            Self::EmptyMessages => f.write_str("messages must be non-empty"),
            Self::InvalidMessage(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse a chat.completions body. Extra keys (`tools`, `tool_choice`, ...) are ignored.
pub fn parse_request(data: &[u8]) -> Result<ChatCompletionRequest, ParseError> {
    let root: Value = serde_json::from_slice(data).map_err(|_| ParseError::InvalidJson)?;
    let root_obj = root.as_object().ok_or(ParseError::InvalidJson)?;
    let raw_messages = root_obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(ParseError::MissingMessages)?;
    if raw_messages.is_empty() {
        return Err(ParseError::EmptyMessages);
    }

    let mut messages = Vec::with_capacity(raw_messages.len());
    for (i, item) in raw_messages.iter().enumerate() {
        let m = item.as_object().ok_or_else(|| {
            ParseError::InvalidMessage(format!("messages[{i}] must be an object"))
        })?;
        let mut role = match m.get("role").and_then(Value::as_str) {
            Some(r) if !r.is_empty() => r.to_owned(),
            _ => {
                return Err(ParseError::InvalidMessage(format!(
                    "messages[{i}] missing role"
                )));
            }
        };
        if role == "developer" {
            role = "system".into();
        }
        let content = normalize_content(m.get("content"));
        let mut calls = Vec::new();
        if let Some(raw_calls) = m.get("tool_calls").and_then(Value::as_array) {
            for c in raw_calls {
                let Some(cd) = c.as_object() else { continue };
                let id = cd
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let func = cd.get("function").and_then(Value::as_object);
                let name = func
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let arguments = match func.and_then(|f| f.get("arguments")) {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => compact_json(other),
                    None => String::new(),
                };
                calls.push(ToolCallSpec {
                    id,
                    name,
                    arguments,
                });
            }
        }
        messages.push(ChatMessage {
            role,
            content,
            tool_calls: calls,
        });
    }

    let max_tokens = [
        root_obj.get("max_completion_tokens"),
        root_obj.get("max_tokens"),
    ]
    .into_iter()
    .find_map(|v| int_value(v).filter(|&n| n > 0))
    .unwrap_or(256);

    let tools_lines = root_obj
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| tools.iter().map(compact_json).collect())
        .unwrap_or_default();

    Ok(ChatCompletionRequest {
        model: root_obj
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        messages,
        tools_lines,
        max_tokens,
        temperature: double_value(root_obj.get("temperature")).unwrap_or(0.0),
        top_p: double_value(root_obj.get("top_p")).unwrap_or(1.0),
        presence_penalty: double_value(root_obj.get("presence_penalty")).unwrap_or(0.0),
        frequency_penalty: double_value(root_obj.get("frequency_penalty")).unwrap_or(0.0),
        repetition_penalty: double_value(root_obj.get("repetition_penalty")).unwrap_or(1.0),
        stream: bool_value(root_obj.get("stream")).unwrap_or(false),
        stream_options_include_usage: root_obj
            .get("stream_options")
            .and_then(Value::as_object)
            .and_then(|o| bool_value(o.get("include_usage")))
            .unwrap_or(false),
        n: int_value(root_obj.get("n")).unwrap_or(1),
        seed: uint64_value(root_obj.get("seed")),
        structured_constraint: structured_outputs::parse_constraint(&root),
    })
}

/// Coerce OpenAI `content` (string | null | multimodal parts) to a single string.
pub fn normalize_content(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    Value::String(s) => out.push_str(s),
                    Value::Object(d) => {
                        if let Some(t) = d.get("text").and_then(Value::as_str) {
                            out.push_str(t);
                        } else if let Some(t) = d.get("content").and_then(Value::as_str) {
                            out.push_str(t);
                        }
                    }
                    _ => {}
                }
            }
            out
        }
        Some(other) => compact_json(other),
    }
}

/// `stop` if generation ended early (EOS / short); `length` if `max_tokens`
/// filled without EOS.
pub fn finish_reason(
    completion_tokens: &[TokenId],
    max_tokens: i64,
    eos_token_ids: &[TokenId],
) -> &'static str {
    if completion_tokens
        .last()
        .is_some_and(|t| eos_token_ids.contains(t))
    {
        return "stop";
    }
    if i64::try_from(completion_tokens.len()).is_ok_and(|n| n >= max_tokens) {
        return "length";
    }
    "stop"
}

/// Tokens to decode for `message.content`. EOS terminates generation: cut at
/// the first occurrence, not only the trailing one (a speculative chunk could
/// carry EOS mid-sequence).
pub fn content_token_ids(tokens: &[TokenId], eos_token_ids: &[TokenId]) -> Vec<TokenId> {
    match tokens.iter().position(|t| eos_token_ids.contains(t)) {
        Some(i) => tokens[..i].to_vec(),
        None => tokens.to_vec(),
    }
}

pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

#[allow(clippy::too_many_arguments)]
pub fn non_stream_json(
    model_id: &str,
    content: &str,
    finish_reason: &str,
    prompt_tokens: i64,
    completion_tokens: i64,
    tool_calls: &[ToolCallSpec],
    id: &str,
    created: i64,
) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": if tool_calls.is_empty() || !content.is_empty() {
            Value::String(content.to_owned())
        } else {
            Value::Null
        },
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(
            tool_calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    })
                })
                .collect(),
        );
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
}

pub(crate) fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

// MARK: JSON helpers (lenient coercion, like the NSNumber-based originals)

fn int_value(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn uint64_value(v: Option<&Value>) -> Option<u64> {
    match v? {
        Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn double_value(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn bool_value(v: Option<&Value>) -> Option<bool> {
    match v? {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<ChatCompletionRequest, ParseError> {
        parse_request(s.as_bytes())
    }

    #[test]
    fn ignores_unknown_request_fields() {
        let req = parse(
            r#"{"model":"maple","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"ls","parameters":{"type":"object"}}}],
            "tool_choice":"auto","structured_outputs":{"json":{"type":"object"}},
            "response_format":{"type":"json_object"},"user":"u","metadata":{"k":1},
            "max_tokens":16}"#,
        )
        .unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content, "hi");
        assert_eq!(req.max_tokens, 16);
        assert_eq!(req.model.as_deref(), Some("maple"));
        assert_eq!(req.tools_lines.len(), 1);
        match req.structured_constraint {
            Some(Constraint::JsonSchema(s)) => assert!(s.contains("object")),
            other => panic!("expected jsonSchema constraint, got {other:?}"),
        }
    }

    #[test]
    fn parses_guided_grammar_literal_union() {
        let req = parse(r#"{"messages":[{"role":"user","content":"x"}],"guided_grammar":"root ::= \"ZQX\"","max_tokens":8}"#).unwrap();
        assert_eq!(
            req.structured_constraint,
            Some(Constraint::Choice(vec!["ZQX".into()]))
        );
    }

    #[test]
    fn parses_stream_options_include_usage() {
        let req = parse(r#"{"messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}"#).unwrap();
        assert!(req.stream);
        assert!(req.stream_options_include_usage);
        let plain = parse(r#"{"messages":[{"role":"user","content":"x"}],"stream":true}"#).unwrap();
        assert!(!plain.stream_options_include_usage);
    }

    #[test]
    fn parses_structured_outputs_choice() {
        let req = parse(r#"{"messages":[{"role":"user","content":"x"}],"structured_outputs":{"choice":["red","green","blue"]}}"#).unwrap();
        assert_eq!(
            req.structured_constraint,
            Some(Constraint::Choice(vec![
                "red".into(),
                "green".into(),
                "blue".into()
            ]))
        );
    }

    #[test]
    fn top_level_grammar_ignored() {
        let req =
            parse(r#"{"messages":[{"role":"user","content":"x"}],"grammar":"root ::= \"ZQX\""}"#)
                .unwrap();
        assert_eq!(req.structured_constraint, None);
    }

    #[test]
    fn null_content_is_empty_string() {
        let req = parse(r#"{"messages":[{"role":"assistant","content":null}]}"#).unwrap();
        assert_eq!(req.messages[0].content, "");
    }

    #[test]
    fn developer_role_maps_to_system() {
        let req = parse(
            r#"{"messages":[{"role":"developer","content":"sys"},{"role":"user","content":"u"}]}"#,
        )
        .unwrap();
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].role, "user");
    }

    #[test]
    fn multimodal_content_parts_flatten_to_text() {
        let req = parse(r#"{"messages":[{"role":"user","content":[{"type":"text","text":"<tools>"},{"type":"text","text":"</tools>"}]}]}"#).unwrap();
        assert_eq!(req.messages[0].content, "<tools></tools>");
    }

    #[test]
    fn max_completion_tokens_preferred() {
        let req = parse(r#"{"messages":[{"role":"user","content":"x"}],"max_tokens":8,"max_completion_tokens":32}"#).unwrap();
        assert_eq!(req.max_tokens, 32);
        let req = parse(r#"{"messages":[{"role":"user","content":"x"}]}"#).unwrap();
        assert_eq!(req.max_tokens, 256);
    }

    #[test]
    fn markup_roundtrip_preserved_in_messages() {
        let system = "You have tools.\n<tools>\n[{\"name\":\"ls\"}]\n</tools>";
        let assistant = r#"<tool_call>{"name":"ls","arguments":{"path":"."}}</tool_call>"#;
        let user = "<tool_response>\nok\n</tool_response>";
        let body = json!({"messages": [
            {"role": "system", "content": system},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": user},
        ], "max_tokens": 64});
        let req = parse_request(body.to_string().as_bytes()).unwrap();
        assert_eq!(req.messages[0].content, system);
        assert_eq!(req.messages[1].content, assistant);
        assert_eq!(req.messages[2].content, user);
    }

    #[test]
    fn tool_calls_are_parsed_with_object_arguments_compacted() {
        let req = parse(
            r#"{"messages":[{"role":"assistant","content":null,"tool_calls":[
            {"id":"c1","type":"function","function":{"name":"ls","arguments":{"path":"."}}},
            {"id":"c2","type":"function","function":{"name":"cat","arguments":"{\"p\": 1}"}},
            "junk"]}]}"#,
        )
        .unwrap();
        let calls = &req.messages[0].tool_calls;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], ToolCallSpec::new("c1", "ls", r#"{"path":"."}"#));
        assert_eq!(calls[1].arguments, r#"{"p": 1}"#);
    }

    #[test]
    fn parse_errors() {
        assert_eq!(parse("not json"), Err(ParseError::InvalidJson));
        assert_eq!(parse("[1]"), Err(ParseError::InvalidJson));
        assert_eq!(parse("{}"), Err(ParseError::MissingMessages));
        assert_eq!(parse(r#"{"messages":[]}"#), Err(ParseError::EmptyMessages));
        assert_eq!(
            parse(r#"{"messages":[1]}"#),
            Err(ParseError::InvalidMessage(
                "messages[0] must be an object".into()
            ))
        );
        assert_eq!(
            parse(r#"{"messages":[{"content":"x"}]}"#),
            Err(ParseError::InvalidMessage(
                "messages[0] missing role".into()
            ))
        );
    }

    #[test]
    fn numeric_fields_accept_strings_and_floats() {
        let req = parse(r#"{"messages":[{"role":"user","content":"x"}],"temperature":"0.5","max_tokens":12.0,"seed":"7","n":2}"#).unwrap();
        assert_eq!(req.temperature, 0.5);
        assert_eq!(req.max_tokens, 12);
        assert_eq!(req.seed, Some(7));
        assert_eq!(req.n, 2);
    }

    #[test]
    fn finish_reason_length_when_max_filled() {
        let tokens = vec![1; 16];
        assert_eq!(finish_reason(&tokens, 16, &DEFAULT_EOS_TOKEN_IDS), "length");
    }

    #[test]
    fn finish_reason_stop_on_eos() {
        let tokens = [10, 11, 151_645];
        assert_eq!(finish_reason(&tokens, 64, &DEFAULT_EOS_TOKEN_IDS), "stop");
        assert_eq!(
            content_token_ids(&tokens, &DEFAULT_EOS_TOKEN_IDS),
            vec![10, 11]
        );
    }

    #[test]
    fn content_token_ids_cuts_at_first_eos() {
        let tokens = [1, 151_645, 2, 3];
        assert_eq!(content_token_ids(&tokens, &DEFAULT_EOS_TOKEN_IDS), vec![1]);
    }

    #[test]
    fn non_stream_response_shape() {
        let obj = non_stream_json(
            "maple-preview",
            "pong",
            "stop",
            3,
            1,
            &[],
            "chatcmpl-test",
            1,
        );
        assert_eq!(obj["object"], "chat.completion");
        assert_eq!(obj["model"], "maple-preview");
        let msg = &obj["choices"][0]["message"];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "pong");
        assert_eq!(obj["choices"][0]["finish_reason"], "stop");
        assert_eq!(obj["usage"]["total_tokens"], 4);
    }

    #[test]
    fn non_stream_response_with_tool_calls_has_null_content() {
        let calls = [ToolCallSpec::new("call_1", "ls", "{}")];
        let obj = non_stream_json("m", "", "tool_calls", 1, 1, &calls, "id", 1);
        let msg = &obj["choices"][0]["message"];
        assert!(msg["content"].is_null());
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "ls");
        assert_eq!(msg["tool_calls"][0]["type"], "function");
        let with_text = non_stream_json("m", "hi", "tool_calls", 1, 1, &calls, "id", 1);
        assert_eq!(with_text["choices"][0]["message"]["content"], "hi");
    }
}
