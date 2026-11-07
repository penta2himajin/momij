//! Model-execution seam. The Swift `LLMBackend` / `TokenizerAdapter` pair
//! becomes one trait here: everything the serve pipeline needs from "the
//! model side" (tokenizer, chat template, token generation).

use std::fmt;

use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use momij_core::TokenId;
use momij_core::generate_options::GenerateOptions;
use momij_core::openai_compat::ChatMessage;
use momij_core::structured_outputs::Constraint;
use serde_json::Value;

#[derive(Debug)]
pub enum BackendError {
    /// Transport failure (connect, read, timeout).
    Http(String),
    /// Non-2xx answer from the model server.
    Status { code: u16, body: String },
    /// The server answered, but not in the shape we expect.
    Protocol(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(e) => write!(f, "backend request failed: {e}"),
            Self::Status { code, body } => write!(f, "backend returned HTTP {code}: {body}"),
            Self::Protocol(e) => write!(f, "backend protocol error: {e}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Server-side constrained decoding (llama.cpp samplers).
#[derive(Debug, Clone, PartialEq)]
pub enum Grammar {
    /// GBNF text (`root ::= ...`).
    Gbnf(String),
    /// JSON Schema object.
    JsonSchema(Value),
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: Vec<TokenId>,
    pub options: GenerateOptions,
    pub grammar: Option<Grammar>,
}

/// One streamed increment: the new token ids and their text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Chunk {
    pub tokens: Vec<TokenId>,
    pub text: String,
}

/// A fully collected generation. `tokens` ends with the EOS token when the
/// model stopped on its own (llama-server includes it), like the Swift
/// backends.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Completion {
    pub tokens: Vec<TokenId>,
    pub text: String,
}

pub async fn collect(
    mut stream: BoxStream<'static, Result<Chunk, BackendError>>,
) -> Result<Completion, BackendError> {
    let mut out = Completion::default();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        out.tokens.extend(chunk.tokens);
        out.text.push_str(&chunk.text);
    }
    Ok(out)
}

#[async_trait]
pub trait Backend: Send + Sync {
    async fn tokenize(&self, text: &str, parse_special: bool)
    -> Result<Vec<TokenId>, BackendError>;
    async fn detokenize(&self, ids: &[TokenId]) -> Result<String, BackendError>;
    /// Render `messages` with the model's chat template (generation prompt
    /// included) and tokenize the result.
    async fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<TokenId>, BackendError>;
    fn generate(&self, request: GenerateRequest)
    -> BoxStream<'static, Result<Chunk, BackendError>>;
    /// Prompt capacity for pre-flight rejection (OpenAI-style 400 instead of
    /// a mid-stream abort). `None` = no known hard limit.
    fn max_prompt_tokens(&self) -> Option<usize> {
        None
    }
}

/// GBNF string literal for `s` (quotes, backslashes and control characters
/// escaped).
pub fn gbnf_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02X}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Map a request constraint to a llama.cpp sampler grammar.
///
/// `Err` carries the message for an OpenAI-style 400.
pub fn grammar_for(constraint: &Constraint) -> Result<Grammar, String> {
    match constraint {
        Constraint::Choice(options) => {
            let alts: Vec<String> = options.iter().map(|o| gbnf_literal(o)).collect();
            Ok(Grammar::Gbnf(format!("root ::= {}", alts.join(" | "))))
        }
        Constraint::JsonSchema(text) => serde_json::from_str::<Value>(text)
            .map(Grammar::JsonSchema)
            .map_err(|e| format!("structured_outputs.json is not valid JSON Schema text: {e}")),
        Constraint::Ebnf(text) => Ok(Grammar::Gbnf(text.clone())),
        Constraint::Regex(_) => Err(
            "structured output not supported yet: regex (only literal unions, JSON schema and grammar are supported by the llama.cpp backend)"
                .to_owned(),
        ),
        Constraint::Unsupported(why) => Err(format!("structured output not supported yet: {why}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbnf_literal_escapes() {
        assert_eq!(gbnf_literal("red"), "\"red\"");
        assert_eq!(gbnf_literal("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
        assert_eq!(gbnf_literal("\u{1}"), "\"\\x01\"");
        assert_eq!(gbnf_literal("日本"), "\"日本\"");
    }

    #[test]
    fn choice_becomes_a_literal_union() {
        let g = grammar_for(&Constraint::Choice(vec!["red".into(), "g\"reen".into()])).unwrap();
        assert_eq!(g, Grammar::Gbnf("root ::= \"red\" | \"g\\\"reen\"".into()));
    }

    #[test]
    fn json_schema_text_is_parsed() {
        let g = grammar_for(&Constraint::JsonSchema(r#"{"type":"object"}"#.into())).unwrap();
        assert_eq!(
            g,
            Grammar::JsonSchema(serde_json::json!({"type": "object"}))
        );
        assert!(grammar_for(&Constraint::JsonSchema("{oops".into())).is_err());
    }

    #[test]
    fn ebnf_passes_through_and_regex_is_rejected() {
        assert_eq!(
            grammar_for(&Constraint::Ebnf("root ::= [a-z]+".into())).unwrap(),
            Grammar::Gbnf("root ::= [a-z]+".into())
        );
        assert!(
            grammar_for(&Constraint::Regex("a+".into()))
                .unwrap_err()
                .contains("regex")
        );
        assert!(
            grammar_for(&Constraint::Unsupported("x".into()))
                .unwrap_err()
                .contains('x')
        );
    }
}
