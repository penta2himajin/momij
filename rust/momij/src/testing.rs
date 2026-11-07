//! Scripted in-memory backend for pipeline and server tests.
//!
//! Byte-level tokenizer (one token per UTF-8 byte), `<|im_start|>role` chat
//! rendering, and generation that replays queued replies followed by EOS.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream::{self, BoxStream};
use momij_core::TokenId;
use momij_core::generate_options::GenerateOptions;
use momij_core::openai_compat::ChatMessage;

use crate::backend::{Backend, BackendError, Chunk, GenerateRequest, Grammar};

pub const EOS: TokenId = 151_645;

#[derive(Debug, Clone)]
pub struct SeenRequest {
    pub prompt_text: String,
    pub options: GenerateOptions,
    pub grammar: Option<Grammar>,
}

#[derive(Default)]
pub struct MockBackend {
    replies: Mutex<VecDeque<Result<String, String>>>,
    seen: Mutex<Vec<SeenRequest>>,
    pub max_prompt: Option<usize>,
}

impl MockBackend {
    pub fn new(replies: &[&str]) -> Self {
        let m = Self::default();
        for r in replies {
            m.replies.lock().unwrap().push_back(Ok((*r).to_owned()));
        }
        m
    }

    /// A backend whose context holds at most `max_prompt` prompt tokens.
    pub fn with_max_prompt(max_prompt: usize) -> Self {
        Self {
            max_prompt: Some(max_prompt),
            ..Self::default()
        }
    }

    pub fn push_error(&self, message: &str) {
        self.replies
            .lock()
            .unwrap()
            .push_back(Err(message.to_owned()));
    }

    pub fn seen(&self) -> Vec<SeenRequest> {
        self.seen.lock().unwrap().clone()
    }
}

fn bytes_to_ids(s: &str) -> Vec<TokenId> {
    s.bytes().map(TokenId::from).collect()
}

fn ids_to_text(ids: &[TokenId]) -> String {
    let bytes: Vec<u8> = ids.iter().filter_map(|&t| u8::try_from(t).ok()).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[async_trait]
impl Backend for MockBackend {
    async fn tokenize(
        &self,
        text: &str,
        _parse_special: bool,
    ) -> Result<Vec<TokenId>, BackendError> {
        Ok(bytes_to_ids(text))
    }

    async fn detokenize(&self, ids: &[TokenId]) -> Result<String, BackendError> {
        Ok(ids_to_text(ids))
    }

    async fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<TokenId>, BackendError> {
        let mut p = String::new();
        for m in messages {
            p += &format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content);
        }
        p += "<|im_start|>assistant\n";
        Ok(bytes_to_ids(&p))
    }

    fn generate(
        &self,
        request: GenerateRequest,
    ) -> BoxStream<'static, Result<Chunk, BackendError>> {
        self.seen.lock().unwrap().push(SeenRequest {
            prompt_text: ids_to_text(&request.prompt),
            options: request.options.clone(),
            grammar: request.grammar.clone(),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok(String::new()));
        let text = match reply {
            Ok(t) => t,
            Err(e) => return stream::once(async move { Err(BackendError::Http(e)) }).boxed(),
        };
        // One chunk per character; EOS unless the token budget cuts it short.
        let mut budget = request.options.max_tokens;
        let mut truncated = false;
        let mut items: Vec<Chunk> = Vec::new();
        for c in text.chars() {
            let mut tokens = bytes_to_ids(&c.to_string());
            if budget == 0 {
                truncated = true;
                break;
            }
            if tokens.len() > budget {
                tokens.truncate(budget);
                truncated = true;
            }
            budget -= tokens.len();
            items.push(Chunk {
                tokens,
                text: c.to_string(),
            });
            if truncated {
                break;
            }
        }
        if !truncated && budget > 0 {
            items.push(Chunk {
                tokens: vec![EOS],
                text: String::new(),
            });
        }
        stream::iter(items.into_iter().map(Ok)).boxed()
    }

    fn max_prompt_tokens(&self) -> Option<usize> {
        self.max_prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::collect;

    fn req(max_tokens: usize) -> GenerateRequest {
        GenerateRequest {
            prompt: vec![1],
            options: GenerateOptions {
                max_tokens,
                ..GenerateOptions::default()
            },
            grammar: None,
        }
    }

    #[tokio::test]
    async fn replays_replies_with_eos_and_records_requests() {
        let m = MockBackend::new(&["hi"]);
        let done = collect(m.generate(req(10))).await.unwrap();
        assert_eq!(done.tokens, [104, 105, EOS]);
        assert_eq!(done.text, "hi");
        assert_eq!(m.seen().len(), 1);
        // Exhausted script -> empty reply (just EOS).
        assert_eq!(collect(m.generate(req(10))).await.unwrap().tokens, [EOS]);
    }

    #[tokio::test]
    async fn token_budget_truncates_without_eos() {
        let m = MockBackend::new(&["hello"]);
        let done = collect(m.generate(req(3))).await.unwrap();
        assert_eq!(done.tokens, [104, 101, 108]);
        assert_eq!(done.text, "hel");
        let exact = MockBackend::new(&["ab"]);
        assert_eq!(
            collect(exact.generate(req(2))).await.unwrap().tokens,
            [97, 98],
            "no room left for EOS"
        );
    }

    #[tokio::test]
    async fn errors_and_templates() {
        let m = MockBackend::new(&[]);
        m.push_error("down");
        assert!(collect(m.generate(req(5))).await.is_err());
        let ids = m
            .apply_chat_template(&[ChatMessage::new("user", "q")])
            .await
            .unwrap();
        assert_eq!(
            m.detokenize(&ids).await.unwrap(),
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
