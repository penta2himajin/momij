//! HTTP backend for the official DeepGrove `llama.cpp` server
//! (`llama-server --jinja`). Replaces the Swift Seedless / MLX / oracle
//! backends: all model execution happens in that process.
//!
//! Endpoints used: `/health`, `/props`, `/tokenize`, `/detokenize`,
//! `/apply-template`, `/completion` (token-id prompts, SSE stream).

use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use momij_core::TokenId;
use momij_core::chat_template_patch;
use momij_core::openai_compat::ChatMessage;
use serde_json::{Value, json};

use crate::backend::{Backend, BackendError, Chunk, GenerateRequest, Grammar};

#[derive(Debug, Clone)]
pub struct LlamaServerBackend {
    client: reqwest::Client,
    base_url: String,
    n_ctx: Option<usize>,
    thinking_enabled: bool,
}

fn http_err(e: &reqwest::Error) -> BackendError {
    BackendError::Http(e.to_string())
}

impl LlamaServerBackend {
    /// Connect, check `/health` and read the context size from `/props`.
    pub async fn connect(base_url: &str, thinking_enabled: bool) -> Result<Self, BackendError> {
        let mut backend = Self::new(base_url, thinking_enabled);
        let health = backend
            .client
            .get(backend.url("/health"))
            .send()
            .await
            .map_err(|e| http_err(&e))?;
        if !health.status().is_success() {
            return Err(BackendError::Status {
                code: health.status().as_u16(),
                body: health.text().await.unwrap_or_default(),
            });
        }
        let props: Value = backend.get_json("/props").await?;
        backend.n_ctx = props
            .pointer("/default_generation_settings/n_ctx")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok());
        Ok(backend)
    }

    /// No network I/O; `n_ctx` unknown until [`LlamaServerBackend::connect`].
    pub fn new(base_url: &str, thinking_enabled: bool) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_owned(),
            n_ctx: None,
            thinking_enabled,
        }
    }

    pub fn n_ctx(&self) -> Option<usize> {
        self.n_ctx
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    async fn get_json(&self, path: &str) -> Result<Value, BackendError> {
        let resp = self
            .client
            .get(self.url(path))
            .send()
            .await
            .map_err(|e| http_err(&e))?;
        Self::read_json(resp).await
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value, BackendError> {
        let resp = self
            .client
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(|e| http_err(&e))?;
        Self::read_json(resp).await
    }

    async fn read_json(resp: reqwest::Response) -> Result<Value, BackendError> {
        let status = resp.status();
        if !status.is_success() {
            return Err(BackendError::Status {
                code: status.as_u16(),
                body: resp.text().await.unwrap_or_default(),
            });
        }
        resp.json::<Value>()
            .await
            .map_err(|e| BackendError::Protocol(format!("invalid JSON body: {e}")))
    }
}

/// `/completion` request body for `request`.
///
/// llama-server applies its own sampler defaults (`top_k 40`, `top_p 0.95`,
/// `min_p 0.05`); they are overridden explicitly so the OpenAI request fields
/// mean what they say (`top_p: 1` is really 1).
pub fn completion_body(request: &GenerateRequest) -> Value {
    let o = &request.options;
    let mut body = json!({
        "prompt": request.prompt,
        "n_predict": o.max_tokens,
        "stream": true,
        "return_tokens": true,
        "cache_prompt": true,
        "temperature": o.temperature,
        "top_p": o.top_p,
        "top_k": 0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": o.repetition_penalty,
        "presence_penalty": o.presence_penalty,
        "frequency_penalty": o.frequency_penalty,
        "seed": o.seed.map_or(Value::from(-1), Value::from),
    });
    if !o.banned_token_ids.is_empty() {
        body["logit_bias"] = Value::Array(
            o.banned_token_ids
                .iter()
                .map(|id| json!([id, false]))
                .collect(),
        );
    }
    match &request.grammar {
        Some(Grammar::Gbnf(g)) => body["grammar"] = Value::String(g.clone()),
        Some(Grammar::JsonSchema(schema)) => body["json_schema"] = schema.clone(),
        None => {}
    }
    body
}

/// One decoded SSE `data:` payload from `/completion`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamEvent {
    pub chunk: Chunk,
    pub stop: bool,
}

/// Parse one `/completion` stream payload. An `error` object is an `Err`.
pub fn parse_stream_event(data: &str) -> Result<StreamEvent, BackendError> {
    let v: Value = serde_json::from_str(data)
        .map_err(|e| BackendError::Protocol(format!("bad stream event: {e}")))?;
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| err.to_string(), str::to_owned);
        return Err(BackendError::Protocol(format!("server error: {msg}")));
    }
    let tokens = v
        .get("tokens")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_i64)
                .filter_map(|t| TokenId::try_from(t).ok())
                .collect()
        })
        .unwrap_or_default();
    Ok(StreamEvent {
        chunk: Chunk {
            tokens,
            text: v
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
        },
        stop: v.get("stop").and_then(Value::as_bool).unwrap_or(false),
    })
}

/// Incremental SSE line splitter: feed bytes, get complete `data:` payloads.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if let Some(data) = line.strip_prefix("data:") {
                out.push(data.trim_start().to_owned());
            }
        }
        out
    }
}

/// Server-side timings of one non-stream completion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timings {
    pub prompt_tokens: usize,
    pub prompt_tps: f64,
    pub predicted_tokens: usize,
    pub predicted_tps: f64,
}

/// `/completion` body for a benchmark trial: exactly `n_predict` tokens
/// (EOS ignored, like `mlx_lm.benchmark`), prompt cache off so every trial
/// pays the prefill, greedy.
pub fn bench_body(prompt: &[TokenId], n_predict: usize) -> Value {
    json!({
        "prompt": prompt,
        "n_predict": n_predict,
        "ignore_eos": true,
        "cache_prompt": false,
        "temperature": 0,
        "top_k": 0,
        "min_p": 0.0,
        "stream": false,
    })
}

pub fn parse_timings(v: &Value) -> Result<Timings, BackendError> {
    let t = v
        .get("timings")
        .ok_or_else(|| BackendError::Protocol("response has no `timings`".into()))?;
    let num = |k: &str| {
        t.get(k)
            .and_then(Value::as_f64)
            .ok_or_else(|| BackendError::Protocol(format!("timings.{k} missing")))
    };
    Ok(Timings {
        prompt_tokens: num("prompt_n")? as usize,
        prompt_tps: num("prompt_per_second")?,
        predicted_tokens: num("predicted_n")? as usize,
        predicted_tps: num("predicted_per_second")?,
    })
}

impl LlamaServerBackend {
    /// One benchmark trial (see [`bench_body`]).
    pub async fn bench_once(
        &self,
        prompt: &[TokenId],
        n_predict: usize,
    ) -> Result<Timings, BackendError> {
        let v = self
            .post_json("/completion", &bench_body(prompt, n_predict))
            .await?;
        parse_timings(&v)
    }
}

#[async_trait]
impl Backend for LlamaServerBackend {
    async fn tokenize(
        &self,
        text: &str,
        parse_special: bool,
    ) -> Result<Vec<TokenId>, BackendError> {
        let v = self
            .post_json(
                "/tokenize",
                &json!({"content": text, "add_special": false, "parse_special": parse_special}),
            )
            .await?;
        ids_from(&v, "tokens")
    }

    async fn detokenize(&self, ids: &[TokenId]) -> Result<String, BackendError> {
        let v = self
            .post_json("/detokenize", &json!({"tokens": ids}))
            .await?;
        v.get("content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| BackendError::Protocol("/detokenize: missing content".into()))
    }

    async fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<TokenId>, BackendError> {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();
        let v = self
            .post_json("/apply-template", &json!({"messages": msgs}))
            .await?;
        let mut prompt = v
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| BackendError::Protocol("/apply-template: missing prompt".into()))?
            .to_owned();
        if !self.thinking_enabled {
            prompt = chat_template_patch::without_forced_think_prompt(&prompt);
        }
        self.tokenize(&prompt, true).await
    }

    fn generate(
        &self,
        request: GenerateRequest,
    ) -> BoxStream<'static, Result<Chunk, BackendError>> {
        let client = self.client.clone();
        let url = self.url("/completion");
        let body = completion_body(&request);
        async_stream::try_stream! {
            let resp = client.post(url).json(&body).send().await.map_err(|e| http_err(&e))?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                Err(BackendError::Status { code: status.as_u16(), body })?;
                return;
            }
            let mut bytes = resp.bytes_stream();
            let mut sse = SseDecoder::default();
            'outer: while let Some(piece) = bytes.next().await {
                let piece = piece.map_err(|e| http_err(&e))?;
                for data in sse.push(&piece) {
                    let event = parse_stream_event(&data)?;
                    if !event.chunk.tokens.is_empty() || !event.chunk.text.is_empty() {
                        yield event.chunk;
                    }
                    if event.stop {
                        break 'outer;
                    }
                }
            }
        }
        .boxed()
    }

    fn max_prompt_tokens(&self) -> Option<usize> {
        self.n_ctx
    }
}

fn ids_from(v: &Value, key: &str) -> Result<Vec<TokenId>, BackendError> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| BackendError::Protocol(format!("missing `{key}`")))?
        .iter()
        .map(|t| {
            t.as_i64()
                .and_then(|n| TokenId::try_from(n).ok())
                .ok_or_else(|| BackendError::Protocol(format!("non-integer token in `{key}`")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use momij_core::generate_options::GenerateOptions;

    use super::*;
    use crate::backend::collect;

    fn request(grammar: Option<Grammar>, options: GenerateOptions) -> GenerateRequest {
        GenerateRequest {
            prompt: vec![1, 2, 3],
            options,
            grammar,
        }
    }

    #[test]
    fn body_overrides_server_sampler_defaults() {
        let body = completion_body(&request(
            None,
            GenerateOptions {
                max_tokens: 64,
                ..GenerateOptions::default()
            },
        ));
        assert_eq!(body["prompt"], json!([1, 2, 3]));
        assert_eq!(body["n_predict"], 64);
        assert_eq!(body["stream"], true);
        assert_eq!(body["return_tokens"], true);
        // The three defaults that would otherwise silently change sampling:
        assert_eq!(body["top_k"], 0);
        assert_eq!(body["min_p"], 0.0);
        assert_eq!(body["top_p"], 1.0);
        assert_eq!(body["seed"], -1);
        assert!(body.get("logit_bias").is_none());
        assert!(body.get("grammar").is_none() && body.get("json_schema").is_none());
    }

    #[test]
    fn body_carries_sampling_seed_bans_and_grammar() {
        let opts = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
            presence_penalty: 0.2,
            frequency_penalty: 0.3,
            seed: Some(42),
            banned_token_ids: vec![151_667, 151_644],
            ..GenerateOptions::default()
        };
        let body = completion_body(&request(Some(Grammar::Gbnf("root ::= \"a\"".into())), opts));
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["repeat_penalty"], 1.1);
        assert_eq!(body["presence_penalty"], 0.2);
        assert_eq!(body["frequency_penalty"], 0.3);
        assert_eq!(body["seed"], 42);
        assert_eq!(
            body["logit_bias"],
            json!([[151_667, false], [151_644, false]])
        );
        assert_eq!(body["grammar"], "root ::= \"a\"");
        let schema = completion_body(&request(
            Some(Grammar::JsonSchema(json!({"type": "object"}))),
            GenerateOptions::default(),
        ));
        assert_eq!(schema["json_schema"], json!({"type": "object"}));
    }

    #[test]
    fn stream_event_parsing() {
        let e = parse_stream_event(r#"{"index":0,"content":"Hi","tokens":[9707],"stop":false}"#)
            .unwrap();
        assert_eq!(
            e.chunk,
            Chunk {
                tokens: vec![9707],
                text: "Hi".into()
            }
        );
        assert!(!e.stop);
        let eos = parse_stream_event(r#"{"content":"","tokens":[151645],"stop":false}"#).unwrap();
        assert_eq!(eos.chunk.tokens, [151_645]);
        let last =
            parse_stream_event(r#"{"content":"","tokens":[],"stop":true,"stop_type":"eos"}"#)
                .unwrap();
        assert!(last.stop && last.chunk.tokens.is_empty());
        assert!(parse_stream_event("not json").is_err());
        let err = parse_stream_event(r#"{"error":{"code":500,"message":"boom"}}"#).unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[test]
    fn sse_decoder_handles_split_and_multiple_events() {
        let mut d = SseDecoder::default();
        assert!(d.push(b"data: {\"a\":").is_empty());
        let out = d.push(b"1}\n\ndata: {\"b\":2}\r\n\n: comment\n");
        assert_eq!(out, ["{\"a\":1}", "{\"b\":2}"]);
        // Multi-byte characters split across reads survive.
        let bytes = "data: {\"c\":\"日\"}\n\n".as_bytes();
        let mut d = SseDecoder::default();
        assert!(d.push(&bytes[..12]).is_empty());
        assert_eq!(d.push(&bytes[12..]), ["{\"c\":\"日\"}"]);
    }

    #[test]
    fn bench_body_pins_the_generation_length() {
        let b = bench_body(&[1, 2], 32);
        assert_eq!(
            (
                b["n_predict"].clone(),
                b["ignore_eos"].clone(),
                b["cache_prompt"].clone()
            ),
            (json!(32), json!(true), json!(false))
        );
        assert_eq!(b["stream"], false);
    }

    #[test]
    fn timings_parse_and_report_missing_fields() {
        let ok = json!({"timings": {"prompt_n": 64, "prompt_per_second": 137.7, "predicted_n": 8, "predicted_per_second": 35.9}});
        assert_eq!(
            parse_timings(&ok).unwrap(),
            Timings {
                prompt_tokens: 64,
                prompt_tps: 137.7,
                predicted_tokens: 8,
                predicted_tps: 35.9
            }
        );
        assert!(parse_timings(&json!({})).is_err());
        assert!(
            parse_timings(&json!({"timings": {"prompt_n": 1}}))
                .unwrap_err()
                .to_string()
                .contains("prompt_per_second")
        );
    }

    // MARK: fake llama-server (exercises the real HTTP client)

    mod fake {
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        use axum::routing::{get, post};
        use axum::{Json, Router};
        use serde_json::{Value, json};
        use std::sync::{Arc, Mutex};

        #[derive(Default)]
        pub struct Seen {
            pub completion: Mutex<Option<Value>>,
        }

        pub async fn spawn(seen: Arc<Seen>) -> String {
            let app = Router::new()
                .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
                .route("/props", get(|| async { Json(json!({"default_generation_settings": {"n_ctx": 8192}})) }))
                .route(
                    "/tokenize",
                    post(|Json(b): Json<Value>| async move {
                        // Byte-level fake tokenizer: one token per byte.
                        let text = b["content"].as_str().unwrap_or("");
                        let toks: Vec<u32> = text.bytes().map(u32::from).collect();
                        Json(json!({"tokens": toks}))
                    }),
                )
                .route(
                    "/detokenize",
                    post(|Json(b): Json<Value>| async move {
                        let s: String = b["tokens"].as_array().unwrap().iter().filter_map(|t| u8::try_from(t.as_u64()?).ok()).map(char::from).collect();
                        Json(json!({"content": s}))
                    }),
                )
                .route(
                    "/apply-template",
                    post(|Json(b): Json<Value>| async move {
                        let mut p = String::new();
                        for m in b["messages"].as_array().unwrap() {
                            p += &format!("<|im_start|>{}\n{}<|im_end|>\n", m["role"].as_str().unwrap(), m["content"].as_str().unwrap());
                        }
                        p += "<|im_start|>assistant\n<think>\n";
                        Json(json!({"prompt": p}))
                    }),
                )
                .route(
                    "/completion",
                    post(|State(seen): State<Arc<Seen>>, Json(b): Json<Value>| async move {
                        *seen.completion.lock().unwrap() = Some(b.clone());
                        if b["n_predict"] == 7 {
                            return (StatusCode::INTERNAL_SERVER_ERROR, "kaboom").into_response();
                        }
                        // Split mid-event on purpose.
                        let body = "data: {\"content\":\"Hel\",\"tokens\":[72,101,108],\"stop\":false}\n\n\
                                    data: {\"content\":\"lo\",\"tokens\":[108,111],\"stop\":false}\n\n\
                                    data: {\"content\":\"\",\"tokens\":[151645],\"stop\":false}\n\n\
                                    data: {\"content\":\"\",\"tokens\":[],\"stop\":true,\"stop_type\":\"eos\"}\n\n\
                                    data: {\"content\":\"never\",\"tokens\":[1],\"stop\":false}\n\n";
                        ([("content-type", "text/event-stream")], body).into_response()
                    }),
                )
                .with_state(seen);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            format!("http://{addr}")
        }
    }

    #[tokio::test]
    async fn talks_to_a_llama_server() {
        let seen = std::sync::Arc::new(fake::Seen::default());
        let url = fake::spawn(seen.clone()).await;
        let backend = LlamaServerBackend::connect(&format!("{url}/"), false)
            .await
            .unwrap();
        assert_eq!(backend.max_prompt_tokens(), Some(8192));

        assert_eq!(backend.tokenize("ab", false).await.unwrap(), [97, 98]);
        assert_eq!(backend.detokenize(&[104, 105]).await.unwrap(), "hi");

        // Thinking disabled: the forced `<think>` opener is stripped before tokenizing.
        let ids = backend
            .apply_chat_template(&[ChatMessage::new("user", "q")])
            .await
            .unwrap();
        let rendered = backend.detokenize(&ids).await.unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
        );

        let done = collect(backend.generate(request(None, GenerateOptions::default())))
            .await
            .unwrap();
        assert_eq!(
            done.tokens,
            [72, 101, 108, 108, 111, 151_645],
            "stops at the stop event"
        );
        assert_eq!(done.text, "Hello");
        let sent = seen.completion.lock().unwrap().clone().unwrap();
        assert_eq!(sent["prompt"], json!([1, 2, 3]));
        assert_eq!(sent["top_k"], 0);
    }

    #[tokio::test]
    async fn thinking_enabled_keeps_the_stock_generation_prompt() {
        let seen = std::sync::Arc::new(fake::Seen::default());
        let url = fake::spawn(seen).await;
        let on = LlamaServerBackend::connect(&url, true).await.unwrap();
        let off = LlamaServerBackend::connect(&url, false).await.unwrap();
        let msgs = [ChatMessage::new("user", "q")];
        let with_think = on.apply_chat_template(&msgs).await.unwrap();
        let without = off.apply_chat_template(&msgs).await.unwrap();
        assert_eq!(
            with_think.len(),
            without.len() + "<think>\n".len(),
            "byte-level fake tokenizer"
        );
        assert!(
            on.detokenize(&with_think)
                .await
                .unwrap()
                .ends_with("<think>\n")
        );
        assert!(!off.detokenize(&without).await.unwrap().contains("<think>"));
    }

    #[tokio::test]
    async fn http_errors_surface_as_status_errors() {
        let seen = std::sync::Arc::new(fake::Seen::default());
        let url = fake::spawn(seen).await;
        let backend = LlamaServerBackend::connect(&url, false).await.unwrap();
        let opts = GenerateOptions {
            max_tokens: 7,
            ..GenerateOptions::default()
        };
        let err = collect(backend.generate(request(None, opts)))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, BackendError::Status { code: 500, body } if body == "kaboom"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn connect_fails_cleanly_when_nothing_listens() {
        let err = LlamaServerBackend::connect("http://127.0.0.1:1", false)
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::Http(_)), "{err}");
    }

    /// Live check against a real llama-server, e.g.
    /// `MOMIJ_LLAMA_URL=http://127.0.0.1:8080 cargo test -p momij -- --ignored`.
    #[tokio::test]
    #[ignore = "needs a running llama-server (set MOMIJ_LLAMA_URL)"]
    async fn live_llama_server_roundtrip() {
        let url = std::env::var("MOMIJ_LLAMA_URL").expect("MOMIJ_LLAMA_URL");
        let backend = LlamaServerBackend::connect(&url, false).await.unwrap();
        assert_eq!(backend.tokenize("<think>", true).await.unwrap(), [151_667]);
        assert_eq!(
            backend.tokenize("<|im_end|>", true).await.unwrap(),
            [151_645]
        );
        let prompt = backend
            .apply_chat_template(&[ChatMessage::new("user", "Reply with exactly: OK")])
            .await
            .unwrap();
        let rendered = backend.detokenize(&prompt).await.unwrap();
        assert!(
            rendered.ends_with("<|im_start|>assistant\n"),
            "{rendered:?}"
        );
        let opts = GenerateOptions {
            max_tokens: 64,
            banned_token_ids: vec![151_667],
            ..GenerateOptions::default()
        };
        let done = collect(backend.generate(GenerateRequest {
            prompt,
            options: opts,
            grammar: None,
        }))
        .await
        .unwrap();
        assert!(!done.tokens.is_empty());
        assert!(
            !done.tokens.contains(&151_667),
            "banned token must not appear"
        );
    }
}
