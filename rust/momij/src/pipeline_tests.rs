//! End-to-end behaviour of the serve pipeline through the HTTP router, with a
//! scripted backend.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use momij_core::trace_store::TraceStore;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::backend::Grammar;
use crate::pipeline::{self, Engine, ServeConfig};
use crate::server;
use crate::testing::{EOS, MockBackend};

struct Harness {
    engine: Arc<Engine>,
    backend: Arc<MockBackend>,
    _tmp: tempfile::TempDir,
}

fn harness_with(backend: MockBackend, tweak: impl FnOnce(&mut ServeConfig)) -> Harness {
    let tmp = tempfile::tempdir().unwrap();
    let backend = Arc::new(backend);
    let mut cfg = ServeConfig::new("maple-test");
    tweak(&mut cfg);
    let engine = Engine::new(
        backend.clone(),
        cfg,
        TraceStore::new(tmp.path().join("traces")),
    );
    Harness {
        engine,
        backend,
        _tmp: tmp,
    }
}

fn harness(replies: &[&str]) -> Harness {
    harness_with(MockBackend::new(replies), |_| {})
}

async fn send(h: &Harness, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, String) {
    let mut req = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(b) => {
            req = req.header("content-type", "application/json");
            Body::from(b.to_string())
        }
        None => Body::empty(),
    };
    let resp = server::router(h.engine.clone())
        .oneshot(req.body(body).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn chat(h: &Harness, body: Value) -> (StatusCode, Value) {
    let (s, text) = send(h, "POST", "/v1/chat/completions", Some(body)).await;
    (s, serde_json::from_str(&text).unwrap_or(Value::Null))
}

/// SSE payloads (`data:` lines) of a streaming request.
async fn chat_stream(h: &Harness, mut body: Value) -> (StatusCode, Vec<String>) {
    body["stream"] = json!(true);
    let (s, text) = send(h, "POST", "/v1/chat/completions", Some(body)).await;
    let events = text
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .map(str::to_owned)
        .collect();
    (s, events)
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn call_block(name: &str, args: &str) -> String {
    format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "write", "description": "Write a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "ls", "description": "List", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    ])
}

fn msg(resp: &Value) -> &Value {
    &resp["choices"][0]["message"]
}

fn finish(resp: &Value) -> &str {
    resp["choices"][0]["finish_reason"].as_str().unwrap()
}

// MARK: plain chat

#[tokio::test]
async fn plain_chat_returns_content_usage_and_model() {
    let h = harness(&["Hello there"]);
    let (status, resp) = chat(&h, json!({"messages": [user("hi")], "max_tokens": 64})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["object"], "chat.completion");
    assert_eq!(resp["model"], "maple-test");
    assert_eq!(resp["id"], "chatcmpl-momij");
    assert_eq!(msg(&resp)["role"], "assistant");
    assert_eq!(msg(&resp)["content"], "Hello there");
    assert_eq!(finish(&resp), "stop");
    // 11 bytes + EOS
    assert_eq!(resp["usage"]["completion_tokens"], 12);
    let prompt_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(
        resp["usage"]["total_tokens"].as_u64().unwrap(),
        prompt_tokens + 12
    );
}

#[tokio::test]
async fn request_reaches_the_backend_as_a_chat_prompt_with_think_guards() {
    let h = harness(&["ok"]);
    let (_, _) = chat(&h, json!({"messages": [{"role": "system", "content": "SYS"}, user("hi")], "max_tokens": 32, "temperature": 0.5, "top_p": 0.9, "seed": 7})).await;
    let seen = &h.backend.seen()[0];
    assert!(
        seen.prompt_text
            .starts_with("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"),
        "{}",
        seen.prompt_text
    );
    assert!(seen.prompt_text.ends_with("<|im_start|>assistant\n"));
    let o = &seen.options;
    assert_eq!(
        (o.max_tokens, o.temperature, o.top_p, o.seed),
        (32, 0.5, 0.9, Some(7))
    );
    assert_eq!(o.eos_token_ids, [EOS]);
    assert_eq!(
        o.banned_token_ids,
        [151_667, 151_644],
        "<think> and <|im_start|> are banned by default"
    );
    assert!(seen.grammar.is_none());
}

#[tokio::test]
async fn thinking_enabled_lifts_the_token_ban() {
    let h = harness_with(MockBackend::new(&["ok"]), |c| c.thinking_enabled = true);
    chat(&h, json!({"messages": [user("hi")]})).await;
    assert!(h.backend.seen()[0].options.banned_token_ids.is_empty());
}

#[tokio::test]
async fn length_finish_when_the_budget_is_exhausted() {
    let h = harness(&["a long answer that does not fit"]);
    let (_, resp) = chat(&h, json!({"messages": [user("hi")], "max_tokens": 5})).await;
    assert_eq!(finish(&resp), "length");
    assert_eq!(msg(&resp)["content"], "a lon");
}

#[tokio::test]
async fn think_block_is_stripped_from_content() {
    let h = harness(&["<think>plan</think>\nthe answer"]);
    let (_, resp) = chat(&h, json!({"messages": [user("hi")]})).await;
    assert_eq!(msg(&resp)["content"], "the answer");
}

#[tokio::test]
async fn structured_choice_maps_to_a_server_side_grammar() {
    let h = harness(&["green"]);
    let (_, resp) = chat(
        &h,
        json!({"messages": [user("pick")], "structured_outputs": {"choice": ["red", "green"]}}),
    )
    .await;
    assert_eq!(msg(&resp)["content"], "green");
    assert_eq!(
        h.backend.seen()[0].grammar,
        Some(Grammar::Gbnf("root ::= \"red\" | \"green\"".into()))
    );
    assert!(
        !h.backend.seen()[0].options.use_suffix_spec,
        "grammar requests leave the suffix-spec path"
    );
}

// MARK: request errors

#[tokio::test]
async fn bad_requests_are_400_with_trace_ids() {
    let h = harness(&[]);
    let (s, text) = send(
        &h,
        "POST",
        "/v1/chat/completions",
        Some(json!("not an object")),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["error"]["type"], "invalid_request_error");
    assert_eq!(e["error"]["message"], "invalid JSON body");
    assert!(e["error"]["trace_id"].as_str().unwrap().starts_with("tr-"));
    assert_eq!(e["error"]["locus"], "upstream");

    let (s, e) = chat(&h, json!({"messages": []})).await;
    assert_eq!(
        (s, e["error"]["message"].as_str()),
        (StatusCode::BAD_REQUEST, Some("messages must be non-empty"))
    );
    let (s, e) = chat(&h, json!({"messages": [user("x")], "n": 2})).await;
    assert_eq!(
        (s, e["error"]["message"].as_str()),
        (StatusCode::BAD_REQUEST, Some("n>1 unsupported"))
    );
    let (s, e) = chat(
        &h,
        json!({"messages": [user("x")], "structured_outputs": {"regex": "a+"}}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(e["error"]["message"].as_str().unwrap().contains("regex"));
    assert!(
        h.backend.seen().is_empty(),
        "rejected requests never reach the model"
    );
}

#[tokio::test]
async fn prompt_longer_than_the_context_is_rejected_before_generation() {
    let h = harness_with(MockBackend::with_max_prompt(40), |_| {});
    let (s, e) = chat(&h, json!({"messages": [user(&"x".repeat(100))]})).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(
        e["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with("prompt too long"),
        "{e}"
    );
    assert!(h.backend.seen().is_empty());
}

#[tokio::test]
async fn backend_failure_is_a_500_generation_error() {
    let h = harness(&[]);
    h.backend.push_error("llama-server down");
    let (s, e) = chat(&h, json!({"messages": [user("hi")]})).await;
    assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        e["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with("generation failed:"),
        "{e}"
    );
}

// MARK: tools

#[tokio::test]
async fn tool_call_reply_becomes_tool_calls() {
    let reply = format!("Listing.\n{}", call_block("ls", r#"{"path": "."}"#));
    let h = harness(&[&reply]);
    let (_, resp) = chat(
        &h,
        json!({"messages": [user("list files")], "tools": tools()}),
    )
    .await;
    assert_eq!(finish(&resp), "tool_calls");
    let calls = msg(&resp)["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "ls");
    assert_eq!(calls[0]["function"]["arguments"], r#"{"path": "."}"#);
    assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(msg(&resp)["content"], "Listing.");
    let prompt = &h.backend.seen()[0].prompt_text;
    assert!(
        prompt.contains("# Tools"),
        "tools instruction is appended to the system turn"
    );
    assert!(prompt.contains("<tool_call>"));
}

#[tokio::test]
async fn tool_history_is_rewritten_into_markup() {
    let h = harness(&["done"]);
    let body = json!({"messages": [
        user("list"),
        {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{\"path\": \".\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ], "tools": tools()});
    chat(&h, body).await;
    let prompt = &h.backend.seen()[0].prompt_text;
    assert!(
        prompt.contains("<tool_result>\na.txt\n</tool_result>"),
        "{prompt}"
    );
    assert!(
        prompt.contains(r#"{"name": "ls", "arguments": {"path": "."}}"#),
        "well-formed call block: {prompt}"
    );
}

#[tokio::test]
async fn protocol_tool_is_absorbed_into_a_clean_stop() {
    let reply = call_block("submit", r#"{"description": "All done, wrote notes.md."}"#);
    let h = harness(&[&reply]);
    let (_, resp) = chat(&h, json!({"messages": [user("go")], "tools": tools()})).await;
    assert_eq!(finish(&resp), "stop");
    assert_eq!(msg(&resp)["content"], "All done, wrote notes.md.");
    assert!(msg(&resp).get("tool_calls").is_none());
    let reply = call_block("submit", "{}");
    let h = harness(&[&reply]);
    let (_, resp) = chat(&h, json!({"messages": [user("go")], "tools": tools()})).await;
    assert_eq!(msg(&resp)["content"], "Task complete.");
}

#[tokio::test]
async fn degenerate_path_is_fixed_from_the_task_text_without_asking_the_model() {
    let reply = call_block(
        "write",
        r#"{"path": "Tower", "content": "hello world, long enough"}"#,
    );
    let h = harness(&[&reply]);
    let (_, resp) = chat(
        &h,
        json!({"messages": [user("Create notes.md with a greeting.")], "tools": tools()}),
    )
    .await;
    assert_eq!(finish(&resp), "tool_calls");
    let args: Value = serde_json::from_str(
        msg(&resp)["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        args,
        json!({"path": "notes.md", "content": "hello world, long enough"})
    );
    assert_eq!(h.backend.seen().len(), 1, "extraction answered; no re-ask");
}

#[tokio::test]
async fn degenerate_path_without_a_task_path_is_fixed_by_a_schema_constrained_re_ask() {
    let reply = call_block(
        "write",
        r#"{"path": "Tower", "content": "hello world, long enough"}"#,
    );
    let h = harness(&[&reply, r#"{"path": "notes.md"}"#]);
    let (_, resp) = chat(
        &h,
        json!({"messages": [user("Save the greeting.")], "tools": tools()}),
    )
    .await;
    assert_eq!(finish(&resp), "tool_calls");
    let args: Value = serde_json::from_str(
        msg(&resp)["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(args["path"], "notes.md");
    assert_eq!(
        args["content"], "hello world, long enough",
        "good fields survive"
    );
    let seen = h.backend.seen();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].options.max_tokens, 400);
    let Some(Grammar::JsonSchema(schema)) = &seen[1].grammar else {
        panic!("expected a JSON schema grammar")
    };
    assert_eq!(schema["required"], json!(["path"]));
    assert!(
        seen[1]
            .prompt_text
            .contains("Fix the arguments of a failed tool call")
    );
}

#[tokio::test]
async fn missing_required_field_is_repaired_on_any_tool() {
    // `write` requires content; the call omits it entirely.
    let reply = call_block("write", r#"{"path": "notes.md"}"#);
    let h = harness(&[&reply, r#"{"content": "hello world, long enough"}"#]);
    let (_, resp) = chat(
        &h,
        json!({"messages": [user("Save the greeting.")], "tools": tools()}),
    )
    .await;
    let args: Value = serde_json::from_str(
        msg(&resp)["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(args["content"], "hello world, long enough");
    assert_eq!(args["path"], "notes.md");
}

#[tokio::test]
async fn empty_tool_name_triggers_a_whole_reply_re_ask() {
    let h = harness(&[
        "<tool_call>\n{\"arguments\": {}}\n</tool_call>",
        &call_block("ls", r#"{"path": "."}"#),
    ]);
    let (_, resp) = chat(&h, json!({"messages": [user("list")], "tools": tools()})).await;
    assert_eq!(finish(&resp), "tool_calls");
    assert_eq!(msg(&resp)["tool_calls"][0]["function"]["name"], "ls");
    let seen = h.backend.seen();
    assert_eq!(seen.len(), 2);
    assert!(
        seen[1]
            .prompt_text
            .contains("Your last reply was malformed"),
        "{}",
        seen[1].prompt_text
    );
}

#[tokio::test]
async fn broken_calls_are_dropped_after_a_failed_repair() {
    let broken = "<tool_call>\n{\"arguments\": {}}\n</tool_call>";
    let h = harness(&[broken, broken]);
    let (_, resp) = chat(&h, json!({"messages": [user("list")], "tools": tools()})).await;
    assert_eq!(finish(&resp), "stop");
    assert!(msg(&resp).get("tool_calls").is_none());
    assert_eq!(
        msg(&resp)["content"],
        "Previous tool call arguments were unusable and were discarded."
    );
}

/// Kept Swift divergence (see `pipeline::Mode`): a call that still misses a
/// required field after a failed field repair is SENT on the non-stream path
/// (the drop only follows a failed whole-reply repair) ...
#[tokio::test]
async fn block_path_sends_a_call_with_a_missing_required_field_when_repair_fails() {
    let reply = call_block("write", r#"{"path": "notes.md"}"#);
    let h = harness(&[&reply, "no json at all"]);
    let (_, resp) = chat(
        &h,
        json!({"messages": [user("Save the greeting.")], "tools": tools()}),
    )
    .await;
    assert_eq!(finish(&resp), "tool_calls");
    assert_eq!(msg(&resp)["tool_calls"][0]["function"]["name"], "write");
}

/// ... and DROPPED on the stream path.
#[tokio::test]
async fn stream_path_drops_a_call_with_a_missing_required_field_when_repair_fails() {
    let reply = call_block("write", r#"{"path": "notes.md"}"#);
    let h = harness(&[&reply, "no json at all"]);
    let (_, events) = chat_stream(
        &h,
        json!({"messages": [user("Save the greeting.")], "tools": tools()}),
    )
    .await;
    let parsed: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    assert!(
        parsed
            .iter()
            .all(|c| c["choices"][0]["delta"].get("tool_calls").is_none()),
        "{events:?}"
    );
    assert_eq!(
        parsed[0]["choices"][0]["delta"]["content"],
        "Previous tool call arguments were unusable and were discarded."
    );
    assert_eq!(
        parsed.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[tokio::test]
async fn repeated_identical_tool_calls_get_a_loop_breaking_nudge() {
    let h = harness(&["ok"]);
    let call = |id: &str| json!({"role": "assistant", "content": "", "tool_calls": [{"id": id, "type": "function", "function": {"name": "chmod", "arguments": "{\"command\": \"chmod +x a.py\"}"}}]});
    let tool = json!({"role": "tool", "content": "error"});
    let body = json!({"messages": [user("run"), call("1"), tool, call("2"), tool, call("3"), tool], "tools": tools()});
    chat(&h, body).await;
    let prompt = &h.backend.seen()[0].prompt_text;
    assert!(prompt.contains("Do NOT retry the same call"), "{prompt}");
    assert!(prompt.contains("`chmod` tool 3 times"), "{prompt}");
}

#[tokio::test]
async fn workspace_mode_resolves_relative_paths_for_the_harness() {
    let reply = call_block(
        "write",
        r#"{"path": "scratch/a.py", "content": "print('hello world')"}"#,
    );
    let h = harness_with(MockBackend::new(&[&reply]), |c| {
        c.workspace_root = Some("/ws".into())
    });
    let body = json!({"messages": [
        user("Write scratch/a.py"),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "ls", "arguments": "{\"path\": \"/ws\"}"}}]},
        {"role": "tool", "content": "listing of /ws/scratch"},
    ], "tools": tools()});
    let (_, resp) = chat(&h, body).await;
    let args: Value = serde_json::from_str(
        msg(&resp)["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        args["path"], "/ws/scratch/a.py",
        "relative path resolved against the root"
    );
    let prompt = &h.backend.seen()[0].prompt_text;
    assert!(
        prompt.contains("RELATIVE to the workspace root"),
        "{prompt}"
    );
    assert!(
        !prompt.contains("/ws/"),
        "root prefix stripped from what the model sees: {prompt}"
    );
}

#[tokio::test]
async fn outside_workspace_path_is_repaired_from_the_task() {
    let reply = call_block(
        "write",
        r#"{"path": "/elsewhere/a.py", "content": "print('hello world')"}"#,
    );
    let h = harness_with(MockBackend::new(&[&reply]), |c| {
        c.workspace_root = Some("/ws".into())
    });
    let (_, resp) = chat(
        &h,
        json!({"messages": [user("Write scratch/a.py please")], "tools": tools()}),
    )
    .await;
    let args: Value = serde_json::from_str(
        msg(&resp)["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(args["path"], "/ws/scratch/a.py");
}

#[tokio::test]
async fn sanitize_truncates_repetition_when_enabled() {
    let looping = format!("start of the answer {}", "tr".repeat(30));
    let off = harness(&[&looping]);
    let (_, resp) = chat(&off, json!({"messages": [user("x")]})).await;
    assert_eq!(msg(&resp)["content"], looping, "measure-only by default");
    let on = harness_with(MockBackend::new(&[&looping]), |c| c.sanitize = true);
    let (_, resp) = chat(&on, json!({"messages": [user("x")]})).await;
    assert_eq!(msg(&resp)["content"], "start of the answer");
}

// MARK: streaming

#[tokio::test]
async fn stream_without_tools_sends_deltas_finish_and_done() {
    let h = harness(&["Hi!"]);
    let (s, events) = chat_stream(&h, json!({"messages": [user("hi")]})).await;
    assert_eq!(s, StatusCode::OK);
    let parsed: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    let deltas: String = parsed
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(deltas, "Hi!");
    assert_eq!(events.last().unwrap(), "[DONE]");
    let last = parsed.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["object"], "chat.completion.chunk");
    assert!(parsed.iter().all(|c| c.get("usage").is_none()));
}

#[tokio::test]
async fn stream_include_usage_adds_a_tail_chunk_with_empty_choices() {
    let h = harness(&["Hi"]);
    let (_, events) = chat_stream(
        &h,
        json!({"messages": [user("hi")], "stream_options": {"include_usage": true}}),
    )
    .await;
    let n = events.len();
    assert_eq!(events[n - 1], "[DONE]");
    let usage: Value = serde_json::from_str(&events[n - 2]).unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 3, "2 bytes + EOS");
    let total = usage["usage"]["total_tokens"].as_u64().unwrap();
    assert_eq!(total, usage["usage"]["prompt_tokens"].as_u64().unwrap() + 3);
    let before: Value = serde_json::from_str(&events[n - 3]).unwrap();
    assert_eq!(
        before["choices"][0]["finish_reason"], "stop",
        "finish chunk precedes the usage tail"
    );
}

#[tokio::test]
async fn stream_with_tools_sends_content_then_indexed_tool_calls() {
    let reply = format!(
        "Two calls.\n{}\n{}",
        call_block("ls", r#"{"path": "a"}"#),
        call_block("ls", r#"{"path": "b"}"#)
    );
    let h = harness(&[&reply]);
    let (_, events) = chat_stream(&h, json!({"messages": [user("list")], "tools": tools()})).await;
    let parsed: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    assert_eq!(parsed[0]["choices"][0]["delta"]["content"], "Two calls.");
    let calls = parsed[1]["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        (calls[0]["index"].as_u64(), calls[1]["index"].as_u64()),
        (Some(0), Some(1)),
        "indexes are positional"
    );
    assert_eq!(calls[1]["function"]["arguments"], r#"{"path": "b"}"#);
    assert_eq!(
        parsed.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert_eq!(events.last().unwrap(), "[DONE]");
}

#[tokio::test]
async fn stream_with_tools_buffers_tag_text_until_parsed() {
    let reply = call_block("ls", r#"{"path": "."}"#);
    let h = harness(&[&reply]);
    let (_, events) = chat_stream(&h, json!({"messages": [user("list")], "tools": tools()})).await;
    assert!(
        events.iter().all(|e| !e.contains("<tool_call>")),
        "markup never leaks into deltas: {events:?}"
    );
}

#[tokio::test]
async fn stream_error_ends_the_stream_without_done_and_fails_the_trace() {
    let h = harness(&[]);
    h.backend.push_error("gone");
    let (s, events) = chat_stream(&h, json!({"messages": [user("hi")]})).await;
    assert_eq!(s, StatusCode::OK);
    assert!(events.is_empty(), "{events:?}");
    let record = wait_for_trace(&h).await;
    assert_eq!(record["ok"], false);
    assert!(
        record["error"]
            .as_str()
            .unwrap()
            .starts_with("stream generation failed"),
        "{record}"
    );
}

// MARK: routes and traces

async fn wait_for_trace(h: &Harness) -> Value {
    for _ in 0..200 {
        if let Some(r) = h.engine.traces.list(1).into_iter().next() {
            return r;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("no trace was written");
}

#[tokio::test]
async fn models_and_health_routes() {
    let h = harness(&[]);
    let (s, body) = send(&h, "GET", "/healthz", None).await;
    assert_eq!((s, body.as_str()), (StatusCode::OK, "ok"));
    let (s, body) = send(&h, "GET", "/v1/models", None).await;
    assert_eq!(s, StatusCode::OK);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(
        v["data"][0],
        json!({"id": "maple-test", "object": "model", "created": 0, "owned_by": "momij"})
    );
}

#[tokio::test]
async fn traces_are_recorded_listed_and_fetchable() {
    let h = harness(&["hello"]);
    chat(&h, json!({"messages": [user("hi")]})).await;
    let record = wait_for_trace(&h).await;
    let id = record["trace_id"].as_str().unwrap().to_owned();
    let stages: Vec<&str> = record["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["stage"].as_str().unwrap())
        .collect();
    assert_eq!(stages, ["accept", "parse", "verify", "decode", "present"]);
    assert_eq!(record["ok"], true);

    let (s, body) = send(&h, "GET", "/v1/traces", None).await;
    assert_eq!(s, StatusCode::OK);
    let list: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"].as_array().unwrap().len(), 1);

    let (s, body) = send(&h, "GET", &format!("/v1/traces/{id}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["trace_id"],
        id.as_str()
    );

    let (s, body) = send(&h, "GET", "/v1/traces/tr-missing", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert!(body.contains("trace not found: tr-missing"));
}

#[tokio::test]
async fn failed_requests_are_traced_too() {
    let h = harness(&[]);
    chat(&h, json!({"messages": []})).await;
    let record = wait_for_trace(&h).await;
    assert_eq!(record["ok"], false);
    assert_eq!(record["error"], "messages must be non-empty");
}

// MARK: helpers and concurrency

#[test]
fn calls_clean_and_workspace_resolution_helpers() {
    use momij_core::openai_compat::ToolCallSpec;
    let tool_lines: Vec<String> = tools()
        .as_array()
        .unwrap()
        .iter()
        .map(Value::to_string)
        .collect();
    let ok = ToolCallSpec::new(
        "1",
        "write",
        r#"{"path": "a.py", "content": "print('hello world')"}"#,
    );
    assert!(pipeline::calls_clean(
        std::slice::from_ref(&ok),
        &tool_lines,
        None
    ));
    let bad_path = ToolCallSpec::new("2", "write", r#"{"path": "Tower", "content": "x"}"#);
    assert!(
        !pipeline::calls_clean(&[bad_path], &tool_lines, None),
        "degenerate"
    );
    let missing = ToolCallSpec::new("3", "write", r#"{"path": "a.py"}"#);
    assert!(
        !pipeline::calls_clean(&[missing], &tool_lines, None),
        "missing required"
    );
    let outside = ToolCallSpec::new(
        "4",
        "write",
        r#"{"path": "/etc/a.py", "content": "print('hello world')"}"#,
    );
    assert!(
        pipeline::calls_clean(std::slice::from_ref(&outside), &tool_lines, None),
        "policy is off without a root"
    );
    assert!(!pipeline::calls_clean(&[outside], &tool_lines, Some("/ws")));

    let (resolved, n) = pipeline::resolve_workspace_paths(
        &[ok, ToolCallSpec::new("5", "bash", r#"{"command": "ls"}"#)],
        "/ws",
    );
    assert_eq!(n, 1);
    assert!(
        resolved[0].arguments.contains(r#""path": "/ws/a.py""#),
        "{}",
        resolved[0].arguments
    );
    assert_eq!(resolved[1].arguments, r#"{"command": "ls"}"#);
}

#[tokio::test]
async fn the_gate_serializes_model_calls_by_default() {
    let h = harness(&[]);
    let first = h.engine.acquire().await;
    let second =
        tokio::time::timeout(std::time::Duration::from_millis(50), h.engine.acquire()).await;
    assert!(
        second.is_err(),
        "a second caller waits while the slot is held"
    );
    drop(first);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(500), h.engine.acquire())
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn config_concurrency_widens_the_gate() {
    let h = harness_with(MockBackend::default(), |c| c.max_concurrency = 2);
    let _a = h.engine.acquire().await;
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(200), h.engine.acquire())
            .await
            .is_ok()
    );
}
