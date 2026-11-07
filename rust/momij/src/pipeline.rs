//! The chat-completions pipeline: request preparation, the model call, and
//! the verify / repair / sanitize stages that turn raw Maple output into an
//! OpenAI response. Port of the logic in `swift/Sources/momij/Server.swift`.
//!
//! The Swift original repeated the tool-call repair logic once for the
//! non-stream path and once for the SSE path. Both now go through
//! [`resolve_turn`]; the few places where the two paths deliberately (or
//! accidentally) differed are selected by [`Mode`] and called out below.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use momij_core::arg_repair;
use momij_core::chat_template_patch;
use momij_core::generate_options::{GenerateOptions, env_on_by_default};
use momij_core::loop_breaker;
use momij_core::openai_compat::{self, ChatCompletionRequest, ChatMessage, ToolCallSpec};
use momij_core::path_policy;
use momij_core::protocol_absorb;
use momij_core::response_verify::{self, Hit, Outcome};
use momij_core::tool_markup::{self, ParsedToolCalls};
use momij_core::trace_store::TraceStore;
use momij_core::{JsonMap, TokenId};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::backend::{self, Backend, BackendError, Completion, GenerateRequest, Grammar};
use crate::trace::RequestTrace;

pub const COMPLETION_ID: &str = "chatcmpl-momij";
const NO_USER_TEXT: &str = "(no user text)";
const DROPPED_PLACEHOLDER: &str = "Previous tool call arguments were unusable and were discarded.";

/// Everything the Swift server read ad hoc from the environment, captured
/// once at startup so requests are deterministic (and testable).
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub model_id: String,
    /// `MOMIJ_WORKSPACE`: fixed workspace root (None = path policy off).
    pub workspace_root: Option<String>,
    /// `MOMIJ_PROTOCOL_TOOLS`: tool names whose calls are absorbed.
    pub protocol_tools: HashSet<String>,
    /// `MOMIJ_ENABLE_THINKING`: keep the stock `<think>` generation prompt.
    pub thinking_enabled: bool,
    /// `MOMIJ_VERIFY_SANITIZE=1`: let the verify pass rewrite the reply.
    pub sanitize: bool,
    /// `MOMIJ_SUFFIX_SPEC` (default on). Forwarded in the options; llama-server
    /// speculation is configured on the server side.
    pub suffix_spec: bool,
    pub eos_token_id: TokenId,
    /// Concurrent model calls (the Swift engine serialized everything).
    pub max_concurrency: usize,
}

impl ServeConfig {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            workspace_root: None,
            protocol_tools: protocol_absorb::tool_names(None),
            thinking_enabled: false,
            sanitize: false,
            suffix_spec: true,
            eos_token_id: openai_compat::DEFAULT_EOS_TOKEN_IDS[0],
            max_concurrency: 1,
        }
    }

    pub fn from_env(model_id: impl Into<String>) -> Self {
        let var = |k: &str| std::env::var(k).ok();
        Self {
            workspace_root: path_policy::workspace_root(),
            protocol_tools: protocol_absorb::tool_names_from_env(),
            thinking_enabled: chat_template_patch::enable_thinking(),
            sanitize: var("MOMIJ_VERIFY_SANITIZE").as_deref() == Some("1"),
            suffix_spec: env_on_by_default(var("MOMIJ_SUFFIX_SPEC").as_deref()),
            max_concurrency: var("MOMIJ_CONCURRENCY")
                .and_then(|v| v.trim().parse().ok())
                .filter(|&n| n > 0)
                .unwrap_or(1),
            ..Self::new(model_id)
        }
    }
}

pub struct Engine {
    pub backend: Arc<dyn Backend>,
    pub config: ServeConfig,
    pub traces: Arc<TraceStore>,
    gate: Arc<Semaphore>,
}

impl Engine {
    pub fn new(backend: Arc<dyn Backend>, config: ServeConfig, traces: TraceStore) -> Arc<Self> {
        let gate = Arc::new(Semaphore::new(config.max_concurrency.max(1)));
        Arc::new(Self {
            backend,
            config,
            traces: Arc::new(traces),
            gate,
        })
    }

    /// Wait for a model slot (released on drop).
    pub async fn acquire(&self) -> OwnedSemaphorePermit {
        // The semaphore is never closed.
        self.gate
            .clone()
            .acquire_owned()
            .await
            .expect("engine gate closed")
    }

    /// One complete generation under the concurrency gate.
    pub async fn generate(&self, request: GenerateRequest) -> Result<Completion, BackendError> {
        let _permit = self.acquire().await;
        backend::collect(self.backend.generate(request)).await
    }

    /// Fire-and-forget persistence: tracing must never fail or delay a request.
    pub fn record_trace(&self, trace: &RequestTrace) {
        let store = self.traces.clone();
        let (record, id) = (trace.record(), trace.id.clone());
        tokio::task::spawn_blocking(move || store.write(&record, &id));
    }
}

/// An error that maps onto an OpenAI-style JSON error response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineError {
    pub status: u16,
    pub message: String,
}

impl PipelineError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

/// A request after validation, constraint mapping, tool-markup rewrite and
/// chat templating.
pub struct Prepared {
    pub request: ChatCompletionRequest,
    /// Messages the model sees (tool history rewritten, loop nudge appended).
    pub effective: Vec<ChatMessage>,
    pub prompt: Vec<TokenId>,
    pub options: GenerateOptions,
    pub grammar: Option<Grammar>,
}

pub async fn prepare(
    engine: &Engine,
    request: ChatCompletionRequest,
    trace: &RequestTrace,
) -> Result<Prepared, PipelineError> {
    let cfg = &engine.config;
    trace.event(
        "parse",
        "ok",
        json!({
            "messages": request.messages.len(),
            "has_tools": !request.tools_lines.is_empty(),
            "stream": request.stream,
            "max_tokens": request.max_tokens,
            // Original (pre-rewrite) role layout: task extraction only
            // considers role "user", so the shape tells us whether the task
            // is even reachable there.
            "roles": request.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
        }),
    );
    if request.n > 1 {
        return Err(PipelineError::new(400, "n>1 unsupported"));
    }
    let grammar = match &request.structured_constraint {
        Some(c) => Some(backend::grammar_for(c).map_err(|e| PipelineError::new(400, e))?),
        None => None,
    };

    // Native Maple tools contract: rewrite tool history into markup form and
    // append the tools instruction (compositor parity). Tool-call loop
    // breaker: a trailing run of identical calls gets a steering message so
    // the retry loop ends this turn (per request only). Detect BEFORE the
    // rewrite — the rewrite folds tool_calls into markup content, leaving
    // nothing for the detector to see.
    let loop_hit = loop_breaker::detect(&request.messages, loop_breaker::DEFAULT_MIN_REPEATS);
    if let Some((tool, count)) = &loop_hit {
        trace.event("loop_break", "ok", json!({"tool": tool, "count": count}));
    }
    let mut effective = request.messages.clone();
    if !request.tools_lines.is_empty() {
        let mut msgs = tool_markup::rewrite_messages(&request.messages);
        let mut suffix = tool_markup::system_suffix(&request.tools_lines);
        if cfg.workspace_root.is_some() {
            suffix.push('\n');
            suffix.push_str(path_policy::suffix_line());
        }
        match msgs.iter_mut().find(|m| m.role == "system") {
            Some(sys) => sys.content = format!("{}\n\n{suffix}", sys.content),
            None => msgs.insert(0, ChatMessage::new("system", suffix)),
        }
        // Workspace mode: strip the root prefix from everything the model
        // sees so its world stays relative and short.
        if let Some(root) = &cfg.workspace_root {
            for m in &mut msgs {
                m.content = path_policy::strip_root_prefix(&m.content, root);
            }
        }
        effective = msgs;
    }
    if let Some((tool, count)) = loop_hit {
        effective.push(ChatMessage::new(
            "user",
            loop_breaker::break_nudge(&tool, count),
        ));
    }
    let prompt = engine
        .backend
        .apply_chat_template(&effective)
        .await
        .map_err(|e| PipelineError::new(500, format!("chat template failed: {e}")))?;

    let options = GenerateOptions {
        max_tokens: usize::try_from(request.max_tokens).unwrap_or(256),
        temperature: request.temperature,
        top_p: request.top_p,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        repetition_penalty: request.repetition_penalty,
        eos_token_ids: vec![cfg.eos_token_id],
        use_suffix_spec: cfg.suffix_spec && grammar.is_none(),
        seed: request.seed,
        banned_token_ids: chat_template_patch::banned_assistant_token_ids_for(cfg.thinking_enabled),
        ..GenerateOptions::default()
    };
    // Pre-flight prompt-capacity check: reject cleanly BEFORE any generation
    // (an OpenAI-style 400 the harness can see) instead of an aborted stream.
    if let Some(cap) = engine
        .backend
        .max_prompt_tokens()
        .filter(|&cap| prompt.len() >= cap)
    {
        return Err(PipelineError::new(
            400,
            format!(
                "prompt too long: {} tokens exceeds the backend context capacity {cap}",
                prompt.len()
            ),
        ));
    }
    Ok(Prepared {
        request,
        effective,
        prompt,
        options,
        grammar,
    })
}

// MARK: - tool-call checks shared by every path

/// Clean check for repaired calls: no degenerate args, no missing required
/// fields, and no outside-workspace path values (the policy is off when no
/// workspace root is configured).
pub fn calls_clean(calls: &[ToolCallSpec], tools_lines: &[String], root: Option<&str>) -> bool {
    degenerate(calls).is_none()
        && !calls.iter().any(|c| {
            !arg_repair::missing_required_fields(
                tools_lines,
                &c.name,
                &arg_repair::parse_args(&c.arguments),
            )
            .is_empty()
        })
        && !calls.iter().any(|c| outside_workspace(c, root))
}

fn outside_workspace(call: &ToolCallSpec, root: Option<&str>) -> bool {
    path_policy::args_outside_workspace(
        &arg_repair::parse_args(&call.arguments),
        root.unwrap_or("/"),
    )
}

fn degenerate(calls: &[ToolCallSpec]) -> Option<Hit> {
    response_verify::degenerate_tool_args(&response_verify::Message::with_tool_calls(
        calls.to_vec(),
    ))
}

/// Workspace mode: resolve relative path fields in the outgoing calls against
/// the workspace root (the model writes relative, the harness executes
/// absolute). Returns the rewritten calls and the change count.
pub fn resolve_workspace_paths(calls: &[ToolCallSpec], root: &str) -> (Vec<ToolCallSpec>, usize) {
    let mut count = 0;
    let out = calls
        .iter()
        .map(|call| {
            let args = arg_repair::parse_args(&call.arguments);
            let mut out_args = args.clone();
            let mut changed = false;
            for field in path_policy::PATH_FIELDS {
                let Some(v) = args
                    .get(field)
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                else {
                    continue;
                };
                let resolved = path_policy::resolve(v, root);
                if resolved != v {
                    out_args.insert(field.to_owned(), Value::String(resolved));
                    changed = true;
                }
            }
            if changed {
                count += 1;
                ToolCallSpec::new(
                    call.id.clone(),
                    call.name.clone(),
                    arg_repair::args_json_string(&out_args),
                )
            } else {
                call.clone()
            }
        })
        .collect();
    (out, count)
}

#[derive(Debug, Clone)]
struct PendingFix {
    idx: usize,
    fields: Vec<String>,
    reasons: HashMap<String, String>,
    kind: String,
}

impl PendingFix {
    fn new(idx: usize, fields: Vec<String>, reason: &str, kind: &str) -> Self {
        let reasons = fields
            .iter()
            .map(|f| (f.clone(), reason.to_owned()))
            .collect();
        Self {
            idx,
            fields,
            reasons,
            kind: kind.to_owned(),
        }
    }
}

/// The fix target comes from the write/edit heuristics OR from schema-required
/// fields missing on ANY tool (the harness-side "missing required property"
/// rejection, caught before it leaves) OR from a path outside the workspace.
fn pending_fix(
    calls: &[ToolCallSpec],
    hit: Option<&Hit>,
    tools_lines: &[String],
    root: Option<&str>,
) -> Option<PendingFix> {
    if let Some(hit) = hit.filter(|h| h.onset < calls.len()) {
        let args = arg_repair::parse_args(&calls[hit.onset].arguments);
        let fields = arg_repair::repair_fields(&hit.detail, &args);
        if !fields.is_empty() {
            let reason = hit
                .detail
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("unusable");
            return Some(PendingFix::new(hit.onset, fields, reason, &hit.kind));
        }
    }
    for (i, call) in calls.iter().enumerate().filter(|(_, c)| !c.name.is_empty()) {
        let args = arg_repair::parse_args(&call.arguments);
        let missing = arg_repair::missing_required_fields(tools_lines, &call.name, &args);
        if !missing.is_empty() {
            return Some(PendingFix::new(
                i,
                missing,
                "missing required property",
                "missing_required_property",
            ));
        }
        // Workspace discipline: absolute paths outside the root would be
        // rejected by the harness sandbox — repair them into relative form.
        if let Some(root) = root {
            for f in path_policy::PATH_FIELDS {
                let outside = args
                    .get(f)
                    .and_then(Value::as_str)
                    .is_some_and(|v| !v.is_empty() && path_policy::is_outside_workspace(v, root));
                if outside {
                    return Some(PendingFix::new(
                        i,
                        vec![f.to_owned()],
                        "absolute path outside the workspace root (use a relative path)",
                        "path_outside_workspace",
                    ));
                }
            }
        }
    }
    None
}

/// Stage-1 targeted field repair: re-ask ONLY the broken argument fields
/// (grammar-constrained to the tool's JSON schema when available), merge into
/// the original arguments, and return them. The untouched fields survive —
/// regenerating the whole reply is what breaks good parts.
#[allow(clippy::too_many_arguments)]
pub async fn field_repair_call(
    engine: &Engine,
    tools_lines: &[String],
    tool: &str,
    args: &JsonMap,
    fields: &[String],
    reasons: &HashMap<String, String>,
    task: &str,
    base: &GenerateOptions,
) -> Result<Option<JsonMap>, PipelineError> {
    if fields.is_empty() {
        return Ok(None);
    }
    let grammar = arg_repair::constrained_object_schema(tools_lines, tool, fields)
        .map(|s| Grammar::JsonSchema(Value::Object(s)));
    let mut prompt = arg_repair::field_repair_prompt(
        task,
        tool,
        &arg_repair::args_json_string(args),
        fields,
        reasons,
    );
    if engine.config.workspace_root.is_some() {
        prompt.push_str("\n\n");
        prompt.push_str(path_policy::suffix_line());
        prompt.push_str(" The workspace root is fixed; reply with paths RELATIVE to it.");
    }
    let prompt_ids = engine
        .backend
        .apply_chat_template(&[ChatMessage::new("user", prompt)])
        .await
        .map_err(|e| PipelineError::new(500, format!("chat template failed: {e}")))?;
    let options = base.with_max_tokens(400);
    let done = engine
        .generate(GenerateRequest {
            prompt: prompt_ids,
            options: options.clone(),
            grammar,
        })
        .await
        .map_err(|e| PipelineError::new(500, format!("generation failed: {e}")))?;
    let text = decode_text(engine, &done.tokens, &options).await?;
    Ok(arg_repair::first_json_object(&text)
        .map(|generated| arg_repair::merge_fields(args, &generated, fields)))
}

/// EOS-trimmed token ids -> text with think blocks handled for the content channel.
async fn decode_text(
    engine: &Engine,
    tokens: &[TokenId],
    options: &GenerateOptions,
) -> Result<String, PipelineError> {
    let (_, text) = decode_raw_and_text(engine, tokens, options).await?;
    Ok(text)
}

/// `(raw, text)`: raw is the detokenized completion; text additionally has
/// the think block removed (unless thinking is enabled).
async fn decode_raw_and_text(
    engine: &Engine,
    tokens: &[TokenId],
    options: &GenerateOptions,
) -> Result<(String, String), PipelineError> {
    let ids = openai_compat::content_token_ids(tokens, &options.eos_token_ids);
    let raw = if ids.is_empty() {
        String::new()
    } else {
        engine
            .backend
            .detokenize(&ids)
            .await
            .map_err(|e| PipelineError::new(500, format!("decode failed: {e}")))?
    };
    let text = chat_template_patch::strip_think_for_content(&raw, engine.config.thinking_enabled);
    Ok((raw, text))
}

/// How the reply is delivered. The two paths share [`resolve_turn`] but the
/// Swift server treated them differently in a few places, kept here:
/// - repair nudge wording (Block names the tool; Stream is generic);
/// - broken calls are dropped only after a failed whole-reply repair (Block)
///   vs whenever they are still broken (Stream);
/// - the opt-in verify sanitize runs on Block only;
/// - a dropped reply finishes `stop` (Block) vs keeps its finish (Stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Block,
    Stream,
}

/// The reply after verify / repair / sanitize.
#[derive(Debug, Clone)]
pub struct Turn {
    pub content: String,
    pub tool_calls: Vec<ToolCallSpec>,
    pub finish: String,
    /// Detokenized completion (think block included).
    pub raw: String,
    pub verify: Outcome,
    /// Some parsed call had an empty name (before drop).
    pub had_empty_name: bool,
}

pub async fn resolve_turn(
    engine: &Engine,
    trace: &RequestTrace,
    prepared: &Prepared,
    mode: Mode,
    completion: &Completion,
) -> Result<Turn, PipelineError> {
    let cfg = &engine.config;
    let opts = &prepared.options;
    let tools_lines = &prepared.request.tools_lines;
    let root = cfg.workspace_root.as_deref();
    let tokens = &completion.tokens;

    let mut finish = openai_compat::finish_reason(
        tokens,
        i64::try_from(opts.max_tokens).unwrap_or(i64::MAX),
        &opts.eos_token_ids,
    )
    .to_owned();
    let (raw, text) = decode_raw_and_text(engine, tokens, opts).await?;
    let mut parsed = tool_markup::parse_pseudo_tool_calls(&text);

    // Protocol-tool interception: the model's "I'm done" intent (a submit /
    // ask_user_question call, often with unusable args) becomes a clean
    // ending instead of a harness rejection loop. All calls drop,
    // finish=stop, and the call's description (or a placeholder) is the
    // final content.
    if let Some(absorb) = protocol_absorb::absorb_call(&parsed.calls, &cfg.protocol_tools) {
        let name = absorb.name.clone();
        parsed = ParsedToolCalls {
            cleaned_content: protocol_absorb::final_content(absorb),
            calls: Vec::new(),
        };
        finish = "stop".into();
        trace.event("protocol_absorb", "ok", json!({"tool": name}));
    }

    // Degenerate-call repair, 3 stages: (1) targeted field re-ask — regenerate
    // ONLY the broken argument fields under the tool's JSON schema and merge
    // (good parts survive); (2) whole-reply re-ask with a corrective nudge;
    // (3) drop the broken calls.
    let mut degenerate_hit = degenerate(&parsed.calls);
    let empty_name_at_start = parsed.calls.iter().any(|c| c.name.is_empty());
    let pending = pending_fix(&parsed.calls, degenerate_hit.as_ref(), tools_lines, root);
    if let (false, Some(pf)) = (empty_name_at_start, pending) {
        field_repair(
            engine,
            trace,
            prepared,
            &mut parsed,
            &mut degenerate_hit,
            &pf,
        )
        .await;
    }

    let mut repair_failed = false;
    if parsed.calls.iter().any(|c| c.name.is_empty()) || degenerate_hit.is_some() {
        let nudge = repair_nudge(mode, degenerate_hit.as_ref());
        let trigger = degenerate_hit
            .as_ref()
            .map_or("empty_name", |h| h.kind.as_str());
        match whole_reply_repair(engine, prepared, &text, &nudge, tokens.len()).await {
            Ok(reparsed) => {
                let clean = !reparsed.calls.iter().any(|c| c.name.is_empty())
                    && degenerate(&reparsed.calls).is_none();
                if clean {
                    parsed = reparsed;
                    trace.event("repair", "ok", json!({"trigger": trigger}));
                } else {
                    repair_failed = true;
                    trace.event("repair", "error", json!({"trigger": trigger}));
                }
            }
            Err(e) => {
                // Keep the original parse on repair failure.
                repair_failed = true;
                trace.event("repair", "error", json!({"error": e.message}));
            }
        }
    }

    // Native verify pass (evprtr verify pipeline parity): every detector runs
    // and records into the trace; sanitize is opt-in (MOMIJ_VERIFY_SANITIZE=1)
    // so the default path only measures.
    let verify = response_verify::verify(&response_verify::Message {
        content: Some(text.clone()),
        reasoning_content: response_verify::think_interior(&raw),
        tool_calls: parsed.calls.clone(),
    });
    let sanitize = mode == Mode::Block && cfg.sanitize;
    let hits: Vec<Value> = verify.hits.iter().map(Hit::event_detail).collect();
    let verify_detail = match mode {
        Mode::Block => json!({"hits": hits, "sanitized": sanitize && verify.first().is_some()}),
        Mode::Stream => json!({"hits": hits}),
    };

    let had_empty_name = parsed.calls.iter().any(|c| c.name.is_empty());
    let mut content = parsed.cleaned_content.clone();
    let mut calls = parsed.calls.clone();
    let mut resp_finish = if calls.is_empty() {
        finish.clone()
    } else {
        "tool_calls".to_owned()
    };

    // Broken-call drop (evprtr sanitize parity): surviving calls that are
    // still unusable (empty name, degenerate write/edit args, missing
    // required fields, paths outside the workspace) must not reach the
    // harness — a tool_calls finish it cannot execute causes retry loops
    // (observed live in the DSH subagent test).
    let broken = had_empty_name
        || degenerate(&calls).is_some()
        || calls.iter().any(|c| {
            !arg_repair::missing_required_fields(
                tools_lines,
                &c.name,
                &arg_repair::parse_args(&c.arguments),
            )
            .is_empty()
        })
        || calls.iter().any(|c| outside_workspace(c, root));
    let drop_now = match mode {
        Mode::Block => repair_failed && broken,
        Mode::Stream => broken,
    };
    if drop_now {
        calls.clear();
        resp_finish = match mode {
            Mode::Block => "stop".to_owned(),
            Mode::Stream => finish.clone(),
        };
        if content
            .trim_matches(|c: char| c == ' ' || c == '\t')
            .is_empty()
        {
            content = DROPPED_PLACEHOLDER.to_owned();
        }
        trace.event("sanitize", "ok", json!({"action": "drop_degenerate_calls"}));
    }
    if let (true, Some(hit)) = (sanitize, verify.first()) {
        if hit.kind == "degenerate_tool_args" {
            // evprtr sanitize: unusable calls are dropped so the harness never
            // sees a tool_calls finish without calls.
            calls.clear();
            resp_finish = "stop".into();
        } else if hit.field == "content"
            && ["word_run", "ngram_run", "char_motif"].contains(&hit.kind.as_str())
        {
            content = response_verify::truncate_before_repetition(&content);
        }
    }
    if let (Some(root), false) = (root, calls.is_empty()) {
        let (resolved, n) = resolve_workspace_paths(&calls, root);
        if n > 0 {
            calls = resolved;
            trace.event("path_resolve", "ok", json!({"resolved": n}));
        }
    }
    trace.event(
        "verify",
        if verify.hits.is_empty() { "ok" } else { "hit" },
        verify_detail,
    );
    Ok(Turn {
        content,
        tool_calls: calls,
        finish: resp_finish,
        raw,
        verify,
        had_empty_name,
    })
}

fn repair_nudge(mode: Mode, hit: Option<&Hit>) -> String {
    let reason = hit
        .and_then(|h| h.detail.get("reason"))
        .and_then(Value::as_str);
    match (mode, hit) {
        (Mode::Block, Some(h)) => {
            let tool = h.detail.get("tool").and_then(Value::as_str).unwrap_or("tool");
            format!(
                "Your previous {tool} tool call had unusable arguments ({}). Retry with valid arguments: for write, path must be a real file path and content the full file text. Or reply in plain text.",
                reason.unwrap_or("unusable")
            )
        }
        (Mode::Block, None) => {
            "Your last reply was malformed. Respond with exactly one tool call in the correct format, or plain text.".to_owned()
        }
        (Mode::Stream, _) => format!(
            "Your previous tool call had unusable arguments ({}). Retry with valid arguments: for write, path must be a real file path and content the full file text. Or reply in plain text.",
            reason.unwrap_or("malformed")
        ),
    }
}

/// Extraction-first path fix, then a model re-ask of just the broken fields.
async fn field_repair(
    engine: &Engine,
    trace: &RequestTrace,
    prepared: &Prepared,
    parsed: &mut ParsedToolCalls,
    degenerate_hit: &mut Option<Hit>,
    pf: &PendingFix,
) {
    let tools_lines = &prepared.request.tools_lines;
    let root = engine.config.workspace_root.as_deref();
    let broken = parsed.calls[pf.idx].clone();
    let broken_args = arg_repair::parse_args(&broken.arguments);
    // Extraction reads the ORIGINAL conversation (role "tool" intact): the
    // effective messages fold tool output into role "user" <tool_result>
    // turns, which the task-selection heuristic would mistake for the task.
    let task = arg_repair::original_user_text(&prepared.request.messages);
    let task_head: String = task.chars().take(200).collect();
    let adopt = |parsed: &mut ParsedToolCalls,
                 degenerate_hit: &mut Option<Hit>,
                 merged: &JsonMap|
     -> bool {
        let mut fixed = parsed.calls.clone();
        fixed[pf.idx] = ToolCallSpec::new(
            broken.id.clone(),
            broken.name.clone(),
            arg_repair::args_json_string(merged),
        );
        if !calls_clean(&fixed, tools_lines, root) {
            return false;
        }
        *degenerate_hit = degenerate(&fixed);
        parsed.calls = fixed;
        true
    };

    // Extraction-first: a path-kind fix is answered deterministically from
    // the task text (it names the exact path; the model re-ask repeats its
    // hallucination). Fall back to the model re-ask only when extraction
    // gives no answer or the merged call fails the clean check.
    let mut adopted_by_task = false;
    if let Some(merged) = arg_repair::task_path_merge(&pf.kind, &pf.fields, &broken_args, &task) {
        if adopt(parsed, degenerate_hit, &merged) {
            trace.event(
                "repair",
                "ok",
                json!({"mode": "field", "source": "task", "kind": pf.kind, "fields": pf.fields, "task_head": task_head}),
            );
            adopted_by_task = true;
        } else {
            // Extracted a path but the call is still not clean — fall
            // through to the model re-ask, but say so.
            trace.event(
                "repair",
                "skip",
                json!({"mode": "field", "source": "task", "kind": pf.kind, "reason": "unclean_after_task_merge", "task_head": task_head}),
            );
        }
    } else if arg_repair::is_path_fix(&pf.kind, &pf.fields) {
        // Path-kind fix with no deterministic answer: record what was read
        // plus the candidate layout for the next unknown shape.
        let user_heads: Vec<String> = prepared
            .request
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .map(|m| m.content.chars().take(60).collect())
            .collect();
        trace.event(
            "repair",
            "skip",
            json!({
                "mode": "field", "source": "task", "kind": pf.kind, "reason": "no_task_path",
                "task_is_placeholder": task == NO_USER_TEXT, "task_head": task_head, "user_heads": user_heads,
            }),
        );
    }
    if adopted_by_task {
        return;
    }
    match field_repair_call(
        engine,
        tools_lines,
        &broken.name,
        &broken_args,
        &pf.fields,
        &pf.reasons,
        &task,
        &prepared.options,
    )
    .await
    {
        Ok(Some(merged)) => {
            if adopt(parsed, degenerate_hit, &merged) {
                trace.event(
                    "repair",
                    "ok",
                    json!({"mode": "field", "kind": pf.kind, "fields": pf.fields}),
                );
            } else {
                trace.event("repair", "skip", json!({"mode": "field", "kind": pf.kind, "reason": "unclean_after_model_merge"}));
            }
        }
        Ok(None) => {
            trace.event(
                "repair",
                "skip",
                json!({"mode": "field", "kind": pf.kind, "reason": "model_reply_unparsable"}),
            );
        }
        Err(e) => {
            trace.event(
                "repair",
                "error",
                json!({"mode": "field", "kind": pf.kind, "error": e.message}),
            );
        }
    }
}

/// Stage 2: re-ask the whole reply with a corrective nudge.
async fn whole_reply_repair(
    engine: &Engine,
    prepared: &Prepared,
    text: &str,
    nudge: &str,
    first_tokens: usize,
) -> Result<ParsedToolCalls, PipelineError> {
    let mut repair = prepared.effective.clone();
    repair.push(ChatMessage::new("assistant", text));
    repair.push(ChatMessage::new("user", nudge));
    let prompt = engine
        .backend
        .apply_chat_template(&repair)
        .await
        .map_err(|e| PipelineError::new(500, format!("chat template failed: {e}")))?;
    let budget = prepared
        .options
        .max_tokens
        .saturating_sub(first_tokens)
        .max(8);
    let options = prepared.options.with_max_tokens(budget);
    let extra = engine
        .generate(GenerateRequest {
            prompt,
            options: options.clone(),
            grammar: prepared.grammar.clone(),
        })
        .await
        .map_err(|e| PipelineError::new(500, format!("generation failed: {e}")))?;
    let repair_text = decode_text(engine, &extra.tokens, &options).await?;
    Ok(tool_markup::parse_pseudo_tool_calls(&repair_text))
}

// MARK: - delivery

fn generation_error(e: &BackendError) -> PipelineError {
    PipelineError::new(500, format!("generation failed: {e}"))
}

/// Non-stream chat completion: the full JSON response body.
pub async fn chat_block(
    engine: &Engine,
    prepared: &Prepared,
    trace: &RequestTrace,
) -> Result<Value, PipelineError> {
    let completion = engine
        .generate(GenerateRequest {
            prompt: prepared.prompt.clone(),
            options: prepared.options.clone(),
            grammar: prepared.grammar.clone(),
        })
        .await
        .map_err(|e| generation_error(&e))?;
    let turn = resolve_turn(engine, trace, prepared, Mode::Block, &completion).await?;
    let (prompt_tokens, completion_tokens) = (prepared.prompt.len(), completion.tokens.len());
    trace.event(
        "decode",
        "ok",
        json!({
            "finish": turn.finish,
            "tokens": completion_tokens,
            "tool_calls": turn.tool_calls.len(),
            "degenerate": turn.had_empty_name,
        }),
    );
    let body = openai_compat::non_stream_json(
        &engine.config.model_id,
        &turn.content,
        &turn.finish,
        i64::try_from(prompt_tokens).unwrap_or(i64::MAX),
        i64::try_from(completion_tokens).unwrap_or(i64::MAX),
        &turn.tool_calls,
        COMPLETION_ID,
        openai_compat::now_unix(),
    );
    trace.event(
        "present",
        "ok",
        json!({
            "tool_calls": turn.tool_calls.len(),
            "tool_args": tool_args_head(&turn.tool_calls),
            "content_head": head(&turn.content, 300),
            "usage": {"prompt": prompt_tokens, "completion": completion_tokens},
        }),
    );
    Ok(body)
}

fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn tool_args_head(calls: &[ToolCallSpec]) -> Vec<Value> {
    calls
        .iter()
        .map(|c| json!({"name": c.name, "arguments": head(&c.arguments, 300)}))
        .collect()
}

fn chunk(model: &str, delta: Value, finish: Value) -> Value {
    json!({
        "id": COMPLETION_ID,
        "object": "chat.completion.chunk",
        "created": openai_compat::now_unix(),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

/// Streaming chat completion: yields the JSON payload of every SSE `data:`
/// line, then the literal `[DONE]`. A generation error ends the stream
/// silently (trace marked failed), like the Swift server.
pub fn chat_stream(
    engine: Arc<Engine>,
    prepared: Prepared,
    trace: Arc<RequestTrace>,
) -> BoxStream<'static, String> {
    async_stream::stream! {
        let model = engine.config.model_id.clone();
        let tools_attached = !prepared.request.tools_lines.is_empty();
        trace.event("route", "ok", json!({"mode": "stream"}));

        let permit = engine.acquire().await;
        let mut upstream = engine.backend.generate(GenerateRequest {
            prompt: prepared.prompt.clone(),
            options: prepared.options.clone(),
            grammar: prepared.grammar.clone(),
        });
        let mut completion = Completion::default();
        let mut failure: Option<BackendError> = None;
        while let Some(item) = upstream.next().await {
            match item {
                Ok(c) => {
                    completion.tokens.extend_from_slice(&c.tokens);
                    completion.text.push_str(&c.text);
                    if !tools_attached && !c.text.is_empty() {
                        yield chunk(&model, json!({"content": c.text}), Value::Null).to_string();
                    }
                }
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        drop(upstream);
        drop(permit);
        if let Some(e) = failure {
            trace.fail(&format!("stream generation failed: {e}"));
            engine.record_trace(&trace);
            return;
        }

        let (prompt_tokens, completion_tokens) = (prepared.prompt.len(), completion.tokens.len());
        let mut finish = openai_compat::finish_reason(
            &completion.tokens,
            i64::try_from(prepared.options.max_tokens).unwrap_or(i64::MAX),
            &prepared.options.eos_token_ids,
        )
        .to_owned();
        let mut tool_call_count = 0;
        let mut tool_args: Vec<Value> = Vec::new();
        let mut content_head = String::new();
        let mut raw_head = String::new();

        if tools_attached {
            match resolve_turn(&engine, &trace, &prepared, Mode::Stream, &completion).await {
                Ok(turn) => {
                    raw_head = head(&turn.raw, 400);
                    content_head = turn.content.clone();
                    finish = turn.finish.clone();
                    if !turn.content.is_empty() {
                        yield chunk(&model, json!({"content": turn.content}), Value::Null).to_string();
                    }
                    if !turn.tool_calls.is_empty() {
                        tool_call_count = turn.tool_calls.len();
                        tool_args = tool_args_head(&turn.tool_calls);
                        let deltas: Vec<Value> = turn
                            .tool_calls
                            .iter()
                            .enumerate()
                            .map(|(i, c)| json!({
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments},
                                // OpenAI streams key tool-call deltas by position.
                                "index": i,
                            }))
                            .collect();
                        yield chunk(&model, json!({"tool_calls": deltas}), Value::Null).to_string();
                    }
                }
                Err(e) => {
                    trace.fail(&format!("stream generation failed: {}", e.message));
                    engine.record_trace(&trace);
                    return;
                }
            }
        } else {
            // Non-tools stream: verify the assembled content too.
            let options = &prepared.options;
            if let Ok((raw, text)) = decode_raw_and_text(&engine, &completion.tokens, options).await {
                let outcome = response_verify::verify(&response_verify::Message {
                    content: Some(text),
                    reasoning_content: response_verify::think_interior(&raw),
                    ..response_verify::Message::default()
                });
                let hits: Vec<Value> = outcome.hits.iter().map(Hit::event_detail).collect();
                trace.event("verify", if outcome.hits.is_empty() { "ok" } else { "hit" }, json!({"hits": hits}));
            }
        }

        yield chunk(&model, json!({}), Value::String(finish.clone())).to_string();
        // OpenAI spec: with stream_options.include_usage the tail chunk carries
        // usage with empty choices (DSH reads chunk.usage for its counters).
        if prepared.request.stream_options_include_usage {
            yield json!({
                "id": COMPLETION_ID,
                "object": "chat.completion.chunk",
                "created": openai_compat::now_unix(),
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
            .to_string();
        }
        yield "[DONE]".to_owned();
        trace.event(
            "present",
            "ok",
            json!({
                "mode": "stream",
                "finish": finish,
                "tokens": completion_tokens,
                "tool_calls": tool_call_count,
                "tool_args": tool_args,
                "content_head": head(&content_head, 300),
                "raw_head": raw_head,
                "usage": {"prompt": prompt_tokens, "completion": completion_tokens},
            }),
        );
        engine.record_trace(&trace);
    }
    .boxed()
}
