//! OpenAI-compatible HTTP surface. Port of `makeRouter` / `runServe` in
//! `swift/Sources/momij/Server.swift`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::StreamExt;
use momij_core::openai_compat;
use serde_json::{Value, json};

use crate::pipeline::{self, Engine};
use crate::trace::RequestTrace;

/// Request bodies above this are refused (8 MB, like the Swift server).
pub const MAX_BODY_BYTES: usize = 8_000_000;

pub fn router(engine: Arc<Engine>) -> Router {
    // Authorization (if present) is ignored — no 401 for Bearer from evprtr.
    Router::new()
        .route("/v1/models", get(models))
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/traces", get(list_traces))
        .route("/v1/traces/{id}", get(get_trace))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(engine)
}

fn json_response(status: StatusCode, body: &Value) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

/// OpenAI-style error body, optionally tagged with the trace id and locus.
pub fn json_error(status: StatusCode, message: &str, trace: Option<&RequestTrace>) -> Response {
    let mut error = json!({"message": message, "type": "invalid_request_error"});
    if let Some(t) = trace {
        error["trace_id"] = json!(t.id);
        error["locus"] = json!(t.locus());
    }
    json_response(status, &json!({"error": error}))
}

fn fail(engine: &Engine, trace: &RequestTrace, status: u16, message: &str) -> Response {
    trace.fail(message);
    engine.record_trace(trace);
    json_error(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        message,
        Some(trace),
    )
}

async fn models(State(engine): State<Arc<Engine>>) -> Response {
    json_response(
        StatusCode::OK,
        &json!({
            "object": "list",
            "data": [{"id": engine.config.model_id, "object": "model", "created": 0, "owned_by": "momij"}],
        }),
    )
}

async fn list_traces(State(engine): State<Arc<Engine>>) -> Response {
    let store = engine.traces.clone();
    match tokio::task::spawn_blocking(move || store.list(20)).await {
        Ok(records) => json_response(StatusCode::OK, &json!({"object": "list", "data": records})),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "trace list failed", None),
    }
}

async fn get_trace(State(engine): State<Arc<Engine>>, Path(id): Path<String>) -> Response {
    let store = engine.traces.clone();
    let lookup = id.clone();
    match tokio::task::spawn_blocking(move || store.get(&lookup)).await {
        Ok(Some(record)) => json_response(StatusCode::OK, &record),
        Ok(None) => json_error(
            StatusCode::NOT_FOUND,
            &format!("trace not found: {id}"),
            None,
        ),
        Err(_) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "trace encode failed",
            None,
        ),
    }
}

async fn chat_completions(State(engine): State<Arc<Engine>>, body: Bytes) -> Response {
    let trace = Arc::new(RequestTrace::new());
    let request = match openai_compat::parse_request(&body) {
        Ok(r) => r,
        Err(e) => return fail(&engine, &trace, 400, &e.to_string()),
    };
    let prepared = match pipeline::prepare(&engine, request, &trace).await {
        Ok(p) => p,
        Err(e) => return fail(&engine, &trace, e.status, &e.message),
    };
    if prepared.request.stream {
        let events = pipeline::chat_stream(engine, prepared, trace)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        return Sse::new(events).into_response();
    }
    match pipeline::chat_block(&engine, &prepared, &trace).await {
        Ok(body) => {
            engine.record_trace(&trace);
            json_response(StatusCode::OK, &body)
        }
        Err(e) => fail(&engine, &trace, e.status, &e.message),
    }
}

/// Bind and serve until Ctrl-C.
pub async fn serve(engine: Arc<Engine>, host: &str, port: u16) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    eprintln!(
        "[momij] listening on http://{host}:{port} model={}",
        engine.config.model_id
    );
    axum::serve(listener, router(engine))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}
