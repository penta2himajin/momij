# Native trace store (compositor stage 3)

Branch `agent/traces` (cut from `agent/native-tools`). Lightweight
structured tracing for the OpenAI server, ported from the evprtr
compositor TraceStore concept and kept minimal.

## Contract

- One JSON record per request, written **fire-and-forget after the
  response** (`Task { ... }`): tracing can never fail or delay a request.
- Store: `MomijCore.TraceStore` — `write(_ record:, id:)` (atomic
  tmp+move), `list(limit:)` newest-first by mtime, `get(id:)`.
- Directory: `MOMIJ_TRACE_DIR` env override, default `~/.momij/traces`.
- Record shape: `trace_id`, `started_at`, `finished_at`, `ok`, `locus`,
  `error`, `events[{stage, status, at, detail}]`.
- Event stages: `accept` (request received) → `parse` (request decoded,
  message/tool counts) → `route` (stream marker) → `decode` (tokens,
  finish reason, tool-call count, degenerate flag) → `present`
  (usage, streamed tool calls) or `fail` (error message, `locus=upstream`).

## Endpoints

- `GET /v1/traces` — newest 20 records (`{"object":"list","data":[...]}`).
- `GET /v1/traces/:id` — single record; 404 if unknown.
- Every error body embeds `"trace_id"` + `"locus"` alongside the OpenAI
  error object, so a failing client can hand back a concrete trace id.

## Error coverage

All server error paths route through `fail(...)` (records + embeds
trace_id): invalid JSON / parse errors, `n>1`, structured-output
unsupported/compile failure, chat-template failure, generation failure,
decode failure, response encode failure. Streaming requests are recorded
after `[DONE]` (or on generation failure inside the stream task).

## Implementation notes

- `RequestTrace` is a locked final class (`@unchecked Sendable`) so it
  can be captured across task boundaries under Swift 6 concurrency.
- Stream trace writes happen inside the stream task directly; the shared
  `TraceStore` (NSLock-guarded) is created once in `makeRouter`.

## Verification

- Unit: `TraceStoreTests` (write/list/get roundtrip, newest-first, miss)
  and `ToolMarkupTests` — 10 green.
- Full suite: 117 tests, 0 failures.
- Live smoke on seedless backend (real Maple model): invalid JSON → 400
  with `trace_id` + trace persisted; tiny completion → 200 with
  accept/parse/decode/present events; stream completion → trace written
  after `[DONE]` with `mode=stream`; `/v1/traces` + `/v1/traces/:id`
  return the records.