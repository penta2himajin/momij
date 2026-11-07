//! Per-request structured trace (lightweight compositor parity). Shared by
//! reference across task boundaries, so interior mutability via a lock.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use momij_core::trace_store::TraceStore;
use serde_json::{Value, json};

/// Seconds since the Unix epoch (evprtr records `time.time()`; the Swift
/// original used `CFAbsoluteTime`, which counts from 2001).
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

#[derive(Debug)]
struct Inner {
    ok: bool,
    locus: String,
    error: Option<String>,
    events: Vec<Value>,
}

#[derive(Debug)]
pub struct RequestTrace {
    pub id: String,
    started_at: f64,
    inner: Mutex<Inner>,
}

impl Default for RequestTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestTrace {
    pub fn new() -> Self {
        let trace = Self {
            id: TraceStore::new_id(),
            started_at: now(),
            inner: Mutex::new(Inner {
                ok: true,
                locus: "none".into(),
                error: None,
                events: Vec::new(),
            }),
        };
        trace.event("accept", "ok", json!({}));
        trace
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn event(&self, stage: &str, status: &str, detail: Value) {
        self.lock()
            .events
            .push(json!({"stage": stage, "status": status, "at": now(), "detail": detail}));
    }

    pub fn fail(&self, message: &str) {
        let mut g = self.lock();
        g.ok = false;
        g.locus = "upstream".into();
        g.error = Some(message.to_owned());
        g.events.push(json!({"stage": "fail", "status": "error", "at": now(), "detail": {"message": message}}));
    }

    pub fn locus(&self) -> String {
        self.lock().locus.clone()
    }

    pub fn record(&self) -> Value {
        let g = self.lock();
        json!({
            "trace_id": self.id,
            "started_at": self.started_at,
            "finished_at": now(),
            "ok": g.ok,
            "locus": g.locus,
            "error": g.error,
            "events": g.events,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_an_accept_event_and_records_ok() {
        let t = RequestTrace::new();
        t.event("parse", "ok", json!({"messages": 2}));
        let r = t.record();
        assert!(t.id.starts_with("tr-"));
        assert_eq!(r["trace_id"], t.id);
        assert_eq!(r["ok"], true);
        assert_eq!(r["locus"], "none");
        assert!(r["error"].is_null());
        let events = r["events"].as_array().unwrap();
        assert_eq!(events[0]["stage"], "accept");
        assert_eq!(events[1]["detail"]["messages"], 2);
        assert!(r["finished_at"].as_f64().unwrap() >= r["started_at"].as_f64().unwrap());
    }

    #[test]
    fn fail_marks_upstream_and_keeps_the_message() {
        let t = RequestTrace::new();
        t.fail("boom");
        let r = t.record();
        assert_eq!(r["ok"], false);
        assert_eq!(r["locus"], "upstream");
        assert_eq!(r["error"], "boom");
        assert_eq!(t.locus(), "upstream");
        assert_eq!(
            r["events"].as_array().unwrap().last().unwrap()["stage"],
            "fail"
        );
    }

    #[test]
    fn timestamps_are_unix_epoch_seconds() {
        let r = RequestTrace::new().record();
        assert!(r["started_at"].as_f64().unwrap() > 1.7e9);
    }
}
