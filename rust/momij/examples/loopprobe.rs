//! loopprobe: how often does Maple loop or hit the token limit, per sampler
//! condition, on a set of chat requests?
//!
//! Each input line is an OpenAI chat-completions body (optionally with
//! `tools`) plus optional `id` / `category` keys. Prompts are prepared by the
//! production pipeline (tool markup, chat template, non-thinking guards), so
//! what is measured is what `momij serve` would send. One JSON row per run is
//! appended to `--out`; progress goes to stderr.
//!
//! ```text
//! cargo run --release -p momij --example loopprobe -- \
//!   --llama-url http://127.0.0.1:8080 --prompts prompts.jsonl --out runs.jsonl \
//!   [--conditions greedy,official,dry,wsp,momij_sampled] [--samples 3] \
//!   [--max-tokens 400] [--thinking] [--no-cache] [--skip-done] [--only id1,id2]
//! ```

use std::collections::HashSet;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use momij::backend::GenerateRequest;
use momij::llama::{LlamaServerBackend, SseDecoder, completion_body, parse_stream_event};
use momij::pipeline::{self, Engine, Prepared, ServeConfig};
use momij::trace::RequestTrace;
use momij_core::trace_store::TraceStore;
use momij_core::{TokenId, chat_template_patch, openai_compat, response_verify, tool_markup};
use serde_json::{Value, json};

struct Condition {
    name: &'static str,
    temperature: f64,
    top_p: f64,
    presence_penalty: f64,
    /// Extra `/completion` fields on top of momij's own body.
    extra: Value,
}

fn condition(name: &str) -> Option<Condition> {
    let c = |name, temperature, top_p, presence_penalty, extra| {
        Some(Condition {
            name,
            temperature,
            top_p,
            presence_penalty,
            extra,
        })
    };
    match name {
        // What momij did before: deterministic.
        "greedy" => c("greedy", 0.0, 1.0, 0.0, json!({})),
        // The fork README's command: --temp 1.0 --top-p 0.95 with llama.cpp's
        // default top_k 40 / min_p 0.05 left on.
        "official" => c(
            "official",
            1.0,
            0.95,
            0.0,
            json!({"top_k": 40, "min_p": 0.05}),
        ),
        // Same temperature/top_p, but with momij's own top_k 0 / min_p 0 override.
        "momij_sampled" => c("momij_sampled", 1.0, 0.95, 0.0, json!({})),
        "dry" => c(
            "dry",
            0.0,
            1.0,
            0.0,
            json!({"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": 4096}),
        ),
        // The Weight-Space Programming evaluation settings.
        "wsp" => c("wsp", 0.6, 0.95, 1.5, json!({"top_k": 20})),
        _ => None,
    }
}

struct Args {
    llama_url: String,
    prompts: String,
    out: String,
    conditions: Vec<String>,
    samples: usize,
    max_tokens: usize,
    thinking: bool,
    /// Send `cache_prompt: false` so a run never depends on what the slot cached before.
    no_cache: bool,
    /// Skip (id, condition, sample) rows already in `--out`, so a restarted run resumes.
    skip_done: bool,
    only: Option<HashSet<String>>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        llama_url: "http://127.0.0.1:8080".into(),
        prompts: String::new(),
        out: "loopprobe.jsonl".into(),
        conditions: vec!["greedy".into()],
        samples: 1,
        max_tokens: 400,
        thinking: false,
        no_cache: false,
        skip_done: false,
        only: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--llama-url" => a.llama_url = val()?,
            "--prompts" => a.prompts = val()?,
            "--out" => a.out = val()?,
            "--conditions" => a.conditions = val()?.split(',').map(str::to_owned).collect(),
            "--samples" => a.samples = val()?.parse().map_err(|e| format!("--samples: {e}"))?,
            "--max-tokens" => {
                a.max_tokens = val()?.parse().map_err(|e| format!("--max-tokens: {e}"))?
            }
            "--only" => a.only = Some(val()?.split(',').map(str::to_owned).collect()),
            "--thinking" => a.thinking = true,
            "--no-cache" => a.no_cache = true,
            "--skip-done" => a.skip_done = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.prompts.is_empty() {
        return Err("--prompts FILE is required".into());
    }
    for c in &a.conditions {
        if condition(c).is_none() {
            return Err(format!(
                "unknown condition {c} (greedy, official, momij_sampled, dry, wsp)"
            ));
        }
    }
    Ok(a)
}

async fn stream_tokens(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
) -> Result<Vec<TokenId>, String> {
    let resp = client
        .post(format!("{url}/completion"))
        .json(body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!(
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }
    let (mut tokens, mut sse, mut bytes) = (Vec::new(), SseDecoder::default(), resp.bytes_stream());
    'outer: while let Some(piece) = bytes.next().await {
        for data in sse.push(&piece.map_err(|e| e.to_string())?) {
            let ev = parse_stream_event(&data).map_err(|e| e.to_string())?;
            tokens.extend(ev.chunk.tokens);
            if ev.stop {
                break 'outer;
            }
        }
    }
    Ok(tokens)
}

/// Share of repeated 4-grams among all 4-grams of `tokens` (0 = all distinct,
/// near 1 = the same few 4-grams over and over). Needs no text detector, so it
/// also sees line-/number-separated repetition and long periods.
fn rep4(tokens: &[TokenId]) -> f64 {
    let grams: Vec<&[TokenId]> = tokens.windows(4).collect();
    if grams.is_empty() {
        return 0.0;
    }
    let distinct: HashSet<&[TokenId]> = grams.iter().copied().collect();
    1.0 - distinct.len() as f64 / grams.len() as f64
}

fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn tail(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

#[allow(clippy::too_many_arguments)]
async fn run_one(
    engine: &Engine,
    client: &reqwest::Client,
    url: &str,
    prepared: &Prepared,
    cond: &Condition,
    sample: usize,
    max_tokens: usize,
    no_cache: bool,
) -> Result<Value, String> {
    let mut options = prepared.options.clone();
    options.max_tokens = max_tokens;
    options.temperature = cond.temperature;
    options.top_p = cond.top_p;
    options.presence_penalty = cond.presence_penalty;
    options.seed = Some(1000 + sample as u64);
    let mut body = completion_body(&GenerateRequest {
        prompt: prepared.prompt.clone(),
        options: options.clone(),
        grammar: prepared.grammar.clone(),
    });
    if let (Some(map), Some(extra)) = (body.as_object_mut(), cond.extra.as_object()) {
        map.extend(extra.clone());
    }
    if no_cache {
        body["cache_prompt"] = json!(false);
    }
    let started = Instant::now();
    let tokens = stream_tokens(client, url, &body).await?;
    let secs = started.elapsed().as_secs_f64();
    let eos = options.eos_token_ids[0];
    let finish = if tokens.last() == Some(&eos) {
        "eos"
    } else {
        "length"
    };
    let ids = openai_compat::content_token_ids(&tokens, &options.eos_token_ids);
    let rep4_all = rep4(&ids);
    let rep4_tail = rep4(&ids[ids.len().saturating_sub(200)..]);
    let raw = if ids.is_empty() {
        String::new()
    } else {
        engine
            .backend
            .detokenize(&ids)
            .await
            .map_err(|e| e.to_string())?
    };
    let text = chat_template_patch::strip_think_for_content(&raw, engine.config.thinking_enabled);
    let parsed = tool_markup::parse_pseudo_tool_calls(&text);
    let outcome = response_verify::verify(&response_verify::Message {
        content: Some(text.clone()),
        reasoning_content: response_verify::think_interior(&raw),
        tool_calls: parsed.calls.clone(),
    });
    let hits: Vec<String> = outcome
        .hits
        .iter()
        .map(|h| {
            format!(
                "{}:{}",
                h.policy_id.rsplit('.').next().unwrap_or(""),
                h.kind
            )
        })
        .collect();
    let looped = outcome
        .hits
        .iter()
        .any(|h| h.policy_id == response_verify::GENERIC_REPETITION_ID);
    let first_onset = outcome
        .hits
        .iter()
        .find(|h| h.policy_id == response_verify::GENERIC_REPETITION_ID)
        .map(|h| h.onset);
    Ok(json!({
        "condition": cond.name, "sample": sample, "tokens": tokens.len(), "finish": finish, "seconds": (secs * 10.0).round() / 10.0,
        "looped": looped, "loop_onset_chars": first_onset,
        "rep4_all": (rep4_all * 1000.0).round() / 1000.0, "rep4_tail": (rep4_tail * 1000.0).round() / 1000.0, "hits": hits,
        "tool_calls": parsed.calls.len(),
        "degenerate_tool_args": parsed.calls.iter().any(|c| c.name.is_empty()) || outcome.hits.iter().any(|h| h.kind == "degenerate_tool_args"),
        "chars": text.chars().count(), "text_head": head(&text, 160), "text_tail": tail(&text, 160),
    }))
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = parse_args()?;
    let backend = LlamaServerBackend::connect(&args.llama_url, args.thinking)
        .await
        .map_err(|e| e.to_string())?;
    let tmp = std::env::temp_dir().join(format!("loopprobe-traces-{}", std::process::id()));
    let mut cfg = ServeConfig::new("loopprobe");
    cfg.thinking_enabled = args.thinking;
    let engine = Engine::new(Arc::new(backend), cfg, TraceStore::new(tmp));
    let client = reqwest::Client::new();
    let done: HashSet<(String, String, u64)> = if args.skip_done {
        std::fs::read_to_string(&args.out)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter_map(|r| {
                Some((
                    r["id"].as_str()?.to_owned(),
                    r["condition"].as_str()?.to_owned(),
                    r["sample"].as_u64()?,
                ))
            })
            .collect()
    } else {
        HashSet::new()
    };
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)
        .map_err(|e| e.to_string())?;

    let text =
        std::fs::read_to_string(&args.prompts).map_err(|e| format!("{}: {e}", args.prompts))?;
    for (lineno, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let spec: Value =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", lineno + 1))?;
        let id = spec
            .get("id")
            .and_then(Value::as_str)
            .map_or_else(|| format!("p{}", lineno + 1), str::to_owned);
        if args.only.as_ref().is_some_and(|s| !s.contains(&id)) {
            continue;
        }
        let category = spec
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let request =
            openai_compat::parse_request(line.as_bytes()).map_err(|e| format!("{id}: {e}"))?;
        let prepared = match pipeline::prepare(&engine, request, &RequestTrace::new()).await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[loopprobe] {id}: skipped ({})", e.message);
                continue;
            }
        };
        for cname in &args.conditions {
            let cond = condition(cname).expect("validated");
            let samples = if cond.temperature == 0.0 && cond.extra.get("top_k").is_none() {
                1
            } else {
                args.samples
            };
            for s in 0..samples {
                if done.contains(&(id.clone(), cname.clone(), s as u64)) {
                    continue;
                }
                match run_one(
                    &engine,
                    &client,
                    &args.llama_url,
                    &prepared,
                    &cond,
                    s,
                    args.max_tokens,
                    args.no_cache,
                )
                .await
                {
                    Ok(mut row) => {
                        row["id"] = json!(id);
                        row["category"] = json!(category);
                        row["thinking"] = json!(args.thinking);
                        row["prompt_tokens"] = json!(prepared.prompt.len());
                        let _ = writeln!(out, "{row}");
                        let _ = out.flush();
                        eprintln!(
                            "[loopprobe] {id} {cname}#{s}: {} tok, {}, looped={}",
                            row["tokens"], row["finish"], row["looped"]
                        );
                    }
                    Err(e) => eprintln!("[loopprobe] {id} {cname}#{s}: ERROR {e}"),
                }
            }
        }
    }
    Ok(())
}
