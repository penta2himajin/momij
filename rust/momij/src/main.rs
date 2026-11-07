//! momij CLI: `serve` (OpenAI-compatible server), `generate` and `bench`,
//! all backed by the official DeepGrove llama.cpp server.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use futures_util::StreamExt;
use momij::backend::{Backend, GenerateRequest};
use momij::llama::LlamaServerBackend;
use momij::pipeline::{Engine, ServeConfig};
use momij::server;
use momij_core::chat_template_patch;
use momij_core::generate_options::GenerateOptions;
use momij_core::trace_store::TraceStore;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "momij",
    about = "momij — Maple-Preview high-speed inference (oMLX replacement)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct Llama {
    /// Base URL of the DeepGrove llama-server (`llama-server --jinja ...`).
    #[arg(long, env = "MOMIJ_LLAMA_URL", default_value = "http://127.0.0.1:8080")]
    llama_url: String,
}

#[derive(Subcommand)]
enum Command {
    /// Run the OpenAI-compatible server.
    Serve {
        #[command(flatten)]
        llama: Llama,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8742)]
        port: u16,
        /// Model id reported by /v1/models and in responses.
        #[arg(long, default_value = "maple-preview")]
        model_id: String,
    },
    /// Generate from a raw prompt (no chat template) and stream the text.
    Generate {
        #[command(flatten)]
        llama: Llama,
        #[arg(long, default_value = "Write a haiku about a maple grove.")]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long, default_value_t = 1.0)]
        top_p: f64,
        #[arg(long, default_value_t = 0.0)]
        presence_penalty: f64,
        #[arg(long, default_value_t = 0.0)]
        frequency_penalty: f64,
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f64,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Measure prefill / decode throughput of the llama-server.
    Bench {
        #[command(flatten)]
        llama: Llama,
        /// Prompt tokens.
        #[arg(short = 'p', long = "prompt-tokens", default_value_t = 128)]
        prompt_tokens: usize,
        /// Generated tokens.
        #[arg(short = 'g', long = "generation-tokens", default_value_t = 256)]
        generation_tokens: usize,
        /// Trials (reported as mean ± std).
        #[arg(short = 'n', long = "num-trials", default_value_t = 3)]
        num_trials: usize,
    },
}

fn trace_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("MOMIJ_TRACE_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".momij").join("traces")
}

async fn connect(llama: &Llama) -> Result<LlamaServerBackend, String> {
    LlamaServerBackend::connect(&llama.llama_url, chat_template_patch::enable_thinking())
        .await
        .map_err(|e| format!("cannot reach llama-server at {}: {e}", llama.llama_url))
}

async fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Serve {
            llama,
            host,
            port,
            model_id,
        } => {
            let backend = connect(&llama).await?;
            eprintln!(
                "[momij] llama-server {} n_ctx={:?}",
                llama.llama_url,
                backend.n_ctx()
            );
            let engine = Engine::new(
                Arc::new(backend),
                ServeConfig::from_env(model_id),
                TraceStore::new(trace_dir()),
            );
            server::serve(engine, &host, port)
                .await
                .map_err(|e| format!("server error: {e}"))
        }
        Command::Generate {
            llama,
            prompt,
            max_tokens,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            seed,
        } => {
            let backend = connect(&llama).await?;
            let ids = backend
                .tokenize(&prompt, true)
                .await
                .map_err(|e| e.to_string())?;
            let options = GenerateOptions {
                max_tokens,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
                repetition_penalty,
                seed,
                banned_token_ids: chat_template_patch::banned_assistant_token_ids(),
                ..GenerateOptions::default()
            };
            let started = Instant::now();
            let mut stream = backend.generate(GenerateRequest {
                prompt: ids,
                options,
                grammar: None,
            });
            let (mut tokens, mut stdout) = (0usize, std::io::stdout());
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| e.to_string())?;
                tokens += chunk.tokens.len();
                let _ = write!(stdout, "{}", chunk.text);
                let _ = stdout.flush();
            }
            let secs = started.elapsed().as_secs_f64().max(1e-9);
            println!(
                "\n[momij] tokens={tokens}  tok/s={:.1}",
                tokens as f64 / secs
            );
            Ok(())
        }
        Command::Bench {
            llama,
            prompt_tokens,
            generation_tokens,
            num_trials,
        } => {
            if prompt_tokens == 0 || generation_tokens == 0 || num_trials == 0 {
                return Err("-p, -g and -n must be positive".into());
            }
            let backend = connect(&llama).await?;
            let seed_ids = backend
                .tokenize("The quick brown fox jumps over the lazy dog. ", false)
                .await
                .map_err(|e| e.to_string())?;
            let prompt: Vec<_> = seed_ids
                .iter()
                .copied()
                .cycle()
                .take(prompt_tokens)
                .collect();
            let (mut prefill, mut decode) = (Vec::new(), Vec::new());
            for trial in 1..=num_trials {
                let t = backend
                    .bench_once(&prompt, generation_tokens)
                    .await
                    .map_err(|e| e.to_string())?;
                eprintln!(
                    "[momij] trial {trial}: prompt_tps={:.3} generation_tps={:.3} ({} prompt / {} generated)",
                    t.prompt_tps, t.predicted_tps, t.prompt_tokens, t.predicted_tokens
                );
                prefill.push(t.prompt_tps);
                decode.push(t.predicted_tps);
            }
            let (pm, ps) = mean_std(&prefill);
            let (dm, ds) = mean_std(&decode);
            println!(
                "backend=llama-server prompt_tokens={prompt_tokens} generation_tokens={generation_tokens} trials={num_trials} \
                 prompt_tps={pm:.3} (±{ps:.3}) generation_tps={dm:.3} (±{ds:.3})"
            );
            Ok(())
        }
    }
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_population_std() {
        let (m, s) = mean_std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert_eq!((m, s), (5.0, 2.0));
        assert_eq!(mean_std(&[3.0]), (3.0, 0.0));
    }

    #[test]
    fn cli_parses_the_documented_commands() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        let cli =
            Cli::try_parse_from(["momij", "bench", "-p", "128", "-g", "256", "-n", "5"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Bench {
                prompt_tokens: 128,
                generation_tokens: 256,
                num_trials: 5,
                ..
            }
        ));
        let cli = Cli::try_parse_from([
            "momij",
            "serve",
            "--port",
            "9000",
            "--llama-url",
            "http://h:1",
        ])
        .unwrap();
        assert!(
            matches!(&cli.command, Command::Serve { port: 9000, llama, .. } if llama.llama_url == "http://h:1")
        );
        assert!(Cli::try_parse_from(["momij", "nope"]).is_err());
    }
}
