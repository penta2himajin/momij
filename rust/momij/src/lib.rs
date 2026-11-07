//! momij: OpenAI-compatible server (and CLI) in front of the official
//! DeepGrove `llama.cpp` server. Rust port of `swift/Sources/momij`.

pub mod backend;
pub mod llama;
pub mod pipeline;
pub mod server;
pub mod trace;

#[cfg(test)]
mod testing;

#[cfg(test)]
mod pipeline_tests;
