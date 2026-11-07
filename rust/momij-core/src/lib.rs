//! Engine core for momij. Rust port of `swift/Sources/MomijCore`, minus the
//! model-execution layer (Seedless Metal / MLX), which is replaced by an HTTP
//! backend talking to the official DeepGrove `llama.cpp` fork.

pub mod arg_repair;
pub mod chat_template_patch;
pub mod json;
pub mod loop_breaker;
pub mod openai_compat;
pub mod path_policy;
pub mod protocol_absorb;
pub mod structured_outputs;
pub mod token_recycle_index;
pub mod tool_markup;
pub mod trace_store;

/// Vocabulary token id (`-1` is used as "no token" by a few callers).
pub type TokenId = i32;

/// JSON object (`[String: Any]` in the Swift original).
pub type JsonMap = serde_json::Map<String, serde_json::Value>;
