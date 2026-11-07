//! Generation options shared by every backend. Port of `GenerateOptions` and
//! the engine-independent env helpers in `LLMBackend.swift` (the Seedless
//! routing table, KV-capacity knobs and concrete backends belong to the
//! Metal/MLX execution layer and are not ported).

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use crate::TokenId;
use crate::openai_compat::DEFAULT_EOS_TOKEN_IDS;

/// Finite-string / grammar frontier: allowed next token ids given the prefix.
pub type AllowedNext = Arc<dyn Fn(&[TokenId]) -> HashSet<TokenId> + Send + Sync>;

#[derive(Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    /// OpenAI presence_penalty (-2...2).
    pub presence_penalty: f64,
    /// OpenAI frequency_penalty (-2...2).
    pub frequency_penalty: f64,
    /// HF / vLLM-style; 1.0 = off. Values >1 penalize repeats.
    pub repetition_penalty: f64,
    pub eos_token_ids: Vec<TokenId>,
    pub use_suffix_spec: bool,
    pub draft_k: usize,
    /// Optional RNG seed for sampled decode (`None` -> nondeterministic).
    pub seed: Option<u64>,
    pub allowed_next: Option<AllowedNext>,
    /// Token ids never sampled (e.g. `<think>` when thinking is disabled).
    pub banned_token_ids: Vec<TokenId>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            eos_token_ids: DEFAULT_EOS_TOKEN_IDS.to_vec(),
            use_suffix_spec: false,
            draft_k: 8,
            seed: None,
            allowed_next: None,
            banned_token_ids: Vec::new(),
        }
    }
}

impl fmt::Debug for GenerateOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenerateOptions")
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("presence_penalty", &self.presence_penalty)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("eos_token_ids", &self.eos_token_ids)
            .field("use_suffix_spec", &self.use_suffix_spec)
            .field("draft_k", &self.draft_k)
            .field("seed", &self.seed)
            .field("allowed_next", &self.allowed_next.as_ref().map(|_| "<fn>"))
            .field("banned_token_ids", &self.banned_token_ids)
            .finish()
    }
}

impl GenerateOptions {
    /// Copy with a replaced `max_tokens` (repair retries use the remaining budget).
    #[must_use]
    pub fn with_max_tokens(&self, budget: usize) -> Self {
        Self {
            max_tokens: budget,
            ..self.clone()
        }
    }

    /// Copy with a grammar guide (structured / constrained re-ask).
    #[must_use]
    pub fn with_allowed_next(&self, guide: Option<AllowedNext>) -> Self {
        Self {
            allowed_next: guide,
            ..self.clone()
        }
    }

    /// Fast SuffixSpec / M-row greedy path is valid only for deterministic greedy.
    pub fn is_greedy_compatible(&self) -> bool {
        self.temperature <= 1e-5
            && self.top_p >= 1.0 - 1e-6
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && (self.repetition_penalty - 1.0).abs() < 1e-6
    }
}

/// Default-on flags: unset or any value except `"0"`.
pub fn env_on_by_default(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Default-off flags: only `"1"` enables.
pub fn env_off_by_default(value: Option<&str>) -> bool {
    value == Some("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_compatible_requires_default_sampling() {
        assert!(GenerateOptions::default().is_greedy_compatible());
        let with = |f: fn(&mut GenerateOptions)| {
            let mut o = GenerateOptions::default();
            f(&mut o);
            o.is_greedy_compatible()
        };
        assert!(!with(|o| o.temperature = 0.7));
        assert!(!with(|o| o.top_p = 0.9));
        assert!(!with(|o| o.presence_penalty = 0.1));
        assert!(!with(|o| o.frequency_penalty = 0.1));
        assert!(!with(|o| o.repetition_penalty = 1.1));
    }

    #[test]
    fn routing_greedy_uses_suffix_flag_only_when_compatible() {
        let sampled = GenerateOptions {
            temperature: 0.8,
            use_suffix_spec: true,
            ..GenerateOptions::default()
        };
        assert!(!sampled.is_greedy_compatible());
        let greedy = GenerateOptions {
            temperature: 0.0,
            use_suffix_spec: true,
            ..GenerateOptions::default()
        };
        assert!(greedy.is_greedy_compatible());
    }

    #[test]
    fn env_flag_defaults() {
        assert!(env_on_by_default(None));
        assert!(env_on_by_default(Some("1")));
        assert!(!env_on_by_default(Some("0")));
        assert!(!env_off_by_default(None));
        assert!(env_off_by_default(Some("1")));
        assert!(!env_off_by_default(Some("true")));
    }

    #[test]
    fn with_helpers_copy_and_replace() {
        let base = GenerateOptions {
            max_tokens: 100,
            seed: Some(3),
            ..GenerateOptions::default()
        };
        let shorter = base.with_max_tokens(10);
        assert_eq!((shorter.max_tokens, shorter.seed), (10, Some(3)));
        assert_eq!(base.max_tokens, 100);
        let guide: AllowedNext = Arc::new(|_| HashSet::from([1]));
        let guided = base.with_allowed_next(Some(guide));
        assert_eq!(
            guided.allowed_next.as_ref().map(|g| g(&[])),
            Some(HashSet::from([1]))
        );
        assert!(format!("{guided:?}").contains("<fn>"));
        assert!(base.allowed_next.is_none());
    }

    #[test]
    fn defaults_match_swift() {
        let o = GenerateOptions::default();
        assert_eq!((o.max_tokens, o.draft_k), (256, 8));
        assert_eq!(o.eos_token_ids, [151_645]);
    }
}
