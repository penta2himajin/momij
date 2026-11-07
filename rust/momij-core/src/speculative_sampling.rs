//! Speculative sampling (Leviathan et al., ICML 2023) over a model-free draft.
//! Port of `SpeculativeSampling.swift`.
//!
//! Default **deterministic** draft: `q(x*)=1` for the proposed token
//! (tree/PLD/recycle emit one chain). Optional per-token `draft_q` from
//! suffix-tree child frequencies.

use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::TokenId;
use crate::logits_processor::LogitsProcessor;

/// One candidate distribution (a row of target logits).
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub ids: Vec<TokenId>,
    pub logits: Vec<f32>,
}

/// Probability of `token` under prepared candidate logits (temperature
/// softmax, no nucleus).
pub fn prob_of(
    token: TokenId,
    ids: &[TokenId],
    logits: &[f32],
    processor: &LogitsProcessor,
) -> f32 {
    ids.iter()
        .position(|&t| t == token)
        .map_or(0.0, |idx| processor.probabilities(ids, logits)[idx])
}

/// Accept `draft_token` with min(1, p/q). On reject, sample residual
/// max(0, p-q). `draft_q` is the draft density on `draft_token` (1 =
/// deterministic proposal). Returns `(accepted, token)`.
pub fn try_accept(
    draft_token: TokenId,
    ids: &[TokenId],
    logits: &[f32],
    processor: &LogitsProcessor,
    rng: &mut impl Rng,
    draft_q: f32,
) -> (bool, TokenId) {
    let p = prob_of(draft_token, ids, logits, processor);
    let q = draft_q.max(0.0);
    if q > 0.0 && p > 0.0 {
        let u = rng.random::<f32>();
        if u <= (p / q).min(1.0) {
            return (true, draft_token);
        }
    }
    let mut mass = processor.probabilities(ids, logits);
    if let (true, Some(idx)) = (q > 0.0, ids.iter().position(|&t| t == draft_token)) {
        mass[idx] = (mass[idx] - q).max(0.0);
    }
    let sum: f32 = mass.iter().sum();
    if sum > 0.0 {
        let u = rng.random::<f32>() * sum;
        let mut acc = 0.0;
        for (i, m) in mass.iter().enumerate() {
            acc += m;
            if u <= acc {
                return (false, ids[i]);
            }
        }
        return (false, ids[ids.len() - 1]);
    }
    (false, processor.sample_prepared(ids, logits, rng))
}

/// Sequential Leviathan-style walk over **precomputed** target rows.
/// `rows[i]` is the candidate distribution after feeding `feeds[i]`
/// (`feeds = [y] + draft`). `rows.len()` must be `>= draft.len() + 1` (last
/// row = bonus). `draft_q[i]` is q on `draft[i]`; empty -> all 1.
/// Returns `(accepted, next)`.
///
/// # Panics
/// If `rows.len() < draft.len() + 1`.
pub fn walk_draft(
    draft: &[TokenId],
    rows: &[Row],
    processor: &LogitsProcessor,
    seen: &HashSet<TokenId>,
    counts: &HashMap<TokenId, usize>,
    rng: &mut impl Rng,
    draft_q: &[f32],
) -> (usize, TokenId) {
    assert!(rows.len() > draft.len(), "need draft.len() + 1 rows");
    let mut seen = seen.clone();
    let mut counts = counts.clone();
    for (i, &token) in draft.iter().enumerate() {
        let (ids, logits) = processor.prepare(&rows[i].ids, &rows[i].logits, &seen, &counts);
        let q = draft_q.get(i).copied().unwrap_or(1.0);
        let (accepted, picked) = try_accept(token, &ids, &logits, processor, rng, q);
        if !accepted {
            return (i, picked);
        }
        seen.insert(token);
        *counts.entry(token).or_default() += 1;
    }
    let bonus = &rows[draft.len()];
    let (ids, logits) = processor.prepare(&bonus.ids, &bonus.logits, &seen, &counts);
    (draft.len(), processor.sample_prepared(&ids, &logits, rng))
}

/// When to spend draft work on the sampled path. Cold 1-step is the floor;
/// probe periodically (1-forward early-exit); M-row batch only when accepts
/// land.
pub mod sampled_spec_policy {
    pub const PROBE_EVERY: usize = 8;
    pub const BATCH_MEAN: f64 = 1.5;
    pub const BATCH_HOLD_MEAN: f64 = 0.8;
    pub const BATCH_MIN_WINDOW: usize = 4;

    pub fn use_batch(mean_accept: f64, window_count: usize, currently_batching: bool) -> bool {
        if currently_batching {
            return window_count >= 2 && mean_accept >= BATCH_HOLD_MEAN;
        }
        window_count >= BATCH_MIN_WINDOW && mean_accept >= BATCH_MEAN
    }

    /// Short drafts: sampled Leviathan rarely copies 6-8 tokens, and long
    /// M-row plus restore is slower than 1-step. Cap well below greedy
    /// `adaptive_draft_k`.
    pub fn draft_k(mean_accept: f64, draft_k: usize) -> usize {
        let k = draft_k.max(1);
        if mean_accept >= 2.5 {
            k.min(4)
        } else if mean_accept >= 1.5 {
            k.min(3)
        } else {
            k.min(2)
        }
    }

    /// Draft/verify this step? `false` -> caller should step sampled only.
    /// After a probe lands, draft every step until the window is full so we
    /// can decide to batch; collapsed / not-hot returns to periodic probes.
    pub fn use_draft(
        generated: usize,
        mean_accept: f64,
        window_count: usize,
        probe_every: usize,
        currently_batching: bool,
    ) -> bool {
        if use_batch(mean_accept, window_count, currently_batching) {
            return true;
        }
        if (1..BATCH_MIN_WINDOW).contains(&window_count) && mean_accept >= 0.5 {
            return true;
        }
        probe_every > 0 && generated > 0 && generated.is_multiple_of(probe_every)
    }
}

#[cfg(test)]
mod tests {
    use super::sampled_spec_policy as policy;
    use super::*;
    use crate::logits_processor::SplitMix64;

    fn greedy() -> LogitsProcessor {
        LogitsProcessor::new(0.0, 1.0)
    }

    fn row(ids: &[TokenId], logits: &[f32]) -> Row {
        Row {
            ids: ids.to_vec(),
            logits: logits.to_vec(),
        }
    }

    #[test]
    fn accepts_when_p_dominates_uniform_q() {
        // Single candidate -> q=1, p=1 -> always accept
        let mut rng = SplitMix64::new(1);
        let proc = LogitsProcessor::new(1.0, 1.0);
        assert_eq!(try_accept(7, &[7], &[0.0], &proc, &mut rng, 1.0), (true, 7));
    }

    #[test]
    fn reject_resamples_from_target() {
        // draft token missing from support -> p=0 -> reject + sample
        let mut rng = SplitMix64::new(99);
        assert_eq!(
            try_accept(99, &[1, 2, 3], &[0.0, 5.0, 0.0], &greedy(), &mut rng, 1.0),
            (false, 2)
        );
    }

    /// Deterministic draft (q=1): low-p proposal is rejected; residual is the argmax.
    #[test]
    fn deterministic_q_rejects_low_prob_draft() {
        let mut rng = SplitMix64::new(1);
        assert_eq!(
            try_accept(1, &[1, 2, 3], &[0.0, 9.0, 0.0], &greedy(), &mut rng, 1.0),
            (false, 2)
        );
    }

    /// Tree-peaked q < p -> always accept (p/q >= 1).
    #[test]
    fn tree_q_accepts_when_p_at_least_q() {
        let mut rng = SplitMix64::new(1);
        assert_eq!(
            try_accept(2, &[1, 2, 3], &[0.0, 5.0, 0.0], &greedy(), &mut rng, 0.4),
            (true, 2)
        );
    }

    #[test]
    fn acceptance_rate_matches_p_over_q() {
        // p(token 1) = 0.25 under uniform-ish softmax([0, ln 3]); q = 1 -> accept ~25%.
        let proc = LogitsProcessor::new(1.0, 1.0);
        let logits = [0.0, 3f32.ln()];
        let mut rng = SplitMix64::new(11);
        let accepted = (0..4000)
            .filter(|_| try_accept(1, &[1, 2], &logits, &proc, &mut rng, 1.0).0)
            .count();
        let rate = accepted as f64 / 4000.0;
        assert!((rate - 0.25).abs() < 0.04, "{rate}");
    }

    #[test]
    fn walk_draft_greedy_full_accept() {
        // temp=0 + draft matches per-row argmax -> full accept, bonus from last row.
        let mut rng = SplitMix64::new(1);
        let rows = [
            row(&[10, 20, 30], &[0.0, 5.0, 1.0]),
            row(&[10, 21, 30], &[0.0, 9.0, 1.0]),
            row(&[11, 22, 30], &[0.0, 1.0, 8.0]),
        ];
        let r = walk_draft(
            &[20, 21],
            &rows,
            &greedy(),
            &HashSet::new(),
            &HashMap::new(),
            &mut rng,
            &[],
        );
        assert_eq!(r, (2, 30));
    }

    #[test]
    fn walk_draft_rejects_at_first_mismatch() {
        let mut rng = SplitMix64::new(1);
        let rows = [
            row(&[10, 20, 30], &[0.0, 5.0, 1.0]),
            row(&[10, 21, 30], &[0.0, 9.0, 1.0]),
            row(&[11, 22, 30], &[8.0, 1.0, 0.0]),
        ];
        let r = walk_draft(
            &[99, 21],
            &rows,
            &greedy(),
            &HashSet::new(),
            &HashMap::new(),
            &mut rng,
            &[],
        );
        assert_eq!(r, (0, 20));
    }

    #[test]
    fn walk_draft_partial_accept_then_reject() {
        let mut rng = SplitMix64::new(1);
        let rows = [
            row(&[10, 20, 30], &[0.0, 5.0, 1.0]),
            row(&[10, 21, 30], &[0.0, 9.0, 1.0]),
            row(&[11, 22, 30], &[8.0, 1.0, 0.0]),
        ];
        let r = walk_draft(
            &[20, 99],
            &rows,
            &greedy(),
            &HashSet::new(),
            &HashMap::new(),
            &mut rng,
            &[],
        );
        assert_eq!(r, (1, 21));
    }

    #[test]
    #[should_panic(expected = "need draft.len() + 1 rows")]
    fn walk_draft_requires_a_bonus_row() {
        let mut rng = SplitMix64::new(1);
        let rows = [row(&[1], &[0.0])];
        let _ = walk_draft(
            &[1],
            &rows,
            &greedy(),
            &HashSet::new(),
            &HashMap::new(),
            &mut rng,
            &[],
        );
    }

    #[test]
    fn sampled_spec_policy_stays_sequential_until_probe() {
        assert!(!policy::use_draft(0, 1.0, 0, policy::PROBE_EVERY, false));
        assert!(!policy::use_draft(3, 1.0, 0, policy::PROBE_EVERY, false));
        assert!(policy::use_draft(8, 1.0, 0, policy::PROBE_EVERY, false));
    }

    #[test]
    fn sampled_spec_policy_batches_only_when_hot() {
        assert!(!policy::use_batch(2.0, 2, false));
        assert!(!policy::use_batch(0.2, 8, false));
        assert!(!policy::use_batch(1.0, 4, false));
        assert!(policy::use_batch(1.5, 4, false));
        assert!(policy::use_draft(9, 1.5, 4, policy::PROBE_EVERY, false));
        assert!(policy::use_draft(9, 1.0, 1, policy::PROBE_EVERY, false));
        assert!(!policy::use_draft(9, 1.0, 4, policy::PROBE_EVERY, false));
        assert!(!policy::use_draft(9, 0.0, 1, policy::PROBE_EVERY, false));
        assert!(policy::use_batch(0.9, 4, true));
        assert!(!policy::use_batch(0.5, 4, true));
        assert_eq!(policy::draft_k(0.4, 8), 2);
        assert_eq!(policy::draft_k(1.6, 8), 3);
        assert_eq!(policy::draft_k(3.0, 8), 4);
    }
}
