//! Temperature / top-p / OpenAI-style penalties over a candidate logit set
//! (FlashHead probes or server top-k, not necessarily the full vocab).
//! Port of `LogitsProcessor.swift`.

use std::collections::{HashMap, HashSet};

use rand::{Rng, RngCore};

use crate::TokenId;
use crate::generate_options::GenerateOptions;

/// Deterministic RNG for sampling tests / reproducible decode.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }
}

impl RngCore for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitsProcessor {
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// HuggingFace-style; 1.0 = off. Values >1 penalize repeats.
    pub repetition_penalty: f32,
}

impl Default for LogitsProcessor {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Index of the first maximum (`-inf` start, strict compare: all `-inf` -> 0).
fn argmax_first(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

impl LogitsProcessor {
    pub fn new(temperature: f32, top_p: f32) -> Self {
        Self {
            temperature,
            top_p,
            ..Self::default()
        }
    }

    pub fn from_options(opts: &GenerateOptions) -> Self {
        Self {
            temperature: opts.temperature as f32,
            top_p: opts.top_p as f32,
            presence_penalty: opts.presence_penalty as f32,
            frequency_penalty: opts.frequency_penalty as f32,
            repetition_penalty: opts.repetition_penalty as f32,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-5
            && self.top_p >= 1.0 - 1e-6
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && (self.repetition_penalty - 1.0).abs() < 1e-6
    }

    pub fn apply_presence_penalty(
        logits: &mut [f32],
        token_ids: &[TokenId],
        seen: &HashSet<TokenId>,
        penalty: f32,
    ) {
        if penalty == 0.0 {
            return;
        }
        for (l, id) in logits.iter_mut().zip(token_ids) {
            if seen.contains(id) {
                *l -= penalty;
            }
        }
    }

    pub fn apply_frequency_penalty(
        logits: &mut [f32],
        token_ids: &[TokenId],
        counts: &HashMap<TokenId, usize>,
        penalty: f32,
    ) {
        if penalty == 0.0 {
            return;
        }
        for (l, id) in logits.iter_mut().zip(token_ids) {
            if let Some(&c) = counts.get(id).filter(|&&c| c > 0) {
                *l -= penalty * c as f32;
            }
        }
    }

    /// HF: score > 0 -> score/penalty; else score*penalty (penalty>=1 shrinks positives).
    pub fn apply_repetition_penalty(
        logits: &mut [f32],
        token_ids: &[TokenId],
        seen: &HashSet<TokenId>,
        penalty: f32,
    ) {
        if (penalty - 1.0).abs() <= 1e-6 || penalty <= 0.0 {
            return;
        }
        for (l, id) in logits.iter_mut().zip(token_ids) {
            if seen.contains(id) {
                if *l > 0.0 {
                    *l /= penalty;
                } else {
                    *l *= penalty;
                }
            }
        }
    }

    /// Apply repetition, presence and frequency penalties (in that order).
    ///
    /// # Panics
    /// If `token_ids` and `logits` differ in length or are empty.
    pub fn prepare(
        &self,
        token_ids: &[TokenId],
        logits: &[f32],
        seen: &HashSet<TokenId>,
        counts: &HashMap<TokenId, usize>,
    ) -> (Vec<TokenId>, Vec<f32>) {
        assert!(token_ids.len() == logits.len() && !token_ids.is_empty());
        let mut scores = logits.to_vec();
        Self::apply_repetition_penalty(&mut scores, token_ids, seen, self.repetition_penalty);
        Self::apply_presence_penalty(&mut scores, token_ids, seen, self.presence_penalty);
        Self::apply_frequency_penalty(&mut scores, token_ids, counts, self.frequency_penalty);
        (token_ids.to_vec(), scores)
    }

    /// Softmax -> optional nucleus -> multinomial. `temperature<=0` -> argmax.
    pub fn sample(
        &self,
        token_ids: &[TokenId],
        logits: &[f32],
        seen: &HashSet<TokenId>,
        counts: &HashMap<TokenId, usize>,
        rng: &mut impl Rng,
    ) -> TokenId {
        let (ids, scores) = self.prepare(token_ids, logits, seen, counts);
        self.sample_prepared(&ids, &scores, rng)
    }

    /// # Panics
    /// If `token_ids` and `logits` differ in length or are empty.
    pub fn sample_prepared(
        &self,
        token_ids: &[TokenId],
        logits: &[f32],
        rng: &mut impl Rng,
    ) -> TokenId {
        assert!(token_ids.len() == logits.len() && !token_ids.is_empty());
        if self.temperature <= 1e-5 {
            return token_ids[argmax_first(logits)];
        }
        let inv_t = 1.0 / self.temperature.max(1e-5);
        let scaled: Vec<f32> = logits.iter().map(|l| l * inv_t).collect();
        let max_l = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = scaled.iter().map(|s| (s - max_l).exp()).collect();
        let mut sum: f32 = exps.iter().sum();
        if sum <= 0.0 {
            return token_ids[0];
        }

        // Nucleus (top-p): keep highest-prob tokens until cumulative >= top_p.
        if self.top_p < 1.0 - 1e-6 {
            let mut order: Vec<usize> = (0..exps.len()).collect();
            order.sort_by(|&a, &b| exps[b].total_cmp(&exps[a]));
            let cutoff = self.top_p.clamp(0.0, 1.0) * sum;
            let mut cum = 0.0;
            let mut keep = HashSet::new();
            for &i in &order {
                keep.insert(i);
                cum += exps[i];
                if cum >= cutoff {
                    break;
                }
            }
            for (i, e) in exps.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    sum -= *e;
                    *e = 0.0;
                }
            }
            if sum <= 0.0 {
                return token_ids[order[0]];
            }
        }

        let u = rng.random::<f32>() * sum;
        let mut acc = 0.0;
        for (i, e) in exps.iter().enumerate() {
            acc += e;
            if u <= acc {
                return token_ids[i];
            }
        }
        token_ids[token_ids.len() - 1]
    }

    /// Softmax probabilities after temperature (no nucleus). For speculative sampling.
    pub fn probabilities(&self, token_ids: &[TokenId], logits: &[f32]) -> Vec<f32> {
        if self.temperature <= 1e-5 {
            let mut p = vec![0.0; logits.len()];
            if !logits.is_empty() {
                p[argmax_first(logits)] = 1.0;
            }
            return p;
        }
        let inv_t = 1.0 / self.temperature.max(1e-5);
        let scaled: Vec<f32> = logits.iter().map(|l| l * inv_t).collect();
        let max_l = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|s| (s - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 {
            return vec![1.0 / token_ids.len() as f32; token_ids.len()];
        }
        exps.iter().map(|e| e / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> (HashSet<TokenId>, HashMap<TokenId, usize>) {
        (HashSet::new(), HashMap::new())
    }

    #[test]
    fn temp_zero_is_argmax() {
        let p = LogitsProcessor::new(0.0, 1.0);
        let mut rng = SplitMix64::new(1);
        let (seen, counts) = none();
        assert_eq!(
            p.sample(&[10, 20, 30], &[1.0, 5.0, 2.0], &seen, &counts, &mut rng),
            20
        );
    }

    #[test]
    fn presence_penalty_lowers_seen_tokens() {
        let mut logits = [3.0f32, 3.0, 3.0];
        LogitsProcessor::apply_presence_penalty(&mut logits, &[1, 2, 3], &HashSet::from([2]), 2.0);
        assert_eq!(logits, [3.0, 1.0, 3.0]);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        let mut logits = [0.0f32, 0.0, 0.0];
        LogitsProcessor::apply_frequency_penalty(
            &mut logits,
            &[7, 8, 9],
            &HashMap::from([(8, 3)]),
            0.5,
        );
        assert_eq!(logits[1], -1.5);
        assert_eq!((logits[0], logits[2]), (0.0, 0.0));
    }

    #[test]
    fn repetition_penalty_shrinks_positive_logits() {
        let mut logits = [4.0f32, -2.0];
        LogitsProcessor::apply_repetition_penalty(
            &mut logits,
            &[1, 2],
            &HashSet::from([1, 2]),
            2.0,
        );
        assert_eq!(logits, [2.0, -4.0]);
    }

    #[test]
    fn top_p_drops_low_mass_tail() {
        // With extreme scores only the top token survives the nucleus.
        let p = LogitsProcessor::new(1.0, 0.5);
        let mut rng = SplitMix64::new(42);
        let (seen, counts) = none();
        let mut hits: HashMap<TokenId, usize> = HashMap::new();
        for _ in 0..200 {
            let id = p.sample(&[1, 2, 3], &[10.0, 0.0, -10.0], &seen, &counts, &mut rng);
            *hits.entry(id).or_default() += 1;
        }
        assert!(hits.get(&1).copied().unwrap_or(0) > 180);
        assert_eq!(hits.get(&3).copied().unwrap_or(0), 0);
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let p = LogitsProcessor::new(1.0, 1.0);
        let mut rng = SplitMix64::new(7);
        let (seen, counts) = none();
        let mut ones = 0;
        for _ in 0..2000 {
            // softmax([0, ln 3]) = [0.25, 0.75]
            if p.sample(&[1, 2], &[0.0, 3f32.ln()], &seen, &counts, &mut rng) == 2 {
                ones += 1;
            }
        }
        let freq = f64::from(ones) / 2000.0;
        assert!((freq - 0.75).abs() < 0.05, "{freq}");
    }

    #[test]
    fn probabilities_normalize_and_greedy_is_one_hot() {
        let p = LogitsProcessor::new(1.0, 1.0).probabilities(&[1, 2, 3], &[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[2] > p[1] && p[1] > p[0]);
        let g = LogitsProcessor::new(0.0, 1.0).probabilities(&[1, 2], &[0.1, 0.9]);
        assert_eq!(g, [0.0, 1.0]);
    }

    #[test]
    fn prepare_applies_penalties_in_order() {
        let p = LogitsProcessor {
            repetition_penalty: 2.0,
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            ..LogitsProcessor::default()
        };
        let (ids, scores) = p.prepare(
            &[1, 2],
            &[4.0, 4.0],
            &HashSet::from([1]),
            &HashMap::from([(1, 2)]),
        );
        assert_eq!(ids, [1, 2]);
        // 4 / 2 = 2; - 1 (presence) = 1; - 0.5 * 2 (frequency) = 0
        assert_eq!(scores, [0.0, 4.0]);
    }

    #[test]
    fn from_options_and_is_greedy() {
        assert!(LogitsProcessor::from_options(&GenerateOptions::default()).is_greedy());
        let hot = GenerateOptions {
            temperature: 0.7,
            ..GenerateOptions::default()
        };
        assert!(!LogitsProcessor::from_options(&hot).is_greedy());
    }

    #[test]
    fn splitmix_is_deterministic_and_seed_zero_is_remapped() {
        let mut a = SplitMix64::new(5);
        let mut b = SplitMix64::new(5);
        assert_eq!((a.next_u64(), a.next_u64()), (b.next_u64(), b.next_u64()));
        assert_eq!(SplitMix64::new(0).state, 0xDEAD_BEEF_CAFE_BABE);
        // Reference value for seed 1 (canonical SplitMix64).
        assert_eq!(SplitMix64::new(1).next_u64(), 0x910A_2DEC_8902_5CC1);
        let mut buf = [0u8; 11];
        SplitMix64::new(9).fill_bytes(&mut buf);
        assert!(buf.iter().any(|&b| b != 0));
    }
}
