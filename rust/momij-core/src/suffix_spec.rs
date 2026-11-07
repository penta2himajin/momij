//! Learning-free suffix speculative decoding (Prompt Lookup / SuffixDecoding
//! style). Port of `SuffixSpec.swift`.
//!
//! Draft: find the longest recent n-gram that already occurred earlier in
//! history (prompt + generated), then copy the contiguous tokens that followed
//! that match. Verify: run the target model on the draft; accept matching
//! greedy tokens.

use crate::TokenId;
use crate::suffix_draft_index::SuffixDraftIndex;

/// Default alpha for `MAX_SPEC = alpha * p`.
pub const DEFAULT_SPEC_ALPHA: f64 = 1.0;

/// Adaptive draft length from recent mean accepted tokens per attempt.
pub fn adaptive_draft_k(mean_accept: f64, draft_k: usize) -> usize {
    let k = draft_k.max(1);
    if mean_accept >= 2.5 {
        k
    } else if mean_accept >= 1.5 {
        k.min(6)
    } else if mean_accept >= 0.8 {
        k.min(4)
    } else if mean_accept >= 0.4 {
        k.min(3)
    } else {
        k.min(2)
    }
}

/// Find a draft of up to `k` tokens by matching the longest repeating suffix (PLD).
///
/// Searches **all** earlier starts (not only the abutting window). The
/// continuation is always a contiguous slice after the match. When
/// `prompt_len` is set, prefer matches whose pattern starts inside the prompt
/// (classic Prompt Lookup Decoding).
pub fn suffix_draft(
    history: &[TokenId],
    k: usize,
    max_ngram: usize,
    prompt_len: Option<usize>,
) -> Vec<TokenId> {
    if history.len() < 2 || k == 0 {
        return Vec::new();
    }
    let n = history.len();
    let max_pattern = max_ngram.min(n - 1);
    let p_len = prompt_len.unwrap_or(0);

    let search = |prefer_prompt: bool| -> Vec<TokenId> {
        for pattern_len in (1..=max_pattern).rev() {
            let pattern_start = n - pattern_len;
            let pattern = &history[pattern_start..];
            for start in (0..pattern_start).rev() {
                if prefer_prompt && p_len > 0 && start >= p_len {
                    continue;
                }
                if &history[start..start + pattern_len] != pattern {
                    continue;
                }
                let cont_start = start + pattern_len;
                let avail = n - cont_start;
                if avail == 0 {
                    continue;
                }
                return history[cont_start..cont_start + k.min(avail)].to_vec();
            }
        }
        Vec::new()
    };

    if p_len > 0 {
        let from_prompt = search(true);
        if !from_prompt.is_empty() {
            return from_prompt;
        }
    }
    search(false)
}

/// [`suffix_draft`] with the default 16-gram window.
pub fn suffix_draft_default(
    history: &[TokenId],
    k: usize,
    prompt_len: Option<usize>,
) -> Vec<TokenId> {
    suffix_draft(history, k, 16, prompt_len)
}

/// Tree-backed draft with PLD linear-scan fallback.
pub fn tree_draft(
    history: &[TokenId],
    k: usize,
    local: Option<&SuffixDraftIndex>,
    global: Option<&SuffixDraftIndex>,
    alpha: f64,
    prompt_len: Option<usize>,
) -> Vec<TokenId> {
    let tree = SuffixDraftIndex::best_draft(local, global, history, k, alpha);
    if tree.tokens.is_empty() {
        suffix_draft_default(history, k, prompt_len)
    } else {
        tree.tokens
    }
}

/// `step` returns the next greedy token given the full id sequence.
pub type Step<'a, E> = &'a mut dyn FnMut(&[TokenId]) -> Result<Option<TokenId>, E>;
/// `multi_step` verifies a draft in one call (returns the greedy continuation
/// of the requested length).
pub type MultiStep<'a, E> = &'a mut dyn FnMut(&[TokenId], usize) -> Result<Vec<TokenId>, E>;

/// Run the speculative loop.
pub fn run<E>(
    prompt: &[TokenId],
    max_tokens: usize,
    draft_k: usize,
    eos: Option<TokenId>,
    step: Step<'_, E>,
    mut multi_step: Option<MultiStep<'_, E>>,
) -> Result<Vec<TokenId>, E> {
    let mut ids = prompt.to_vec();
    let mut out: Vec<TokenId> = Vec::new();
    let mut accept_window: Vec<usize> = Vec::new();
    while out.len() < max_tokens {
        let mean = if accept_window.is_empty() {
            1.0
        } else {
            accept_window.iter().sum::<usize>() as f64 / accept_window.len() as f64
        };
        let k = adaptive_draft_k(mean, draft_k.min(max_tokens - out.len()));
        let draft = suffix_draft_default(&ids, k, Some(prompt.len()));
        if draft.is_empty() {
            let Some(t) = step(&ids)? else { break };
            ids.push(t);
            out.push(t);
            if eos == Some(t) {
                break;
            }
            continue;
        }
        // Verify draft: greedy continuation of length draft.len()
        let verified: Vec<TokenId> = if let Some(multi) = multi_step.as_mut() {
            multi(&ids, draft.len())?
        } else {
            let mut cur = ids.clone();
            let mut v = Vec::new();
            for _ in 0..draft.len() {
                let Some(t) = step(&cur)? else { break };
                v.push(t);
                cur.push(t);
            }
            v
        };
        let accepted = draft
            .iter()
            .zip(&verified)
            .take_while(|(d, v)| d == v)
            .count();
        accept_window.push(accepted);
        if accept_window.len() > 8 {
            accept_window.remove(0);
        }
        if accepted == 0 {
            // reject all — commit first verified token if any
            let committed = match verified.first() {
                Some(&t) => Some(t),
                None => step(&ids)?,
            };
            let Some(t) = committed else { break };
            ids.push(t);
            out.push(t);
            if eos == Some(t) {
                break;
            }
        } else {
            let chunk = &verified[..accepted];
            ids.extend_from_slice(chunk);
            out.extend_from_slice(chunk);
            if eos.is_some_and(|e| chunk.contains(&e)) {
                break;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::convert::Infallible;

    use super::*;

    #[test]
    fn finds_repeated_suffix_draft() {
        // history ends with [1,2,3] and earlier contains [1,2,3,9,8]
        let history = [7, 1, 2, 3, 9, 8, 1, 2, 3];
        assert_eq!(suffix_draft_default(&history, 2, None), [9, 8]);
    }

    /// Match need not abut the current suffix (classic PLD long-distance hit).
    #[test]
    fn finds_distant_prompt_lookup_draft() {
        let history = [10, 11, 12, 13, 14, 99, 98, 10, 11];
        assert_eq!(suffix_draft_default(&history, 3, None), [12, 13, 14]);
    }

    /// Prefer continuation from the prompt span when `prompt_len` is set.
    #[test]
    fn prefers_prompt_region_match() {
        let history = [1, 2, 3, 4, 5, 1, 2, 90, 91, 1, 2];
        assert_eq!(suffix_draft_default(&history, 3, Some(5)), [3, 4, 5]);
    }

    /// Bigram fallback must be a contiguous continuation, not a bag of successors.
    #[test]
    fn bigram_fallback_is_contiguous() {
        let history = [5, 7, 8, 9, 1, 2, 5];
        assert_eq!(suffix_draft_default(&history, 3, None), [7, 8, 9]);
    }

    /// Longer pattern match -> longer adaptive draft (capped by k).
    #[test]
    fn adaptive_draft_length_tracks_match() {
        let history = [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4];
        assert_eq!(
            suffix_draft_default(&history, 8, None),
            [5, 6, 0, 1, 2, 3, 4]
        );
        assert_eq!(adaptive_draft_k(0.0, 8), 2);
        assert_eq!(adaptive_draft_k(0.5, 8), 3);
        assert_eq!(adaptive_draft_k(0.9, 8), 4);
        assert_eq!(adaptive_draft_k(1.6, 8), 6);
        assert_eq!(adaptive_draft_k(3.0, 8), 8);
        assert_eq!(adaptive_draft_k(3.0, 0), 1);
    }

    #[test]
    fn suffix_draft_degenerate_inputs() {
        assert!(suffix_draft_default(&[], 4, None).is_empty());
        assert!(suffix_draft_default(&[1], 4, None).is_empty());
        assert!(suffix_draft_default(&[1, 2, 3], 0, None).is_empty());
        assert!(suffix_draft_default(&[1, 2, 3], 4, None).is_empty());
    }

    #[test]
    fn tree_draft_falls_back_to_pld() {
        let history = [7, 1, 2, 3, 9, 8, 1, 2, 3];
        assert_eq!(
            tree_draft(&history, 2, None, None, DEFAULT_SPEC_ALPHA, None),
            [9, 8]
        );
    }

    #[test]
    fn tree_draft_prefers_the_tree_when_it_has_a_draft() {
        let mut global = SuffixDraftIndex::new(32);
        global.insert(&[1, 2, 3, 40, 41]);
        let history = [7, 1, 2, 3, 9, 8, 1, 2, 3];
        let d = tree_draft(&history, 2, None, Some(&global), DEFAULT_SPEC_ALPHA, None);
        assert_eq!(d, [40, 41]);
    }

    #[test]
    fn spec_accepts_matching_draft() {
        // Deterministic "model": always emits the next integer from 10.
        let seq = Cell::new(10);
        let mut step = |_ids: &[TokenId]| -> Result<Option<TokenId>, Infallible> {
            let t = seq.get();
            seq.set(t + 1);
            Ok(Some(t))
        };
        let mut multi = |_ids: &[TokenId], k: usize| -> Result<Vec<TokenId>, Infallible> {
            Ok((0..k)
                .map(|_| {
                    let t = seq.get();
                    seq.set(t + 1);
                    t
                })
                .collect())
        };
        let out = run(&[1, 2, 3, 1, 2, 3], 4, 2, None, &mut step, Some(&mut multi)).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn run_accepts_a_correct_draft_and_stops_at_eos() {
        // Greedy continuation of the pattern 1 2 3 4 5 repeating.
        let cycle = [1, 2, 3, 4, 5];
        let next = |ids: &[TokenId]| -> TokenId {
            let last = *ids.last().unwrap();
            cycle[(cycle.iter().position(|&c| c == last).unwrap() + 1) % cycle.len()]
        };
        let mut step =
            |ids: &[TokenId]| -> Result<Option<TokenId>, Infallible> { Ok(Some(next(ids))) };
        let mut multi = |ids: &[TokenId], k: usize| -> Result<Vec<TokenId>, Infallible> {
            let mut cur = ids.to_vec();
            let mut v = Vec::new();
            for _ in 0..k {
                let t = next(&cur);
                v.push(t);
                cur.push(t);
            }
            Ok(v)
        };
        let prompt = [1, 2, 3, 4, 5, 1, 2, 3];
        // Without EOS the loop runs to the token budget.
        let full = run(&prompt, 10, 4, None, &mut step, Some(&mut multi)).unwrap();
        assert_eq!(full, [4, 5, 1, 2, 3, 4, 5, 1, 2, 3]);
        // With EOS = 5 the accepted chunk that contains it is committed whole
        // (tokens after EOS included); the consumer cuts at the first EOS.
        let out = run(&prompt, 10, 4, Some(5), &mut step, Some(&mut multi)).unwrap();
        assert_eq!(out, [4, 5, 1, 2]);
        assert_eq!(crate::openai_compat::content_token_ids(&out, &[5]), [4]);
        // Plain sequential verify (no multi_step) gives the same tokens.
        let out2 = run(&prompt, 10, 4, Some(5), &mut step, None).unwrap();
        assert_eq!(out2, out);
    }

    #[test]
    fn run_propagates_step_errors_and_handles_none() {
        let mut fail = |_: &[TokenId]| -> Result<Option<TokenId>, &'static str> { Err("boom") };
        assert_eq!(run(&[1, 2], 3, 2, None, &mut fail, None), Err("boom"));
        let mut done = |_: &[TokenId]| -> Result<Option<TokenId>, Infallible> { Ok(None) };
        assert_eq!(
            run(&[1, 2], 3, 2, None, &mut done, None).unwrap(),
            Vec::<TokenId>::new()
        );
    }
}
