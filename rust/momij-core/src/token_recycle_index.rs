//! FlashHead-approximated Token Recycling (Luo et al., ACL 2025 /
//! arXiv:2408.08696). Port of `TokenRecycleIndex.swift`.
//!
//! Classic TR stores top-k from the **full** vocab distribution in an adjacency
//! matrix and BFS-drafts a tree. Candidates here come from a cheaper source
//! (cluster logits / server top-k) — enough to help open-ended chat where
//! suffix trees alone are weak, without an exact lm_head top-k pass.

use std::collections::{HashMap, HashSet};

use crate::TokenId;

#[derive(Debug, Clone)]
pub struct TokenRecycleIndex {
    adj: HashMap<TokenId, Vec<TokenId>>,
    pub top_k: usize,
    observe_count: usize,
}

impl TokenRecycleIndex {
    pub fn new(top_k: usize) -> Self {
        Self {
            adj: HashMap::new(),
            top_k: top_k.max(2),
            observe_count: 0,
        }
    }

    pub fn clear(&mut self) {
        self.adj.clear();
        self.observe_count = 0;
    }

    pub fn observe_count(&self) -> usize {
        self.observe_count
    }

    pub fn entry_count(&self) -> usize {
        self.adj.len()
    }

    /// Update candidates for `from_token` (most recent distribution wins).
    pub fn observe(&mut self, from_token: TokenId, candidates: &[TokenId]) {
        if from_token < 0 || candidates.is_empty() {
            return;
        }
        self.observe_count += 1;
        let previous = self.adj.get(&from_token).map_or(&[][..], Vec::as_slice);
        let mut merged = Vec::with_capacity(self.top_k);
        let mut seen = HashSet::new();
        for &t in candidates.iter().chain(previous) {
            if t < 0 || t == from_token {
                continue;
            }
            if seen.insert(t) {
                merged.push(t);
                if merged.len() >= self.top_k {
                    break;
                }
            }
        }
        if !merged.is_empty() {
            self.adj.insert(from_token, merged);
        }
    }

    /// Single-chain draft: repeatedly take the top adjacency child (TR tree
    /// collapsed to a path — matches a chain verify). Stops on missing/cycle.
    pub fn draft(&self, last: TokenId, max_k: usize) -> Vec<TokenId> {
        if max_k == 0 || last < 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(max_k);
        let mut cur = last;
        let mut seen = HashSet::from([last]);
        for _ in 0..max_k {
            let Some(next) = self.adj.get(&cur).and_then(|c| c.first().copied()) else {
                break;
            };
            if !seen.insert(next) {
                break;
            }
            out.push(next);
            cur = next;
        }
        out
    }
}

impl Default for TokenRecycleIndex {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_at_least_two() {
        assert_eq!(TokenRecycleIndex::new(0).top_k, 2);
        assert_eq!(TokenRecycleIndex::default().top_k, 8);
    }

    #[test]
    fn draft_follows_the_top_child_chain() {
        let mut idx = TokenRecycleIndex::new(4);
        idx.observe(1, &[2, 9]);
        idx.observe(2, &[3]);
        idx.observe(3, &[4]);
        assert_eq!(idx.draft(1, 8), vec![2, 3, 4]);
        assert_eq!(idx.draft(1, 2), vec![2, 3]);
        assert_eq!(idx.observe_count(), 3);
        assert_eq!(idx.entry_count(), 3);
    }

    #[test]
    fn draft_stops_on_cycle_and_missing() {
        let mut idx = TokenRecycleIndex::new(4);
        idx.observe(1, &[2]);
        idx.observe(2, &[1]);
        assert_eq!(idx.draft(1, 8), vec![2], "cycle back to the start stops");
        assert!(idx.draft(99, 8).is_empty());
        assert!(idx.draft(1, 0).is_empty());
        assert!(idx.draft(-1, 8).is_empty());
    }

    #[test]
    fn observe_merges_newest_first_and_dedups() {
        let mut idx = TokenRecycleIndex::new(3);
        idx.observe(1, &[2, 3]);
        idx.observe(1, &[4, 2, 1, -5]);
        // newest candidates first, self (1) and negatives dropped, capped at 3
        assert_eq!(idx.draft(1, 1), vec![4]);
        idx.observe(1, &[7]);
        assert_eq!(idx.draft(1, 1), vec![7]);
    }

    #[test]
    fn observe_ignores_invalid_input() {
        let mut idx = TokenRecycleIndex::new(4);
        idx.observe(-1, &[2]);
        idx.observe(1, &[]);
        idx.observe(1, &[1, -2]);
        assert_eq!(idx.observe_count(), 1);
        assert_eq!(idx.entry_count(), 0);
        idx.clear();
        assert_eq!(idx.observe_count(), 0);
    }
}
