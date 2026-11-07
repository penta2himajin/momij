//! Online suffix-tree draft index (SuffixDecoding / ArcticInference style).
//! Port of `SuffixDraftIndex.swift`.
//!
//! Indexes all depth-limited suffixes of observed token sequences. Drafts walk
//! the longest matching suffix of `history`, then follow highest-count
//! children. Speculation length is `MAX_SPEC = alpha * matchLen` (capped by
//! `max_k`). A **single chain** is drafted (vLLM's production SuffixDecoding
//! verifies a chain, not a tree).
//!
//! Sources: Oliaro et al. NeurIPS 2025; Snowflake ArcticInference blog (max
//! depth 64).

use std::collections::BTreeMap;

use crate::TokenId;

#[derive(Debug, Default, Clone)]
struct Node {
    /// Ordered map: frequency ties break toward the lowest token id, so drafts
    /// are deterministic (the Swift original iterated an unordered dictionary).
    children: BTreeMap<TokenId, Node>,
    /// Occurrences of the path ending at this node.
    count: u32,
}

#[derive(Debug, Clone)]
pub struct SuffixDraftIndex {
    root: Node,
    pub max_depth: usize,
    /// Total tokens inserted (approx; for diagnostics).
    pub token_count: usize,
}

/// Result of a draft lookup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Draft {
    pub match_len: usize,
    pub tokens: Vec<TokenId>,
}

impl SuffixDraftIndex {
    pub fn new(max_depth: usize) -> Self {
        Self {
            root: Node::default(),
            max_depth: max_depth.max(4),
            token_count: 0,
        }
    }

    pub fn clear(&mut self) {
        self.root.children.clear();
        self.token_count = 0;
    }

    /// Insert every suffix of `tokens` (truncated to `max_depth`).
    pub fn insert(&mut self, tokens: &[TokenId]) {
        if tokens.is_empty() {
            return;
        }
        self.token_count += tokens.len();
        for i in 0..tokens.len() {
            let end = tokens.len().min(i + self.max_depth);
            let mut node = &mut self.root;
            for &t in &tokens[i..end] {
                node = node.children.entry(t).or_default();
                node.count += 1;
            }
        }
    }

    /// `MAX_SPEC = alpha * p` (SuffixDecoding). At least 1 when a match exists.
    pub fn max_spec(match_len: usize, max_k: usize, alpha: f64) -> usize {
        if match_len == 0 || max_k == 0 {
            return 0;
        }
        let raw = (alpha * match_len as f64).trunc().max(0.0) as usize;
        raw.min(max_k).max(1)
    }

    /// Node reached by walking the last `p` tokens of `history` from the root.
    fn walk(&self, history: &[TokenId], p: usize) -> Option<&Node> {
        let mut node = &self.root;
        for t in &history[history.len() - p..] {
            node = node.children.get(t)?;
        }
        Some(node)
    }

    /// Longest suffix match -> frequency-greedy continuation.
    pub fn draft(&self, history: &[TokenId], max_k: usize, alpha: f64) -> Draft {
        let n = history.len();
        if n == 0 || max_k == 0 {
            return Draft::default();
        }
        let found = (1..=self.max_depth.min(n)).rev().find_map(|p| {
            self.walk(history, p)
                .filter(|node| !node.children.is_empty())
                .map(|node| (p, node))
        });
        let Some((best_p, start)) = found else {
            return Draft::default();
        };
        let spec = Self::max_spec(best_p, max_k, alpha);
        let mut out = Vec::with_capacity(spec);
        let mut node = start;
        for _ in 0..spec {
            // Highest count wins; first (lowest id) on ties.
            let mut best: Option<(&TokenId, &Node)> = None;
            for (t, child) in &node.children {
                if best.is_none_or(|(_, b)| child.count > b.count) {
                    best = Some((t, child));
                }
            }
            let Some((&t, child)) = best else { break };
            out.push(t);
            node = child;
        }
        Draft {
            match_len: best_p,
            tokens: out,
        }
    }

    /// Merge drafts from local + global: prefer longer match, then longer draft.
    pub fn best_draft(
        local: Option<&Self>,
        global: Option<&Self>,
        history: &[TokenId],
        max_k: usize,
        alpha: f64,
    ) -> Draft {
        let mut best = Draft::default();
        for idx in [local, global].into_iter().flatten() {
            let d = idx.draft(history, max_k, alpha);
            if d.match_len > best.match_len
                || (d.match_len == best.match_len && d.tokens.len() > best.tokens.len())
            {
                best = d;
            }
        }
        best
    }

    /// Draft-model q along `draft` from this tree: child_count / sibling_mass.
    /// Positions that leave the tree stay 1 (deterministic). Empty draft -> `[]`.
    pub fn q_along(&self, history: &[TokenId], draft: &[TokenId]) -> Vec<f32> {
        if draft.is_empty() {
            return Vec::new();
        }
        let mut q = vec![1.0f32; draft.len()];
        let n = history.len();
        if n == 0 {
            return q;
        }
        let start = (1..=self.max_depth.min(n))
            .rev()
            .find_map(|p| self.walk(history, p));
        let Some(mut node) = start else { return q };
        for (i, t) in draft.iter().enumerate() {
            let total: u32 = node.children.values().map(|c| c.count).sum();
            let Some(child) = node.children.get(t).filter(|_| total > 0) else {
                break;
            };
            q[i] = (child.count as f32 / total as f32).max(1e-6);
            node = child;
        }
        q
    }
}

impl Default for SuffixDraftIndex {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_tree_draft_follows_frequency() {
        let mut idx = SuffixDraftIndex::new(16);
        // Paths through [1,2,3]: ->[4,4] twice and ->[9] once -> prefer 4
        idx.insert(&[0, 1, 2, 3, 4, 4]);
        idx.insert(&[7, 1, 2, 3, 4, 5]);
        idx.insert(&[8, 1, 2, 3, 9]);
        let d = idx.draft(&[1, 2, 3], 4, 1.0);
        assert_eq!(d.match_len, 3);
        assert_eq!(d.tokens.first(), Some(&4));
        assert_eq!(SuffixDraftIndex::max_spec(d.match_len, 4, 1.0), 3);
    }

    #[test]
    fn max_spec_scales_with_match_len() {
        assert_eq!(SuffixDraftIndex::max_spec(4, 16, 1.0), 4);
        assert_eq!(SuffixDraftIndex::max_spec(4, 16, 2.0), 8);
        assert_eq!(SuffixDraftIndex::max_spec(4, 3, 2.0), 3);
        assert_eq!(SuffixDraftIndex::max_spec(0, 8, 1.0), 0);
        assert_eq!(
            SuffixDraftIndex::max_spec(3, 8, 0.1),
            1,
            "at least 1 when a match exists"
        );
    }

    #[test]
    fn global_index_beats_empty_local() {
        let mut global = SuffixDraftIndex::new(32);
        global.insert(&[1, 2, 3, 4, 5, 6]);
        let d = SuffixDraftIndex::best_draft(None, Some(&global), &[1, 2, 3], 4, 1.0);
        assert_eq!(d.match_len, 3);
        assert_eq!(d.tokens, [4, 5, 6]);
        let empty = SuffixDraftIndex::new(32);
        let d = SuffixDraftIndex::best_draft(Some(&empty), Some(&global), &[1, 2, 3], 4, 1.0);
        assert_eq!(d.tokens, [4, 5, 6]);
    }

    #[test]
    fn best_draft_prefers_longer_match_then_longer_draft() {
        let mut a = SuffixDraftIndex::new(32);
        a.insert(&[9, 1, 2, 3, 7]);
        let mut b = SuffixDraftIndex::new(32);
        b.insert(&[2, 3, 8, 8, 8]);
        let d = SuffixDraftIndex::best_draft(Some(&a), Some(&b), &[1, 2, 3], 4, 1.0);
        assert_eq!(d.match_len, 3);
        assert_eq!(d.tokens, [7]);
    }

    #[test]
    fn draft_edge_cases() {
        let mut idx = SuffixDraftIndex::new(16);
        idx.insert(&[1, 2, 3]);
        assert_eq!(idx.draft(&[], 4, 1.0), Draft::default());
        assert_eq!(idx.draft(&[1], 0, 1.0), Draft::default());
        assert_eq!(idx.draft(&[99], 4, 1.0), Draft::default());
        // A match at the very end of the indexed data has no continuation.
        assert_eq!(idx.draft(&[3], 4, 1.0), Draft::default());
        assert_eq!(idx.token_count, 3);
        idx.clear();
        assert_eq!(idx.token_count, 0);
        assert_eq!(idx.draft(&[1, 2], 4, 1.0), Draft::default());
    }

    #[test]
    fn frequency_ties_break_to_lowest_token_id() {
        let mut idx = SuffixDraftIndex::new(16);
        idx.insert(&[1, 9]);
        idx.insert(&[1, 4]);
        assert_eq!(idx.draft(&[1], 1, 1.0).tokens, [4]);
    }

    #[test]
    fn max_depth_is_at_least_four() {
        assert_eq!(SuffixDraftIndex::new(1).max_depth, 4);
        assert_eq!(SuffixDraftIndex::default().max_depth, 64);
    }

    #[test]
    fn q_along_follows_child_frequency() {
        let mut idx = SuffixDraftIndex::new(16);
        idx.insert(&[1, 2, 3, 4]);
        idx.insert(&[1, 2, 3, 4]);
        idx.insert(&[1, 2, 3, 9]);
        let q = idx.q_along(&[1, 2, 3], &[4]);
        assert_eq!(q.len(), 1);
        assert!((q[0] - 2.0 / 3.0).abs() < 1e-5, "{}", q[0]);
    }

    #[test]
    fn q_along_defaults_to_one_off_tree() {
        let mut idx = SuffixDraftIndex::new(16);
        idx.insert(&[1, 2, 3, 4]);
        assert!(idx.q_along(&[1, 2, 3], &[]).is_empty());
        assert_eq!(idx.q_along(&[], &[4]), [1.0]);
        assert_eq!(idx.q_along(&[42], &[4, 5]), [1.0, 1.0]);
        // Leaves the tree after the first position.
        let q = idx.q_along(&[1, 2, 3], &[4, 77]);
        assert_eq!(q, [1.0, 1.0]);
    }
}
