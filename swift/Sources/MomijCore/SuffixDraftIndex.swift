import Foundation

/// Online suffix-tree draft index (SuffixDecoding / ArcticInference style).
///
/// Indexes all depth-limited suffixes of observed token sequences. Drafts walk the
/// longest matching suffix of `history`, then follow highest-count children.
/// Speculation length is `MAX_SPEC = α · matchLen` (capped by `maxK`).
///
/// vLLM's production SuffixDecoding also verifies a **single chain** (not a tree);
/// we match that shape so existing `stepChainFeeds` / early-exit verify stay valid.
///
/// Sources: Oliaro et al. NeurIPS 2025; Snowflake ArcticInference blog (max depth 64).
public final class SuffixDraftIndex: @unchecked Sendable {
    final class Node {
        var children: [Int: Node] = [:]
        /// Occurrences of the path ending at this node.
        var count: Int = 0
    }

    private let root = Node()
    public let maxDepth: Int
    /// Total tokens inserted (approx; for diagnostics).
    public private(set) var tokenCount: Int = 0

    public init(maxDepth: Int = 64) {
        self.maxDepth = max(4, maxDepth)
    }

    public func clear() {
        root.children.removeAll(keepingCapacity: true)
        tokenCount = 0
    }

    /// Insert every suffix of `tokens` (truncated to `maxDepth`).
    public func insert(_ tokens: [Int]) {
        guard !tokens.isEmpty else { return }
        tokenCount += tokens.count
        for i in 0 ..< tokens.count {
            var node = root
            let end = min(tokens.count, i + maxDepth)
            for j in i ..< end {
                let t = tokens[j]
                if node.children[t] == nil { node.children[t] = Node() }
                node = node.children[t]!
                node.count += 1
            }
        }
    }

    /// `MAX_SPEC = α · p` (SuffixDecoding). At least 1 when a match exists.
    public static func maxSpec(matchLen p: Int, maxK: Int, alpha: Double) -> Int {
        guard p > 0, maxK > 0 else { return 0 }
        let raw = Int((alpha * Double(p)).rounded(.towardZero))
        return max(1, min(maxK, raw))
    }

    /// Longest suffix match → frequency-greedy continuation.
    public func draft(
        from history: [Int], maxK: Int, alpha: Double = 1.0
    ) -> (matchLen: Int, tokens: [Int]) {
        let n = history.count
        guard n > 0, maxK > 0 else { return (0, []) }

        var bestP = 0
        var bestNode: Node?

        let maxP = min(maxDepth, n)
        for p in stride(from: maxP, through: 1, by: -1) {
            var node = root
            var ok = true
            for j in (n - p) ..< n {
                guard let child = node.children[history[j]] else {
                    ok = false
                    break
                }
                node = child
            }
            if ok, !node.children.isEmpty {
                bestP = p
                bestNode = node
                break
            }
        }
        guard let start = bestNode, bestP > 0 else { return (0, []) }

        let spec = Self.maxSpec(matchLen: bestP, maxK: maxK, alpha: alpha)
        var out: [Int] = []
        out.reserveCapacity(spec)
        var node = start
        for _ in 0 ..< spec {
            var bestT: Int?
            var bestC = -1
            for (t, child) in node.children {
                if child.count > bestC {
                    bestC = child.count
                    bestT = t
                }
            }
            guard let t = bestT, let child = node.children[t] else { break }
            out.append(t)
            node = child
        }
        return (bestP, out)
    }

    /// Merge drafts from local + global: prefer longer match, then longer draft.
    public static func bestDraft(
        local: SuffixDraftIndex?,
        global: SuffixDraftIndex?,
        history: [Int],
        maxK: Int,
        alpha: Double
    ) -> (matchLen: Int, tokens: [Int]) {
        var best = (matchLen: 0, tokens: [Int]())
        for idx in [local, global].compactMap({ $0 }) {
            let d = idx.draft(from: history, maxK: maxK, alpha: alpha)
            if d.matchLen > best.matchLen
                || (d.matchLen == best.matchLen && d.tokens.count > best.tokens.count)
            {
                best = d
            }
        }
        return best
    }
}

extension SuffixSpec {
    /// Default α for `MAX_SPEC = αp`. Override with env in the decode engine.
    public static let defaultSpecAlpha: Double = 1.0

    /// Tree-backed draft with PLD linear-scan fallback.
    public static func treeDraft(
        history: [Int],
        k: Int,
        local: SuffixDraftIndex?,
        global: SuffixDraftIndex?,
        alpha: Double = defaultSpecAlpha,
        promptLen: Int? = nil
    ) -> [Int] {
        let tree = SuffixDraftIndex.bestDraft(
            local: local, global: global, history: history, maxK: k, alpha: alpha)
        if !tree.tokens.isEmpty { return tree.tokens }
        return suffixDraft(history: history, k: k, promptLen: promptLen)
    }
}
