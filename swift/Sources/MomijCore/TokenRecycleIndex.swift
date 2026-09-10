import Foundation

/// FlashHead-approximated Token Recycling (Luo et al., ACL 2025 / arXiv:2408.08696).
///
/// Classic TR stores top-k from the **full** vocab distribution in an adjacency matrix
/// and BFS-drafts a tree. We approximate candidates with FlashHead's gathered cluster
/// logits (probes×clusterSize ≪ V) — enough to help open-ended chat where suffix
/// trees alone are weak, without an exact lm_head top-k pass.
public final class TokenRecycleIndex: @unchecked Sendable {
    private var adj: [Int: [Int]] = [:]
    public let topK: Int
    public private(set) var observeCount: Int = 0

    public init(topK: Int = 8) {
        self.topK = max(2, topK)
    }

    public func clear() {
        adj.removeAll(keepingCapacity: true)
        observeCount = 0
    }

    /// Update candidates for `fromToken` (most recent distribution wins).
    public func observe(fromToken: Int, candidates: [Int]) {
        guard fromToken >= 0, !candidates.isEmpty else { return }
        observeCount += 1
        var merged: [Int] = []
        merged.reserveCapacity(topK)
        var seen = Set<Int>()
        for t in candidates + (adj[fromToken] ?? []) {
            if t < 0 || t == fromToken { continue }
            if seen.insert(t).inserted {
                merged.append(t)
                if merged.count >= topK { break }
            }
        }
        if !merged.isEmpty { adj[fromToken] = merged }
    }

    /// Single-chain draft: repeatedly take the top adjacency child (TR tree collapsed
    /// to a path — matches our chain verify). Stops on missing/cycle.
    public func draft(from last: Int, maxK: Int) -> [Int] {
        guard maxK > 0, last >= 0 else { return [] }
        var out: [Int] = []
        out.reserveCapacity(maxK)
        var cur = last
        var seen = Set<Int>([last])
        for _ in 0 ..< maxK {
            guard let cands = adj[cur], let next = cands.first else { break }
            if !seen.insert(next).inserted { break }
            out.append(next)
            cur = next
        }
        return out
    }

    public var entryCount: Int { adj.count }
}
