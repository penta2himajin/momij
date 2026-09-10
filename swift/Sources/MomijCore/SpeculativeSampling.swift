import Foundation

/// Speculative sampling accept/reject over a draft proposal (Leviathan/Chen-style).
/// Draft `q` is model-free: uniform over the target candidate set (FlashHead probes).
public enum SpeculativeSampling {
    public struct Decision: Sendable {
        /// How many draft tokens were accepted in order.
        public var accepted: Int
        /// Token chosen when the draft stops (rejected resample or bonus after full accept).
        public var next: Int
        /// True if `next` came from sampling after a reject (draft truncated).
        public var rejected: Bool
    }

    /// Probability of `token` under prepared candidate logits (temperature softmax, no nucleus).
    public static func prob(
        of token: Int, ids: [Int], logits: [Float], processor: LogitsProcessor
    ) -> Float {
        guard let idx = ids.firstIndex(of: token) else { return 0 }
        let p = processor.probabilities(tokenIds: ids, logits: logits)
        return p[idx]
    }

    /// Uniform draft density on the candidate support.
    public static func uniformQ(token: Int, ids: [Int]) -> Float {
        guard ids.contains(token), !ids.isEmpty else { return 0 }
        return 1 / Float(ids.count)
    }

    /// Accept `draftToken` with min(1, p/q); on fail sample from target.
    public static func tryAccept(
        draftToken: Int,
        ids: [Int],
        logits: [Float],
        processor: LogitsProcessor,
        rng: inout some RandomNumberGenerator
    ) -> (accepted: Bool, token: Int) {
        let p = prob(of: draftToken, ids: ids, logits: logits, processor: processor)
        let q = uniformQ(token: draftToken, ids: ids)
        if q > 0, p > 0 {
            let u = Float.random(in: 0 ..< 1, using: &rng)
            if u <= min(1, p / q) {
                return (true, draftToken)
            }
        }
        let sampled = processor.samplePrepared(tokenIds: ids, logits: logits, rng: &rng)
        return (false, sampled)
    }
}
