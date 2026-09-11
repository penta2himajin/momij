import Foundation

/// Speculative sampling (Leviathan et al., ICML 2023) over a model-free draft.
/// Default **deterministic** draft: `q(x*)=1` for the proposed token (tree/PLD/recycle
/// emit one chain). Optional per-token `draftQ` from suffix-tree child frequencies.
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

    /// Accept `draftToken` with min(1, p/q). On reject, sample residual max(0, p−q).
    /// `draftQ` is the draft density on `draftToken` (1 = deterministic proposal).
    public static func tryAccept(
        draftToken: Int,
        ids: [Int],
        logits: [Float],
        processor: LogitsProcessor,
        rng: inout some RandomNumberGenerator,
        draftQ: Float = 1
    ) -> (accepted: Bool, token: Int) {
        let p = prob(of: draftToken, ids: ids, logits: logits, processor: processor)
        let q = max(0, draftQ)
        if q > 0, p > 0 {
            let u = Float.random(in: 0 ..< 1, using: &rng)
            if u <= min(1, p / q) {
                return (true, draftToken)
            }
        }
        var mass = processor.probabilities(tokenIds: ids, logits: logits)
        if let idx = ids.firstIndex(of: draftToken), q > 0 {
            mass[idx] = max(0, mass[idx] - q)
        }
        let sum = mass.reduce(0, +)
        if sum > 0 {
            let u = Float.random(in: 0 ..< 1, using: &rng) * sum
            var acc: Float = 0
            for i in mass.indices {
                acc += mass[i]
                if u <= acc { return (false, ids[i]) }
            }
            return (false, ids[ids.count - 1])
        }
        let sampled = processor.samplePrepared(tokenIds: ids, logits: logits, rng: &rng)
        return (false, sampled)
    }

    /// Sequential Leviathan-style walk over **precomputed** target rows.
    /// `rows[i]` is the candidate distribution after feeding `feeds[i]` (`feeds = [y]+draft`).
    /// `rows.count` must be `draft.count + 1` (last row = bonus).
    /// `draftQ[i]` is q on `draft[i]`; empty → all 1 (deterministic).
    public static func walkDraft(
        draft: [Int],
        rows: [(ids: [Int], logits: [Float])],
        processor: LogitsProcessor,
        seen: Set<Int>,
        counts: [Int: Int],
        rng: inout some RandomNumberGenerator,
        draftQ: [Float] = []
    ) -> (accepted: Int, next: Int) {
        precondition(rows.count >= draft.count + 1)
        var seen = seen
        var counts = counts
        for i in 0 ..< draft.count {
            let prepared = processor.prepare(
                tokenIds: rows[i].ids, logits: rows[i].logits,
                seen: seen, counts: counts)
            let q = i < draftQ.count ? draftQ[i] : 1
            let d = tryAccept(
                draftToken: draft[i],
                ids: prepared.ids, logits: prepared.logits,
                processor: processor, rng: &rng, draftQ: q)
            if !d.accepted {
                return (i, d.token)
            }
            seen.insert(draft[i])
            counts[draft[i], default: 0] += 1
        }
        let bonus = rows[draft.count]
        let prepared = processor.prepare(
            tokenIds: bonus.ids, logits: bonus.logits,
            seen: seen, counts: counts)
        let next = processor.samplePrepared(
            tokenIds: prepared.ids, logits: prepared.logits, rng: &rng)
        return (draft.count, next)
    }

    /// When to spend draft work on the sampled path. Cold 1-step is the 150 tok/s floor;
    /// probe periodically (1-forward early-exit); M-row batch only when accepts land.
    public enum SampledSpecPolicy {
        public static let probeEvery = 8
        public static let batchMean = 1.5
        public static let batchHoldMean = 0.8
        public static let batchMinWindow = 4

        public static func useBatch(
            meanAccept: Double, windowCount: Int, currentlyBatching: Bool = false
        ) -> Bool {
            if currentlyBatching {
                return windowCount >= 2 && meanAccept >= batchHoldMean
            }
            return windowCount >= batchMinWindow && meanAccept >= batchMean
        }

        /// Short drafts: sampled Leviathan rarely copies 6–8 tokens, and long M-row
        /// plus restore is slower than 1-step. Cap well below greedy `adaptiveDraftK`.
        public static func draftK(meanAccept: Double, draftK: Int) -> Int {
            let k = max(1, draftK)
            if meanAccept >= 2.5 { return min(k, 4) }
            if meanAccept >= 1.5 { return min(k, 3) }
            return min(k, 2)
        }

        /// Draft/verify this step? False → caller should `stepSampled` only.
        /// After a probe lands, draft every step until the window is full so we can
        /// decide to batch; collapsed / not-hot returns to periodic probes.
        public static func useDraft(
            generated: Int, meanAccept: Double, windowCount: Int,
            probeEvery: Int = probeEvery,
            currentlyBatching: Bool = false
        ) -> Bool {
            if useBatch(
                meanAccept: meanAccept, windowCount: windowCount,
                currentlyBatching: currentlyBatching)
            { return true }
            if windowCount >= 1 && windowCount < batchMinWindow && meanAccept >= 0.5 {
                return true
            }
            guard probeEvery > 0, generated > 0 else { return false }
            return generated % probeEvery == 0
        }
    }
}
