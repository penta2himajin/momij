import Foundation

/// Deterministic RNG for sampling tests / reproducible decode.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64
    public init(seed: UInt64) { state = seed == 0 ? 0xDEAD_BEEF_CAFE_BABE : seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Temperature / top-p / OpenAI-style penalties over a candidate logit set
/// (FlashHead probes, not necessarily full vocab).
public struct LogitsProcessor: Sendable {
    public var temperature: Float
    public var topP: Float
    public var presencePenalty: Float
    public var frequencyPenalty: Float
    /// HuggingFace-style; 1.0 = off. Values >1 penalize repeats.
    public var repetitionPenalty: Float

    public init(
        temperature: Float = 0,
        topP: Float = 1,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        repetitionPenalty: Float = 1
    ) {
        self.temperature = temperature
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty
    }

    public static func from(_ opts: GenerateOptions) -> LogitsProcessor {
        LogitsProcessor(
            temperature: Float(opts.temperature),
            topP: Float(opts.topP),
            presencePenalty: Float(opts.presencePenalty),
            frequencyPenalty: Float(opts.frequencyPenalty),
            repetitionPenalty: Float(opts.repetitionPenalty))
    }

    public var isGreedy: Bool {
        temperature <= 1e-5
            && topP >= 1 - 1e-6
            && presencePenalty == 0
            && frequencyPenalty == 0
            && abs(repetitionPenalty - 1) < 1e-6
    }

    public static func applyPresencePenalty(
        logits: inout [Float], tokenIds: [Int], seen: Set<Int>, penalty: Float
    ) {
        guard penalty != 0 else { return }
        for i in tokenIds.indices where seen.contains(tokenIds[i]) {
            logits[i] -= penalty
        }
    }

    public static func applyFrequencyPenalty(
        logits: inout [Float], tokenIds: [Int], counts: [Int: Int], penalty: Float
    ) {
        guard penalty != 0 else { return }
        for i in tokenIds.indices {
            if let c = counts[tokenIds[i]], c > 0 {
                logits[i] -= penalty * Float(c)
            }
        }
    }

    /// HF: score > 0 → score/penalty; else score*penalty (penalty≥1 shrinks positives).
    public static func applyRepetitionPenalty(
        logits: inout [Float], tokenIds: [Int], seen: Set<Int>, penalty: Float
    ) {
        guard abs(penalty - 1) > 1e-6, penalty > 0 else { return }
        for i in tokenIds.indices where seen.contains(tokenIds[i]) {
            if logits[i] > 0 {
                logits[i] /= penalty
            } else {
                logits[i] *= penalty
            }
        }
    }

    public func prepare(
        tokenIds: [Int], logits: [Float],
        seen: Set<Int>, counts: [Int: Int]
    ) -> (ids: [Int], logits: [Float]) {
        precondition(tokenIds.count == logits.count && !tokenIds.isEmpty)
        var scores = logits
        Self.applyRepetitionPenalty(
            logits: &scores, tokenIds: tokenIds, seen: seen, penalty: repetitionPenalty)
        Self.applyPresencePenalty(
            logits: &scores, tokenIds: tokenIds, seen: seen, penalty: presencePenalty)
        Self.applyFrequencyPenalty(
            logits: &scores, tokenIds: tokenIds, counts: counts, penalty: frequencyPenalty)
        return (tokenIds, scores)
    }

    /// Softmax → optional nucleus → multinomial. `temperature<=0` → argmax.
    public func sample(
        tokenIds: [Int], logits: [Float],
        seen: Set<Int> = [], counts: [Int: Int] = [:],
        rng: inout some RandomNumberGenerator
    ) -> Int {
        let prepared = prepare(tokenIds: tokenIds, logits: logits, seen: seen, counts: counts)
        return samplePrepared(tokenIds: prepared.ids, logits: prepared.logits, rng: &rng)
    }

    public func samplePrepared(
        tokenIds: [Int], logits: [Float],
        rng: inout some RandomNumberGenerator
    ) -> Int {
        precondition(tokenIds.count == logits.count && !tokenIds.isEmpty)
        if temperature <= 1e-5 {
            var best = 0
            var bestV = -Float.infinity
            for i in logits.indices where logits[i] > bestV {
                bestV = logits[i]
                best = i
            }
            return tokenIds[best]
        }
        let invT = 1 / max(temperature, 1e-5)
        var scaled = logits.map { $0 * invT }
        let maxL = scaled.max() ?? 0
        var exps = scaled.map { exp($0 - maxL) }
        var sum = exps.reduce(0, +)
        guard sum > 0 else { return tokenIds[0] }

        // Nucleus (top-p): keep highest-prob tokens until cumulative ≥ topP.
        if topP < 1 - 1e-6 {
            var order = Array(exps.indices)
            order.sort { exps[$0] > exps[$1] }
            let cutoff = max(0, min(1, topP)) * sum
            var cum: Float = 0
            var keep = Set<Int>()
            for i in order {
                keep.insert(i)
                cum += exps[i]
                if cum >= cutoff { break }
            }
            for i in exps.indices where !keep.contains(i) {
                sum -= exps[i]
                exps[i] = 0
            }
            if sum <= 0 {
                return tokenIds[order[0]]
            }
        }

        let u = Float.random(in: 0 ..< 1, using: &rng) * sum
        var acc: Float = 0
        for i in exps.indices {
            acc += exps[i]
            if u <= acc { return tokenIds[i] }
        }
        return tokenIds[tokenIds.count - 1]
    }

    /// Softmax probabilities after temperature (no nucleus). For speculative sampling.
    public func probabilities(tokenIds: [Int], logits: [Float]) -> [Float] {
        if temperature <= 1e-5 {
            var best = 0
            var bestV = -Float.infinity
            for i in logits.indices where logits[i] > bestV {
                bestV = logits[i]
                best = i
            }
            var p = [Float](repeating: 0, count: logits.count)
            p[best] = 1
            return p
        }
        let invT = 1 / max(temperature, 1e-5)
        let scaled = logits.map { $0 * invT }
        let maxL = scaled.max() ?? 0
        let exps = scaled.map { exp($0 - maxL) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else {
            return [Float](repeating: 1 / Float(tokenIds.count), count: tokenIds.count)
        }
        return exps.map { $0 / sum }
    }
}
