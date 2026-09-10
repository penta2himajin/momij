import Foundation

/// Learning-free suffix speculative decoding (Prompt Lookup / SuffixDecoding style).
///
/// Draft: find the longest recent n-gram that already occurred earlier in history
/// (prompt + generated), then copy the contiguous tokens that followed that match.
/// Verify: run the target model on the draft; accept matching greedy tokens.
public enum SuffixSpec {
    public enum SpecError: Error { case stepFailed }

    /// Adaptive draft length from recent mean accepted tokens per attempt.
    public static func adaptiveDraftK(meanAccept: Double, draftK: Int) -> Int {
        let k = max(1, draftK)
        if meanAccept >= 2.5 { return k }
        if meanAccept >= 1.5 { return min(k, 6) }
        if meanAccept >= 0.8 { return min(k, 4) }
        if meanAccept >= 0.4 { return min(k, 3) }
        return min(k, 2)
    }

    /// Find a draft of up to `k` tokens by matching the longest repeating suffix (PLD).
    ///
    /// Searches **all** earlier starts (not only the abutting window). Continuation is
    /// always a contiguous slice after the match. When `promptLen` is set, prefer matches
    /// whose pattern starts inside the prompt (classic Prompt Lookup Decoding).
    public static func suffixDraft(
        history: [Int], k: Int, maxNgram: Int = 16, promptLen: Int? = nil
    ) -> [Int] {
        guard history.count >= 2, k > 0 else { return [] }
        let n = history.count
        let maxPattern = min(maxNgram, n - 1)
        let pLen = promptLen ?? 0

        func search(preferPrompt: Bool) -> [Int] {
            for patternLen in stride(from: maxPattern, through: 1, by: -1) {
                let patternStart = n - patternLen
                for start in stride(from: patternStart - 1, through: 0, by: -1) {
                    if preferPrompt, pLen > 0, start >= pLen { continue }
                    var ok = true
                    for j in 0 ..< patternLen {
                        if history[start + j] != history[patternStart + j] {
                            ok = false
                            break
                        }
                    }
                    guard ok else { continue }
                    let contStart = start + patternLen
                    let avail = n - contStart
                    guard avail > 0 else { continue }
                    let take = min(k, avail)
                    return Array(history[contStart ..< (contStart + take)])
                }
            }
            return []
        }

        if pLen > 0 {
            let fromPrompt = search(preferPrompt: true)
            if !fromPrompt.isEmpty { return fromPrompt }
        }
        return search(preferPrompt: false)
    }

    /// Run speculative loop. `step` returns next greedy token given full prompt ids.
    /// `multiStep` optionally verifies a draft in one call (returns greedy continuation).
    public static func run(
        prompt: [Int],
        maxTokens: Int,
        draftK: Int,
        eos: Int?,
        step: ([Int]) throws -> Int?,
        multiStep: (([Int], Int) throws -> [Int])? = nil
    ) throws -> [Int] {
        var ids = prompt
        var out: [Int] = []
        var acceptWindow: [Int] = []
        while out.count < maxTokens {
            let mean = acceptWindow.isEmpty ? 1.0
                : Double(acceptWindow.reduce(0, +)) / Double(acceptWindow.count)
            let k = adaptiveDraftK(meanAccept: mean, draftK: min(draftK, maxTokens - out.count))
            let draft = suffixDraft(history: ids, k: k, promptLen: prompt.count)
            if draft.isEmpty {
                guard let t = try step(ids) else { break }
                ids.append(t); out.append(t)
                if let eos, t == eos { break }
                continue
            }
            // Verify draft: get greedy continuation of length draft.count
            let verified: [Int]
            if let multiStep {
                verified = try multiStep(ids, draft.count)
            } else {
                var cur = ids
                var v: [Int] = []
                for _ in 0 ..< draft.count {
                    guard let t = try step(cur) else { break }
                    v.append(t); cur.append(t)
                }
                verified = v
            }
            var accepted = 0
            for i in 0 ..< min(draft.count, verified.count) {
                if draft[i] == verified[i] {
                    accepted += 1
                } else {
                    break
                }
            }
            acceptWindow.append(accepted)
            if acceptWindow.count > 8 { acceptWindow.removeFirst() }
            if accepted == 0 {
                // reject all — commit first verified token if any
                if let t = verified.first {
                    ids.append(t); out.append(t)
                    if let eos, t == eos { break }
                } else if let t = try step(ids) {
                    ids.append(t); out.append(t)
                    if let eos, t == eos { break }
                } else {
                    break
                }
            } else {
                let chunk = Array(verified.prefix(accepted))
                ids.append(contentsOf: chunk)
                out.append(contentsOf: chunk)
                if let eos, chunk.contains(eos) { break }
            }
        }
        return out
    }
}
