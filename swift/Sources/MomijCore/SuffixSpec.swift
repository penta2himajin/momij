import Foundation

/// Learning-free suffix speculative decoding (Qwisp Tell / SuffixSpec style).
///
/// Draft: copy a repeated suffix from recent history when one is found.
/// Verify: run the target model on the draft prefix; accept matching greedy tokens.
public enum SuffixSpec {
    public enum SpecError: Error { case stepFailed }

    /// Find a draft of up to `k` tokens by matching the longest repeating suffix.
    public static func suffixDraft(history: [Int], k: Int) -> [Int] {
        guard history.count >= 4, k > 0 else { return [] }
        let n = history.count
        let maxLen = min(k, n / 2)
        for len in stride(from: maxLen, through: 1, by: -1) {
            let suffix = Array(history[(n - len) ..< n])
            // search earlier occurrence
            if n - 2 * len < 0 { continue }
            for start in stride(from: n - 2 * len, through: 0, by: -1) {
                if Array(history[start ..< (start + len)]) == suffix {
                    let draftStart = start + len
                    let avail = min(k, n - draftStart)
                    if avail > 0 {
                        return Array(history[draftStart ..< (draftStart + avail)])
                    }
                }
            }
        }
        // n-gram fallback: last token repeated patterns of length 1..k from bigrams
        if let last = history.last {
            var draft: [Int] = []
            for i in stride(from: n - 2, through: 0, by: -1) {
                if history[i] == last, i + 1 < n {
                    draft.append(history[i + 1])
                    if draft.count >= k { break }
                    // continue chain
                }
            }
            return draft
        }
        return []
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
        while out.count < maxTokens {
            let draft = suffixDraft(history: ids, k: min(draftK, maxTokens - out.count))
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
