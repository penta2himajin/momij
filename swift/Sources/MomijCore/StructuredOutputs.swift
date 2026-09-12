import Foundation
import MLX

/// oMLX-compatible constrained-decoding request fields.
///
/// Supported enforcement (P1): finite languages via token-frontier masking —
/// `choice`, literal-union EBNF (`root ::= "a" | "b"`), and `guided_grammar` alias.
/// Open JSON Schema / complex regex need a fuller grammar engine (xgrammar); those
/// parse into `.unsupported` so the serve path can 400 clearly rather than ignore.
public enum StructuredOutputs {
    public enum Constraint: Equatable, Sendable {
        /// One of the exact strings (then EOS).
        case choice([String])
        /// JSON Schema string (xgrammar).
        case jsonSchema(String)
        /// Full EBNF / regex via xgrammar (non-literal).
        case ebnf(String)
        case regex(String)
        /// Opaque / not yet enforced.
        case unsupported(String)
    }

    /// Extract constraint from a chat.completions JSON object.
    /// Prefer `structured_outputs.*`; `guided_grammar` is an alias for grammar.
    public static func parseConstraint(from root: [String: Any]) -> Constraint? {
        if let so = root["structured_outputs"] as? [String: Any] {
            if let choices = so["choice"] as? [Any] {
                let strs = choices.compactMap { $0 as? String }.filter { !$0.isEmpty }
                if !strs.isEmpty { return .choice(strs) }
            }
            if let g = so["grammar"] as? String {
                if let c = parseLiteralUnionEBNF(g) { return c }
                return .ebnf(g)
            }
            if let r = so["regex"] as? String {
                if let c = parseLiteralUnionRegex(r) { return c }
                return .regex(r)
            }
            if let json = so["json"] {
                if let s = json as? String {
                    return .jsonSchema(s)
                }
                if JSONSerialization.isValidJSONObject(json),
                   let data = try? JSONSerialization.data(withJSONObject: json),
                   let s = String(data: data, encoding: .utf8)
                {
                    return .jsonSchema(s)
                }
                return .unsupported("structured_outputs.json (unserializable)")
            }
        }
        if let g = root["guided_grammar"] as? String {
            if let c = parseLiteralUnionEBNF(g) { return c }
            return .ebnf(g)
        }
        // Top-level `grammar` is ignored (oMLX parity).
        return nil
    }

    /// `root ::= "a" | "b"` (optional whitespace). Returns nil if not this shape.
    public static func parseLiteralUnionEBNF(_ src: String) -> Constraint? {
        var s = src.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("root") {
            s = String(s.drop(while: { $0 != "=" }))
            if s.hasPrefix("::=") {
                s = String(s.dropFirst(3))
            } else if s.hasPrefix("=") {
                s = String(s.dropFirst())
            }
            s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return parseQuotedAlternation(s)
    }

    /// `^(?:a|b)$` or `a|b` of quoted/plain literals without regex metachars.
    public static func parseLiteralUnionRegex(_ src: String) -> Constraint? {
        var s = src.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("^") { s.removeFirst() }
        if s.hasSuffix("$") { s.removeLast() }
        if s.hasPrefix("(?:"), s.hasSuffix(")") {
            s = String(s.dropFirst(3).dropLast())
        }
        return parseQuotedAlternation(s)
    }

    private static func parseQuotedAlternation(_ src: String) -> Constraint? {
        let parts = splitTopLevelAlternation(src)
        guard !parts.isEmpty else { return nil }
        var out: [String] = []
        for p in parts {
            let t = p.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let lit = unwrapQuotedLiteral(t) else { return nil }
            // Reject regex/ebnf metachar leftovers inside bare tokens.
            if lit.contains(where: { "[](){}*+?.\\".contains($0) }) { return nil }
            out.append(lit)
        }
        return out.isEmpty ? nil : .choice(out)
    }

    private static func unwrapQuotedLiteral(_ s: String) -> String? {
        if s.count >= 2 {
            let a = s.first!, b = s.last!
            if (a == "\"" && b == "\"") || (a == "'" && b == "'") {
                return String(s.dropFirst().dropLast())
            }
        }
        // Bare token (ZQX, red, …) — only [A-Za-z0-9_]+
        guard !s.isEmpty, s.unicodeScalars.allSatisfy({
            CharacterSet.alphanumerics.contains($0) || $0 == "_"
        }) else { return nil }
        return s
    }

    private static func splitTopLevelAlternation(_ src: String) -> [String] {
        var parts: [String] = []
        var cur = ""
        var depth = 0
        var inQuote: Character? = nil
        for ch in src {
            if let q = inQuote {
                cur.append(ch)
                if ch == q { inQuote = nil }
                continue
            }
            if ch == "\"" || ch == "'" {
                inQuote = ch
                cur.append(ch)
                continue
            }
            if ch == "(" { depth += 1; cur.append(ch); continue }
            if ch == ")" { depth = max(0, depth - 1); cur.append(ch); continue }
            if ch == "|", depth == 0 {
                parts.append(cur)
                cur = ""
                continue
            }
            cur.append(ch)
        }
        if !cur.isEmpty { parts.append(cur) }
        return parts
    }
}

/// Greedy constrained decode over a finite set of UTF-8 strings.
///
/// Builds token sequences with `encode` (no special tokens), then at each step
/// only tokens that continue at least one remaining alternative are allowed.
public struct FiniteStringGuide: Sendable {
    public let alternatives: [[Int]]
    public let eosTokenId: Int

    public init(strings: [String], encode: (String) throws -> [Int], eosTokenId: Int) throws {
        var alts: [[Int]] = []
        for s in strings {
            let ids = try encode(s)
            guard !ids.isEmpty || s.isEmpty else {
                throw GuideError.encodeEmpty(s)
            }
            alts.append(ids)
        }
        self.alternatives = alts
        self.eosTokenId = eosTokenId
    }

    public enum GuideError: Error, Equatable, CustomStringConvertible {
        case encodeEmpty(String)
        case deadEnd
        case noAllowedToken

        public var description: String {
            switch self {
            case .encodeEmpty(let s): return "encode produced no tokens for \(s.debugDescription)"
            case .deadEnd: return "constraint dead-end"
            case .noAllowedToken: return "no allowed next token under constraint"
            }
        }
    }

    /// Token ids still compatible with `prefix` (generated so far).
    public func surviving(prefix: [Int]) -> [[Int]] {
        alternatives.filter { alt in
            alt.count >= prefix.count && Array(alt.prefix(prefix.count)) == prefix
        }
    }

    /// Allowed next token ids given generated `prefix`. Includes EOS when a
    /// surviving alternative is exactly complete.
    public func allowedNext(prefix: [Int]) -> Set<Int> {
        var next = Set<Int>()
        for alt in surviving(prefix: prefix) {
            if alt.count == prefix.count {
                next.insert(eosTokenId)
            } else {
                next.insert(alt[prefix.count])
            }
        }
        return next
    }

    /// Pick argmax among `scores` restricted to `allowed`. `ids[i]` pairs with `scores[i]`.
    public static func argmaxAllowed(ids: [Int], scores: [Float], allowed: Set<Int>) -> Int? {
        var bestId: Int?
        var bestScore = -Float.greatestFiniteMagnitude
        for i in ids.indices where allowed.contains(ids[i]) {
            if scores[i] > bestScore {
                bestScore = scores[i]
                bestId = ids[i]
            }
        }
        return bestId
    }
}

/// Host-side argmax among a grammar-allowed set. Used by mlx (after one logits
/// copy) and seedless constrained decode — never per-id GPU `.item()`.
public enum ConstrainedPick {
    public static func argmax(allowed: Set<Int>, scores: [Float]) -> Int? {
        var bestId: Int?
        var bestScore = -Float.greatestFiniteMagnitude
        for id in allowed {
            guard id >= 0, id < scores.count else { continue }
            let s = scores[id]
            if s > bestScore {
                bestScore = s
                bestId = id
            }
        }
        return bestId
    }

    public static func applyBanned(_ scores: inout [Float], banned: [Int]) {
        for id in banned where id >= 0 && id < scores.count {
            scores[id] = -.infinity
        }
    }

    public static func argmaxAll(_ scores: [Float]) -> Int {
        var bestId = 0
        var best = -Float.greatestFiniteMagnitude
        for i in scores.indices where scores[i] > best {
            best = scores[i]
            bestId = i
        }
        return bestId
    }

    public static func hostScores(_ logits: MLXArray) -> [Float] {
        var row = logits
        while row.ndim > 1 { row = row[0] }
        let flat = row.asType(.float32)
        MLX.eval(flat)
        let n = flat.dim(0)
        var scores = [Float](repeating: -Float.infinity, count: n)
        flat.asData(access: .copy).data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float.self)
            let m = min(n, raw.count / MemoryLayout<Float>.size)
            for i in 0 ..< m { scores[i] = src[i] }
        }
        return scores
    }
}
