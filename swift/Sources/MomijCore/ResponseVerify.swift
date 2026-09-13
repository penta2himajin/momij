import Foundation

/// Verify/repair policies ported from evprtr `compositor/verify/`.
///
/// `policy_id` / `kind` / `detail` values match the Python originals
/// byte-for-byte so trace records can be compared across both
/// implementations (evprtr as oracle, momij as native runtime).
/// Offsets are UTF-16 code units (NSRegularExpression); Python's are
/// code points — identical for the ASCII fixtures used in parity tests.
public enum ResponseVerify {
    // MARK: - Types (parity with compositor/verify/types.py + repetition.py)

    /// A verify hit (evprtr `Diagnosis`).
    public struct Hit {
        public let policyId: String
        public let field: String
        public let kind: String
        public let onset: Int
        public let detail: [String: Any]

        public func eventDetail() -> [String: Any] {
            [
                "policy_id": policyId,
                "field": field,
                "kind": kind,
                "onset": onset,
                "detail": detail,
            ]
        }
    }

    /// Raw repetition finding (evprtr `RepetitionHit`).
    public struct RepetitionHit {
        public let kind: String
        public let onset: Int
        public let detail: [String: Any]
    }

    /// Assistant-message view the detectors operate on (evprtr passes the
    /// OpenAI message dict; momij passes decoded channels explicitly).
    public struct Message {
        public let content: String?
        public let reasoningContent: String?
        public let toolCalls: [OpenAIChatCompat.ToolCallSpec]

        public init(
            content: String? = nil, reasoningContent: String? = nil,
            toolCalls: [OpenAIChatCompat.ToolCallSpec] = []
        ) {
            self.content = content
            self.reasoningContent = reasoningContent
            self.toolCalls = toolCalls
        }
    }

    /// Per-request verdicts: every fired detector (bundle order) plus the
    /// evprtr-equivalent first hit (their pipeline short-circuits).
    public struct Outcome {
        public let hits: [Hit]
        public var first: Hit? { hits.first }
    }

    public static let genericRepetitionId = "generic.repetition"
    public static let thinContentId = "maple_preview.thin_content"
    public static let emptyLongReasoningId = "maple_preview.empty_content_long_reasoning"
    public static let pseudoToolMarkupId = "maple_preview.pseudo_tool_markup"
    public static let degenerateToolArgsId = "maple_preview.degenerate_tool_args"

    // Markup markers, assembled at runtime so source text never embeds a
    // contiguous marker (same trick ToolMarkup uses for the call tags).
    static let toolCallOpen = "<" + "tool_call"
    static let thinkOpen = "<" + "think" + ">"
    static let thinkClose = "<" + "/" + "think" + ">"

    // MARK: - repetition.py port

    /// Detect runaway token/phrase repetition. Port of
    /// `compositor/verify/repetition.py:find_repetition` (same thresholds:
    /// word_run 8, ngram 3x5, char motif 2..24 chars repeated 8+ times).
    public static func findRepetition(_ text: String) -> RepetitionHit? {
        guard !text.isEmpty, text.count >= 32 else { return nil }
        let ns = text as NSString
        var hits: [RepetitionHit] = []

        let tokenRe = try! NSRegularExpression(pattern: "\\S+")
        let matches = tokenRe.matches(in: text, range: NSRange(location: 0, length: ns.length))
        let tokens = matches.map { ns.substring(with: $0.range) }

        // Word run: >= 8 identical consecutive tokens.
        if tokens.count >= 8 {
            var run = 1
            for i in 1 ..< tokens.count {
                if tokens[i] == tokens[i - 1] {
                    run += 1
                    if run >= 8 {
                        let onset = matches[i - run + 1].range.location
                        hits.append(RepetitionHit(
                            kind: "word_run", onset: onset,
                            detail: [
                                "token": String(ns.substring(with: matches[i].range).pyPrefix(80)),
                                "run": run,
                            ]))
                        break
                    }
                } else {
                    run = 1
                }
            }
        }

        // N-gram run: >= 5 consecutive identical 3-grams (space-joined tokens).
        if tokens.count >= 3 * 5 {
            let grams = (0 ... (tokens.count - 3)).map { i in
                tokens[i ... i + 2].joined(separator: " ")
            }
            var run = 1
            for i in 1 ..< grams.count {
                if grams[i] == grams[i - 1] {
                    run += 1
                    if run >= 5 {
                        let onset = matches[i - run + 1].range.location
                        hits.append(RepetitionHit(
                            kind: "ngram_run", onset: onset,
                            detail: [
                                "ngram": String(grams[i].pyPrefix(120)),
                                "run": run,
                                "n": 3,
                            ]))
                        break
                    }
                } else {
                    run = 1
                }
            }
        }

        // Char motif: a 2..24 char unit repeated >= 8 times contiguously.
        // Like Python's re, the dot must not match "\n".
        // Guard (evprtr parity, 2026-09-13 fix): whitespace-only motifs
        // (markdown hard breaks, code indentation runs) are idiomatic
        // formatting, not model collapse.
        let motifRe = try! NSRegularExpression(pattern: "(.{2,24}?)\\1{7,}")
        if let m = motifRe.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) {
            let motif = ns.substring(with: m.range(at: 1))
            if !motif.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                hits.append(RepetitionHit(
                    kind: "char_motif", onset: m.range.location,
                    detail: [
                        "motif": String(motif.pyPrefix(80)),
                        "span": m.range.length,
                    ]))
            }
        }

        guard !hits.isEmpty else { return nil }
        var best = hits[0]
        for h in hits.dropFirst() where h.onset < best.onset { best = h }
        return best
    }

    /// `text[: onset].rstrip()` (evprtr `truncate_before_repetition`).
    public static func truncateBeforeRepetition(_ text: String) -> String {
        guard let hit = findRepetition(text) else { return text }
        let ns = text as NSString
        let cut = ns.substring(to: min(hit.onset, ns.length))
        return (cut as NSString).trimmingTrailingWhitespacePy()
    }

    // MARK: - detectors_maple_preview.py port

    /// Preamble-only answers (ends with ':' and never delivers bullets).
    public static func thinContent(_ message: Message) -> Hit? {
        guard let content = message.content else { return nil }
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return nil }
        let hasListMarker = ["\n-", "\n*", "\n1.", "\n1)"].contains { text.contains($0) }
        if text.hasSuffix(":") && !hasListMarker {
            return Hit(
                policyId: thinContentId, field: "content", kind: "thin_content", onset: 0,
                detail: ["len": text.count])
        }
        return nil
    }

    /// Empty content with a long reasoning blob (common Maple collapse shape).
    public static func emptyContentLongReasoning(
        _ message: Message, minReasoningLen: Int = 800
    ) -> Hit? {
        let contentS = (message.content ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !contentS.isEmpty { return nil }
        guard let reasoning = message.reasoningContent,
              reasoning.count > minReasoningLen
        else { return nil }
        if let hit = findRepetition(reasoning) {
            var detail = hit.detail
            detail["reasoning_len"] = reasoning.count
            return Hit(
                policyId: emptyLongReasoningId, field: "reasoning_content",
                kind: hit.kind, onset: hit.onset, detail: detail)
        }
        return Hit(
            policyId: emptyLongReasoningId, field: "reasoning_content",
            kind: "empty_content_long_reasoning",
            onset: min(400, reasoning.count),
            detail: ["reasoning_len": reasoning.count])
    }

    /// Tool markup leaked into assistant content instead of structured calls.
    /// (`_PSEUDO_TOOL_BLOCK` — full block or a bare open/close tag; lazy body,
    /// DOTALL via bracketed classes, case-insensitive.)
    static let pseudoToolBlockRe = try! NSRegularExpression(
        pattern: toolCallOpen + "\\b[^>]*>[\\s\\S]*?<" + "/" + "tool_call\\b[^>]*>"
            + "|</?" + "tool_call\\b[^>]*>",
        options: .caseInsensitive)

    public static func pseudoToolCallInContent(_ message: Message) -> Hit? {
        guard let content = message.content,
              content.lowercased().contains(toolCallOpen)
        else { return nil }
        let ns = content as NSString
        let match = pseudoToolBlockRe.firstMatch(
            in: content, range: NSRange(location: 0, length: ns.length))
        let onset: Int
        if let match {
            onset = match.range.location
        } else {
            onset = (content.lowercased() as NSString).range(of: toolCallOpen).location
        }
        return Hit(
            policyId: pseudoToolMarkupId, field: "content", kind: "pseudo_tool_markup",
            onset: onset,
            detail: ["has_structured_tool_calls": !message.toolCalls.isEmpty])
    }

    /// `strip_pseudo_tool_markup`: remove markup blocks/tags, trim.
    public static func stripPseudoToolMarkup(_ text: String) -> String {
        let ns = text as NSString
        let cleaned = pseudoToolBlockRe.stringByReplacingMatches(
            in: text, range: NSRange(location: 0, length: ns.length), withTemplate: "")
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - detectors_tools.py port

    static let codeishSuffixes = [
        ".toml", ".rs", ".py", ".ts", ".js", ".json", ".md", ".txt", ".yml", ".yaml",
    ]

    /// Parse `function.arguments` (JSON object string) like evprtr `_fn_args`.
    static func fnArgs(_ call: OpenAIChatCompat.ToolCallSpec) -> (name: String, args: [String: Any]) {
        guard !call.arguments.isEmpty,
              let obj = (try? JSONSerialization.jsonObject(
                  with: Data(call.arguments.utf8), options: [.fragmentsAllowed])) as? [String: Any]
        else { return (call.name, [:]) }
        return (call.name, obj)
    }

    static func looksLikePath(_ path: String) -> Bool {
        if path.isEmpty || path == "." || path == ".." { return false }
        if path.contains("/") || path.contains("\\") { return true }
        let lower = path.lowercased()
        return codeishSuffixes.contains { lower.hasSuffix($0) }
    }

    /// Reject structured tool_calls whose arguments cannot be a real edit.
    public static func degenerateToolArgs(_ message: Message) -> Hit? {
        if message.toolCalls.isEmpty { return nil }
        for (idx, call) in message.toolCalls.enumerated() {
            let (name, args) = fnArgs(call)
            if name == "write", let hit = writeProblem(args) {
                var detail = hit
                detail["tool"] = "write"
                return Hit(
                    policyId: degenerateToolArgsId, field: "tool_calls",
                    kind: "degenerate_tool_args", onset: idx, detail: detail)
            }
            if name == "edit", let hit = editProblem(args) {
                var detail = hit
                detail["tool"] = "edit"
                return Hit(
                    policyId: degenerateToolArgsId, field: "tool_calls",
                    kind: "degenerate_tool_args", onset: idx, detail: detail)
            }
        }
        return nil
    }

    static func writeProblem(_ args: [String: Any]) -> [String: Any]? {
        let path = ((args["path"] as? String) ?? (args["file_path"] as? String) ?? "")
            .trimmingCharacters(in: .whitespaces)
        let content = args["content"] as? String ?? ""
        if !looksLikePath(path) {
            return ["reason": "path_not_filelike", "path": path, "content_len": content.count]
        }
        if content.trimmingCharacters(in: .whitespaces) == path.trimmingCharacters(in: .whitespaces) {
            return ["reason": "content_equals_path", "path": path, "content_len": content.count]
        }
        let lowerPath = path.lowercased().replacingOccurrences(of: "\\", with: "/")
        if lowerPath.hasSuffix("cargo.toml") {
            let s = content as NSString
            if s.range(of: "[package]").location == NSNotFound,
               s.range(of: "axum").location == NSNotFound,
               content.count < 80 {
                return ["reason": "cargo_toml_too_thin", "path": path, "content_len": content.count]
            }
        }
        if [".rs", ".toml", ".py", ".ts"].contains(where: lowerPath.hasSuffix) {
            if content.trimmingCharacters(in: .whitespaces).count < 12 {
                return ["reason": "source_content_too_short", "path": path, "content_len": content.count]
            }
        }
        return nil
    }

    static func editProblem(_ args: [String: Any]) -> [String: Any]? {
        let path = ((args["path"] as? String) ?? (args["file_path"] as? String) ?? "")
            .trimmingCharacters(in: .whitespaces)
        if !path.isEmpty && !looksLikePath(path) {
            return ["reason": "path_not_filelike", "path": path]
        }
        guard let edits = args["edits"] as? [[String: Any]], !edits.isEmpty else {
            return ["reason": "empty_edits", "path": path]
        }
        var usable = 0
        for edit in edits {
            let old = edit["oldText"] as? String ?? edit["old_string"] as? String ?? ""
            let new = edit["newText"] as? String ?? edit["new_string"] as? String ?? ""
            if old.isEmpty && new.isEmpty {
                return ["reason": "empty_edit_texts", "path": path]
            }
            if !old.isEmpty && old == new {
                return ["reason": "noop_edit", "path": path, "old_len": old.count]
            }
            let low = old.lowercased()
            let inventedTokens = [
                "non-empty oldtext", "oldtext and newtext", "mutation tool",
                "use the edit", "do not only",
            ]
            if inventedTokens.contains(where: low.contains) {
                return ["reason": "invented_old_text", "path": path]
            }
            if !old.isEmpty && old.count < 80
                && old.replacingOccurrences(of: " ", with: "").lowercased()
                    .contains("tokio{version") {
                return ["reason": "invented_old_text", "path": path]
            }
            usable += 1
        }
        if usable == 0 {
            return ["reason": "empty_edit_texts", "path": path]
        }
        return nil
    }

    // MARK: - pipeline.py port (VerifyBundle.maple_preview)

    /// Bundle order matches `VerifyBundle.maple_preview`:
    /// repetition -> degenerate tool args -> pseudo markup -> thin -> empty+long.
    /// `Outcome.first` reproduces their short-circuit semantics.
    public static func verify(_ message: Message) -> Outcome {
        var hits: [Hit] = []
        for (field, text) in messageFields(message) {
            if let hit = findRepetition(text) {
                hits.append(Hit(
                    policyId: genericRepetitionId, field: field, kind: hit.kind,
                    onset: hit.onset, detail: hit.detail))
            }
        }
        if let hit = degenerateToolArgs(message) { hits.append(hit) }
        if let hit = pseudoToolCallInContent(message) { hits.append(hit) }
        if let hit = thinContent(message) { hits.append(hit) }
        if let hit = emptyContentLongReasoning(message) { hits.append(hit) }
        return Outcome(hits: hits)
    }

    /// evprtr `message_fields`: text-bearing channels, content first.
    static func messageFields(_ message: Message) -> [(field: String, text: String)] {
        var out: [(String, String)] = []
        if let content = message.content,
           !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            out.append(("content", content))
        }
        if let reasoning = message.reasoningContent,
           !reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            out.append(("reasoning_content", reasoning))
        }
        return out
    }

    /// The interior of the first think block, if any — momij's stand-in for
    /// `reasoning_content` (Seedless decodes the think region into raw text).
    public static func thinkInterior(of raw: String) -> String? {
        guard let open = raw.range(of: thinkOpen) else { return nil }
        if let close = raw.range(of: thinkClose, range: open.upperBound ..< raw.endIndex) {
            return String(raw[open.upperBound ..< close.lowerBound])
        }
        return String(raw[open.upperBound...])
    }
}

extension String {
    /// Prefix by code points, mirroring Python's slice notation.
    func pyPrefix(_ n: Int) -> String {
        String(prefix(n))
    }
}

/// Python `rstrip()` semantics (whitespace + newlines) on an NSString.
extension NSString {
    func trimmingTrailingWhitespacePy() -> String {
        var len = length
        let ws = CharacterSet.whitespacesAndNewlines
        while len > 0, ws.contains(UnicodeScalar(character(at: len - 1)) ?? UnicodeScalar(0)) {
            len -= 1
        }
        return substring(to: len)
    }
}