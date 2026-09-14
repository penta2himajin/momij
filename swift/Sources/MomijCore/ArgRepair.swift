import Foundation

/// Targeted field repair for degenerate tool-call arguments.
///
/// Instead of regenerating the whole reply (which breaks good parts),
/// re-ask the model for ONLY the broken fields, merge them into the
/// existing arguments, and re-validate. Ported concept from evprtr's
/// FreshConstrainedRepair narrowed to field scope; the field/reason
/// mapping follows `ResponseVerify.degenerateToolArgs` detail keys.
public enum ArgRepair {
    // MARK: - Which fields to re-ask for

    /// Map a degenerate hit's reasons to the argument fields to fix.
    /// `detail` is the hit's detail dict (reason / tool keys).
    public static func fieldsToFix(detail: [String: Any]) -> [String] {
        switch detail["reason"] as? String {
        case "path_not_filelike":
            return ["path"]
        case "content_equals_path", "cargo_toml_too_thin", "source_content_too_short":
            return ["content"]
        case "empty_edits", "empty_edit_texts", "noop_edit", "invented_old_text":
            return ["edits"]
        default:
            return []
        }
    }

    /// fieldsToFix adapted to the field names the call actually uses
    /// (path <-> file_path aliasing — live calls use either key).
    public static func repairFields(detail: [String: Any], args: [String: Any]) -> [String] {
        let base = fieldsToFix(detail: detail)
        guard !base.isEmpty else { return [] }
        return base.map { f in
            if args[f] != nil { return f }
            if f == "path", args["file_path"] != nil { return "file_path" }
            if f == "file_path", args["path"] != nil { return "path" }
            return f
        }
    }

    /// Required argument fields (per the tool's JSON schema) that are absent
    /// or null — the harness-side "missing required property" rejection,
    /// caught here so the repair can fill them BEFORE the call leaves.
    public static func missingRequiredFields(
        toolLines: [String], tool: String, args: [String: Any]
    ) -> [String] {
        guard let schema = toolSchema(toolLines: toolLines, tool: tool) else { return [] }
        let required = schema["required"] as? [String] ?? []
        return required.filter { f in
            guard let v = args[f] else { return true }
            if v is NSNull { return true }
            if let s = v as? String { return s.isEmpty }
            return false
        }
    }

    /// evprtr `original_user_text`: prefer the primary user task (the
    /// longest early user message), skipping tiny placeholder turns AND
    /// harness-generated scaffolding turns. The latter are longer than the
    /// task, so the length heuristic would otherwise select them and path
    /// extraction would read paths quoted from injected context or tool
    /// output instead of the task (both measured live).
    public static func originalUserText(_ messages: [OpenAIChatCompat.ChatMessage]) -> String {
        var candidates: [(idx: Int, len: Int, text: String)] = []
        for (idx, m) in messages.enumerated() where m.role == "user" {
            let text = m.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            let lowered = text.lowercased()
            if ["agent", "(no user text)", "no user text"].contains(lowered) { continue }
            if isNonTaskUserTurn(text) { continue }
            candidates.append((idx, text.count, text))
        }
        if candidates.isEmpty { return "(no user text)" }
        let maxLen = candidates.map { $0.len }.max() ?? 0
        let threshold = max(80, Int(Double(maxLen) * 0.6))
        let primary = candidates.filter { $0.len >= threshold }
        let pool = primary.isEmpty ? candidates : primary
        return pool.min { ($0.idx, -$0.len) < ($1.idx, -$1.len) }?.text ?? "(no user text)"
    }

    /// A user turn that is harness-generated scaffolding rather than task
    /// text. Markers observed live in DSH subagent requests:
    /// - `<system-reminder>`: DSH-injected workspace instructions.
    /// - `<tool_result>`: tool output after `ToolMarkup.rewriteMessages`
    ///   folds role "tool" into role "user" for the model-visible history.
    /// - `Current runtime context`: the DSH runtime/policy preamble turn.
    /// Only the WRAPPER form (turn starts with the marker) is excluded; a
    /// task message that merely quotes one stays a candidate.
    public static func isNonTaskUserTurn(_ text: String) -> Bool {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return nonTaskTurnPrefixes.contains { t.hasPrefix($0) }
    }

    /// Prefix markers of harness scaffolding turns (see `isNonTaskUserTurn`).
    public static let nonTaskTurnPrefixes = [
        "<system-reminder>", "<tool_result>", "Current runtime context",
    ]

    // MARK: - Schema for the constrained re-ask

    /// Extract the tool's `parameters` JSON schema from the request tool lines.
    public static func toolSchema(toolLines: [String], tool: String) -> [String: Any]? {
        for line in toolLines {
            guard let obj = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
                  let fn = obj["function"] as? [String: Any],
                  (fn["name"] as? String) == tool,
                  let params = fn["parameters"] as? [String: Any]
            else { continue }
            return params
        }
        return nil
    }

    /// Build a strict object schema limited to `fields`, pulling each
    /// property's schema from the tool definition (fallback: string).
    public static func constrainedObjectSchema(
        toolLines: [String], tool: String, fields: [String]
    ) -> [String: Any]? {
        guard !fields.isEmpty else { return nil }
        let schema = toolSchema(toolLines: toolLines, tool: tool)
        let props = schema?["properties"] as? [String: Any]
        var outProps: [String: Any] = [:]
        for f in fields {
            if let p = props?[f] as? [String: Any] {
                outProps[f] = p
            } else {
                outProps[f] = ["type": "string"]
            }
        }
        return [
            "type": "object",
            "properties": outProps,
            "required": fields,
            "additionalProperties": false,
        ]
    }

    // MARK: - Prompt

    /// English-only (Maple). Asks for exactly the broken fields as a JSON
    /// object; everything else in the call is preserved.
    public static func fieldRepairPrompt(
        task: String, tool: String, argsJSON: String,
        fields: [String], reasons: [String: String]
    ) -> String {
        let problemLines = fields.map { f in
            "- \(f): \(reasons[f] ?? "unusable")"
        }.joined(separator: "\n")
        return """
        Fix the arguments of a failed tool call. Reply with ONLY a JSON object and nothing else.

        Task (what the user asked for):
        \(task)

        The assistant tried to call the `\(tool)` tool with these arguments:
        \(argsJSON)

        Problems found:
        \(problemLines)

        Reply with ONLY a JSON object that has exactly these keys: \(fields.joined(separator: ", "))
        Use corrected values that make the `\(tool)` call usable. No prose, no markdown.
        """
    }

    // MARK: - Merge + serialize

    /// Merge the generated field values into the original arguments.
    /// Only `fields` are taken from `generated`; everything else is kept.
    public static func mergeFields(
        into args: [String: Any], generated: [String: Any], fields: [String]
    ) -> [String: Any] {
        var out = args
        for f in fields {
            if let v = generated[f] { out[f] = v }
        }
        return out
    }

    /// Serialize merged args back to the OpenAI arguments JSON string
    /// (ToolMarkup.jsonDumps keeps the contract's python-style spacing).
    public static func argsJSONString(_ args: [String: Any]) -> String {
        ToolMarkup.jsonDumps(args)
    }

    /// Parse `function.arguments` (JSON object string) leniently; falls
    /// back to an empty dict like evprtr `_fn_args`.
    public static func parseArgs(_ arguments: String) -> [String: Any] {
        guard let obj = ToolMarkup.jsonParseValue(arguments) as? [String: Any] else {
            return [:]
        }
        return obj
    }

    /// Extract the first JSON object from a model reply (the constrained
    /// generation should be pure JSON; tolerate stray prose/markdown and
    /// literal newlines inside strings).
    public static func firstJSONObject(in text: String) -> [String: Any]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let obj = ToolMarkup.jsonParseValue(trimmed) as? [String: Any] {
            return obj
        }
        guard let start = trimmed.firstIndex(of: "{"),
              let end = trimmed.lastIndex(of: "}"), start < end
        else { return nil }
        let slice = String(trimmed[start ... end])
        return ToolMarkup.jsonParseValue(slice) as? [String: Any]
    }

    /// Extract the most likely target path from a task text (deterministic —
    /// the task names the exact relative path; the model only needs to copy
    /// it, but live sessions showed it corrupting instead). Codeish suffixes
    /// define candidate tokens; the longest slash-joined candidate wins.
    /// Returns nil when the task names no path (the caller falls back to the
    /// model re-ask).
    public static func pathFromTask(_ task: String) -> String? {
        let wordChars = CharacterSet(
            charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-")
        var tokens: [String] = []
        var current = ""
        for ch in task {
            let scalars = Array(ch.unicodeScalars)
            let isWord = scalars.allSatisfy { wordChars.contains($0) }
            if isWord {
                current.append(ch)
            } else if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }
        if !current.isEmpty { tokens.append(current) }
        var candidates: [String] = []
        for tok in tokens {
            // Leading "./" pairs and bare slashes are debris; a lone leading
            // dot is hidden-directory semantics and must survive
            // (".github/PULL_REQUEST_TEMPLATE.md" is not "github/...").
            var t = Substring(tok)
            while t.hasPrefix("./") { t = t.dropFirst(2) }
            while t.hasPrefix("/") { t = t.dropFirst() }
            while let last = t.last, last == "." || last == "/" { t = t.dropLast() }
            let cleaned = String(t)
            if cleaned.isEmpty { continue }
            if pathSuffixSet.contains(where: { cleaned.lowercased().hasSuffix($0) }) {
                candidates.append(cleaned)
            }
        }
        let slashed = candidates.filter { $0.contains("/") }
        if let best = slashed.max(by: { $0.count < $1.count }) { return best }
        return candidates.max(by: { $0.count < $1.count })
    }

    private static let pathSuffixSet: Set<String> = [
        ".toml", ".rs", ".py", ".ts", ".js", ".json", ".md", ".txt", ".yml", ".yaml",
    ]

    // MARK: - Extraction-first path fix (pathFromTask wiring)

    /// True when a pending fix targets a path field: the workspace scan
    /// (path_outside_workspace), an aliased path field from a degenerate
    /// hit, or a schema-required path property missing on the call.
    public static func isPathFix(kind: String, fields: [String]) -> Bool {
        if kind == "path_outside_workspace" { return true }
        return !Set(fields).isDisjoint(with: PathPolicy.pathFields)
    }

    /// Extraction-first answer for a path-kind pending fix: the task text
    /// names the exact relative path, so merge `pathFromTask(task)` into
    /// the pending path field(s) instead of re-asking the model (which
    /// repeats its hallucination). Returns nil when the fix is not
    /// path-kind, the task names no path, or the fix also covers non-path
    /// fields (extraction cannot answer those — the caller falls back to
    /// the model re-ask). The adoption decision stays with the caller's
    /// clean check (degenerate / required / outside-workspace).
    public static func taskPathMerge(
        kind: String, fields: [String], args: [String: Any], task: String
    ) -> [String: Any]? {
        guard isPathFix(kind: kind, fields: fields) else { return nil }
        guard let extracted = pathFromTask(task) else { return nil }
        let pathFieldsTouched = fields.filter { PathPolicy.pathFields.contains($0) }
        guard !pathFieldsTouched.isEmpty, pathFieldsTouched.count == fields.count
        else { return nil }
        var out = args
        for f in pathFieldsTouched { out[f] = extracted }
        return out
    }
}