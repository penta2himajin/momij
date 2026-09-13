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

    /// evprtr `original_user_text`: prefer the primary user task (the
    /// longest early user message), skipping tiny placeholder turns.
    public static func originalUserText(_ messages: [OpenAIChatCompat.ChatMessage]) -> String {
        var candidates: [(idx: Int, len: Int, text: String)] = []
        for (idx, m) in messages.enumerated() where m.role == "user" {
            let text = m.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            let lowered = text.lowercased()
            if ["agent", "(no user text)", "no user text"].contains(lowered) { continue }
            candidates.append((idx, text.count, text))
        }
        if candidates.isEmpty { return "(no user text)" }
        let maxLen = candidates.map { $0.len }.max() ?? 0
        let threshold = max(80, Int(Double(maxLen) * 0.6))
        let primary = candidates.filter { $0.len >= threshold }
        let pool = primary.isEmpty ? candidates : primary
        return pool.min { ($0.idx, -$0.len) < ($1.idx, -$1.len) }?.text ?? "(no user text)"
    }

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
    /// object; everything else in the call is preserved by the caller.
    public static func fieldRepairPrompt(
        task: String, tool: String, argsJSON: String, fields: [String], reasons: [String: String]
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
        guard !arguments.isEmpty,
              let obj = (try? JSONSerialization.jsonObject(
                  with: Data(arguments.utf8), options: [.fragmentsAllowed])) as? [String: Any]
        else { return [:] }
        return obj
    }

    /// Extract the first JSON object from a model reply (the constrained
    /// generation should be pure JSON; tolerate stray prose/markdown).
    public static func firstJSONObject(in text: String) -> [String: Any]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let obj = (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))) as? [String: Any] {
            return obj
        }
        guard let start = trimmed.firstIndex(of: "{"),
              let end = trimmed.lastIndex(of: "}"), start < end
        else { return nil }
        let slice = String(trimmed[start ... end])
        return (try? JSONSerialization.jsonObject(with: Data(slice.utf8))) as? [String: Any]
    }
}