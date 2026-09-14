import Foundation

/// Workspace-relative path policy: the model writes paths RELATIVE to a
/// fixed workspace root (so it never fumbles long absolute paths), and momij
/// resolves them to absolute before the harness executes the call. Absolute
/// paths inside the workspace are normalized back to the root, and the
/// conversation history is stripped of the root prefix so the model's world
/// stays purely relative.
public enum PathPolicy {
    /// MOMIJ_WORKSPACE env: the fixed workspace root. nil = mode off.
    public static func workspaceRoot() -> String? {
        guard let raw = getenv("MOMIJ_WORKSPACE") else { return nil }
        let s = String(cString: raw).trimmingCharacters(in: .whitespaces)
        return s.isEmpty ? nil : s
    }

    /// Path-bearing argument fields known to harness tools.
    public static let pathFields = ["path", "file_path"]

    /// Strip escape debris (backslashes, quotes) the model tends to add to
    /// paths, and collapse doubled separators. Path fields only — never used
    /// on free-text arguments.
    public static func normalizeEscapes(_ path: String) -> String {
        var dropped = ""
        for ch in path where ch != "\\" && ch != "\"" {
            dropped.append(ch)
        }
        var collapsed = ""
        var prevSlash = false
        for ch in dropped {
            if ch == "/" {
                if prevSlash { continue }
                prevSlash = true
            } else {
                prevSlash = false
            }
            collapsed.append(ch)
        }
        return collapsed
    }

    /// Resolve a tool-call path value against the workspace root.
    /// - relative -> root + "/" + value
    /// - absolute inside root -> kept (normalized)
    /// - absolute outside root -> kept unchanged (harness's business;
    ///   logged by the caller)
    public static func resolve(_ value: String, in root: String) -> String {
        let cleaned = normalizeEscapes(value)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.isEmpty { return cleaned }
        if cleaned.hasPrefix("/") {
            // Absolute: normalize the root prefix to exactly one form.
            if cleaned.hasPrefix(root) { return cleaned }
            return cleaned
        }
        if root.hasSuffix("/") { return root + cleaned }
        return root + "/" + cleaned
    }

    /// Rewrite path fields inside an arguments JSON string. Returns nil when
    /// nothing changed (caller keeps the original).
    public static func rewrittenArgs(_ argsJSON: String, root: String) -> String? {
        guard let args = ToolMarkup.jsonParseValue(argsJSON) as? [String: Any] else { return nil }
        var changed = false
        var out = args
        for field in pathFields {
            guard let v = args[field] as? String, !v.isEmpty else { continue }
            let resolved = resolve(v, in: root)
            if resolved != v {
                out[field] = resolved
                changed = true
            }
        }
        guard changed else { return nil }
        return ToolMarkup.jsonDumps(out)
    }

    /// Strip the workspace-root prefix from text the model sees (tool
    /// results, history) so its world stays relative and short.
    public static func stripRootPrefix(in text: String, root: String) -> String {
        guard !root.isEmpty else { return text }
        let withSlash = root.hasSuffix("/") ? root : root + "/"
        return text.replacingOccurrences(of: withSlash, with: "")
            .replacingOccurrences(of: root, with: "")
    }

    /// True when the path (after escape normalization) is absolute and
    /// OUTSIDE the workspace root — a workspace-discipline violation that
    /// the repair pipeline should re-ask for, and the sandbox would reject.
    public static func isOutsideWorkspace(_ value: String, root: String) -> Bool {
        let cleaned = normalizeEscapes(value)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard cleaned.hasPrefix("/") else { return false }
        return !cleaned.hasPrefix(root)
    }


    /// True when ANY path field in the parsed args resolves outside the
    /// workspace root (workspace mode only) — a repairable violation.
    public static func argsOutsideWorkspace(
        _ args: [String: Any], root: String
    ) -> Bool {
        for f in pathFields {
            if let v = args[f] as? String, !v.isEmpty,
               isOutsideWorkspace(v, root: root) {
                return true
            }
        }
        return false
    }

    /// The instruction line appended to the tools suffix in workspace mode.
    public static func suffixLine() -> String {
        "Paths in tool calls are RELATIVE to the workspace root. "
            + "Never write absolute paths; never escape slashes."
    }
}