import Foundation

/// Maple's shipped `chat_template.jinja` always opens assistant with `<think>\n`.
/// That forces a reasoning channel and, on long prompts, collapses into loops /
/// raw `<|im_start|>` text. Serve defaults to non-thinking generation; set
/// `MOMIJ_ENABLE_THINKING=1` to keep the stock template.
public enum ChatTemplatePatch {
    public static var enableThinking: Bool {
        guard let raw = getenv("MOMIJ_ENABLE_THINKING") else { return false }
        let s = String(cString: raw)
        return s == "1" || s.lowercased() == "true"
    }

    /// Replace the generation-prompt suffix that injects `<think>`.
    /// Maple's jinja stores escaped `\n` inside the string literal (backslash + n).
    public static func withoutForcedThink(_ template: String) -> String {
        let from = "{{- '<|im_start|>assistant\\n<think>\\n' }}"
        let to = "{{- '<|im_start|>assistant\\n' }}"
        guard template.contains(from) else { return template }
        return template.replacingOccurrences(of: from, with: to)
    }

    public static func loadModelTemplate(modelDir: String) -> String? {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("chat_template.jinja")
        return try? String(contentsOf: url, encoding: .utf8)
    }

    public static var bannedAssistantTokenIds: [Int] {
        enableThinking ? [] : [151_667, 151_644]  // <think>, <|im_start|>
    }

    /// Template string for applyChatTemplate, or nil to use the tokenizer default.
    public static func serveTemplate(modelDir: String) -> String? {
        if enableThinking { return nil }
        guard let raw = loadModelTemplate(modelDir: modelDir) else { return nil }
        let patched = withoutForcedThink(raw)
        return patched == raw ? nil : patched
    }

    /// If thinking is disabled and the model still emitted a think block, keep the
    /// visible answer only (client content channel). Unclosed `<think>` is dropped
    /// rather than leaked as assistant text.
    public static func stripThinkForContent(_ text: String, thinkingEnabled: Bool = enableThinking) -> String {
        guard !thinkingEnabled else { return text }
        var s = text
        while let open = s.range(of: "<think>") {
            if let close = s.range(of: "</think>", range: open.upperBound..<s.endIndex) {
                s.removeSubrange(open.lowerBound..<close.upperBound)
            } else {
                s = String(s[..<open.lowerBound])
                break
            }
        }
        s = s.replacingOccurrences(of: "</think>", with: "")
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
