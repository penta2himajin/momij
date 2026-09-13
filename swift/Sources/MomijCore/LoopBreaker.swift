import Foundation

/// Tool-call loop breaker: detect a trailing run of identical tool calls
/// (same tool, arguments equal after escape-normalization) and produce a
/// steering message that breaks the retry loop.
///
/// Motivated by a live DSH-subagent session (2026-09-13): the model retried
/// `chmod +x` 12+ times, doubling backslash escapes after each failure —
/// the error echoed the mangled path and the model "fixed" it by escaping
/// harder. Normalization makes those variants compare equal so the loop is
/// visible before it burns the budget.
public enum LoopBreaker {
    /// Collapse escape-escalation variants to one comparable form:
    /// remove backslashes and quotes, collapse whitespace.
    public static func normalizedArgs(_ arguments: String) -> String {
        var out = ""
        var lastWasSpace = false
        for ch in arguments {
            if ch == "\\" || ch == "\"" || ch == "'" { continue }
            if ch == " " || ch == "\t" || ch == "\n" {
                if !lastWasSpace { out.append(" ") }
                lastWasSpace = true
            } else {
                out.append(ch)
                lastWasSpace = false
            }
        }
        return out.trimmingCharacters(in: .whitespaces)
    }

    /// Trailing run of assistant tool calls with the same tool name and
    /// equal normalized arguments. Returns (tool, repeat count) when the
    /// run reaches `minRepeats`, else nil.
    public static func detect(
        in messages: [OpenAIChatCompat.ChatMessage], minRepeats: Int = 3
    ) -> (tool: String, count: Int)? {
        var seq: [(name: String, norm: String)] = []
        for m in messages where m.role == "assistant" {
            for call in m.toolCalls {
                seq.append((call.name, normalizedArgs(call.arguments)))
            }
        }
        guard let last = seq.last else { return nil }
        var run = 0
        for entry in seq.reversed() {
            if entry.name == last.name && entry.norm == last.norm { run += 1 }
            else { break }
        }
        guard run >= minRepeats, !last.norm.isEmpty else { return nil }
        return (last.name, run)
    }

    /// The per-request steering message that breaks the loop. English only
    /// (Maple); states the concrete lesson (plain forward slashes) and the
    /// instruction to move on.
    public static func breakNudge(tool: String, count: Int) -> String {
        """
        You have now called the `\(tool)` tool \(count) times in a row with effectively the same arguments (only the escaping changed). Repeating it will not change the result. If a path is involved, use plain forward slashes with no backslashes and no extra quotes. Do NOT retry the same call. Continue with the next step of the task.
        """
    }
}