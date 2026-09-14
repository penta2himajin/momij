import Foundation

/// Protocol-tool interception (the ending fumble absorb): after long
/// tool-call runs Maple signals "I'm done" by calling a protocol tool
/// (`submit`, `ask_user_question`) — often with unusable arguments, which
/// the harness rejects, looping the turn. momij absorbs the intent: drop
/// every parsed call and end the turn with a clean stop + the call's
/// description (or a placeholder) as the final content.
public enum ProtocolAbsorb {
    /// The tool names whose calls get absorbed. MOMIJ_PROTOCOL_TOOLS
    /// overrides the default (comma-separated); an empty value disables
    /// interception entirely.
    public static func toolNames(env: String? = ProcessInfo.processInfo.environment["MOMIJ_PROTOCOL_TOOLS"]) -> Set<String> {
        guard let raw = env else {
            return ["submit", "ask_user_question"]
        }
        let names = raw.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }
        return Set(names)
    }

    /// The first parsed call targeting a protocol tool (nil = no absorb).
    /// Empty-named calls can never match.
    public static func absorbCall(
        in calls: [OpenAIChatCompat.ToolCallSpec], names: Set<String>
    ) -> OpenAIChatCompat.ToolCallSpec? {
        calls.first { !$0.name.isEmpty && names.contains($0.name) }
    }

    /// Final content for an absorbed call: its `description` argument
    /// (when usable) or the placeholder — the model's "I'm done" intent
    /// becomes the final assistant message.
    public static func finalContent(for call: OpenAIChatCompat.ToolCallSpec) -> String {
        let args = ArgRepair.parseArgs(call.arguments)
        if let d = args["description"] as? String {
            let trimmed = d.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return "Task complete."
    }
}