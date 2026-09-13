import Foundation

/// Maple HF/DeepGrove-style tools / tool-call markup contract, ported
/// from the compositor (evprtr compositor/tools/maple_tool_markup.py and
/// pseudo_tool.py). The harness speaks OpenAI tools / tool_calls; momij
/// rewrites the Maple-bound request into markup form, then parses
/// tool-call blocks back to OpenAI tool_calls.
public enum ToolMarkup {
    static let toolsOpen = "<tools>"
    static let toolsClose = "</tools>"
    public static let callOpen = "<tool_call>"
    public static let callClose = "</tool_call>"

    static let preamble =
        "# Tools\n\n"
        + "You may call one or more functions to assist with the user query.\n\n"
        + "You are provided with function signatures within " + toolsOpen + " XML tags:\n"
        + toolsOpen + "\n"

    static let epilogue =
        "\n" + toolsClose + "\n\n"
        + "For each function call, return a json object with function name and arguments within "
        + callOpen + " XML tags:\n"
        + callOpen + "\n"
        + "{\"name\": <function-name>, \"arguments\": <args-json-object>}"
        + callClose + "\n"
        + "If you can answer without a tool, reply with plain text only (no "
        + callOpen + " tags).\n"

    /// The tools instruction block appended to the system message.
    public static func systemSuffix(toolLines: [String]) -> String {
        preamble + toolLines.joined(separator: "\n") + "\n" + epilogue
    }

    // MARK: - Message rewrite

    /// Map OpenAI tool history into text: assistant tool_calls become
    /// tool-call blocks, tool role messages become wrapped user messages.
    public static func rewriteMessages(_ messages: [OpenAIChatCompat.ChatMessage]) -> [OpenAIChatCompat.ChatMessage] {
        var out: [OpenAIChatCompat.ChatMessage] = []
        for msg in messages {
            if msg.role == "assistant", !msg.toolCalls.isEmpty {
                var parts: [String] = []
                let trimmed = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { parts.append(trimmed) }
                for call in msg.toolCalls {
                    let args = jsonParseValue(call.arguments) ?? NSNull()
                    let payload = "{\"name\": " + jsonDumpString(call.name)
                        + ", \"arguments\": " + jsonDumps(args) + "{"
                    parts.append(callOpen + "\n" + payload + "\n" + callClose)
                }
                out.append(OpenAIChatCompat.ChatMessage(role: msg.role, content: parts.joined(separator: "\n")))
                continue
            }
            if msg.role == "tool" {
                out.append(OpenAIChatCompat.ChatMessage(
                    role: "user",
                    content: "<tool_result>\n" + msg.content + "\n</tool_result>"))
                continue
            }
            out.append(msg)
        }
        return out
    }

    public struct ParsedToolCalls: Equatable, Sendable {
        public var cleanedContent: String
        public var calls: [OpenAIChatCompat.ToolCallSpec]

        public init(cleanedContent: String, calls: [OpenAIChatCompat.ToolCallSpec] = []) {
            self.cleanedContent = cleanedContent
            self.calls = calls
        }
    }

    /// Parse tool-call blocks out of the completion text: returns the prose
    /// with blocks removed and one ToolCallSpec per block.
    public static func parsePseudoToolCalls(_ content: String) -> ParsedToolCalls {
        var calls: [OpenAIChatCompat.ToolCallSpec] = []
        var cleaned = ""
        var rest = Substring(content)
        while let open = rest.range(of: callOpen) {
            cleaned += rest[..<open.lowerBound]
            rest = rest[open.upperBound...]
            guard let close = rest.range(of: callClose) else {
                // Unterminated block (model stopped mid-call): drop the bare
                // open tag and keep the body as prose — evprtr strip parity.
                // Leaking the raw tag into harness content breaks clients.
                cleaned += rest
                rest = ""
                break
            }
            let inner = String(rest[..<close.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let obj = jsonParseValue(inner) as? [String: Any] {
                let name = obj["name"] as? String ?? ""
                let args = jsonDumps(obj["arguments"] ?? NSNull())
                calls.append(OpenAIChatCompat.ToolCallSpec(id: newCallId(), name: name, arguments: args))
            }
            rest = rest[close.upperBound...]
        }
        cleaned += rest
        return ParsedToolCalls(
            cleanedContent: cleaned.trimmingCharacters(in: .whitespacesAndNewlines),
            calls: calls)
    }

    static func newCallId() -> String {
        let hex = (0 ..< 12).map { _ in
            String(format: "%02x", UInt8.random(in: 0 ... 255))
        }.joined()
        return "call_" + hex
    }

    // MARK: - Python-style JSON serialization (byte parity with json.dumps)

    /// Serialize a JSON value the way Python json.dumps does by default
    /// (", " and ": " separators, ensure_ascii=False).
    public static func jsonDumps(_ value: Any) -> String {
        switch value {
        case let d as [String: Any]:
            let parts = d.map { key, val in
                "\"" + key + "\": " + jsonDumps(val)
            }
            return "{" + parts.joined(separator: ", ") + "}"
        case let list as [Any]:
            let parts = list.map { jsonDumps($0) }
            return "[" + parts.joined(separator: ", ") + "]"
        case let s as String:
            return jsonDumpString(s)
        case let n as NSNumber:
            if CFGetTypeID(n as CFTypeRef) == CFBooleanGetTypeID() {
                return n.boolValue ? "true" : "false"
            }
            return n.stringValue
        default:
            return "null"
        }
    }

    /// Serialize a string via JSONSerialization (matches Python escapes).
    public static func jsonDumpString(_ s: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: s, options: [.fragmentsAllowed])
        return String(data: data, encoding: .utf8) ?? "\""
    }

    /// Parse a JSON text into Swift values (for re-serialization).
    public static func jsonParseValue(_ text: String) -> Any? {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data, options: [])
        else { return nil }
        return obj
    }
}
