import Foundation

/// OpenAI-compatible chat request/response helpers for the momij serve path.
/// Unknown JSON fields are ignored (evprtr / Pi may attach tools, structured_outputs, …).
public enum OpenAIChatCompat {
    public struct ChatMessage: Equatable, Sendable {
        public var role: String
        public var content: String

        public init(role: String, content: String) {
            self.role = role
            self.content = content
        }
    }

    public struct ChatCompletionRequest: Equatable, Sendable {
        public var model: String?
        public var messages: [ChatMessage]
        public var maxTokens: Int
        public var temperature: Double
        public var topP: Double
        public var presencePenalty: Double
        public var frequencyPenalty: Double
        public var repetitionPenalty: Double
        public var stream: Bool
        public var n: Int
        public var seed: UInt64?
        /// From `structured_outputs` / `guided_grammar` (nil = unconstrained).
        public var structuredConstraint: StructuredOutputs.Constraint?

        public init(
            model: String? = nil,
            messages: [ChatMessage],
            maxTokens: Int = 256,
            temperature: Double = 0,
            topP: Double = 1,
            presencePenalty: Double = 0,
            frequencyPenalty: Double = 0,
            repetitionPenalty: Double = 1,
            stream: Bool = false,
            n: Int = 1,
            seed: UInt64? = nil,
            structuredConstraint: StructuredOutputs.Constraint? = nil
        ) {
            self.model = model
            self.messages = messages
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.presencePenalty = presencePenalty
            self.frequencyPenalty = frequencyPenalty
            self.repetitionPenalty = repetitionPenalty
            self.stream = stream
            self.n = n
            self.seed = seed
            self.structuredConstraint = structuredConstraint
        }
    }

    public enum ParseError: Error, Equatable, CustomStringConvertible {
        case invalidJSON
        case missingMessages
        case emptyMessages
        case invalidMessage(String)

        public var description: String {
            switch self {
            case .invalidJSON: return "invalid JSON body"
            case .missingMessages: return "missing messages"
            case .emptyMessages: return "messages must be non-empty"
            case .invalidMessage(let s): return s
            }
        }
    }

    /// Parse a chat.completions body. Extra keys (`tools`, `tool_choice`, …) are ignored.
    public static func parseRequest(from data: Data) throws -> ChatCompletionRequest {
        let obj: Any
        do {
            obj = try JSONSerialization.jsonObject(with: data, options: [])
        } catch {
            throw ParseError.invalidJSON
        }
        guard let root = obj as? [String: Any] else { throw ParseError.invalidJSON }
        guard let rawMessages = root["messages"] as? [Any] else { throw ParseError.missingMessages }
        guard !rawMessages.isEmpty else { throw ParseError.emptyMessages }

        var messages: [ChatMessage] = []
        messages.reserveCapacity(rawMessages.count)
        for (i, item) in rawMessages.enumerated() {
            guard let m = item as? [String: Any] else {
                throw ParseError.invalidMessage("messages[\(i)] must be an object")
            }
            guard var role = m["role"] as? String, !role.isEmpty else {
                throw ParseError.invalidMessage("messages[\(i)] missing role")
            }
            if role == "developer" { role = "system" }
            let content = normalizeContent(m["content"])
            messages.append(ChatMessage(role: role, content: content))
        }

        let maxTok: Int = {
            if let v = intValue(root["max_completion_tokens"]), v > 0 { return v }
            if let v = intValue(root["max_tokens"]), v > 0 { return v }
            return 256
        }()

        return ChatCompletionRequest(
            model: root["model"] as? String,
            messages: messages,
            maxTokens: maxTok,
            temperature: doubleValue(root["temperature"]) ?? 0,
            topP: doubleValue(root["top_p"]) ?? 1,
            presencePenalty: doubleValue(root["presence_penalty"]) ?? 0,
            frequencyPenalty: doubleValue(root["frequency_penalty"]) ?? 0,
            repetitionPenalty: doubleValue(root["repetition_penalty"]) ?? 1,
            stream: boolValue(root["stream"]) ?? false,
            n: intValue(root["n"]) ?? 1,
            seed: uint64Value(root["seed"]),
            structuredConstraint: StructuredOutputs.parseConstraint(from: root)
        )
    }

    /// Coerce OpenAI `content` (string | null | multimodal parts) to a single string.
    public static func normalizeContent(_ value: Any?) -> String {
        guard let value, !(value is NSNull) else { return "" }
        if let s = value as? String { return s }
        if let parts = value as? [Any] {
            var out = ""
            for part in parts {
                if let s = part as? String {
                    out += s
                } else if let d = part as? [String: Any] {
                    if let t = d["text"] as? String {
                        out += t
                    } else if let t = d["content"] as? String {
                        out += t
                    }
                }
            }
            return out
        }
        return String(describing: value)
    }

    /// `stop` if generation ended early (EOS / short); `length` if maxTokens filled without EOS.
    public static func finishReason(
        completionTokens: [Int],
        maxTokens: Int,
        eosTokenIds: [Int] = [151_645]
    ) -> String {
        if let last = completionTokens.last, eosTokenIds.contains(last) {
            return "stop"
        }
        if completionTokens.count >= maxTokens {
            return "length"
        }
        return "stop"
    }

    /// Tokens to decode for `message.content`. eos terminates generation: cut
    /// at the first occurrence, not only the trailing one (a speculative chunk
    /// could carry eos mid-sequence).
    public static func contentTokenIds(
        _ tokens: [Int],
        eosTokenIds: [Int] = [151_645]
    ) -> [Int] {
        if let ei = tokens.firstIndex(where: { eosTokenIds.contains($0) }) {
            return Array(tokens.prefix(ei))
        }
        return tokens
    }

    public static func nonStreamJSON(
        modelID: String,
        content: String,
        finishReason: String,
        promptTokens: Int,
        completionTokens: Int,
        id: String = "chatcmpl-momij",
        created: Int = Int(Date().timeIntervalSince1970)
    ) throws -> Data {
        let payload: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": modelID,
            "choices": [[
                "index": 0,
                "message": [
                    "role": "assistant",
                    "content": content,
                ],
                "finish_reason": finishReason,
            ]],
            "usage": [
                "prompt_tokens": promptTokens,
                "completion_tokens": completionTokens,
                "total_tokens": promptTokens + completionTokens,
            ],
        ]
        return try JSONSerialization.data(withJSONObject: payload)
    }

    // MARK: - JSON helpers

    private static func intValue(_ v: Any?) -> Int? {
        switch v {
        case let i as Int: return i
        case let n as NSNumber: return n.intValue
        case let s as String: return Int(s)
        default: return nil
        }
    }

    private static func uint64Value(_ v: Any?) -> UInt64? {
        switch v {
        case let i as UInt64: return i
        case let i as Int where i >= 0: return UInt64(i)
        case let n as NSNumber: return n.uint64Value
        case let s as String: return UInt64(s)
        default: return nil
        }
    }

    private static func doubleValue(_ v: Any?) -> Double? {
        switch v {
        case let d as Double: return d
        case let i as Int: return Double(i)
        case let n as NSNumber: return n.doubleValue
        case let s as String: return Double(s)
        default: return nil
        }
    }

    private static func boolValue(_ v: Any?) -> Bool? {
        switch v {
        case let b as Bool: return b
        case let n as NSNumber: return n.boolValue
        default: return nil
        }
    }
}
