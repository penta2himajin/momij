import Foundation
import Hummingbird
import HTTPTypes
import MomijCore


enum MomijHTTP {
    struct ModelObject: ResponseEncodable, Codable {
        let id: String
        let object = "model"
        let created = 0
        let owned_by = "momij"
    }

    struct ModelsResponse: ResponseEncodable, Codable {
        let object = "list"
        let data: [ModelObject]
    }

    struct ChatMessage: Codable {
        var role: String
        var content: String?
    }

    struct ChatCompletionRequest: Codable {
        var model: String?
        var messages: [ChatMessage]
        var max_tokens: Int?
        var max_completion_tokens: Int?
        var temperature: Double?
        var top_p: Double?
        var presence_penalty: Double?
        var frequency_penalty: Double?
        /// HF / vLLM-style (not official OpenAI); 1.0 = off.
        var repetition_penalty: Double?
        var stream: Bool?
        var n: Int?
        var seed: UInt64?
    }

    actor AsyncLock {
        private var locked = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        func acquire() async {
            if !locked { locked = true; return }
            await withCheckedContinuation { waiters.append($0) }
        }
        func release() {
            if waiters.isEmpty { locked = false } else { waiters.removeFirst().resume() }
        }
    }

    final class MomijEngine: @unchecked Sendable {
        let tokenizer: any TokenizerAdapter
        let backend: any LLMBackend
        let modelID: String
        private let lock = AsyncLock()

        init(tokenizer: any TokenizerAdapter, backend: any LLMBackend, modelID: String) {
            self.tokenizer = tokenizer
            self.backend = backend
            self.modelID = modelID
        }

        func withLock<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
            await lock.acquire()
            defer { Task { await lock.release() } }
            return try await body()
        }
    }

    protocol TokenizerAdapter: Sendable {
        func encode(_ text: String) throws -> [Int]
        func decode(_ ids: [Int]) throws -> String
        func applyChatTemplate(_ messages: [ChatMessage]) throws -> [Int]
    }

    /// Minimal byte-fallback tokenizer when swift-transformers Hub load is unavailable.
    struct ByteTokenizer: TokenizerAdapter {
        func encode(_ text: String) throws -> [Int] {
            Array(text.utf8).map { Int($0) }
        }
        func decode(_ ids: [Int]) throws -> String {
            let bytes = ids.compactMap { UInt8(exactly: $0) }
            return String(bytes: bytes, encoding: .utf8) ?? ""
        }
        func applyChatTemplate(_ messages: [ChatMessage]) throws -> [Int] {
            let text = messages.map { "\($0.role): \($0.content ?? "")" }.joined(separator: "\n")
            return try encode(text)
        }
    }

    static func makeRouter(engine: MomijEngine) -> Router<BasicRequestContext> {
        let router = Router(context: BasicRequestContext.self)
        router.get("/v1/models") { _, _ -> ModelsResponse in
            ModelsResponse(data: [ModelObject(id: engine.modelID)])
        }
        router.get("/healthz") { _, _ -> String in "ok" }
        router.post("/v1/chat/completions") { req, _ -> Response in
            let body = try await req.body.collect(upTo: 8_000_000)
            let chatReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(body.readableBytesView))
            if (chatReq.n ?? 1) > 1 {
                return Response(status: .badRequest, body: .init(byteBuffer: .init(string: #"{"error":"n>1 unsupported"}"#)))
            }
            let maxTok = chatReq.max_completion_tokens ?? chatReq.max_tokens ?? 256
            let ids = try engine.tokenizer.applyChatTemplate(chatReq.messages)
            let opts = GenerateOptions(
                maxTokens: maxTok,
                temperature: chatReq.temperature ?? 0,
                topP: chatReq.top_p ?? 1,
                presencePenalty: chatReq.presence_penalty ?? 0,
                frequencyPenalty: chatReq.frequency_penalty ?? 0,
                repetitionPenalty: chatReq.repetition_penalty ?? 1,
                useSuffixSpec: ProcessInfo.processInfo.environment["MOMIJ_SUFFIX_SPEC"] == "1",
                seed: chatReq.seed
            )
            if chatReq.stream == true {
                return try await streamSSE(engine: engine, prompt: ids, options: opts)
            }
            let tokens = try await engine.withLock {
                var out: [Int] = []
                for try await t in engine.backend.generate(ids, options: opts) {
                    out.append(t)
                }
                return out
            }
            let text = try engine.tokenizer.decode(tokens)
            let payload: [String: Any] = [
                "id": "chatcmpl-momij",
                "object": "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": engine.modelID,
                "choices": [[
                    "index": 0,
                    "message": ["role": "assistant", "content": text],
                    "finish_reason": "stop",
                ]],
                "usage": [
                    "prompt_tokens": ids.count,
                    "completion_tokens": tokens.count,
                    "total_tokens": ids.count + tokens.count,
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: payload)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: data))
            )
        }
        return router
    }

    static func streamSSE(engine: MomijEngine, prompt: [Int], options: GenerateOptions) async throws -> Response {
        let stream = AsyncStream<ByteBuffer> { cont in
            Task {
                do {
                    let tokens = try await engine.withLock { () -> [Int] in
                        var out: [Int] = []
                        for try await t in engine.backend.generate(prompt, options: options) {
                            out.append(t)
                            let piece = (try? engine.tokenizer.decode([t])) ?? ""
                            let chunk: [String: Any] = [
                                "id": "chatcmpl-momij",
                                "object": "chat.completion.chunk",
                                "created": Int(Date().timeIntervalSince1970),
                                "model": engine.modelID,
                                "choices": [[
                                    "index": 0,
                                    "delta": ["content": piece],
                                    "finish_reason": NSNull(),
                                ]],
                            ]
                            if let data = try? JSONSerialization.data(withJSONObject: chunk),
                               let line = String(data: data, encoding: .utf8) {
                                var buf = ByteBufferAllocator().buffer(capacity: line.count + 16)
                                buf.writeString("data: \(line)\n\n")
                                cont.yield(buf)
                            }
                        }
                        return out
                    }
                    _ = tokens
                    var done = ByteBufferAllocator().buffer(capacity: 16)
                    done.writeString("data: [DONE]\n\n")
                    cont.yield(done)
                    cont.finish()
                } catch {
                    cont.finish()
                }
            }
        }
        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream"],
            body: .init(asyncSequence: stream)
        )
    }

    static func runServe(engine: MomijEngine, host: String, port: Int) async throws {
        let router = makeRouter(engine: engine)
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )
        fputs("[momij] listening on http://\(host):\(port) model=\(engine.modelID)\n", stderr)
        try await app.runService()
    }

}
