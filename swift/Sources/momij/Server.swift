import Foundation
import Hummingbird
import HTTPTypes
import MomijCore
import XGrammar

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

    /// Alias kept for TokenizerAdapter / HF bridge.
    typealias ChatMessage = OpenAIChatCompat.ChatMessage

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
        let modelDir: String
        private let lock = AsyncLock()
        private var xgrammarTokenizer: TokenizerInfo?

        init(
            tokenizer: any TokenizerAdapter,
            backend: any LLMBackend,
            modelID: String,
            modelDir: String
        ) {
            self.tokenizer = tokenizer
            self.backend = backend
            self.modelID = modelID
            self.modelDir = modelDir
        }

        func withLock<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
            await lock.acquire()
            defer { Task { await lock.release() } }
            return try await body()
        }

        func grammarTokenizerInfo(eos: Int) throws -> TokenizerInfo {
            if let cached = xgrammarTokenizer { return cached }
            let info = try XGrammarTokenizer.load(modelDir: modelDir, eosTokenId: eos)
            xgrammarTokenizer = info
            return info
        }
    }

    protocol TokenizerAdapter: Sendable {
        func encode(_ text: String) throws -> [Int]
        func decode(_ ids: [Int]) throws -> String
        func applyChatTemplate(_ messages: [ChatMessage]) throws -> [Int]
    }

    /// Dev / offline only. Serve path must not use this (HF chat_template required).
    struct ByteTokenizer: TokenizerAdapter {
        func encode(_ text: String) throws -> [Int] {
            Array(text.utf8).map { Int($0) }
        }
        func decode(_ ids: [Int]) throws -> String {
            let bytes = ids.compactMap { UInt8(exactly: $0) }
            return String(bytes: bytes, encoding: .utf8) ?? ""
        }
        func applyChatTemplate(_ messages: [ChatMessage]) throws -> [Int] {
            throw TemplateError.byteFallbackForbidden
        }
    }

    enum TemplateError: Error, CustomStringConvertible {
        case byteFallbackForbidden
        case applyFailed(String)

        var description: String {
            switch self {
            case .byteFallbackForbidden:
                return "HF chat_template required; byte tokenizer cannot serve"
            case .applyFailed(let s):
                return "chat_template failed: \(s)"
            }
        }
    }

    static func makeRouter(engine: MomijEngine) -> Router<BasicRequestContext> {
        let router = Router(context: BasicRequestContext.self)
        // Authorization (if present) is ignored — no 401 for Bearer from evprtr.
        router.get("/v1/models") { _, _ -> ModelsResponse in
            ModelsResponse(data: [ModelObject(id: engine.modelID)])
        }
        router.get("/healthz") { _, _ -> String in "ok" }
        router.post("/v1/chat/completions") { req, _ -> Response in
            let body = try await req.body.collect(upTo: 8_000_000)
            let chatReq: OpenAIChatCompat.ChatCompletionRequest
            do {
                chatReq = try OpenAIChatCompat.parseRequest(from: Data(body.readableBytesView))
            } catch let e as OpenAIChatCompat.ParseError {
                return jsonError(status: .badRequest, message: e.description)
            } catch {
                return jsonError(status: .badRequest, message: "invalid request")
            }
            if chatReq.n > 1 {
                return jsonError(status: .badRequest, message: "n>1 unsupported")
            }
            let eos = 151_645
            var allowedNext: (@Sendable ([Int]) -> Set<Int>)? = nil
            do {
                switch chatReq.structuredConstraint {
                case .choice(let strs):
                    let guide = try FiniteStringGuide(
                        strings: strs,
                        encode: { try engine.tokenizer.encode($0) },
                        eosTokenId: eos)
                    allowedNext = { prefix in guide.allowedNext(prefix: prefix) }
                case .jsonSchema(let schema):
                    let tokInfo = try engine.grammarTokenizerInfo(eos: eos)
                    let guide = try await XGrammarTokenGuide.compileJSONSchema(
                        schema, tokenizerInfo: tokInfo, eosTokenId: eos)
                    allowedNext = { prefix in
                        (try? guide.allowedNext(prefix: prefix)) ?? [eos]
                    }
                case .ebnf(let ebnf):
                    let tokInfo = try engine.grammarTokenizerInfo(eos: eos)
                    let guide = try await XGrammarTokenGuide.compileEBNF(
                        ebnf, tokenizerInfo: tokInfo, eosTokenId: eos)
                    allowedNext = { prefix in
                        (try? guide.allowedNext(prefix: prefix)) ?? [eos]
                    }
                case .regex(let pattern):
                    // Compile as EBNF root wrapping the regex via Grammar(regex:).
                    let tokInfo = try engine.grammarTokenizerInfo(eos: eos)
                    let grammar = Grammar(regex: pattern)
                    let compiled = await grammar.compiled(for: tokInfo)
                    let guide = XGrammarTokenGuide(
                        compiled: compiled, vocabSize: tokInfo.vocabulary.size, eosTokenId: eos)
                    allowedNext = { prefix in
                        (try? guide.allowedNext(prefix: prefix)) ?? [eos]
                    }
                case .unsupported(let why):
                    return jsonError(
                        status: .badRequest,
                        message: "structured output not supported yet: \(why)")
                case nil:
                    break
                }
            } catch {
                return jsonError(
                    status: .badRequest,
                    message: "structured output setup failed: \(error)")
            }

            let promptIds: [Int]
            do {
                promptIds = try engine.tokenizer.applyChatTemplate(chatReq.messages)
            } catch {
                return jsonError(status: .internalServerError, message: "\(error)")
            }

            let opts = GenerateOptions(
                maxTokens: chatReq.maxTokens,
                temperature: chatReq.temperature,
                topP: chatReq.topP,
                presencePenalty: chatReq.presencePenalty,
                frequencyPenalty: chatReq.frequencyPenalty,
                repetitionPenalty: chatReq.repetitionPenalty,
                eosTokenIds: [eos],
                useSuffixSpec: ProcessInfo.processInfo.environment["MOMIJ_SUFFIX_SPEC"] == "1"
                    && allowedNext == nil,
                seed: chatReq.seed,
                allowedNext: allowedNext
            )
            if chatReq.stream {
                return try await streamSSE(engine: engine, prompt: promptIds, options: opts)
            }

            let tokens: [Int]
            do {
                tokens = try await engine.withLock {
                    var out: [Int] = []
                    for try await t in engine.backend.generate(promptIds, options: opts) {
                        out.append(t)
                    }
                    return out
                }
            } catch {
                return jsonError(status: .internalServerError, message: "generation failed: \(error)")
            }

            let finish = OpenAIChatCompat.finishReason(
                completionTokens: tokens, maxTokens: opts.maxTokens, eosTokenIds: opts.eosTokenIds)
            let decodeIds = OpenAIChatCompat.contentTokenIds(tokens, eosTokenIds: opts.eosTokenIds)
            let text: String
            do {
                let raw = try engine.tokenizer.decode(decodeIds)
                text = ChatTemplatePatch.stripThinkForContent(raw)
            } catch {
                return jsonError(status: .internalServerError, message: "decode failed: \(error)")
            }
            do {
                let data = try OpenAIChatCompat.nonStreamJSON(
                    modelID: engine.modelID,
                    content: text,
                    finishReason: finish,
                    promptTokens: promptIds.count,
                    completionTokens: tokens.count)
                return Response(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: .init(data: data))
                )
            } catch {
                return jsonError(status: .internalServerError, message: "response encode failed")
            }
        }
        return router
    }

    static func jsonError(status: HTTPResponse.Status, message: String) -> Response {
        let payload: [String: Any] = [
            "error": [
                "message": message,
                "type": "invalid_request_error",
            ],
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload))
            ?? Data(#"{"error":{"message":"error","type":"invalid_request_error"}}"#.utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: data))
        )
    }

    static func streamSSE(engine: MomijEngine, prompt: [Int], options: GenerateOptions) async throws -> Response {
        let stream = AsyncStream<ByteBuffer> { cont in
            Task {
                do {
                    let tokens = try await engine.withLock { () -> [Int] in
                        var out: [Int] = []
                        for try await t in engine.backend.generate(prompt, options: options) {
                            out.append(t)
                            let pieceIds = OpenAIChatCompat.contentTokenIds(
                                [t], eosTokenIds: options.eosTokenIds)
                            let piece = pieceIds.isEmpty
                                ? ""
                                : ((try? engine.tokenizer.decode(pieceIds)) ?? "")
                            if !piece.isEmpty {
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
                        }
                        return out
                    }
                    let finish = OpenAIChatCompat.finishReason(
                        completionTokens: tokens,
                        maxTokens: options.maxTokens,
                        eosTokenIds: options.eosTokenIds)
                    let finalChunk: [String: Any] = [
                        "id": "chatcmpl-momij",
                        "object": "chat.completion.chunk",
                        "created": Int(Date().timeIntervalSince1970),
                        "model": engine.modelID,
                        "choices": [[
                            "index": 0,
                            "delta": [:] as [String: Any],
                            "finish_reason": finish,
                        ]],
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: finalChunk),
                       let line = String(data: data, encoding: .utf8) {
                        var buf = ByteBufferAllocator().buffer(capacity: line.count + 16)
                        buf.writeString("data: \(line)\n\n")
                        cont.yield(buf)
                    }
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
