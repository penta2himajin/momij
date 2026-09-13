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
        let traces = TraceStore(
            dir: URL(fileURLWithPath: ProcessInfo.processInfo.environment["MOMIJ_TRACE_DIR"]
                ?? NSString("~/.momij/traces").expandingTildeInPath))
        // Fire-and-forget persistence: tracing must never fail or delay a request.
        @Sendable func recordTrace(_ t: RequestTrace) {
            Task { traces.write(t.record(), id: t.id) }
        }
        router.get("/v1/models") { _, _ -> ModelsResponse in
            ModelsResponse(data: [ModelObject(id: engine.modelID)])
        }
        router.get("/healthz") { _, _ -> String in "ok" }
        router.get("/v1/traces") { _, _ -> Response in
            let records = traces.list(limit: 20)
            guard let data = try? JSONSerialization.data(withJSONObject: ["object": "list", "data": records])
            else {
                return jsonError(status: .internalServerError, message: "trace list failed")
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: data))
            )
        }
        router.get("/v1/traces/:id") { _, context -> Response in
            let id = context.parameters.get("id", as: String.self) ?? ""
            guard let record = traces.get(id) else {
                return jsonError(status: .notFound, message: "trace not found: \(id)")
            }
            guard let data = try? JSONSerialization.data(withJSONObject: record) else {
                return jsonError(status: .internalServerError, message: "trace encode failed")
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: data))
            )
        }

        @Sendable func fail(_ status: HTTPResponse.Status, _ message: String, trace: RequestTrace) -> Response {
            trace.fail(message)
            recordTrace(trace)
            return jsonError(status: status, message: message, traceId: trace.id, locus: trace.locus)
        }

        router.post("/v1/chat/completions") { req, _ -> Response in
            var trace = RequestTrace()
            let body = try await req.body.collect(upTo: 8_000_000)
            let chatReq: OpenAIChatCompat.ChatCompletionRequest
            do {
                chatReq = try OpenAIChatCompat.parseRequest(from: Data(body.readableBytesView))
                trace.event("parse", "ok", detail: [
                    "messages": chatReq.messages.count,
                    "has_tools": !chatReq.toolsLines.isEmpty,
                    "stream": chatReq.stream,
                    "max_tokens": chatReq.maxTokens,
                ])
            } catch let e as OpenAIChatCompat.ParseError {
                return fail(.badRequest, e.description, trace: trace)
            } catch {
                return fail(.badRequest, "invalid request", trace: trace)
            }
            if chatReq.n > 1 {
                return fail(.badRequest, "n>1 unsupported", trace: trace)
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
                    return fail(
                        .badRequest, "structured output not supported yet: \(why)", trace: trace)
                case nil:
                    break
                }
            } catch {
                return fail(.badRequest, "structured output setup failed: \(error)", trace: trace)
            }

            // Native Maple tools contract: rewrite tool history into markup
            // form and append the tools instruction (compositor parity).
            var effective = chatReq
            if !chatReq.toolsLines.isEmpty {
                var msgs = ToolMarkup.rewriteMessages(chatReq.messages)
                let suffix = ToolMarkup.systemSuffix(toolLines: chatReq.toolsLines)
                if let idx = msgs.firstIndex(where: { $0.role == "system" }) {
                    msgs[idx].content = msgs[idx].content + "\n\n" + suffix
                } else {
                    msgs.insert(OpenAIChatCompat.ChatMessage(role: "system", content: suffix), at: 0)
                }
                effective.messages = msgs
            }
            let promptIds: [Int]
            do {
                promptIds = try engine.tokenizer.applyChatTemplate(effective.messages)
            } catch {
                return fail(.internalServerError, "chat template failed: \(error)", trace: trace)
            }

            let opts = GenerateOptions(
                maxTokens: chatReq.maxTokens,
                temperature: chatReq.temperature,
                topP: chatReq.topP,
                presencePenalty: chatReq.presencePenalty,
                frequencyPenalty: chatReq.frequencyPenalty,
                repetitionPenalty: chatReq.repetitionPenalty,
                eosTokenIds: [eos],
                useSuffixSpec: SeedlessServeDefaults.suffixSpecFromEnv
                    && allowedNext == nil,
                seed: chatReq.seed,
                allowedNext: allowedNext,
                bannedTokenIds: ChatTemplatePatch.bannedAssistantTokenIds
            )
            if chatReq.stream {
                trace.event("route", "ok", detail: ["mode": "stream"])
                return try await streamSSE(
                    engine: engine, prompt: promptIds, options: opts,
                    toolsAttached: !chatReq.toolsLines.isEmpty,
                    effectiveMessages: effective.messages,
                    trace: trace, traces: traces)
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
                return fail(.internalServerError, "generation failed: \(error)", trace: trace)
            }

            var finish = OpenAIChatCompat.finishReason(
                completionTokens: tokens, maxTokens: opts.maxTokens, eosTokenIds: opts.eosTokenIds)
            let decodeIds = OpenAIChatCompat.contentTokenIds(tokens, eosTokenIds: opts.eosTokenIds)
            let text: String
            do {
                let raw = try engine.tokenizer.decode(decodeIds)
                text = ChatTemplatePatch.stripThinkForContent(raw)
            } catch {
                return fail(.internalServerError, "decode failed: \(error)", trace: trace)
            }
            var parsed = ToolMarkup.parsePseudoToolCalls(text)
            // Degenerate-call repair (compositor parity): an empty tool name
            // is the collapse signature — retry once with a corrective nudge.
            if parsed.calls.contains(where: { $0.name.isEmpty }) {
                var repair = effective.messages
                repair.append(OpenAIChatCompat.ChatMessage(
                    role: "assistant", content: text))
                repair.append(OpenAIChatCompat.ChatMessage(
                    role: "user",
                    content: "Your last reply was malformed. Respond with exactly one tool call in the correct format, or plain text."))
                do {
                    let repairPrompt = try engine.tokenizer.applyChatTemplate(repair)
                    let repairOpts = opts
                    let repairBudget = max(8, opts.maxTokens - tokens.count)
                    let extra = try await engine.withLock { () -> [Int] in
                        var out: [Int] = []
                        for try await t in engine.backend.generate(repairPrompt, options: opts.withMaxTokens(repairBudget)) {
                            out.append(t)
                        }
                        return out
                    }
                    let repairIds = OpenAIChatCompat.contentTokenIds(
                        extra, eosTokenIds: opts.eosTokenIds)
                    let repairRaw = try engine.tokenizer.decode(repairIds)
                    let repairText = ChatTemplatePatch.stripThinkForContent(repairRaw)
                    let reparsed = ToolMarkup.parsePseudoToolCalls(repairText)
                    if !reparsed.calls.contains(where: { $0.name.isEmpty }) {
                        parsed = reparsed
                        finish = parsed.calls.isEmpty ? finish : "tool_calls"
                    }
                } catch {
                    // Keep the original parse on repair failure.
                }
            }
            let respFinish = parsed.calls.isEmpty ? finish : "tool_calls"
            trace.event("decode", "ok", detail: [
                "finish": respFinish,
                "tokens": tokens.count,
                "tool_calls": parsed.calls.count,
                "degenerate": parsed.calls.contains(where: { $0.name.isEmpty }),
            ])
            do {
                let data = try OpenAIChatCompat.nonStreamJSON(
                    modelID: engine.modelID,
                    content: parsed.cleanedContent,
                    finishReason: respFinish,
                    promptTokens: promptIds.count,
                    completionTokens: tokens.count,
                    toolCalls: parsed.calls)
                trace.event("present", "ok", detail: [
                    "tool_calls": parsed.calls.count,
                    "usage": ["prompt": promptIds.count, "completion": tokens.count],
                ])
                recordTrace(trace)
                return Response(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: .init(data: data))
                )
            } catch {
                return fail(.internalServerError, "response encode failed", trace: trace)
            }
        }
        return router
    }

    /// Per-request structured trace (lightweight compositor parity).
    /// Class + lock so it can be captured across task boundaries.
    final class RequestTrace: @unchecked Sendable {
        let id = TraceStore.newID()
        private let startedAt = CFAbsoluteTimeGetCurrent()
        private let gate = NSLock()
        private var ok = true
        private(set) var locus = "none"
        private var error: String?
        private var events: [[String: Any]] = []

        init() {
            event("accept", "ok")
        }

        func event(_ stage: String, _ status: String, detail: [String: Any] = [:]) {
            gate.lock()
            defer { gate.unlock() }
            events.append([
                "stage": stage,
                "status": status,
                "at": CFAbsoluteTimeGetCurrent(),
                "detail": detail,
            ])
        }

        func fail(_ message: String) {
            gate.lock()
            defer { gate.unlock() }
            ok = false
            locus = "upstream"
            error = message
            events.append([
                "stage": "fail",
                "status": "error",
                "at": CFAbsoluteTimeGetCurrent(),
                "detail": ["message": message],
            ])
        }

        func record() -> [String: Any] {
            gate.lock()
            defer { gate.unlock() }
            return [
                "trace_id": id,
                "started_at": startedAt,
                "finished_at": CFAbsoluteTimeGetCurrent(),
                "ok": ok,
                "locus": locus,
                "error": error ?? NSNull(),
                "events": events,
            ]
        }
    }

    static func jsonError(
        status: HTTPResponse.Status, message: String,
        traceId: String? = nil, locus: String? = nil
    ) -> Response {
        var errorBody: [String: Any] = [
            "message": message,
            "type": "invalid_request_error",
        ]
        if let traceId { errorBody["trace_id"] = traceId }
        if let locus { errorBody["locus"] = locus }
        let payload: [String: Any] = ["error": errorBody]
        let data = (try? JSONSerialization.data(withJSONObject: payload))
            ?? Data(#"{"error":{"message":"error","type":"invalid_request_error"}}"#.utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: data))
        )
    }
    static func streamSSE(
        engine: MomijEngine, prompt: [Int], options: GenerateOptions,
        toolsAttached: Bool, effectiveMessages: [OpenAIChatCompat.ChatMessage],
        trace: RequestTrace, traces: TraceStore
    ) async throws -> Response {
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
                            if !piece.isEmpty, !toolsAttached {
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
                    var finish = OpenAIChatCompat.finishReason(
                        completionTokens: tokens,
                        maxTokens: options.maxTokens,
                        eosTokenIds: options.eosTokenIds)
                    var streamToolCalls = 0
                    if toolsAttached {
                        let decodeIds = OpenAIChatCompat.contentTokenIds(
                            tokens, eosTokenIds: options.eosTokenIds)
                        let raw = try engine.tokenizer.decode(decodeIds)
                        let text = ChatTemplatePatch.stripThinkForContent(raw)
                        let parsed = ToolMarkup.parsePseudoToolCalls(text)
                        streamToolCalls = parsed.calls.count
                        if !parsed.cleanedContent.isEmpty {
                            let contentChunk: [String: Any] = [
                                "id": "chatcmpl-momij",
                                "object": "chat.completion.chunk",
                                "created": Int(Date().timeIntervalSince1970),
                                "model": engine.modelID,
                                "choices": [[
                                    "index": 0,
                                    "delta": ["content": parsed.cleanedContent],
                                    "finish_reason": NSNull(),
                                ]],
                            ]
                            if let data = try? JSONSerialization.data(withJSONObject: contentChunk),
                               let line = String(data: data, encoding: .utf8) {
                                var buf = ByteBufferAllocator().buffer(capacity: line.count + 16)
                                buf.writeString("data: \(line)\n\n")
                                cont.yield(buf)
                            }
                        }
                        if !parsed.calls.isEmpty {
                            finish = "tool_calls"
                            let callsDelta: [String: Any] = [
                                "id": "chatcmpl-momij",
                                "object": "chat.completion.chunk",
                                "created": Int(Date().timeIntervalSince1970),
                                "model": engine.modelID,
                                "choices": [[
                                    "index": 0,
                                    "delta": ["tool_calls": parsed.calls.map { call in
                                        [
                                            "id": call.id,
                                            "type": "function",
                                            "function": [
                                                "name": call.name,
                                                "arguments": call.arguments,
                                            ],
                                            "index": 0,
                                        ]
                                    }],
                                    "finish_reason": NSNull(),
                                ]],
                            ]
                            if let data = try? JSONSerialization.data(withJSONObject: callsDelta),
                               let line = String(data: data, encoding: .utf8) {
                                var buf = ByteBufferAllocator().buffer(capacity: line.count + 16)
                                buf.writeString("data: \(line)\n\n")
                                cont.yield(buf)
                            }
                        }
                    }
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
                    // Fire-and-forget trace write after the stream completes.
                    trace.event("present", "ok", detail: [
                        "mode": "stream",
                        "finish": finish,
                        "tokens": tokens.count,
                        "tool_calls": streamToolCalls,
                    ])
                    traces.write(trace.record(), id: trace.id)
                } catch {
                    trace.fail("stream generation failed: \(error)")
                    traces.write(trace.record(), id: trace.id)
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
