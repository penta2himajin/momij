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
            // Tool-call loop breaker: a trailing run of identical calls
            // (escape-escalation variants normalize equal) gets a steering
            // message so the retry loop ends this turn. Per-request only —
            // the harness's own history is untouched. NOTE: detect BEFORE
            // rewriteMessages — the rewrite folds tool_calls into markup
            // content, leaving nothing for the detector to see.
            if let loop = LoopBreaker.detect(in: chatReq.messages) {
                trace.event("loop_break", "ok", detail: [
                    "tool": loop.tool, "count": loop.count])
            }
            var effective = chatReq
            let workspaceRoot = PathPolicy.workspaceRoot()
            if !chatReq.toolsLines.isEmpty {
                var msgs = ToolMarkup.rewriteMessages(chatReq.messages)
                var suffix = ToolMarkup.systemSuffix(toolLines: chatReq.toolsLines)
                if workspaceRoot != nil {
                    suffix += "\n" + PathPolicy.suffixLine()
                }
                if let idx = msgs.firstIndex(where: { $0.role == "system" }) {
                    msgs[idx].content = msgs[idx].content + "\n\n" + suffix
                } else {
                    msgs.insert(OpenAIChatCompat.ChatMessage(role: "system", content: suffix), at: 0)
                }
                // Workspace mode: strip the root prefix from everything the
                // model sees (tool results, history) so its world stays
                // relative and short.
                if let root = workspaceRoot {
                    for i in msgs.indices {
                        msgs[i].content = PathPolicy.stripRootPrefix(in: msgs[i].content, root: root)
                    }
                }
                effective.messages = msgs
            }
            if let loop = LoopBreaker.detect(in: chatReq.messages) {
                effective.messages.append(OpenAIChatCompat.ChatMessage(
                    role: "user", content: LoopBreaker.breakNudge(
                        tool: loop.tool, count: loop.count)))
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
                    toolsLines: chatReq.toolsLines,
                    effectiveMessages: effective.messages,
                    includeUsage: chatReq.streamOptionsIncludeUsage,
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
            let raw: String
            let text: String
            do {
                raw = try engine.tokenizer.decode(decodeIds)
                text = ChatTemplatePatch.stripThinkForContent(raw)
            } catch {
                return fail(.internalServerError, "decode failed: \(error)", trace: trace)
            }
            var parsed = ToolMarkup.parsePseudoToolCalls(text)
            // Protocol-tool interception: the model's "I'm done" intent (a
            // submit / ask_user_question call, often with unusable args)
            // becomes a clean ending instead of a harness rejection loop.
            // All calls drop, finish=stop, and the call's description (or a
            // placeholder) is the final content.
            if let absorb = ProtocolAbsorb.absorbCall(
                in: parsed.calls, names: ProtocolAbsorb.toolNames()) {
                parsed = ToolMarkup.ParsedToolCalls(
                    cleanedContent: ProtocolAbsorb.finalContent(for: absorb), calls: [])
                finish = "stop"
                trace.event("protocol_absorb", "ok", detail: ["tool": absorb.name])
            }
            // Degenerate-call repair, 3 stages: (1) targeted field re-ask —
            // regenerate ONLY the broken argument fields under the tool's
            // JSON schema and merge (good parts survive); (2) whole-reply
            // re-ask with a corrective nudge; (3) drop the broken calls.
            // The fix target comes from the write/edit heuristics OR from
            // schema-required fields missing on ANY tool (the harness-side
            // "missing required property" rejection, caught before it leaves).
            var degenerateHit = ResponseVerify.degenerateToolArgs(.init(toolCalls: parsed.calls))
            let emptyNameAtStart = parsed.calls.contains(where: { $0.name.isEmpty })
            var pendingFix: (idx: Int, fields: [String], reasons: [String: String], kind: String)? = nil
            if let hit = degenerateHit, hit.onset < parsed.calls.count {
                let hitArgs = ArgRepair.parseArgs(parsed.calls[hit.onset].arguments)
                let fields = ArgRepair.repairFields(detail: hit.detail, args: hitArgs)
                if !fields.isEmpty {
                    let reason = (hit.detail["reason"] as? String) ?? "unusable"
                    pendingFix = (hit.onset, fields,
                        Dictionary(uniqueKeysWithValues: fields.map { ($0, reason) }), hit.kind)
                }
            }
            if pendingFix == nil {
                for (i, call) in parsed.calls.enumerated() where !call.name.isEmpty {
                    let args = ArgRepair.parseArgs(call.arguments)
                    let missing = ArgRepair.missingRequiredFields(
                        toolLines: chatReq.toolsLines, tool: call.name, args: args)
                    if !missing.isEmpty {
                        pendingFix = (i, missing,
                            Dictionary(uniqueKeysWithValues: missing.map {
                                ($0, "missing required property") }),
                            "missing_required_property")
                        break
                    }
                    // Workspace discipline: absolute paths outside the root
                    // would be rejected by the harness sandbox — repair them
                    // here into relative form instead.
                    if let root = workspaceRoot {
                        for f in PathPolicy.pathFields {
                            if let v = args[f] as? String, !v.isEmpty,
                               PathPolicy.isOutsideWorkspace(v, root: root) {
                                pendingFix = (i, [f], [f: "absolute path outside the workspace root (use a relative path)"], "path_outside_workspace")
                                break
                            }
                        }
                        if pendingFix != nil { break }
                    }
                }
            }
            if !emptyNameAtStart, let pf = pendingFix {
                let brokenCall = parsed.calls[pf.idx]
                let task = ArgRepair.originalUserText(effective.messages)
                var adoptedByTask = false
                // Extraction-first: a path-kind fix is answered
                // deterministically from the task text (it names the exact
                // path; the model re-ask repeats its hallucination). Fall
                // back to the model re-ask only when extraction gives no
                // answer or the merged call fails the clean check.
                if let taskMerged = ArgRepair.taskPathMerge(
                    kind: pf.kind, fields: pf.fields,
                    args: ArgRepair.parseArgs(brokenCall.arguments), task: task) {
                    var fixedCalls = parsed.calls
                    fixedCalls[pf.idx] = OpenAIChatCompat.ToolCallSpec(
                        id: brokenCall.id, name: brokenCall.name,
                        arguments: ArgRepair.argsJSONString(taskMerged))
                    if callsClean(fixedCalls, toolsLines: chatReq.toolsLines) {
                        parsed = ToolMarkup.ParsedToolCalls(
                            cleanedContent: parsed.cleanedContent, calls: fixedCalls)
                        degenerateHit = ResponseVerify.degenerateToolArgs(
                            .init(toolCalls: fixedCalls))
                        trace.event("repair", "ok", detail: [
                            "mode": "field", "source": "task",
                            "kind": pf.kind, "fields": pf.fields])
                        adoptedByTask = true
                    }
                }
                if !adoptedByTask {
                    do {
                        let merged = try await MomijHTTP.fieldRepairCall(
                            engine: engine, eos: eos, toolsLines: chatReq.toolsLines,
                            tool: brokenCall.name, args: ArgRepair.parseArgs(brokenCall.arguments),
                            fields: pf.fields, reasons: pf.reasons,
                            task: task,
                            baseOpts: opts)
                        if let merged {
                            var fixedCalls = parsed.calls
                            fixedCalls[pf.idx] = OpenAIChatCompat.ToolCallSpec(
                                id: brokenCall.id, name: brokenCall.name,
                                arguments: ArgRepair.argsJSONString(merged))
                            if callsClean(fixedCalls, toolsLines: chatReq.toolsLines) {
                                parsed = ToolMarkup.ParsedToolCalls(
                                    cleanedContent: parsed.cleanedContent, calls: fixedCalls)
                                degenerateHit = ResponseVerify.degenerateToolArgs(
                                    .init(toolCalls: fixedCalls))
                                trace.event("repair", "ok", detail: [
                                    "mode": "field", "kind": pf.kind, "fields": pf.fields])
                            }
                        }
                    } catch {
                        trace.event("repair", "error", detail: [
                            "mode": "field", "kind": pf.kind, "error": "\(error)"])
                    }
                }
            }
            let repairNudge: String
            if let hit = degenerateHit {
                let reason = (hit.detail["reason"] as? String) ?? "unusable"
                let tool = (hit.detail["tool"] as? String) ?? "tool"
                repairNudge = "Your previous \(tool) tool call had unusable arguments (\(reason)). "
                    + "Retry with valid arguments: for write, path must be a real file path "
                    + "and content the full file text. Or reply in plain text."
            } else {
                repairNudge = "Your last reply was malformed. Respond with exactly one tool call "
                    + "in the correct format, or plain text."
            }
            var repairFailed = false
            if parsed.calls.contains(where: { $0.name.isEmpty }) || degenerateHit != nil {
                var repair = effective.messages
                repair.append(OpenAIChatCompat.ChatMessage(
                    role: "assistant", content: text))
                repair.append(OpenAIChatCompat.ChatMessage(
                    role: "user",
                    content: repairNudge))
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
                    let reparsedClean = reparsed.calls.contains(where: { $0.name.isEmpty }) == false
                        && ResponseVerify.degenerateToolArgs(.init(toolCalls: reparsed.calls)) == nil
                    if reparsedClean {
                        parsed = reparsed
                        finish = parsed.calls.isEmpty ? finish : "tool_calls"
                        trace.event("repair", "ok", detail: ["trigger": degenerateHit?.kind ?? "empty_name"])
                    } else {
                        repairFailed = true
                        trace.event("repair", "error", detail: ["trigger": degenerateHit?.kind ?? "empty_name"])
                    }
                } catch {
                    // Keep the original parse on repair failure.
                    repairFailed = true
                    trace.event("repair", "error", detail: ["error": "\(error)"])
                }
            }
            // Native verify pass (evprtr verify pipeline parity): every
            // detector runs and records into the trace; sanitize is opt-in
            // (MOMIJ_VERIFY_SANITIZE=1) so the default path only measures.
            let sanitizeEnabled = ProcessInfo.processInfo.environment["MOMIJ_VERIFY_SANITIZE"] == "1"
            let verifyMsg = ResponseVerify.Message(
                content: text, reasoningContent: ResponseVerify.thinkInterior(of: raw),
                toolCalls: parsed.calls)
            let verifyOutcome = ResponseVerify.verify(verifyMsg)
            var respContent = parsed.cleanedContent
            var respToolCalls = parsed.calls
            var respFinish = parsed.calls.isEmpty ? finish : "tool_calls"
            // Broken-call drop (evprtr sanitize parity): when repair failed
            // and the surviving calls are still unusable (empty name or
            // degenerate write/edit args), they must not reach the harness —
            // a tool_calls finish the harness cannot execute causes retry
            // loops (observed live in the DSH subagent test).
            let brokenCalls = parsed.calls.contains(where: { $0.name.isEmpty })
                || ResponseVerify.degenerateToolArgs(.init(toolCalls: parsed.calls)) != nil
                || parsed.calls.contains {
                    !ArgRepair.missingRequiredFields(
                        toolLines: chatReq.toolsLines, tool: $0.name,
                        args: ArgRepair.parseArgs($0.arguments)).isEmpty
                }
                || parsed.calls.contains {
                    PathPolicy.argsOutsideWorkspace(
                        ArgRepair.parseArgs($0.arguments),
                        root: PathPolicy.workspaceRoot() ?? "/")
                }
            if repairFailed && brokenCalls {
                respToolCalls = []
                respFinish = "stop"
                if respContent.trimmingCharacters(in: .whitespaces).isEmpty {
                    respContent = "Previous tool call arguments were unusable and were discarded."
                }
                trace.event("sanitize", "ok", detail: ["action": "drop_degenerate_calls"])
            }
            if sanitizeEnabled, let hit = verifyOutcome.first {
                if hit.kind == "degenerate_tool_args" {
                    // evprtr sanitize: unusable calls are dropped so the
                    // harness never sees a tool_calls finish without calls.
                    respToolCalls = []
                    respFinish = "stop"
                } else if hit.field == "content",
                          ["word_run", "ngram_run", "char_motif"].contains(hit.kind) {
                    respContent = ResponseVerify.truncateBeforeRepetition(respContent)
                }
            }
            if let root = workspaceRoot {
                let (resolved, n) = MomijHTTP.resolveWorkspacePaths(respToolCalls, root: root)
                if n > 0 {
                    respToolCalls = resolved
                    trace.event("path_resolve", "ok", detail: ["resolved": n])
                }
            }
            trace.event("verify", verifyOutcome.hits.isEmpty ? "ok" : "hit", detail: [
                "hits": verifyOutcome.hits.map { $0.eventDetail() },
                "sanitized": sanitizeEnabled && verifyOutcome.first != nil,
            ])
            trace.event("decode", "ok", detail: [
                "finish": respFinish,
                "tokens": tokens.count,
                "tool_calls": respToolCalls.count,
                "degenerate": parsed.calls.contains(where: { $0.name.isEmpty }),
            ])
            do {
                let data = try OpenAIChatCompat.nonStreamJSON(
                    modelID: engine.modelID,
                    content: respContent,
                    finishReason: respFinish,
                    promptTokens: promptIds.count,
                    completionTokens: tokens.count,
                    toolCalls: respToolCalls)
                trace.event("present", "ok", detail: [
                    "tool_calls": respToolCalls.count,
                    "tool_args": respToolCalls.map { call in
                        ["name": call.name, "arguments": String(call.arguments.prefix(300))]
                    },
                    "content_head": String(respContent.prefix(300)),
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

    /// Workspace mode: resolve relative path fields in the outgoing calls
    /// against the workspace root (the model writes relative, the harness
    /// executes absolute). Returns the rewritten calls and the change count.
    static func resolveWorkspacePaths(
        _ calls: [OpenAIChatCompat.ToolCallSpec], root: String
    ) -> ([OpenAIChatCompat.ToolCallSpec], Int) {
        var out: [OpenAIChatCompat.ToolCallSpec] = []
        var count = 0
        for call in calls {
            let args = ArgRepair.parseArgs(call.arguments)
            var changed = false
            var outArgs = args
            for field in PathPolicy.pathFields {
                guard let v = args[field] as? String, !v.isEmpty else { continue }
                let resolved = PathPolicy.resolve(v, in: root)
                if resolved != v {
                    outArgs[field] = resolved
                    changed = true
                }
            }
            if changed {
                out.append(OpenAIChatCompat.ToolCallSpec(
                    id: call.id, name: call.name, arguments: ArgRepair.argsJSONString(outArgs)))
                count += 1
            } else {
                out.append(call)
            }
        }
        return (out, count)
    }

    /// Clean check for repaired calls: no degenerate args, no missing
    /// required fields, and no outside-workspace path values (workspace
    /// mode only — the policy is off when MOMIJ_WORKSPACE is unset).
    /// Shared by the field-repair adoption sites on both paths.
    static func callsClean(
        _ calls: [OpenAIChatCompat.ToolCallSpec], toolsLines: [String]
    ) -> Bool {
        return ResponseVerify.degenerateToolArgs(.init(toolCalls: calls)) == nil
            && calls.contains {
                !ArgRepair.missingRequiredFields(
                    toolLines: toolsLines, tool: $0.name,
                    args: ArgRepair.parseArgs($0.arguments)).isEmpty
            } == false
            && calls.contains {
                PathPolicy.argsOutsideWorkspace(
                    ArgRepair.parseArgs($0.arguments),
                    root: PathPolicy.workspaceRoot() ?? "/")
            } == false
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
    /// Stage-1 targeted field repair: re-ask ONLY the broken argument fields
    /// (grammar-constrained to the tool's JSON schema when available), merge
    /// into the original arguments, and return them. The untouched fields
    /// survive — regenerating the whole reply is what breaks good parts.
    static func fieldRepairCall(
        engine: MomijEngine, eos: Int, toolsLines: [String], tool: String,
        args: [String: Any], fields: [String], reasons: [String: String],
        task: String, baseOpts: GenerateOptions
    ) async throws -> [String: Any]? {
        guard !fields.isEmpty else { return nil }
        var opts = baseOpts.withMaxTokens(400)
        if let schema = ArgRepair.constrainedObjectSchema(
            toolLines: toolsLines, tool: tool, fields: fields),
           let schemaJSON = try? JSONSerialization.data(withJSONObject: schema),
           let schemaText = String(data: schemaJSON, encoding: .utf8),
           let tokInfo = try? engine.grammarTokenizerInfo(eos: eos),
           let guide = try? await XGrammarTokenGuide.compileJSONSchema(
               schemaText, tokenizerInfo: tokInfo, eosTokenId: eos) {
            opts = opts.withAllowedNext { prefix in
                (try? guide.allowedNext(prefix: prefix)) ?? [eos]
            }
        }
        var prompt = ArgRepair.fieldRepairPrompt(
            task: task, tool: tool,
            argsJSON: ArgRepair.argsJSONString(args),
            fields: fields, reasons: reasons)
        if let root = PathPolicy.workspaceRoot() {
            prompt += "\n\n" + PathPolicy.suffixLine()
                + " The workspace root is fixed; reply with paths RELATIVE to it."
        }
        let promptIds = try engine.tokenizer.applyChatTemplate([
            OpenAIChatCompat.ChatMessage(role: "user", content: prompt)])
        let genOpts = opts
        let out = try await engine.withLock { () -> [Int] in
            var acc: [Int] = []
            for try await t in engine.backend.generate(promptIds, options: genOpts) {
                acc.append(t)
            }
            return acc
        }
        let outIds = OpenAIChatCompat.contentTokenIds(out, eosTokenIds: genOpts.eosTokenIds)
        let text = ChatTemplatePatch.stripThinkForContent(try engine.tokenizer.decode(outIds))
        guard let generated = ArgRepair.firstJSONObject(in: text) else { return nil }
        return ArgRepair.mergeFields(into: args, generated: generated, fields: fields)
    }

    static func streamSSE(
        engine: MomijEngine, prompt: [Int], options: GenerateOptions,
        toolsAttached: Bool, toolsLines: [String],
        effectiveMessages: [OpenAIChatCompat.ChatMessage],
        includeUsage: Bool,
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
                    var streamToolArgs: [[String: Any]] = []
                    var respContentHead = ""
                    var rawHead = ""
                    if toolsAttached {
                        let decodeIds = OpenAIChatCompat.contentTokenIds(
                            tokens, eosTokenIds: options.eosTokenIds)
                        let raw = try engine.tokenizer.decode(decodeIds)
                        rawHead = raw
                        let text = ChatTemplatePatch.stripThinkForContent(raw)
                        var parsed = ToolMarkup.parsePseudoToolCalls(text)
                        // Protocol-tool interception on the stream path too:
                        // the "I'm done" intent becomes a clean stop instead
                        // of a rejection loop.
                        if let absorb = ProtocolAbsorb.absorbCall(
                            in: parsed.calls, names: ProtocolAbsorb.toolNames()) {
                            parsed = ToolMarkup.ParsedToolCalls(
                                cleanedContent: ProtocolAbsorb.finalContent(for: absorb),
                                calls: [])
                            finish = "stop"
                            trace.event("protocol_absorb", "ok", detail: ["tool": absorb.name])
                        }
                        // Degenerate-call repair on the stream path too, 3
                        // stages: field re-ask (merge), whole-reply nudge,
                        // then drop — a broken write/edit must not reach the
                        // harness as a tool_calls chunk.
                        var degenerateHit = ResponseVerify.degenerateToolArgs(
                            .init(toolCalls: parsed.calls))
                        let streamEmptyName = parsed.calls.contains(where: { $0.name.isEmpty })
                        var pendingFix: (idx: Int, fields: [String], reasons: [String: String], kind: String)? = nil
                        if let hit = degenerateHit, hit.onset < parsed.calls.count {
                            let hitArgs = ArgRepair.parseArgs(parsed.calls[hit.onset].arguments)
                            let fields = ArgRepair.repairFields(detail: hit.detail, args: hitArgs)
                            if !fields.isEmpty {
                                let reason = (hit.detail["reason"] as? String) ?? "unusable"
                                pendingFix = (hit.onset, fields,
                                    Dictionary(uniqueKeysWithValues: fields.map { ($0, reason) }),
                                    hit.kind)
                            }
                        }
                        if pendingFix == nil {
                            for (i, call) in parsed.calls.enumerated() where !call.name.isEmpty {
                                let args = ArgRepair.parseArgs(call.arguments)
                                let missing = ArgRepair.missingRequiredFields(
                                    toolLines: toolsLines, tool: call.name, args: args)
                                if !missing.isEmpty {
                                    pendingFix = (i, missing,
                                        Dictionary(uniqueKeysWithValues: missing.map {
                                            ($0, "missing required property") }),
                                        "missing_required_property")
                                    break
                                }
                                // Workspace discipline: absolute paths outside
                                // the root get repaired into relative form.
                                if let root = PathPolicy.workspaceRoot() {
                                    for f in PathPolicy.pathFields {
                                        if let v = args[f] as? String, !v.isEmpty,
                                           PathPolicy.isOutsideWorkspace(v, root: root) {
                                            pendingFix = (i, [f], [f: "absolute path outside the workspace root (use a relative path)"], "path_outside_workspace")
                                            break
                                        }
                                    }
                                    if pendingFix != nil { break }
                                }
                            }
                        }
                        if !streamEmptyName, let pf = pendingFix {
                            let brokenCall = parsed.calls[pf.idx]
                            let task = ArgRepair.originalUserText(effectiveMessages)
                            var adoptedByTask = false
                            // Extraction-first (same contract as the
                            // non-stream path): a path-kind fix is answered
                            // from the task text before any model re-ask.
                            if let taskMerged = ArgRepair.taskPathMerge(
                                kind: pf.kind, fields: pf.fields,
                                args: ArgRepair.parseArgs(brokenCall.arguments), task: task) {
                                var fixedCalls = parsed.calls
                                fixedCalls[pf.idx] = OpenAIChatCompat.ToolCallSpec(
                                    id: brokenCall.id, name: brokenCall.name,
                                    arguments: ArgRepair.argsJSONString(taskMerged))
                                if callsClean(fixedCalls, toolsLines: toolsLines) {
                                    parsed = ToolMarkup.ParsedToolCalls(
                                        cleanedContent: parsed.cleanedContent,
                                        calls: fixedCalls)
                                    degenerateHit = ResponseVerify.degenerateToolArgs(
                                        .init(toolCalls: fixedCalls))
                                    trace.event("repair", "ok", detail: [
                                        "mode": "field", "source": "task",
                                        "kind": pf.kind, "fields": pf.fields])
                                    adoptedByTask = true
                                }
                            }
                            if !adoptedByTask {
                                do {
                                    let merged = try await MomijHTTP.fieldRepairCall(
                                        engine: engine,
                                        eos: options.eosTokenIds.first ?? 151_645,
                                        toolsLines: toolsLines,
                                        tool: brokenCall.name,
                                        args: ArgRepair.parseArgs(brokenCall.arguments),
                                        fields: pf.fields, reasons: pf.reasons,
                                        task: task,
                                        baseOpts: options)
                                    if let merged {
                                        var fixedCalls = parsed.calls
                                        fixedCalls[pf.idx] = OpenAIChatCompat.ToolCallSpec(
                                            id: brokenCall.id, name: brokenCall.name,
                                            arguments: ArgRepair.argsJSONString(merged))
                                        if callsClean(fixedCalls, toolsLines: toolsLines) {
                                            parsed = ToolMarkup.ParsedToolCalls(
                                                cleanedContent: parsed.cleanedContent,
                                                calls: fixedCalls)
                                            degenerateHit = ResponseVerify.degenerateToolArgs(
                                                .init(toolCalls: fixedCalls))
                                            trace.event("repair", "ok", detail: [
                                                "mode": "field", "kind": pf.kind,
                                                "fields": pf.fields])
                                        }
                                    }
                                } catch {
                                    trace.event("repair", "error", detail: [
                                        "mode": "field", "kind": pf.kind,
                                        "error": "\(error)"])
                                }
                            }
                        }
                        let needsRepair = degenerateHit != nil || streamEmptyName
                        if needsRepair {
                            let reason = (degenerateHit?.detail["reason"] as? String) ?? "malformed"
                            let nudge = "Your previous tool call had unusable arguments (\(reason)). "
                                + "Retry with valid arguments: for write, path must be a real "
                                + "file path and content the full file text. Or reply in plain text."
                            do {
                                var repair = effectiveMessages
                                repair.append(OpenAIChatCompat.ChatMessage(role: "assistant", content: text))
                                repair.append(OpenAIChatCompat.ChatMessage(role: "user", content: nudge))
                                let repairPrompt = try engine.tokenizer.applyChatTemplate(repair)
                                let repairBudget = max(8, options.maxTokens - tokens.count)
                                let extra = try await engine.withLock { () -> [Int] in
                                    var out: [Int] = []
                                    for try await t in engine.backend.generate(
                                        repairPrompt, options: options.withMaxTokens(repairBudget)) {
                                        out.append(t)
                                    }
                                    return out
                                }
                                let repairIds = OpenAIChatCompat.contentTokenIds(
                                    extra, eosTokenIds: options.eosTokenIds)
                                let repairRaw = try engine.tokenizer.decode(repairIds)
                                let repairText = ChatTemplatePatch.stripThinkForContent(repairRaw)
                                let reparsed = ToolMarkup.parsePseudoToolCalls(repairText)
                                let reparsedClean = reparsed.calls.contains(where: { $0.name.isEmpty }) == false
                                    && ResponseVerify.degenerateToolArgs(.init(toolCalls: reparsed.calls)) == nil
                                if reparsedClean {
                                    parsed = reparsed
                                    trace.event("repair", "ok", detail: [
                                        "trigger": degenerateHit?.kind ?? "empty_name"])
                                } else {
                                    trace.event("repair", "error", detail: [
                                        "trigger": degenerateHit?.kind ?? "empty_name"])
                                }
                            } catch {
                                trace.event("repair", "error", detail: ["error": "\(error)"])
                            }
                        }
                        respContentHead = parsed.cleanedContent
                        let verifyOutcome = ResponseVerify.verify(.init(
                            content: text, reasoningContent: ResponseVerify.thinkInterior(of: raw),
                            toolCalls: parsed.calls))
                        trace.event("verify", verifyOutcome.hits.isEmpty ? "ok" : "hit", detail: [
                            "hits": verifyOutcome.hits.map { $0.eventDetail() },
                        ])
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
                        // Broken-call drop (evprtr sanitize parity): when the
                        // surviving calls are unusable (empty name or degenerate
                        // write/edit args) — after a failed repair — they must
                        // not reach the harness as a tool_calls chunk.
                        let streamBrokenCalls = parsed.calls.contains(where: { $0.name.isEmpty })
                            || ResponseVerify.degenerateToolArgs(.init(toolCalls: parsed.calls)) != nil
                            || parsed.calls.contains {
                                !ArgRepair.missingRequiredFields(
                                    toolLines: toolsLines, tool: $0.name,
                                    args: ArgRepair.parseArgs($0.arguments)).isEmpty
                            }
                            || parsed.calls.contains {
                                PathPolicy.argsOutsideWorkspace(
                                    ArgRepair.parseArgs($0.arguments),
                                    root: PathPolicy.workspaceRoot() ?? "/")
                            }
                        if streamBrokenCalls {
                            if parsed.cleanedContent.trimmingCharacters(in: .whitespaces).isEmpty {
                                parsed = ToolMarkup.ParsedToolCalls(
                                    cleanedContent:
                                    "Previous tool call arguments were unusable and were discarded.",
                                    calls: [])
                            } else {
                                parsed = ToolMarkup.ParsedToolCalls(
                                    cleanedContent: parsed.cleanedContent, calls: [])
                            }
                            trace.event("sanitize", "ok", detail: ["action": "drop_degenerate_calls"])
                        }
                        if !parsed.calls.isEmpty {
                            finish = "tool_calls"
                            if let root = PathPolicy.workspaceRoot() {
                                let (resolved, n) = MomijHTTP.resolveWorkspacePaths(
                                    parsed.calls, root: root)
                                if n > 0 {
                                    parsed = ToolMarkup.ParsedToolCalls(
                                        cleanedContent: parsed.cleanedContent, calls: resolved)
                                    trace.event("path_resolve", "ok", detail: ["resolved": n])
                                }
                            }
                            streamToolCalls = parsed.calls.count
                            streamToolArgs = parsed.calls.map { call in
                                ["name": call.name, "arguments": String(call.arguments.prefix(300))]
                            }
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
                    } else {
                        // Non-tools stream: verify the assembled content too.
                        let decodeIds = OpenAIChatCompat.contentTokenIds(
                            tokens, eosTokenIds: options.eosTokenIds)
                        if let raw = try? engine.tokenizer.decode(decodeIds) {
                            let text = ChatTemplatePatch.stripThinkForContent(raw)
                            let verifyOutcome = ResponseVerify.verify(.init(
                                content: text, reasoningContent: ResponseVerify.thinkInterior(of: raw)))
                            trace.event("verify", verifyOutcome.hits.isEmpty ? "ok" : "hit", detail: [
                                "hits": verifyOutcome.hits.map { $0.eventDetail() },
                            ])
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
                    // OpenAI spec: with stream_options.include_usage the tail
                    // chunk carries usage with empty choices. DSH's client
                    // reads chunk.usage on any chunk for its token counters
                    // (0toks display fix, fed back from the subagent test).
                    if includeUsage {
                        let usageChunk: [String: Any] = [
                            "id": "chatcmpl-momij",
                            "object": "chat.completion.chunk",
                            "created": Int(Date().timeIntervalSince1970),
                            "model": engine.modelID,
                            "choices": [] as [[String: Any]],
                            "usage": [
                                "prompt_tokens": prompt.count,
                                "completion_tokens": tokens.count,
                                "total_tokens": prompt.count + tokens.count,
                            ],
                        ]
                        if let data = try? JSONSerialization.data(withJSONObject: usageChunk),
                           let line = String(data: data, encoding: .utf8) {
                            var ub = ByteBufferAllocator().buffer(capacity: line.count + 16)
                            ub.writeString("data: \(line)\n\n")
                            cont.yield(ub)
                        }
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
                        "tool_args": streamToolArgs,
                        "content_head": String(respContentHead.prefix(300)),
                        "raw_head": String(rawHead.prefix(400)),
                        "usage": ["prompt": prompt.count, "completion": tokens.count],
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
