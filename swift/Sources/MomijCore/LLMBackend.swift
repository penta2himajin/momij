import Foundation

public struct GenerateOptions: Sendable {
    public var maxTokens: Int = 256
    public var temperature: Double = 0
    public var topP: Double = 1
    /// OpenAI presence_penalty (−2…2). Non-zero leaves the greedy SuffixSpec path.
    public var presencePenalty: Double = 0
    /// OpenAI frequency_penalty (−2…2).
    public var frequencyPenalty: Double = 0
    /// HF / vLLM-style; 1.0 = off. Values >1 penalize repeats.
    public var repetitionPenalty: Double = 1
    public var eosTokenIds: [Int] = [151_645]
    public var useSuffixSpec: Bool = false
    public var draftK: Int = 8
    /// Optional RNG seed for sampled decode (`nil` → nondeterministic).
    public var seed: UInt64? = nil
    /// Finite-string / grammar frontier: allowed next token ids given prefix.
    public var allowedNext: (@Sendable ([Int]) -> Set<Int>)? = nil

    public init(
        maxTokens: Int = 256, temperature: Double = 0, topP: Double = 1,
        presencePenalty: Double = 0, frequencyPenalty: Double = 0,
        repetitionPenalty: Double = 1,
        eosTokenIds: [Int] = [151_645], useSuffixSpec: Bool = false, draftK: Int = 8,
        seed: UInt64? = nil,
        allowedNext: (@Sendable ([Int]) -> Set<Int>)? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty
        self.eosTokenIds = eosTokenIds
        self.useSuffixSpec = useSuffixSpec
        self.draftK = draftK
        self.seed = seed
        self.allowedNext = allowedNext
    }

    /// Fast SuffixSpec / M-row greedy path is valid only for deterministic greedy.
    public var isGreedyCompatible: Bool {
        temperature <= 1e-5
            && topP >= 1 - 1e-6
            && presencePenalty == 0
            && frequencyPenalty == 0
            && abs(repetitionPenalty - 1) < 1e-6
    }
}

public protocol LLMBackend: AnyObject {
    func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncThrowingStream<Int, Error>
}

/// Production Seedless Metal backend. Greedy-compatible requests keep SuffixSpec/M-row;
/// sampling / penalties use FlashHead candidate sampling (+ optional speculative).
public final class SeedlessBackend: LLMBackend, @unchecked Sendable {
    public let engine: SeedlessDecodeEngine
    /// When true (default), non-greedy paths use rejection-sampling drafts.
    public let speculativeSample: Bool

    public init(modelDir: String, fullMaxLen: Int = 16_384) throws {
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        try SeedlessMetal.ensureCompiled()
        self.engine = try SeedlessDecodeEngine(store: store, fullMaxLen: fullMaxLen)
        self.speculativeSample = ProcessInfo.processInfo.environment["MOMIJ_SPEC_SAMPLE"] != "0"
    }

    public init(engine: SeedlessDecodeEngine, speculativeSample: Bool = true) {
        self.engine = engine
        self.speculativeSample = speculativeSample
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncThrowingStream<Int, Error> {
        AsyncThrowingStream { cont in
            Task {
                do {
                    let eos = options.eosTokenIds.first
                    let tokens: [Int]
                    if let allowedNext = options.allowedNext {
                        // Finite grammar: walk the frontier (deterministic). Prefer
                        // continuing tokens over EOS; among ties, lowest id.
                        tokens = Self.generateConstrained(
                            maxTokens: options.maxTokens,
                            eos: eos ?? 151_645,
                            allowedNext: allowedNext)
                    } else if options.isGreedyCompatible {
                        if options.useSuffixSpec {
                            let r = try self.engine.generateSuffixSpec(
                                prompt: prompt, maxTokens: options.maxTokens,
                                draftK: options.draftK, eos: eos)
                            tokens = r.tokens
                        } else {
                            tokens = try self.engine.generate(
                                prompt: prompt, maxTokens: options.maxTokens, eos: eos)
                        }
                    } else {
                        let proc = LogitsProcessor.from(options)
                        if self.speculativeSample {
                            tokens = try self.engine.generateSampledSpeculative(
                                prompt: prompt, maxTokens: options.maxTokens,
                                processor: proc, draftK: options.draftK,
                                eos: eos, seed: options.seed)
                        } else {
                            tokens = try self.engine.generateSampled(
                                prompt: prompt, maxTokens: options.maxTokens,
                                processor: proc, eos: eos, seed: options.seed)
                        }
                    }
                    for t in tokens { cont.yield(t) }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
        }
    }

    /// Deterministic finite-language walk (no logits). Used when Seedless cannot
    /// cheaply mask the full vocab; MLX backend prefers score-argmax among allowed.
    static func generateConstrained(
        maxTokens: Int,
        eos: Int,
        allowedNext: @Sendable ([Int]) -> Set<Int>
    ) -> [Int] {
        var out: [Int] = []
        var prefix: [Int] = []
        while out.count < maxTokens {
            let allow = allowedNext(prefix)
            if allow.isEmpty { break }
            let cont = allow.filter { $0 != eos }.sorted()
            let pick: Int
            if let first = cont.first {
                pick = first
            } else if allow.contains(eos) {
                pick = eos
            } else {
                break
            }
            out.append(pick)
            prefix.append(pick)
            if pick == eos { break }
        }
        return out
    }
}

/// MLX Maple greedy backend.
public final class MapleMLXBackend: LLMBackend, @unchecked Sendable {
    public let engine: MapleEngine

    public init(modelDir: String) throws {
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        self.engine = MapleEngine(store: store)
    }

    public init(engine: MapleEngine) {
        self.engine = engine
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncThrowingStream<Int, Error> {
        AsyncThrowingStream { cont in
            Task {
                do {
                    let tokens: [Int]
                    if options.useSuffixSpec {
                        tokens = try SuffixSpec.run(
                            prompt: prompt, maxTokens: options.maxTokens,
                            draftK: options.draftK, eos: options.eosTokenIds.first,
                            step: { ids in
                                // Greedy one-step: regenerate from full prefix (correct but slow verify).
                                // Prefill each time is wasteful; good enough for SuffixSpec wiring.
                                let out = self.engine.generate(
                                    prompt: ids, maxTokens: 1, eos: nil)
                                return out.first
                            },
                            multiStep: { ids, k in
                                self.engine.generate(prompt: ids, maxTokens: k, eos: nil)
                            })
                    } else {
                        tokens = self.engine.generate(
                            prompt: prompt, maxTokens: options.maxTokens,
                            eos: options.eosTokenIds.first,
                            allowedNext: options.allowedNext)
                    }
                    for t in tokens { cont.yield(t) }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
        }
    }
}

/// Oracle backend: long-lived Python mlx-lm-deepgrove worker (JSONL).
public final class OracleBackend: LLMBackend, @unchecked Sendable {
    private let process: Process
    private let stdinPipe: Pipe
    private let stdoutPipe: Pipe
    private let lock = NSLock()

    public init(
        modelDir: String,
        python: String = "/Users/penta2himajin/repos/mlx-lm-deepgrove/.venv/bin/python",
        worker: String? = nil,
        flashHead: Bool = false
    ) throws {
        let workerPath: String
        if let worker {
            workerPath = worker
        } else {
            // LLMBackend.swift lives at <repo>/swift/Sources/MomijCore/
            var dir = URL(fileURLWithPath: #filePath)
            for _ in 0 ..< 4 { dir = dir.deletingLastPathComponent() }
            workerPath = dir.appendingPathComponent("python/momij_oracle/worker.py").path
        }
        guard FileManager.default.fileExists(atPath: workerPath) else {
            throw OracleError.remote("worker not found at \(workerPath)")
        }
        guard FileManager.default.isExecutableFile(atPath: python)
                || FileManager.default.fileExists(atPath: python) else {
            throw OracleError.remote("python not found at \(python)")
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        var args = [workerPath, "--model", modelDir]
        if flashHead { args.append("--flash-head") }
        p.arguments = args
        let inn = Pipe(), out = Pipe()
        p.standardInput = inn
        p.standardOutput = out
        p.standardError = FileHandle.standardError
        try p.run()
        self.process = p
        self.stdinPipe = inn
        self.stdoutPipe = out
        // wait ready
        guard let line = readLine(), line.contains("ready") else {
            throw OracleError.notReady
        }
    }

    deinit {
        if process.isRunning {
            process.terminate()
        }
    }

    private func readLine() -> String? {
        var data = Data()
        let handle = stdoutPipe.fileHandleForReading
        while true {
            let chunk = handle.readData(ofLength: 1)
            if chunk.isEmpty { return nil }
            if chunk[0] == UInt8(ascii: "\n") { break }
            data.append(chunk)
        }
        return String(data: data, encoding: .utf8)
    }

    private func request(_ obj: [String: Any]) throws -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        let data = try JSONSerialization.data(withJSONObject: obj)
        stdinPipe.fileHandleForWriting.write(data)
        stdinPipe.fileHandleForWriting.write(Data("\n".utf8))
        guard let line = readLine(),
              let resp = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
        else { throw OracleError.badResponse }
        if let err = resp["error"] as? String { throw OracleError.remote(err) }
        return resp
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncThrowingStream<Int, Error> {
        AsyncThrowingStream { cont in
            Task {
                do {
                    var body: [String: Any] = [
                        "cmd": "generate",
                        "prompt": prompt,
                        "max_tokens": options.maxTokens,
                        "temperature": options.temperature,
                    ]
                    if options.useSuffixSpec {
                        body["suffix_spec"] = true
                        body["draft_k"] = options.draftK
                    }
                    let resp = try self.request(body)
                    let tokens = (resp["tokens"] as? [Int]) ?? []
                    for t in tokens { cont.yield(t) }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
        }
    }

    public func benchmark(promptTokens: Int, genTokens: Int, trials: Int) throws -> [String: Double] {
        var body: [String: Any] = [
            "cmd": "benchmark",
            "prompt_tokens": promptTokens,
            "generation_tokens": genTokens,
            "num_trials": trials,
        ]
        if ProcessInfo.processInfo.environment["MOMIJ_PROFILE_MOE"] == "1" {
            body["profile"] = true
        }
        let resp = try request(body)
        if let prof = resp["profile"] as? [String: Any] {
            let router = (prof["router_ms"] as? Double) ?? 0
            let sw = (prof["switch_ms"] as? Double) ?? 0
            let agg = (prof["agg_ms"] as? Double) ?? 0
            let moe = (prof["moe_ms"] as? Double) ?? 0
            let attn = (prof["attn_ms"] as? Double) ?? 0
            let n = Int((prof["decode_moe_calls"] as? Double) ?? 0)
            fputs(String(format:
                "[moe-profile] oracle sync decode_moe=%d router_ms=%.3f switch_ms=%.3f agg_ms=%.3f moe_ms=%.3f attn_ms=%.3f\n",
                n, router, sw, agg, moe, attn), stderr)
        }
        return [
            "prompt_tps": resp["prompt_tps"] as? Double ?? 0,
            "generation_tps": resp["generation_tps"] as? Double ?? 0,
            "peak_memory": resp["peak_memory"] as? Double ?? 0,
        ]
    }
}

public enum OracleError: Error, CustomStringConvertible {
    case notReady, badResponse, remote(String)
    public var description: String {
        switch self {
        case .notReady: return "oracle worker not ready"
        case .badResponse: return "oracle bad response"
        case .remote(let s): return "oracle: \(s)"
        }
    }
}
