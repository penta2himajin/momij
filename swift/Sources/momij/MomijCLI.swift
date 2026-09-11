import Foundation
import MomijCore
import Tokenizers

@main
struct MomijMain {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let cmd = args.first else {
            printUsage()
            return
        }
        do {
            switch cmd {
            case "bench":
                try await runBench(Array(args.dropFirst()))
            case "serve":
                try await runServeCmd(Array(args.dropFirst()))
            case "seedless-bench":
                try runSeedlessBench(Array(args.dropFirst()))
            case "generate":
                try await runGenerate(Array(args.dropFirst()))
            default:
                printUsage()
            }
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    static func printUsage() {
        print(
            """
            momij — Maple-Preview high-speed inference (oMLX replacement)

            Usage:
              momij bench --model <dir> [--backend mlx|oracle|seedless] [--flash-head] [-p 128] [-g 256] [-n 3]
              momij seedless-bench [--model <dir>]
              momij generate --model <dir> --prompt <text> [--backend mlx|oracle|seedless] [--suffix-spec] [--temperature T] [--top-p P] [--repetition-penalty R]
              momij serve --model <dir> [--backend seedless|mlx|oracle] [--port 8742] [--host 127.0.0.1]
            """
        )
    }

    static func flag(_ args: [String], _ name: String) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func has(_ args: [String], _ name: String) -> Bool { args.contains(name) }

    static func defaultModel() -> String {
        ProcessInfo.processInfo.environment["MOMIJ_MODEL"]
            ?? NSString("~/models/deepgrove/maple-preview-2bit-mlx").expandingTildeInPath
    }

    static func runBench(_ args: [String]) async throws {
        let model = flag(args, "--model") ?? defaultModel()
        let backend = flag(args, "--backend") ?? "oracle"
        let p = Int(flag(args, "-p") ?? flag(args, "--prompt-tokens") ?? "128")!
        let g = Int(flag(args, "-g") ?? flag(args, "--generation-tokens") ?? "256")!
        let n = Int(flag(args, "-n") ?? flag(args, "--num-trials") ?? "3")!
        let flash = has(args, "--flash-head")

        if backend == "oracle" {
            let ob = try OracleBackend(modelDir: model, flashHead: flash)
            let r = try ob.benchmark(promptTokens: p, genTokens: g, trials: n)
            print(String(format: "backend=oracle flash_head=%@ prompt_tps=%.3f generation_tps=%.3f peak_memory=%.3f",
                         flash ? "true" : "false",
                         r["prompt_tps"] ?? 0, r["generation_tps"] ?? 0, r["peak_memory"] ?? 0))
        } else if backend == "seedless" {
            let store = try WeightStore(modelDir: model)
            if has(args, "--sweep-cb") {
                print(try SeedlessDecodeEngine.sweepLayersPerCB(store: store, prompt: p, gen: g))
                return
            }
            let fullLen = max(p + g + 64, 2048)
            let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: fullLen)
            if has(args, "--suffix-spec") {
                var promptIds: [Int]? = nil
                if let promptText = flag(args, "--prompt") {
                    let tokenizer = try await loadTokenizer(modelDir: model)
                    promptIds = try tokenizer.encode(promptText)
                }
                print(try eng.benchmarkSuffixSpec(
                    promptTokens: p, genTokens: g, trials: n, draftK: 8, prompt: promptIds))
            } else if let ts = flag(args, "--temperature"), let temp = Float(ts), temp > 0 {
                var promptIds: [Int]? = nil
                if let promptText = flag(args, "--prompt") {
                    let tokenizer = try await loadTokenizer(modelDir: model)
                    promptIds = try tokenizer.encode(promptText)
                }
                let r = try eng.benchmarkSampled(
                    promptTokens: p, genTokens: g, trials: n, temperature: temp,
                    prompt: promptIds)
                print(String(format: "backend=seedless sampled temp=%.2f seq=%.1f tok/s  spec=%.1f tok/s (gen-only)  accept/attempt=%.2f  batched=%d",
                             temp, r.seqTps, r.specTps, r.acceptPerAttempt, r.batched))
            } else {
                let r = try eng.benchmark(promptTokens: p, genTokens: g, trials: n, profile: true)
                let ph = r.phase
                print(String(format: "backend=seedless layers_per_cb=%d flash_head=%@ fuse=%@ probes=%d chain_k=%d prompt_tps=%.3f generation_tps=%.3f",
                             eng.layersPerCB,
                             eng.useFlashHead ? "true" : "false",
                             eng.flashFused ? "true" : "false",
                             eng.flashProbes,
                             SeedlessDecodeEngine.envChainK,
                             r.promptTps, r.genTps))
                print(String(format: "  phase_ms/tok embed=%.3f layers=%.3f head=%.3f  (sum=%.3f)",
                             ph.embed, ph.layers, ph.head,
                             ph.embed + ph.layers + ph.head))
                if let rel = try? SeedlessDecodeEngine.parityL0(store: store) {
                    print(String(format: "  parity L0 rel_l2=%.4e", rel))
                }
            }
        } else {
            MoEProfile.reset()
            let store = try WeightStore(modelDir: model)
            let engine = MapleEngine(store: store)
            let r = engine.benchmark(promptTokens: p, genTokens: g, trials: n)
            print(String(format: "backend=mlx prompt_tps=%.3f generation_tps=%.3f peak_memory=%.3f",
                         r.promptTps, r.genTps, r.peakGB))
            MoEProfile.dumpIfEnabled()
        }
    }

    static func runSeedlessBench(_ args: [String]) throws {
        let model = flag(args, "--model") ?? defaultModel()
        try SeedlessMetal.ensureCompiled()
        let qmv = try SeedlessMetal.benchQmv2(iters: 200)
        print(String(format: "seedless gqmm2 (H=2048→N=512,Ktop=1) kernel/s=%.1f", qmv))
        print(try SeedlessMetal.benchGqmm2Mrow(iters: 30))
        print(try SeedlessMetal.benchGqmm2Mrow(
            K: 512, N: 2048, Ktop: 8, iters: 30, lhsPerExpert: true))
        let fused = try SeedlessMetal.benchFusedExpert(E: 256, iters: 100)
        print(String(format: "seedless fused-expert (E=256,K=8) steps/s=%.1f", fused))
        print(try SeedlessMetal.benchFusedExpertMrow(iters: 20))
        print(try SeedlessMetal.benchAttnMrow(iters: 20))
        print(try SeedlessMetal.benchAttnBlockMrow(iters: 20))
        print(try SeedlessMetal.benchMoEBlockMrow(iters: 20))
        print(try SeedlessMetal.benchLayerMrowConfigs(iters: 8))
        if FileManager.default.fileExists(atPath: model) {
            do {
                let store = try WeightStore(modelDir: model)
                let real = try SeedlessEngine.benchRealExpert(store: store, iters: 50)
                print(String(format: "seedless fused-expert real-weights steps/s=%.1f", real))
                let moe1cb = try SeedlessEngine.benchMoEBlock(store: store, layer: 0, iters: 50)
                // Extrapolate 24 layers × 1 CB each ≈ decode floor (attn not included yet).
                let layers = store.config.numHiddenLayers
                let tokFloor = moe1cb / Double(layers)
                print(String(format: "seedless moe-block-1cb (L0, rms+gate+top8+experts+resid) blocks/s=%.1f  → ~%.1f tok/s floor @%d layers (no attn)",
                             moe1cb, tokFloor, layers))
                // Milestone B layer profile is the primary claim; skip MoE-only stack to save RAM/time.
                print(try SeedlessEngine.profileLayerStack(store: store, iters: 8))
                if has(args, "--profile-moe") {
                    print(try SeedlessEngine.profileMoEBlock(store: store, layer: 0, iters: 40))
                }
                if has(args, "--profile-floor") {
                    print(try SeedlessEngine.profileDecodeFloor(store: store, iters: 16))
                }
                if has(args, "--sweep-cb") {
                    print(try SeedlessDecodeEngine.sweepLayersPerCB(store: store, prompt: 64, gen: 64))
                }
                if has(args, "--suffix-spec") {
                    let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048)
                    print(try eng.benchmarkSuffixSpec(promptTokens: 128, genTokens: 128, trials: 3, draftK: 8))
                }
            } catch {
                fputs("[momij] skip real-weight seedless bind: \(error)\n", stderr)
            }
        }
    }

    static func runGenerate(_ args: [String]) async throws {
        let model = flag(args, "--model") ?? defaultModel()
        let backendName = flag(args, "--backend") ?? "oracle"
        let promptText = flag(args, "--prompt") ?? "Write a haiku about a maple grove."
        let maxTok = Int(flag(args, "--max-tokens") ?? "64")!
        let suffix = has(args, "--suffix-spec")
        let temperature = Double(flag(args, "--temperature") ?? "0") ?? 0
        let topP = Double(flag(args, "--top-p") ?? "1") ?? 1
        let presence = Double(flag(args, "--presence-penalty") ?? "0") ?? 0
        let frequency = Double(flag(args, "--frequency-penalty") ?? "0") ?? 0
        let repetition = Double(flag(args, "--repetition-penalty") ?? "1") ?? 1

        let tokenizer = try await loadTokenizer(modelDir: model)
        let ids = try tokenizer.encode(promptText)
        let opts = GenerateOptions(
            maxTokens: maxTok,
            temperature: temperature,
            topP: topP,
            presencePenalty: presence,
            frequencyPenalty: frequency,
            repetitionPenalty: repetition,
            useSuffixSpec: suffix)

        let backend: any LLMBackend
        switch backendName {
        case "mlx":
            backend = try MapleMLXBackend(modelDir: model)
        case "seedless":
            backend = try SeedlessBackend(modelDir: model)
        default:
            backend = try OracleBackend(modelDir: model, flashHead: has(args, "--flash-head"))
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        var out: [Int] = []
        for try await t in backend.generate(ids, options: opts) {
            out.append(t)
            if let s = try? tokenizer.decode([t]) {
                print(s, terminator: "")
                fflush(stdout)
            }
        }
        let dt = max(CFAbsoluteTimeGetCurrent() - t0, 1e-9)
        print()
        print(String(format: "[momij] tokens=%d  tok/s=%.1f", out.count, Double(out.count) / dt))
    }

    static func runServeCmd(_ args: [String]) async throws {
        let model = flag(args, "--model") ?? defaultModel()
        let backendName = flag(args, "--backend") ?? "seedless"
        let host = flag(args, "--host") ?? "127.0.0.1"
        let port = Int(flag(args, "--port") ?? "8742")!
        let modelID = flag(args, "--model-id") ?? "maple-preview"

        let tokenizer = try await loadTokenizer(modelDir: model)
        let backend: any LLMBackend
        switch backendName {
        case "mlx":
            backend = try MapleMLXBackend(modelDir: model)
        case "oracle":
            backend = try OracleBackend(modelDir: model, flashHead: has(args, "--flash-head"))
        default:
            backend = try SeedlessBackend(modelDir: model)
        }
        let engine = MomijHTTP.MomijEngine(tokenizer: tokenizer, backend: backend, modelID: modelID)
        try await MomijHTTP.runServe(engine: engine, host: host, port: port)
    }

    static func loadTokenizer(modelDir: String) async throws -> any MomijHTTP.TokenizerAdapter {
        do {
            let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: modelDir))
            return HFTokenizer(tok)
        } catch {
            fputs("[momij] warning: HF tokenizer load failed (\(error)); using byte fallback\n", stderr)
            return MomijHTTP.ByteTokenizer()
        }
    }
}

struct HFTokenizer: MomijHTTP.TokenizerAdapter {
    let inner: any Tokenizer
    init(_ inner: any Tokenizer) { self.inner = inner }
    func encode(_ text: String) throws -> [Int] { inner.encode(text: text) }
    func decode(_ ids: [Int]) throws -> String { inner.decode(tokens: ids) }
    func applyChatTemplate(_ messages: [MomijHTTP.ChatMessage]) throws -> [Int] {
        let dicts: [[String: String]] = messages.map {
            ["role": $0.role, "content": $0.content ?? ""]
        }
        if let ids = try? inner.applyChatTemplate(messages: dicts) {
            return ids
        }
        let text = messages.map { "\($0.role): \($0.content ?? "")" }.joined(separator: "\n")
        return try encode(text)
    }
}
