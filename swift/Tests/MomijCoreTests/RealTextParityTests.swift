import XCTest
import Tokenizers
@testable import MomijCore

/// Real-text losslessness contract for the seedless serve paths.
/// SuffixSpec (both exact-head and serve-default) must equal exact sequential
/// greedy on real-text prompts; the M-row chain verify must match sequential.
final class RealTextParityTests: XCTestCase {
    private var modelDir: String {
        ProcessInfo.processInfo.environment["MOMIJ_MODEL"]
            ?? NSString("~/models/deepgrove/maple-preview-2bit-mlx").expandingTildeInPath
    }

    private func makeStore() throws -> WeightStore {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        return store
    }

    private func promptIds(_ text: String) async throws -> [Int] {
        let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: modelDir))
        return tok.encode(text: text)
    }

    /// Long enough to leave the trivial band, full of repeated phrases so the
    /// suffix-tree and recycle drafters fire like they do on agentic traffic.
    static let agentText = "You are a coding assistant. Use tools when needed. "
        + "List the files in the current directory using the ls tool. "
        + "For each function call, return a json object with function name and arguments "
        + "within tool call markup tags. "
        + "If you can answer without a tool, reply with plain text only. "
        + "You may call one or more functions to assist with the user query. "
        + "List the files in the current directory using the ls tool. "
        + "For each function call, return a json object with function name and arguments "
        + "within tool call markup tags."

    func testExactHeadSuffixSpecMatchesExactGreedy() async throws {
        let store = try makeStore()
        let prompt = try await promptIds(Self.agentText)
        let refEng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        let reference = try refEng.generate(prompt: prompt, maxTokens: 64, eos: nil)
        let eng = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 2048, enableFlashHead: true, enableExactHead: true)
        let spec = try eng.generateSuffixSpec(prompt: prompt, maxTokens: 64, eos: nil)
        XCTAssertLessThanOrEqual(
            spec.tokens.count, 64,
            "SuffixSpec overshot maxTokens: " + String(spec.tokens.count))
        let detail = String(describing: spec.tokens) + " vs " + String(describing: reference)
            + " accepted " + String(spec.accepted) + "/" + String(spec.attempts)
        XCTAssertEqual(spec.tokens, reference, "exact-head SuffixSpec diverged: " + detail)
    }

    /// Forced batching verifies via the one-CB chain on every attempt, which
    /// exercises snapshot/restore+replay on rejects. Must stay lossless.
    func testForcedBatchSuffixSpecMatchesExactGreedy() async throws {
        setenv("MOMIJ_SPEC_BATCH", "1", 1)
        defer { setenv("MOMIJ_SPEC_BATCH", "0", 1) }
        let store = try makeStore()
        let prompt = try await promptIds(Self.agentText)
        let refEng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        let reference = try refEng.generate(prompt: prompt, maxTokens: 64, eos: nil)
        let eng = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 2048, enableFlashHead: true, enableExactHead: true)
        let spec = try eng.generateSuffixSpec(prompt: prompt, maxTokens: 64, eos: nil)
        XCTAssertLessThanOrEqual(spec.tokens.count, 64)
        let detail = String(describing: spec.tokens) + " vs " + String(describing: reference)
            + " accepted " + String(spec.accepted) + "/" + String(spec.attempts)
        XCTAssertEqual(spec.tokens, reference, "forced-batch SuffixSpec diverged: " + detail)
    }

    func testServeDefaultSuffixSpecMatchesExactGreedy() async throws {
        let store = try makeStore()
        let prompt = try await promptIds(Self.agentText)
        let refEng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        let reference = try refEng.generate(prompt: prompt, maxTokens: 64, eos: nil)
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048)
        let spec = try eng.generateSuffixSpec(prompt: prompt, maxTokens: 64, eos: nil)
        let detail = String(describing: spec.tokens) + " vs " + String(describing: reference)
            + " accepted " + String(spec.accepted) + "/" + String(spec.attempts)
        XCTAssertEqual(spec.tokens, reference, "serve-default SuffixSpec diverged: " + detail)
    }

    func testChainEvalsMatchSequential() async throws {
        let store = try makeStore()
        let prompt = try await promptIds(Self.agentText)
        let K = 8
        let refEng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        let gen = try refEng.generate(prompt: prompt, maxTokens: K + 1, eos: nil)
        XCTAssertTrue(gen.count >= K + 1)
        let first = gen[0]
        let seq = Array(gen[1 ..< K + 1])
        let chainEng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        _ = try chainEng.generate(prompt: prompt, maxTokens: 1, eos: nil)
        let feeds = [first] + Array(seq.prefix(K - 1))
        let evals = try chainEng.stepChainFeeds(feeds)
        XCTAssertEqual(evals, seq, String(describing: evals) + " vs " + String(describing: seq))
    }

    /// M-row prefill must keep working after the SWA ring wraps (>512 tokens):
    /// chunks stay large instead of collapsing to M=1, and the hidden state
    /// still matches the MLX reference layer by layer.
    func testWrappedPrefillMatchesMLX() async throws {
        setenv("MOMIJ_SPEC_MAX_M", "64", 1)
        let store = try makeStore()
        let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: modelDir))
        var prompt = tok.encode(text: Self.agentText)
        while prompt.count < 600 { prompt += tok.encode(text: " " + Self.agentText) }
        prompt = Array(prompt.prefix(600))
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        _ = try eng.generate(prompt: prompt, maxTokens: 1, eos: nil)
        var pos = 0
        var minPostWrap = 64
        for m in eng.lastPrefillChunks {
            if pos >= 512 { minPostWrap = min(minPostWrap, m) }
            pos += m
        }
        XCTAssertEqual(pos, prompt.count)
        XCTAssertGreaterThan(minPostWrap, 1, "post-wrap chunks collapsed to M=1: "
            + String(describing: eng.lastPrefillChunks))
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let mlxH = mlx.lastTokenHiddenAfterEachLayer(prompt)
        let seedH = try eng.lastTokenHiddenAfterEachLayer(prompt: prompt)
        for (i, pair) in zip(seedH.layers, mlxH.layers).enumerated() {
            let sArr = pair.0
            let mArr = pair.1
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for j in 0 ..< sArr.count {
                dot += sArr[j] * mArr[j]; na += sArr[j] * sArr[j]; nb += mArr[j] * mArr[j]
            }
            let cos = dot / (sqrt(na) * sqrt(nb) + 1e-12)
            // Cyclic-slot fp16 summation order differs from MLX's batched
            // kernels; the binding contract is greedy token parity (tested
            // separately). 0.96 keeps gross window/math errors out.
            XCTAssertGreaterThan(cos, 0.96, "layer " + String(i) + " cos=" + String(cos))
        }
    }

    /// Greedy continuation after the SWA wrap must match the MLX reference.
    func testWrappedGreedyMatchesMLX() async throws {
        setenv("MOMIJ_SPEC_MAX_M", "64", 1)
        let store = try makeStore()
        let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: modelDir))
        var prompt = tok.encode(text: Self.agentText)
        while prompt.count < 600 { prompt += tok.encode(text: " " + Self.agentText) }
        prompt = Array(prompt.prefix(600))
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let mlxOut = mlx.generate(prompt: prompt, maxTokens: 8, eos: nil)
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        let seedOut = try eng.generate(prompt: prompt, maxTokens: 8, eos: nil)
        XCTAssertEqual(seedOut, mlxOut, String(describing: seedOut) + " vs " + String(describing: mlxOut))
    }

    /// M-row chain evals must match sequential evals on a >512-token prompt.
    func testWrappedChainEvalsMatchSequential() async throws {
        setenv("MOMIJ_SPEC_MAX_M", "64", 1)
        let store = try makeStore()
        let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: modelDir))
        var prompt = tok.encode(text: Self.agentText)
        while prompt.count < 600 { prompt += tok.encode(text: " " + Self.agentText) }
        prompt = Array(prompt.prefix(600))
        let K = 8
        let refEng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        let gen = try refEng.generate(prompt: prompt, maxTokens: K + 1, eos: nil)
        XCTAssertTrue(gen.count >= K + 1)
        let first = gen[0]
        let seq = Array(gen[1 ..< K + 1])
        let chainEng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        _ = try chainEng.generate(prompt: prompt, maxTokens: 1, eos: nil)
        let feeds = [first] + Array(seq.prefix(K - 1))
        let evals = try chainEng.stepChainFeeds(feeds)
        XCTAssertEqual(evals, seq, String(describing: evals) + " vs " + String(describing: seq))
    }
}
