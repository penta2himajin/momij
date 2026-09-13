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
}