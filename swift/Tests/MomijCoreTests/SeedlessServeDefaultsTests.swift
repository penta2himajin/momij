import XCTest
import Foundation
@testable import MomijCore

/// Tests for the seedless serve KV-capacity sizing (`MOMIJ_FULL_MAX_LEN`)
/// and the pre-flight prompt-capacity hook (`LLMBackend.maxPromptTokens`).
final class SeedlessServeDefaultsTests: XCTestCase {
    private final class StubBackend: LLMBackend {
        func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncThrowingStream<Int, Error> {
            AsyncThrowingStream { $0.finish() }
        }
    }

    // MARK: fullMaxLenFromEnv

    func testFullMaxLenDefaultWhenEnvUnset() {
        XCTAssertEqual(SeedlessServeDefaults.fullMaxLenFromEnv(nil), 16_384)
    }

    func testFullMaxLenParsesOverride() {
        XCTAssertEqual(SeedlessServeDefaults.fullMaxLenFromEnv("32768"), 32_768)
        XCTAssertEqual(SeedlessServeDefaults.fullMaxLenFromEnv(" 65536 "), 65_536)
    }

    func testFullMaxLenFallsBackOnGarbage() {
        XCTAssertEqual(SeedlessServeDefaults.fullMaxLenFromEnv("garbage"), 16_384)
        XCTAssertEqual(SeedlessServeDefaults.fullMaxLenFromEnv(""), 16_384)
    }

    func testFullMaxLenFallsBackOnNonPositive() {
        // A zero/negative capacity would make every request fail; fall back.
        XCTAssertEqual(SeedlessServeDefaults.fullMaxLenFromEnv("0"), 16_384)
        XCTAssertEqual(SeedlessServeDefaults.fullMaxLenFromEnv("-4096"), 16_384)
    }

    // MARK: maxPromptTokens (pre-flight capacity hook)

    func testMaxPromptTokensNilByDefault() {
        XCTAssertNil(StubBackend().maxPromptTokens)
    }

    func testMaxPromptTokensMirrorsEngineCapacity() throws {
        // Must dispatch through the existential (the server holds the
        // backend as `any LLMBackend`): a protocol-extension-only
        // declaration would statically return nil for every conformer.
        let store = try? WeightStore(modelDir: Self.modelDir())
        guard let store else {
            throw XCTSkip("maple-preview model dir not available in this environment")
        }
        let engine = try? SeedlessDecodeEngine(store: store, fullMaxLen: 4096)
        guard let engine else { throw XCTSkip("seedless engine unavailable") }
        let backend: any LLMBackend = SeedlessBackend(engine: engine)
        XCTAssertEqual(backend.maxPromptTokens, 4096)
    }

    private static func modelDir() -> String {
        ProcessInfo.processInfo.environment["MOMIJ_MODEL"]
            ?? NSString("~/models/deepgrove/maple-preview-2bit-mlx").expandingTildeInPath
    }
}