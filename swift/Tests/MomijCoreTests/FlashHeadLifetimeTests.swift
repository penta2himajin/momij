import XCTest
import Foundation
import MLX
import Metal
@testable import MomijCore

/// Release-only SIGSEGV in `SeedlessFlashHead.init` (objc_release).
///
/// Bisect: `c94f5bf` (MROW default on) release bench lives (~169 tok/s);
/// `57535f7` (sampling dual-path) dies in this init. Debug never reproduced it.
/// Run: `swift test -c release --filter FlashHeadLifetimeTests` with `mlx.metallib`
/// next to the xctest binary.
final class FlashHeadLifetimeTests: XCTestCase {
    static func mapleDir() -> String {
        ProcessInfo.processInfo.environment["MOMIJ_MODEL"]
            ?? NSString("~/models/deepgrove/maple-preview-2bit-mlx").expandingTildeInPath
    }

    func testMtlBufHostCopyDoesNotDangleAfterArrayRelease() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else {
            throw XCTSkip("no Metal device")
        }
        let n = 2048
        let buf: MTLBuffer
        do {
            let a = MLXArray(Array(repeating: Float16(1.25), count: n))
            MLX.eval(a)
            guard let b = SeedlessMetal.mtlBufHostCopy(a, device) else {
                XCTFail("mtlBufHostCopy returned nil")
                return
            }
            buf = b
        }
        let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
        XCTAssertEqual(Float(p[0]), 1.25, accuracy: 1e-3)
        XCTAssertEqual(Float(p[n - 1]), 1.25, accuracy: 1e-3)
    }

    func testA_FlashHeadInitCompletes() throws {
        let dir = Self.mapleDir()
        guard FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("Maple weights not at \(dir)")
        }
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else {
            throw XCTSkip("no Metal device")
        }
        let store = try WeightStore(modelDir: dir)
        store.residentAll()
        let fh = SeedlessFlashHead(store: store, device: device)
        XCTAssertNotNil(fh, "FlashHead weights present but init returned nil")
        XCTAssertGreaterThan(fh!.nProbes, 0)
        XCTAssertGreaterThan(fh!.clusterSize, 0)
        // Touch GPU buffers so a dangling noCopy alias crashes here, not later.
        XCTAssertGreaterThan(fh!.tokenMapBuf.length, 0)
        _ = fh!.forceCount
    }

    func testDecodeEngineInitCompletes() throws {
        let dir = Self.mapleDir()
        guard FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("Maple weights not at \(dir)")
        }
        try SeedlessMetal.ensureCompiled()
        let store = try WeightStore(modelDir: dir)
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: 256)
        XCTAssertTrue(eng.useFlashHead)
        _ = try eng.generate(prompt: [100, 101, 102], maxTokens: 1, eos: nil)
    }
}
