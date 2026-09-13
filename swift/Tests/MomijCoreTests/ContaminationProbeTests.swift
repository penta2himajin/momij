import XCTest
import Metal
import Tokenizers
@testable import MomijCore

/// Diagnostic probe: is engine output deterministic given process history?
/// Distinguishes dirty-initial-state (read-before-write) from in-run clobber.
final class ContaminationProbeTests: XCTestCase {
    private var modelDir: String {
        ProcessInfo.processInfo.environment["MOMIJ_MODEL"]
            ?? NSString("~/models/deepgrove/maple-preview-2bit-mlx").expandingTildeInPath
    }

    private let agentText = "You are a coding assistant. Use tools when needed. "
        + "List the files in the current directory using the ls tool. "
        + "For each function call, return a json object with function name and arguments "
        + "within tool call markup tags. "
        + "If you can answer without a tool, reply with plain text only. "
        + "You may call one or more functions to assist with the user query. "
        + "List the files in the current directory using the ls tool. "
        + "For each function call, return a json object with function name and arguments "
        + "within tool call markup tags."

    private func contaminate() throws {
        guard let device = SeedlessMetal.device else { throw XCTSkip("no Metal device") }
        do {
            let s = try WeightStore(modelDir: modelDir)
            s.residentAll()
            let fh = SeedlessFlashHead(store: s, device: device)
            XCTAssertNotNil(fh)
        }
    }

    private func gen(_ store: WeightStore, _ prompt: [Int], maxTokens: Int = 64) throws -> [Int] {
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: 2048, enableFlashHead: false)
        return try eng.generate(prompt: prompt, maxTokens: maxTokens, eos: nil)
    }

    private func rep(_ t: [Int]) -> String {
        String(describing: t.prefix(35)) + " len=" + String(t.count)
    }

    func testDeterminismUnderContamination() async throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: modelDir))
        let prompt = tok.encode(text: agentText)

        let store1 = try WeightStore(modelDir: modelDir)
        store1.residentAll()
        let base = try gen(store1, prompt)
        print("[probe] baseline      =", rep(base))

        let eng1 = try SeedlessDecodeEngine(store: store1, fullMaxLen: 2048, enableFlashHead: false)
        let again = try eng1.generate(prompt: prompt, maxTokens: 64, eos: nil)
        print("[probe] same-engine#2 =", rep(again))
        XCTAssertEqual(again, base, "same engine, second run diverged")

        let base2 = try gen(store1, prompt)
        print("[probe] fresh-engine2 =", rep(base2))
        XCTAssertEqual(base2, base, "fresh engine on same store diverged")

        try contaminate()

        let store2 = try WeightStore(modelDir: modelDir)
        store2.residentAll()
        let c1 = try gen(store2, prompt)
        print("[probe] contaminated1 =", rep(c1))

        let eng2 = try SeedlessDecodeEngine(store: store2, fullMaxLen: 2048, enableFlashHead: false)
        let c2 = try eng2.generate(prompt: prompt, maxTokens: 64, eos: nil)
        print("[probe] contam same#2 =", rep(c2))

        let c3 = try gen(store2, prompt)
        print("[probe] contaminated2 =", rep(c3))

        // Snapshot prompt-position KV (K and V) for every layer.
        func kvSnapshot(_ eng: SeedlessDecodeEngine, positions: Int) -> [[[UInt16]]] {
            eng.stack.layers.map { l in
                let kvDim = l.numKV * l.headDim
                var perLayer: [[UInt16]] = []
                for cache in [l.kCache, l.vCache] {
                    let p = cache.contents().bindMemory(to: UInt16.self, capacity: positions * kvDim)
                    perLayer.append(Array(UnsafeBufferPointer(start: p, count: positions * kvDim)))
                }
                return perLayer
            }
        }
        func diffKV(_ a: [[[UInt16]]], _ b: [[[UInt16]]], label: String) {
            for li in 0 ..< min(a.count, b.count) {
                let pa = a[li]
                let pb = b[li]
                var total = 0
                var first = -1
                outer: for ci in 0 ..< min(pa.count, pb.count) {
                    for i in 0 ..< min(pa[ci].count, pb[ci].count) {
                        if pa[ci][i] != pb[ci][i] {
                            total += 1
                            if first < 0 { first = ci * 1_000_000 + i }
                        }
                        if total > 200 { break outer }
                    }
                }
                if first >= 0 {
                    print("[probe] \(label) L\(li): diffs>=\(total) firstElem=\(first)")
                }
            }
        }
        let kvDim0 = eng2.stack.layers[0].numKV * eng2.stack.layers[0].headDim
        let refEng = try SeedlessDecodeEngine(store: store2, fullMaxLen: 2048, enableFlashHead: false)
        let refOut = try refEng.generate(prompt: prompt, maxTokens: 64, eos: nil)
        print("[probe] refRun =", rep(refOut), "isHealthy:", refOut == base)
        let snapRef = kvSnapshot(refEng, positions: 128)
        let snapB = kvSnapshot(eng2, positions: 128)
        diffKV(snapRef, snapB, label: "B-vs-REF")
        let engC = try SeedlessDecodeEngine(store: store2, fullMaxLen: 2048, enableFlashHead: false)
        let cRun = try engC.generate(prompt: prompt, maxTokens: 64, eos: nil)
        print("[probe] engineC run =", rep(cRun))
        let snapC = kvSnapshot(engC, positions: 128)
        diffKV(snapRef, snapC, label: "C-vs-REF")
        // Layer-0 K at pos 0, all 4 kv heads (32 elems each).
        for (label, snap) in [("REF", snapRef), ("B", snapB), ("C", snapC)] {
            print("[probe] L0.K[pos0] \(label) =", Array(snap[0][0].prefix(2 * kvDim0)))
        }

        print("[probe] base==c1:", base == c1, " c1==c2:", c1 == c2, " c1==c3:", c1 == c3)
        XCTAssertEqual(c1, base, "contaminated fresh engine diverged from baseline")
        XCTAssertEqual(c2, c1, "same engine second run diverged under contamination")
        XCTAssertEqual(c3, c1, "second fresh engine diverged under contamination")
    }
}