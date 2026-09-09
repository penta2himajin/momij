import XCTest
@testable import MomijCore

final class SuffixSpecTests: XCTestCase {
    func testFindsRepeatedSuffixDraft() {
        // history ends with [1,2,3] and earlier contains [1,2,3,9,8]
        let history = [7, 1, 2, 3, 9, 8, 1, 2, 3]
        let draft = SuffixSpec.suffixDraft(history: history, k: 2)
        XCTAssertEqual(draft, [9, 8])
    }

    func testSpecAcceptsMatchingDraft() throws {
        // Deterministic "model": always emits next = last+1
        var seq = 10
        let out = try SuffixSpec.run(
            prompt: [1, 2, 3, 1, 2, 3],
            maxTokens: 4,
            draftK: 2,
            eos: nil,
            step: { ids in
                _ = ids
                let t = seq
                seq += 1
                return t
            },
            multiStep: { ids, k in
                _ = ids
                var r: [Int] = []
                for _ in 0 ..< k {
                    r.append(seq)
                    seq += 1
                }
                return r
            }
        )
        XCTAssertEqual(out.count, 4)
    }
}

final class SeedlessMetalTests: XCTestCase {
    func testGqmm2CompilesAndRuns() throws {
        try SeedlessMetal.ensureCompiled()
        let rate = try SeedlessMetal.benchQmv2(iters: 20)
        XCTAssertGreaterThan(rate, 10, "gqmm2 should exceed 10 kernel/s, got \(rate)")
    }

    func testFusedExpertFasterThanNaiveFloor() throws {
        try SeedlessMetal.ensureCompiled()
        let rate = try SeedlessMetal.benchFusedExpert(E: 256, iters: 30)
        // Naive nested kernel was ~0.7/s; tiled gather path should be >> 10/s.
        XCTAssertGreaterThan(rate, 10, "fused expert should exceed 10 steps/s, got \(rate)")
    }

    func testMoEBlockOneCBCompilesAndRuns() throws {
        try SeedlessMetal.ensureCompiled()
        XCTAssertTrue(SeedlessMetal.ready)
        // Synthetic buffers: verify encode path doesn't throw on Maple shapes.
        guard let device = SeedlessMetal.device else {
            throw XCTSkip("no Metal device")
        }
        let H = 2048, I = 512, E = 256, Ktop = 8, gs = 128
        func buf(_ n: Int, _ bpe: Int = 2) -> MTLBuffer {
            device.makeBuffer(length: n * bpe, options: .storageModeShared)!
        }
        // Minimal packed weight stubs (zeros) — shape-valid for dispatch.
        let packedUG = (2 * I) * (H / 16)  // uint32 packs per expert row layout approx unused for empty
        _ = packedUG
        let x = buf(H), normW = buf(H), gateW = buf(E * H)
        let ugW = buf(E * 2 * I * (H / 16), 4), ugS = buf(E * 2 * I * (H / gs)), ugB = buf(E * 2 * I * (H / gs))
        let dW = buf(E * H * (I / 16), 4), dS = buf(E * H * (I / gs)), dB = buf(E * H * (I / gs))
        let xNorm = buf(H), logits = buf(E, 4), inds = buf(Ktop, 4), scores = buf(Ktop, 4)
        let ugOut = buf(Ktop * 2 * I), act = buf(Ktop * I), downOut = buf(Ktop * H), moeOut = buf(H)
        try SeedlessMetal.moeBlockOneCB(
            h: x, normW: normW, gateW: gateW,
            upGateW: ugW, upGateS: ugS, upGateB: ugB,
            downW: dW, downS: dS, downB: dB,
            xNorm: xNorm, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: 1e-6, gs: gs)
        let yp = x.contents().bindMemory(to: Float16.self, capacity: H)
        XCTAssertTrue(yp[0].isFinite)
    }
}

final class SeedlessMoEStackTests: XCTestCase {
    func testEncodeAllCompilesOnSyntheticTwoLayers() throws {
        // Smoke: encodeMoEBlock twice into one CB with distinct scratch buffers.
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 2048, I = 512, E = 256, Ktop = 8, gs = 128
        func buf(_ n: Int, _ bpe: Int = 2) -> MTLBuffer {
            device.makeBuffer(length: n * bpe, options: .storageModeShared)!
        }
        let h = buf(H)
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        for _ in 0 ..< 2 {
            let normW = buf(H), gateW = buf(E * H)
            let ugW = buf(E * 2 * I * (H / 16), 4), ugS = buf(E * 2 * I * (H / gs)), ugB = buf(E * 2 * I * (H / gs))
            let dW = buf(E * H * (I / 16), 4), dS = buf(E * H * (I / gs)), dB = buf(E * H * (I / gs))
            let xNorm = buf(H), logits = buf(E, 4), inds = buf(Ktop, 4), scores = buf(Ktop, 4)
            let ugOut = buf(Ktop * 2 * I), act = buf(Ktop * I), downOut = buf(Ktop * H), moeOut = buf(H)
            try SeedlessMetal.encodeMoEBlock(
                into: enc, h: h, normW: normW, gateW: gateW,
                upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB,
                xNorm: xNorm, logits: logits, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
                H: H, I: I, E: E, Ktop: Ktop, eps: 1e-6, gs: gs)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertTrue(h.contents().bindMemory(to: Float16.self, capacity: 1)[0].isFinite)
    }
}

final class ConfigTests: XCTestCase {
    func testConfigCodingKeys() throws {
        let json = """
        {"model_type":"maple","hidden_size":2048,"num_hidden_layers":24,
         "num_attention_heads":16,"num_key_value_heads":4,"head_dim":128,
         "vocab_size":10,"num_experts":256,"num_experts_per_tok":8,
         "moe_intermediate_size":512,"rms_norm_eps":1e-6,"rope_theta":10000,
         "sliding_window":512,"partial_rotary_factor":0.5,"tie_word_embeddings":false,
         "layer_types":["sliding_attention","full_attention"],
         "quantization":{"bits":2,"group_size":128,"mode":"affine"}}
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MapleConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 2048)
        XCTAssertEqual(cfg.expertBits, 2)
        XCTAssertTrue(cfg.isSliding(0))
        XCTAssertFalse(cfg.isSliding(1))
    }
}
