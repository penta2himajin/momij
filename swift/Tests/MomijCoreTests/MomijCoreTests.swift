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
    func testGateSimdMatchesTGReduce() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let E = 256, H = 2048
        func fillHalf(_ buf: MTLBuffer) {
            let n = buf.length / 2
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = Float16((Float(i % 97) - 48.0) * 1e-3) }
        }
        let w = device.makeBuffer(length: E * H * 2, options: .storageModeShared)!
        let x = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        let y0 = device.makeBuffer(length: E * 4, options: .storageModeShared)!
        let y1 = device.makeBuffer(length: E * 4, options: .storageModeShared)!
        fillHalf(w); fillHalf(x)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeGate(into: enc, w: w, x: x, y: y0, E: E, H: H, simd: false)
        SeedlessMetal.encodeGate(into: enc, w: w, x: x, y: y1, E: E, H: H, simd: true)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let a = y0.contents().bindMemory(to: Float.self, capacity: E)
        let b = y1.contents().bindMemory(to: Float.self, capacity: E)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< E {
            let d = abs(a[i] - b[i])
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(a[i] * a[i])
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-4, "gate simd vs TG; rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

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

    func testGqmm2SplitKMatchesDefault() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 128, E = 4, Ktop = 2, gs = 128
        func fill(_ buf: MTLBuffer) {
            let n = buf.length
            let p = buf.contents().bindMemory(to: UInt8.self, capacity: n)
            for i in 0 ..< n { p[i] = UInt8((i * 17 + 3) & 0xff) }
        }
        func fillHalf(_ buf: MTLBuffer, _ v: Float16) {
            let n = buf.length / 2
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = v }
        }
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let x = buf(H * 2); fillHalf(x, 0.01)
        let w = buf(E * N * packedK * 4); fill(w)
        let s = buf(E * N * nGroups * 2); fillHalf(s, 0.02)
        let b = buf(E * N * nGroups * 2); fillHalf(b, 0.001)
        let inds = buf(Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        ip[0] = 1; ip[1] = 2
        let y0 = buf(Ktop * N * 2)
        let y1 = buf(Ktop * N * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y0,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: true, w16: false)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let a = y0.contents().bindMemory(to: Float16.self, capacity: Ktop * N)
        let bOut = y1.contents().bindMemory(to: Float16.self, capacity: Ktop * N)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (Ktop * N) {
            let d = abs(Float(a[i]) - Float(bOut[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "split-K gqmm2 must match default; rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

    func testGqmm2W16MatchesDefault() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 128, E = 4, Ktop = 2, gs = 128
        func fill(_ buf: MTLBuffer) {
            let n = buf.length
            let p = buf.contents().bindMemory(to: UInt8.self, capacity: n)
            for i in 0 ..< n { p[i] = UInt8((i * 17 + 3) & 0xff) }
        }
        func fillHalf(_ buf: MTLBuffer, _ v: Float16) {
            let n = buf.length / 2
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = v }
        }
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let x = buf(H * 2); fillHalf(x, 0.01)
        let w = buf(E * N * packedK * 4); fill(w)
        let s = buf(E * N * nGroups * 2); fillHalf(s, 0.02)
        let b = buf(E * N * nGroups * 2); fillHalf(b, 0.001)
        let inds = buf(Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        ip[0] = 1; ip[1] = 2
        let y0 = buf(Ktop * N * 2)
        let y1 = buf(Ktop * N * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y0,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: true)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let a = y0.contents().bindMemory(to: Float16.self, capacity: Ktop * N)
        let bOut = y1.contents().bindMemory(to: Float16.self, capacity: Ktop * N)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (Ktop * N) {
            let d = abs(Float(a[i]) - Float(bOut[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "w16 gqmm2 must match default; rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

    func testUpSwigluFusedMatchesSeparate() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, I = 128, E = 4, Ktop = 2, gs = 128
        func fill(_ buf: MTLBuffer) {
            let n = buf.length
            let p = buf.contents().bindMemory(to: UInt8.self, capacity: n)
            for i in 0 ..< n { p[i] = UInt8((i * 17 + 3) & 0xff) }
        }
        func fillHalf(_ buf: MTLBuffer, _ v: Float16) {
            let n = buf.length / 2
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = v }
        }
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedH = H * 2 / 32
        let nGroupsH = H / gs
        let x = buf(H * 2); fillHalf(x, 0.01)
        let ugW = buf(E * 2 * I * packedH * 4); fill(ugW)
        let ugS = buf(E * 2 * I * nGroupsH * 2); fillHalf(ugS, 0.02)
        let ugB = buf(E * 2 * I * nGroupsH * 2); fillHalf(ugB, 0.001)
        let inds = buf(Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        ip[0] = 1; ip[1] = 3
        let ugOut = buf(Ktop * 2 * I * 2)
        let actSep = buf(Ktop * I * 2)
        let actFused = buf(Ktop * I * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: ugW, scales: ugS, biases: ugB, inds: inds, out: ugOut,
            Ktop: Ktop, K: H, N: 2 * I, gs: gs, into: enc)
        SeedlessMetal.encodeClampedSwiglu(into: enc, ug: ugOut, act: actSep, I: I, Ktop: Ktop)
        try SeedlessMetal.gqmm2UpSwiglu(
            x: x, w: ugW, scales: ugS, biases: ugB, inds: inds, out: actFused,
            Ktop: Ktop, K: H, I: I, gs: gs, into: enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let a = actSep.contents().bindMemory(to: Float16.self, capacity: Ktop * I)
        let b = actFused.contents().bindMemory(to: Float16.self, capacity: Ktop * I)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (Ktop * I) {
            let d = abs(Float(a[i]) - Float(b[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

    func testFlashTopKMatchesCPUSelect() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let E = 4748, K = 64
        let nChunks = (E + SeedlessMetal.flashTopKChunk - 1) / SeedlessMetal.flashTopKChunk
        let localTop = min(K, SeedlessMetal.flashTopKChunk)
        let nCand = nChunks * localTop
        let scores = device.makeBuffer(length: E * 4, options: .storageModeShared)!
        let inds = device.makeBuffer(length: K * 4, options: .storageModeShared)!
        let candS = device.makeBuffer(length: nCand * 4, options: .storageModeShared)!
        let candI = device.makeBuffer(length: nCand * 4, options: .storageModeShared)!
        let sp = scores.contents().bindMemory(to: Float.self, capacity: E)
        // Adversarial: put the global top-K entirely in chunk 0.
        for i in 0 ..< E { sp[i] = Float(i) * 1e-6 }
        for i in 0 ..< K { sp[i] = 1000.0 + Float(K - i) }

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeFlashTopK(
            into: enc, scores: scores, inds: inds,
            candScores: candS, candInds: candI, E: E, K: K)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var order = Array(0 ..< E)
        order.sort { sp[$0] > sp[$1] }
        let want = Set(order.prefix(K))
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: K)
        var got = Set<Int>()
        for i in 0 ..< K { got.insert(Int(ip[i])) }
        XCTAssertEqual(got, want, "GPU hierarchical top-k must match CPU top-\(K)")
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

final class SeedlessAttnEncodeTests: XCTestCase {
    func testEncodeAttnBlockCompiles() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 2048, heads = 16, kv = 4, d = 128, gs = 128
        let qDim = heads * d, kvDim = kv * d, qkvN = qDim + 2 * kvDim
        func buf(_ n: Int, _ bpe: Int = 2) -> MTLBuffer {
            device.makeBuffer(length: n * bpe, options: .storageModeShared)!
        }
        let dens = buf(1, 4)
        dens.contents().storeBytes(of: Int32(0), as: Int32.self)
        let x = buf(H), qkvW = buf(qkvN * (H / 16), 4), qkvS = buf(qkvN * (H / gs)), qkvB = buf(qkvN * (H / gs))
        let oW = buf(H * (qDim / 16), 4), oS = buf(H * (qDim / gs)), oB = buf(H * (qDim / gs))
        let qkW = buf((heads + kv) * d), inv = buf(32, 4)
        let qkvOut = buf(qkvN), qkOut = buf(qDim + kvDim), attnTmp = buf(qDim), attnOut = buf(H)
        let maxLen = 8
        let kCache = buf(kv * maxLen * d), vCache = buf(kv * maxLen * d)
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.encodeAttnBlock(
            into: enc, xNorm: x,
            qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
            oW: oW, oS: oS, oB: oB,
            qkW: qkW, invFreq: inv, densInds: dens,
            qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
            kCache: kCache, vCache: vCache,
            H: H, numHeads: heads, numKV: kv, headDim: d,
            ropeDim: 64, ropePos: 0, writePos: 0, maxLen: maxLen, seqLen: 1, eps: 1e-6, gs: gs)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertTrue(attnOut.contents().bindMemory(to: Float16.self, capacity: 1)[0].isFinite)
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
