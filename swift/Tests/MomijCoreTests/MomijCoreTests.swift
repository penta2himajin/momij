import XCTest
import MLX
@testable import MomijCore

final class SuffixSpecTests: XCTestCase {
    func testFindsRepeatedSuffixDraft() {
        // history ends with [1,2,3] and earlier contains [1,2,3,9,8]
        let history = [7, 1, 2, 3, 9, 8, 1, 2, 3]
        let draft = SuffixSpec.suffixDraft(history: history, k: 2)
        XCTAssertEqual(draft, [9, 8])
    }

    /// Match need not abut the current suffix (classic PLD long-distance hit).
    func testFindsDistantPromptLookupDraft() {
        // prompt-like span [10,11,12,13,14] appears early; generation ends with [10,11]
        let history = [10, 11, 12, 13, 14, 99, 98, 10, 11]
        let draft = SuffixSpec.suffixDraft(history: history, k: 3)
        XCTAssertEqual(draft, [12, 13, 14])
    }

    /// Prefer continuation from the prompt span when `promptLen` is set.
    func testPrefersPromptRegionMatch() {
        // prompt = [1,2,3,4,5]; later gen also has [1,2] followed by noise continuation
        let history = [1, 2, 3, 4, 5, 1, 2, 90, 91, 1, 2]
        let draft = SuffixSpec.suffixDraft(history: history, k: 3, promptLen: 5)
        XCTAssertEqual(draft, [3, 4, 5])
    }

    /// Bigram fallback must be a contiguous continuation, not a bag of successors.
    func testBigramFallbackIsContiguous() {
        // last=5; earlier "... 5, 7, 8, 9 ..." then unrelated 5-successors must not scramble
        let history = [5, 7, 8, 9, 1, 2, 5]
        let draft = SuffixSpec.suffixDraft(history: history, k: 3)
        XCTAssertEqual(draft, [7, 8, 9])
    }

    /// Longer pattern match → longer adaptive draft (capped by k).
    func testAdaptiveDraftLengthTracksMatch() {
        let history = [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]
        let d2 = SuffixSpec.suffixDraft(history: history, k: 8)
        // pattern [1,2,3,4] → contiguous continuation after earliest/most-recent match
        XCTAssertEqual(d2, [5, 6, 0, 1, 2, 3, 4])
        XCTAssertEqual(SuffixSpec.adaptiveDraftK(meanAccept: 0.0, draftK: 8), 2)
        XCTAssertEqual(SuffixSpec.adaptiveDraftK(meanAccept: 0.9, draftK: 8), 4)
        XCTAssertEqual(SuffixSpec.adaptiveDraftK(meanAccept: 1.6, draftK: 8), 6)
        XCTAssertEqual(SuffixSpec.adaptiveDraftK(meanAccept: 3.0, draftK: 8), 8)
    }

    func testSuffixTreeDraftFollowsFrequency() {
        let idx = SuffixDraftIndex(maxDepth: 16)
        // Paths through [1,2,3]: →[4,4] twice and →[9] once → prefer 4
        idx.insert([0, 1, 2, 3, 4, 4])
        idx.insert([7, 1, 2, 3, 4, 5])
        idx.insert([8, 1, 2, 3, 9])
        let d = idx.draft(from: [1, 2, 3], maxK: 4, alpha: 1.0)
        XCTAssertEqual(d.matchLen, 3)
        XCTAssertEqual(d.tokens.first, 4)
        XCTAssertEqual(SuffixDraftIndex.maxSpec(matchLen: d.matchLen, maxK: 4, alpha: 1.0), 3)
    }

    func testMaxSpecScalesWithMatchLen() {
        XCTAssertEqual(SuffixDraftIndex.maxSpec(matchLen: 4, maxK: 16, alpha: 1.0), 4)
        XCTAssertEqual(SuffixDraftIndex.maxSpec(matchLen: 4, maxK: 16, alpha: 2.0), 8)
        XCTAssertEqual(SuffixDraftIndex.maxSpec(matchLen: 4, maxK: 3, alpha: 2.0), 3)
        XCTAssertEqual(SuffixDraftIndex.maxSpec(matchLen: 0, maxK: 8, alpha: 1.0), 0)
    }

    func testLiveKVSeqIsPrefixNotFullAllocation() {
        XCTAssertEqual(SeedlessLayerBlock.liveSeq(offset: 128, maxLen: 2048, isSliding: false), 128)
        XCTAssertEqual(SeedlessLayerBlock.liveSeq(offset: 0, maxLen: 2048, isSliding: false), 0)
        XCTAssertEqual(SeedlessLayerBlock.liveSeq(offset: 100, maxLen: 512, isSliding: true), 100)
        XCTAssertEqual(SeedlessLayerBlock.liveSeq(offset: 600, maxLen: 512, isSliding: true), 512)
        XCTAssertEqual(SeedlessLayerBlock.liveBytes(seq: 128, numKV: 8, headDim: 64), 128 * 8 * 64 * 2)
        XCTAssertLessThan(
            SeedlessLayerBlock.liveBytes(seq: 128, numKV: 8, headDim: 64),
            8 * 2048 * 64 * 2)
    }

    func testCanRewindWhenWritesAreAppendOnly() {
        XCTAssertTrue(SeedlessLayerBlock.canRewind(isSliding: false, offset: 200, maxLen: 2048, steps: 8))
        XCTAssertTrue(SeedlessLayerBlock.canRewind(isSliding: true, offset: 100, maxLen: 512, steps: 8))
        XCTAssertFalse(SeedlessLayerBlock.canRewind(isSliding: true, offset: 508, maxLen: 512, steps: 8))
        XCTAssertTrue(SeedlessLayerBlock.canRewind(isSliding: true, offset: 504, maxLen: 512, steps: 8))
        XCTAssertFalse(SeedlessLayerBlock.canRewind(isSliding: true, offset: 512, maxLen: 512, steps: 1))
    }

    /// After the SWA window fills, write into a ring (offset % maxLen). Never pin
    /// every decode step at maxLen-1 — that plus in-place shift races the KV.
    func testSWAWritePosIsRingNotClampedTail() {
        XCTAssertEqual(SeedlessLayerBlock.kvWritePos(offset: 0, maxLen: 512, isSliding: true), 0)
        XCTAssertEqual(SeedlessLayerBlock.kvWritePos(offset: 511, maxLen: 512, isSliding: true), 511)
        XCTAssertEqual(SeedlessLayerBlock.kvWritePos(offset: 512, maxLen: 512, isSliding: true), 0)
        XCTAssertEqual(SeedlessLayerBlock.kvWritePos(offset: 513, maxLen: 512, isSliding: true), 1)
        XCTAssertEqual(SeedlessLayerBlock.kvWritePos(offset: 1024, maxLen: 512, isSliding: true), 0)
        XCTAssertEqual(SeedlessLayerBlock.kvSeqLen(offset: 100, maxLen: 512, isSliding: true), 101)
        XCTAssertEqual(SeedlessLayerBlock.kvSeqLen(offset: 512, maxLen: 512, isSliding: true), 512)
        XCTAssertEqual(SeedlessLayerBlock.kvWritePos(offset: 100, maxLen: 2048, isSliding: false), 100)
        XCTAssertEqual(SeedlessLayerBlock.kvSeqLen(offset: 100, maxLen: 2048, isSliding: false), 101)
    }

    /// MLX SWA cache must keep absolute RoPE offset (oracle RotatingKVCache), not clamp to maxSize.
    func testRotatingKVCacheOffsetStaysAbsolute() {
        let cache = KVCache(maxSize: 8)
        func kv(_ L: Int, mark: Float) -> (MLXArray, MLXArray) {
            // [B=1, H=1, L, D=2]
            var data = [Float](repeating: 0, count: L * 2)
            for i in 0 ..< L { data[i * 2] = mark + Float(i) }
            let k = MLXArray(data).reshaped([1, 1, L, 2])
            return (k, k)
        }
        let (k0, v0) = kv(10, mark: 100)
        let (rk0, _) = cache.update(k0, v0)
        MLX.eval(rk0)
        XCTAssertEqual(cache.offset, 10, "prefill past window must advance absolute offset")
        XCTAssertLessThanOrEqual(rk0.dim(2), 10)
        // Decode steps past the window: offset must keep climbing for RoPE.
        for step in 0 ..< 5 {
            let (k1, v1) = kv(1, mark: Float(200 + step))
            _ = cache.update(k1, v1)
            XCTAssertEqual(cache.offset, 11 + step)
        }
        XCTAssertEqual(cache.offset, 15)
        let kLast = cache.keys!
        let vLast = cache.values!
        MLX.eval(kLast, vLast)
        XCTAssertEqual(kLast.dim(2), 8, "stored KV stays within maxSize")
        _ = vLast
    }

    func testSlidingWindowCausalMask() {
        // N=4, window=2, offset=0 → row i attends [max(0,i-1)...i]
        let m = AttentionMasks.slidingCausal(queryLen: 4, offset: 0, windowSize: 2)
        MLX.eval(m)
        // additive: 0 allowed, -inf blocked. Shape [1,1,4,4] or [4,4]
        let flat = m.asType(.float32).reshaped([-1])
        // positions (q,k): (0,0) ok; (1,0)(1,1) ok; (2,0) blocked; (2,1)(2,2) ok
        func at(_ q: Int, _ k: Int) -> Float {
            flat[q * 4 + k].item(Float.self)
        }
        XCTAssertEqual(at(0, 0), 0)
        XCTAssertTrue(at(2, 0).isInfinite && at(2, 0) < 0)
        XCTAssertEqual(at(2, 1), 0)
        XCTAssertEqual(at(2, 2), 0)
        XCTAssertTrue(at(1, 0) == 0 || at(1, 0).isFinite)
    }

    func testGlobalIndexBeatsEmptyLocal() {
        let global = SuffixDraftIndex(maxDepth: 32)
        global.insert([1, 2, 3, 4, 5, 6])
        let history = [1, 2, 3]
        let d = SuffixDraftIndex.bestDraft(
            local: nil, global: global, history: history, maxK: 4, alpha: 1.0)
        XCTAssertEqual(d.matchLen, 3)
        XCTAssertEqual(d.tokens, [4, 5, 6])
    }

    func testTreeDraftFallsBackToPLD() {
        // Empty trees → PLD contiguous match
        let history = [7, 1, 2, 3, 9, 8, 1, 2, 3]
        let d = SuffixSpec.treeDraft(
            history: history, k: 2, local: nil, global: nil, promptLen: nil)
        XCTAssertEqual(d, [9, 8])
    }

    func testTokenRecycleDraftChain() {
        let r = TokenRecycleIndex(topK: 4)
        r.observe(fromToken: 10, candidates: [20, 21, 22])
        r.observe(fromToken: 20, candidates: [30, 31])
        r.observe(fromToken: 30, candidates: [40])
        XCTAssertEqual(r.draft(from: 10, maxK: 3), [20, 30, 40])
        // Recent observe wins ordering
        r.observe(fromToken: 10, candidates: [99, 20])
        XCTAssertEqual(r.draft(from: 10, maxK: 1).first, 99)
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

    /// Batched M=2 must match two sequential M=1 gathers (same weights, per-token x / inds).
    func testGqmm2MrowMatchesSequential() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, SeedlessMetal.queue != nil else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 128, E = 8, Ktop = 2, M = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let x = buf(M * H * 2)
        let xp = x.contents().bindMemory(to: Float16.self, capacity: M * H)
        for i in 0 ..< (M * H) { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let w = buf(E * N * packedK * 4)
        let wp = w.contents().bindMemory(to: UInt8.self, capacity: E * N * packedK * 4)
        for i in 0 ..< (E * N * packedK * 4) { wp[i] = UInt8((i * 17 + 3) & 0xff) }
        let s = buf(E * N * nGroups * 2)
        let b = buf(E * N * nGroups * 2)
        let sp = s.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        let bp = b.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        for i in 0 ..< (E * N * nGroups) {
            sp[i] = 0.02
            bp[i] = -0.02
        }
        let indsM = buf(M * Ktop * 4)
        let ipM = indsM.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
        ipM[0] = 1; ipM[1] = 3; ipM[2] = 2; ipM[3] = 5
        let yM = buf(M * Ktop * N * 2)
        let ySeq = buf(M * Ktop * N * 2)

        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: indsM, out: yM,
            Ktop: Ktop, K: H, N: N, gs: gs, M: M, splitK: false)
        for m in 0 ..< M {
            let x1 = buf(H * 2)
            let xSrc = x.contents().bindMemory(to: Float16.self, capacity: M * H)
            let xDst = x1.contents().bindMemory(to: Float16.self, capacity: H)
            for i in 0 ..< H { xDst[i] = xSrc[m * H + i] }
            let inds1 = buf(Ktop * 4)
            let iSrc = indsM.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
            let iDst = inds1.contents().bindMemory(to: Int32.self, capacity: Ktop)
            for k in 0 ..< Ktop { iDst[k] = iSrc[m * Ktop + k] }
            let y1 = buf(Ktop * N * 2)
            try SeedlessMetal.gqmm2(
                x: x1, w: w, scales: s, biases: b, inds: inds1, out: y1,
                Ktop: Ktop, K: H, N: N, gs: gs, M: 1, splitK: false)
            let ySrc = y1.contents().bindMemory(to: Float16.self, capacity: Ktop * N)
            let yDst = ySeq.contents().bindMemory(to: Float16.self, capacity: M * Ktop * N)
            for i in 0 ..< (Ktop * N) { yDst[m * Ktop * N + i] = ySrc[i] }
        }
        let a = yM.contents().bindMemory(to: Float16.self, capacity: M * Ktop * N)
        let bOut = ySeq.contents().bindMemory(to: Float16.self, capacity: M * Ktop * N)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (M * Ktop * N) {
            let d = abs(Float(a[i]) - Float(bOut[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "M-row gqmm2 must match sequential M=1; rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

    /// Batched M=2 fused expert must match two sequential M=1 steps (per-token x / inds / scores).
    func testFusedExpertMrowMatchesSequential() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, SeedlessMetal.queue != nil else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, I = 512, E = 8, Ktop = 2, M = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedH = H * 2 / 32
        let packedI = I * 2 / 32
        let nGH = H / gs
        let nGI = I / gs
        let x = buf(M * H * 2)
        let xp = x.contents().bindMemory(to: Float16.self, capacity: M * H)
        for i in 0 ..< (M * H) { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let ugW = buf(E * 2 * I * packedH * 4)
        let dW = buf(E * H * packedI * 4)
        let u8ug = ugW.contents().bindMemory(to: UInt8.self, capacity: E * 2 * I * packedH * 4)
        for i in 0 ..< (E * 2 * I * packedH * 4) { u8ug[i] = UInt8((i * 17 + 3) & 0xff) }
        let u8d = dW.contents().bindMemory(to: UInt8.self, capacity: E * H * packedI * 4)
        for i in 0 ..< (E * H * packedI * 4) { u8d[i] = UInt8((i * 13 + 7) & 0xff) }
        let ugS = buf(E * 2 * I * nGH * 2)
        let ugB = buf(E * 2 * I * nGH * 2)
        let dS = buf(E * H * nGI * 2)
        let dB = buf(E * H * nGI * 2)
        let ugSp = ugS.contents().bindMemory(to: Float16.self, capacity: E * 2 * I * nGH)
        let ugBp = ugB.contents().bindMemory(to: Float16.self, capacity: E * 2 * I * nGH)
        for i in 0 ..< (E * 2 * I * nGH) { ugSp[i] = 0.02; ugBp[i] = -0.02 }
        let dSp = dS.contents().bindMemory(to: Float16.self, capacity: E * H * nGI)
        let dBp = dB.contents().bindMemory(to: Float16.self, capacity: E * H * nGI)
        for i in 0 ..< (E * H * nGI) { dSp[i] = 0.02; dBp[i] = -0.02 }
        let inds = buf(M * Ktop * 4)
        let scores = buf(M * Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
        let sp = scores.contents().bindMemory(to: Float.self, capacity: M * Ktop)
        ip[0] = 1; ip[1] = 3; ip[2] = 2; ip[3] = 5
        for i in 0 ..< (M * Ktop) { sp[i] = 1.0 / Float(Ktop) }
        let ugOut = buf(M * Ktop * 2 * I * 2)
        let act = buf(M * Ktop * I * 2)
        let downOut = buf(M * Ktop * H * 2)
        let yM = buf(M * H * 2)
        let ySeq = buf(M * H * 2)

        try SeedlessMetal.fusedExpertStep(
            x: x, upGateW: ugW, upGateS: ugS, upGateB: ugB,
            downW: dW, downS: dS, downB: dB, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, y: yM,
            H: H, I: I, Ktop: Ktop, gs: gs, M: M)

        for m in 0 ..< M {
            let x1 = buf(H * 2)
            let xSrc = x.contents().bindMemory(to: Float16.self, capacity: M * H)
            let xDst = x1.contents().bindMemory(to: Float16.self, capacity: H)
            for i in 0 ..< H { xDst[i] = xSrc[m * H + i] }
            let inds1 = buf(Ktop * 4)
            let iSrc = inds.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
            let iDst = inds1.contents().bindMemory(to: Int32.self, capacity: Ktop)
            for k in 0 ..< Ktop { iDst[k] = iSrc[m * Ktop + k] }
            let sc1 = buf(Ktop * 4)
            let sSrc = scores.contents().bindMemory(to: Float.self, capacity: M * Ktop)
            let sDst = sc1.contents().bindMemory(to: Float.self, capacity: Ktop)
            for k in 0 ..< Ktop { sDst[k] = sSrc[m * Ktop + k] }
            let ug1 = buf(Ktop * 2 * I * 2)
            let act1 = buf(Ktop * I * 2)
            let dn1 = buf(Ktop * H * 2)
            let y1 = buf(H * 2)
            try SeedlessMetal.fusedExpertStep(
                x: x1, upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB, inds: inds1, scores: sc1,
                ugOut: ug1, act: act1, downOut: dn1, y: y1,
                H: H, I: I, Ktop: Ktop, gs: gs, M: 1)
            let ySrc = y1.contents().bindMemory(to: Float16.self, capacity: H)
            let yDst = ySeq.contents().bindMemory(to: Float16.self, capacity: M * H)
            for i in 0 ..< H { yDst[m * H + i] = ySrc[i] }
        }
        let a = yM.contents().bindMemory(to: Float16.self, capacity: M * H)
        let bOut = ySeq.contents().bindMemory(to: Float16.self, capacity: M * H)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (M * H) {
            let d = abs(Float(a[i]) - Float(bOut[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "M-row fused expert must match sequential M=1; rel_l2=\(rel) maxAbs=\(maxAbs)")
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
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false, ternary: false, fold: false, deferA: false, foldA: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: true, w16: false, ternary: false, fold: false, deferA: false, foldA: false)
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

    /// Maple ternary: codes ∈ {0,1,2}, bias == -scale, α constant across groups.
    /// Ternary path must match stock affine gqmm2 on such weights.
    func testGqmm2TernaryMatchesAffineOnTernaryWeights() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 128, E = 4, Ktop = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedK = H * 2 / 32  // uint32 count along K
        let nGroups = H / gs
        let x = buf(H * 2)
        let xp = x.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }

        let w = buf(E * N * packedK * 4)
        let wp = w.contents().bindMemory(to: UInt32.self, capacity: E * N * packedK)
        // Pack 16 codes {0,1,2} per uint32 (LSB first), never 3.
        var seed: UInt64 = 0xC0FFEE
        for i in 0 ..< (E * N * packedK) {
            var word: UInt32 = 0
            for t in 0 ..< 16 {
                seed = seed &* 6364136223846793005 &+ 1
                let code = UInt32(seed % 3)  // 0,1,2
                word |= code << (UInt32(t) * 2)
            }
            wp[i] = word
        }
        let s = buf(E * N * nGroups * 2)
        let b = buf(E * N * nGroups * 2)
        let sp = s.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        let bp = b.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        for e in 0 ..< E {
            for n in 0 ..< N {
                let alpha = Float16(0.02 + 0.001 * Float((e * N + n) % 13))
                for g in 0 ..< nGroups {
                    let idx = ((e * N) + n) * nGroups + g
                    sp[idx] = alpha
                    bp[idx] = -alpha
                }
            }
        }
        let inds = buf(Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        ip[0] = 1; ip[1] = 2
        let y0 = buf(Ktop * N * 2)
        let y1 = buf(Ktop * N * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y0,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false, ternary: false, fold: false, deferA: false, foldA: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false, ternary: true, fold: false, deferA: false, foldA: false)
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
        // Affine path uses masked·pre-divide; ternary uses shift·plain x.
        // Same real math (bias=-α); half rounding differs at ~1e-3–3e-3.
        XCTAssertLessThan(rel, 5e-3, "ternary gqmm2 must match affine on ternary weights; rel_l2=\(rel) maxAbs=\(maxAbs)")
        XCTAssertLessThan(maxAbs, 5e-3, "ternary maxAbs too large: \(maxAbs)")
    }

    /// Fold path keeps stock qd2/ld16_b2; only hoists row α and uses bias=-α.
    func testGqmm2FoldMatchesAffineOnTernaryWeights() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 128, E = 4, Ktop = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let x = buf(H * 2)
        let xp = x.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }

        let w = buf(E * N * packedK * 4)
        let wp = w.contents().bindMemory(to: UInt32.self, capacity: E * N * packedK)
        var seed: UInt64 = 0xC0FFEE
        for i in 0 ..< (E * N * packedK) {
            var word: UInt32 = 0
            for t in 0 ..< 16 {
                seed = seed &* 6364136223846793005 &+ 1
                word |= UInt32(seed % 3) << (UInt32(t) * 2)
            }
            wp[i] = word
        }
        let s = buf(E * N * nGroups * 2)
        let b = buf(E * N * nGroups * 2)
        let sp = s.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        let bp = b.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        for e in 0 ..< E {
            for n in 0 ..< N {
                let alpha = Float16(0.02 + 0.001 * Float((e * N + n) % 13))
                for g in 0 ..< nGroups {
                    let idx = ((e * N) + n) * nGroups + g
                    sp[idx] = alpha
                    bp[idx] = -alpha
                }
            }
        }
        let inds = buf(Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        ip[0] = 1; ip[1] = 2
        let y0 = buf(Ktop * N * 2)
        let y1 = buf(Ktop * N * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y0,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false,
            ternary: false, fold: false, deferA: false, foldA: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false,
            ternary: false, fold: true, deferA: false, foldA: false)
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
        XCTAssertLessThan(rel, 1e-3, "fold gqmm2 must match affine; rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

    /// Defer-α keeps stock ld16_b2 / masked accum; α applies once after simd_sum.
    func testGqmm2DeferAMatchesFoldOnTernaryWeights() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 128, E = 4, Ktop = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let x = buf(H * 2)
        let xp = x.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }

        let w = buf(E * N * packedK * 4)
        let wp = w.contents().bindMemory(to: UInt32.self, capacity: E * N * packedK)
        var seed: UInt64 = 0xC0FFEE
        for i in 0 ..< (E * N * packedK) {
            var word: UInt32 = 0
            for t in 0 ..< 16 {
                seed = seed &* 6364136223846793005 &+ 1
                word |= UInt32(seed % 3) << (UInt32(t) * 2)
            }
            wp[i] = word
        }
        let s = buf(E * N * nGroups * 2)
        let b = buf(E * N * nGroups * 2)
        let sp = s.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        let bp = b.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        for e in 0 ..< E {
            for n in 0 ..< N {
                let alpha = Float16(0.02 + 0.001 * Float((e * N + n) % 13))
                for g in 0 ..< nGroups {
                    let idx = ((e * N) + n) * nGroups + g
                    sp[idx] = alpha
                    bp[idx] = -alpha
                }
            }
        }
        let inds = buf(Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        ip[0] = 1; ip[1] = 2
        let y0 = buf(Ktop * N * 2)
        let y1 = buf(Ktop * N * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y0,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false,
            ternary: false, fold: true, deferA: false, foldA: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false,
            ternary: false, fold: false, deferA: true, foldA: false)
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
        XCTAssertLessThan(rel, 1e-3, "defer-α gqmm2 must match fold; rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

    /// Variant A: same as fold, epilogue is α*(accum-sum) instead of α*accum+(-α)*sum.
    func testGqmm2FoldAMatchesFoldOnTernaryWeights() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 128, E = 4, Ktop = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let x = buf(H * 2)
        let xp = x.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }

        let w = buf(E * N * packedK * 4)
        let wp = w.contents().bindMemory(to: UInt32.self, capacity: E * N * packedK)
        var seed: UInt64 = 0xC0FFEE
        for i in 0 ..< (E * N * packedK) {
            var word: UInt32 = 0
            for t in 0 ..< 16 {
                seed = seed &* 6364136223846793005 &+ 1
                word |= UInt32(seed % 3) << (UInt32(t) * 2)
            }
            wp[i] = word
        }
        let s = buf(E * N * nGroups * 2)
        let b = buf(E * N * nGroups * 2)
        let sp = s.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        let bp = b.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        for e in 0 ..< E {
            for n in 0 ..< N {
                let alpha = Float16(0.02 + 0.001 * Float((e * N + n) % 13))
                for g in 0 ..< nGroups {
                    let idx = ((e * N) + n) * nGroups + g
                    sp[idx] = alpha
                    bp[idx] = -alpha
                }
            }
        }
        let inds = buf(Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        ip[0] = 1; ip[1] = 2
        let y0 = buf(Ktop * N * 2)
        let y1 = buf(Ktop * N * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y0,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false,
            ternary: false, fold: true, deferA: false, foldA: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false,
            ternary: false, fold: false, deferA: false, foldA: true)
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
        XCTAssertLessThan(rel, 1e-3, "fold-A gqmm2 must match fold; rel_l2=\(rel) maxAbs=\(maxAbs)")
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
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false, ternary: false, fold: false, deferA: false, foldA: false)
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y1,
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: true, ternary: false, fold: false, deferA: false, foldA: false)
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

    /// Qwisp TG shape `(32,2)` must match momij `(64,1)` on the default fold kernel.
    func testGqmm2Tg2dMatchesDefault() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, N = 256, E = 4, Ktop = 2, gs = 128
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
            Ktop: Ktop, K: H, N: N, gs: gs, into: enc, splitK: false, w16: false, tg2d: true)
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
        XCTAssertLessThan(rel, 1e-5, "tg2d gqmm2 must match 1D TG; rel_l2=\(rel) maxAbs=\(maxAbs)")
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
            Ktop: Ktop, K: H, N: 2 * I, gs: gs, into: enc, ternary: false, fold: false, deferA: false, foldA: false)
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

    /// Fold gqmm2 + SwiGLU ≡ Ktop-in-TG fold-up-swiglu (shared-x, two-kernel rounding).
    func testUpKtopSwigluMatchesFoldThenSwiglu() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, I = 128, E = 4, Ktop = 2, M = 2, gs = 128
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
        let x = buf(M * H * 2); fillHalf(x, 0.01)
        let ugW = buf(E * 2 * I * packedH * 4); fill(ugW)
        let ugS = buf(E * 2 * I * nGroupsH * 2); fillHalf(ugS, 0.02)
        let ugB = buf(E * 2 * I * nGroupsH * 2); fillHalf(ugB, 0.001)
        let inds = buf(M * Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
        ip[0] = 1; ip[1] = 3; ip[2] = 2; ip[3] = 0
        let ugOut = buf(M * Ktop * 2 * I * 2)
        let actSep = buf(M * Ktop * I * 2)
        let actFused = buf(M * Ktop * I * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: ugW, scales: ugS, biases: ugB, inds: inds, out: ugOut,
            Ktop: Ktop, K: H, N: 2 * I, gs: gs, lhsPerExpert: false, M: M,
            into: enc, splitK: false, w16: false, ternary: false, fold: true,
            deferA: false, foldA: false)
        SeedlessMetal.encodeClampedSwiglu(into: enc, ug: ugOut, act: actSep, I: I, Ktop: M * Ktop)
        try SeedlessMetal.encodeGqmm2UpKtopSwiglu(
            into: enc, x: x, w: ugW, scales: ugS, biases: ugB, inds: inds, out: actFused,
            Ktop: Ktop, K: H, I: I, gs: gs, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let a = actSep.contents().bindMemory(to: Float16.self, capacity: M * Ktop * I)
        let b = actFused.contents().bindMemory(to: Float16.self, capacity: M * Ktop * I)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (M * Ktop * I) {
            let d = abs(Float(a[i]) - Float(b[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        XCTAssertLessThan(sqrt(num / max(den, 1e-30)), 1e-5, "maxAbs=\(maxAbs)")
    }

    /// Fold gqmm2 + resid_add ≡ fold-add epilogue into `h` (o-proj analog, Ktop=1).
    func testFoldAddMatchesGqmm2ThenResid() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let K = 512, N = 256, E = 4, Ktop = 1, M = 2, gs = 128
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
        let packedK = K * 2 / 32
        let nGroups = K / gs
        let x = buf(M * K * 2); fillHalf(x, 0.01)
        let w = buf(E * N * packedK * 4); fill(w)
        let s = buf(E * N * nGroups * 2); fillHalf(s, 0.02)
        let b = buf(E * N * nGroups * 2); fillHalf(b, -0.02)
        let inds = buf(M * Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
        ip[0] = 1; ip[1] = 2
        let y = buf(M * Ktop * N * 2)
        let hA = buf(M * N * 2)
        let hB = buf(M * N * 2)
        fillHalf(hA, 0.03)
        hB.contents().copyMemory(from: hA.contents(), byteCount: M * N * 2)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: x, w: w, scales: s, biases: b, inds: inds, out: y,
            Ktop: Ktop, K: K, N: N, gs: gs, lhsPerExpert: false, M: M,
            into: enc, splitK: false, w16: false, ternary: false, fold: true,
            deferA: false, foldA: false)
        SeedlessMetal.encodeResid(into: enc, h: hA, delta: y, H: N, M: M)
        try SeedlessMetal.encodeGqmm2FoldAdd(
            into: enc, x: x, w: w, scales: s, biases: b, inds: inds, h: hB,
            Ktop: Ktop, K: K, N: N, gs: gs, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let a = hA.contents().bindMemory(to: Float16.self, capacity: M * N)
        let bp = hB.contents().bindMemory(to: Float16.self, capacity: M * N)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (M * N) {
            let d = abs(Float(a[i]) - Float(bp[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        XCTAssertLessThan(sqrt(num / max(den, 1e-30)), 1e-5, "maxAbs=\(maxAbs)")
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

    /// Qwisp-style combine→resid fold: `score_reduce` + `resid_add` ≡ `score_resid`.
    func testScoreResidMatchesReduceThenAdd() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 2048, Ktop = 8, M = 2
        func buf(_ n: Int, bpe: Int) -> MTLBuffer {
            device.makeBuffer(length: n * bpe, options: .storageModeShared)!
        }
        let down = buf(M * Ktop * H, bpe: 2)
        let scores = buf(M * Ktop, bpe: 4)
        let y = buf(M * H, bpe: 2)
        let hA = buf(M * H, bpe: 2)
        let hB = buf(M * H, bpe: 2)
        let dp = down.contents().bindMemory(to: Float16.self, capacity: M * Ktop * H)
        let sp = scores.contents().bindMemory(to: Float.self, capacity: M * Ktop)
        let ap = hA.contents().bindMemory(to: Float16.self, capacity: M * H)
        let bp = hB.contents().bindMemory(to: Float16.self, capacity: M * H)
        for i in 0 ..< (M * Ktop * H) { dp[i] = Float16(Float((i * 13) % 50) * 0.01 - 0.2) }
        for i in 0 ..< (M * Ktop) { sp[i] = Float((i % 7) + 1) / 28.0 }
        for i in 0 ..< (M * H) {
            let v = Float16(Float(i % 11) * 0.03)
            ap[i] = v
            bp[i] = v
        }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeScoreReduce(
            into: enc, down: down, scores: scores, y: y, H: H, Ktop: Ktop, M: M)
        SeedlessMetal.encodeResid(into: enc, h: hA, delta: y, H: H, M: M)
        SeedlessMetal.encodeScoreResid(
            into: enc, down: down, scores: scores, h: hB, H: H, Ktop: Ktop, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        var num: Double = 0, den: Double = 0
        for i in 0 ..< (M * H) {
            let d = Float(ap[i]) - Float(bp[i])
            num += Double(d * d)
            den += Double(Float(ap[i]) * Float(ap[i]))
        }
        XCTAssertLessThan(sqrt(num / max(den, 1e-30)), 1e-5)
    }

    /// Down gather + `score_resid` ≡ `gqmm2` fold-score epilogue into `h`.
    func testDownScoreResidMatchesGqmm2ThenAdd() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, I = 512, E = 8, Ktop = 2, M = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedI = I * 2 / 32
        let nGI = I / gs
        let act = buf(M * Ktop * I * 2)
        let ap = act.contents().bindMemory(to: Float16.self, capacity: M * Ktop * I)
        for i in 0 ..< (M * Ktop * I) { ap[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let w = buf(E * H * packedI * 4)
        let wp = w.contents().bindMemory(to: UInt8.self, capacity: E * H * packedI * 4)
        for i in 0 ..< (E * H * packedI * 4) { wp[i] = UInt8((i * 17 + 3) & 0xff) }
        let s = buf(E * H * nGI * 2)
        let b = buf(E * H * nGI * 2)
        let sp = s.contents().bindMemory(to: Float16.self, capacity: E * H * nGI)
        let bp = b.contents().bindMemory(to: Float16.self, capacity: E * H * nGI)
        for i in 0 ..< (E * H * nGI) {
            sp[i] = 0.02
            bp[i] = -0.02
        }
        let inds = buf(M * Ktop * 4)
        let scores = buf(M * Ktop * 4)
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
        let scp = scores.contents().bindMemory(to: Float.self, capacity: M * Ktop)
        ip[0] = 1; ip[1] = 3; ip[2] = 2; ip[3] = 5
        for i in 0 ..< (M * Ktop) { scp[i] = Float((i % 7) + 1) / 28.0 }
        let down = buf(M * Ktop * H * 2)
        let hA = buf(M * H * 2)
        let hB = buf(M * H * 2)
        let ha = hA.contents().bindMemory(to: Float16.self, capacity: M * H)
        let hb = hB.contents().bindMemory(to: Float16.self, capacity: M * H)
        for i in 0 ..< (M * H) {
            let v = Float16(Float(i % 11) * 0.03)
            ha[i] = v
            hb[i] = v
        }

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.gqmm2(
            x: act, w: w, scales: s, biases: b, inds: inds, out: down,
            Ktop: Ktop, K: I, N: H, gs: gs, lhsPerExpert: true, M: M,
            into: enc, splitK: false, w16: false, ternary: false, fold: true,
            deferA: false, foldA: false)
        SeedlessMetal.encodeScoreResid(
            into: enc, down: down, scores: scores, h: hA, H: H, Ktop: Ktop, M: M)
        try SeedlessMetal.encodeGqmm2DownScoreResid(
            into: enc, x: act, w: w, scales: s, biases: b,
            inds: inds, scores: scores, h: hB,
            Ktop: Ktop, K: I, N: H, gs: gs, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var num = 0.0, den = 0.0
        var maxAbs: Float = 0
        for i in 0 ..< (M * H) {
            let d = abs(Float(ha[i]) - Float(hb[i]))
            maxAbs = max(maxAbs, d)
            num += Double(d * d)
            den += Double(Float(ha[i]) * Float(ha[i]))
        }
        XCTAssertLessThan(sqrt(num / max(den, 1e-30)), 1e-5, "maxAbs=\(maxAbs)")
    }

    /// Batched MoE block M=2 must match two sequential M=1 steps.
    func testMoEBlockMrowMatchesSequential() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, SeedlessMetal.queue != nil else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, I = 512, E = 16, Ktop = 2, M = 2, gs = 128
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        let packedH = H * 2 / 32
        let packedI = I * 2 / 32
        let nGH = H / gs
        let nGI = I / gs
        let h = buf(M * H * 2)
        let hp = h.contents().bindMemory(to: Float16.self, capacity: M * H)
        for i in 0 ..< (M * H) { hp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let hSeq = buf(M * H * 2)
        let hsp = hSeq.contents().bindMemory(to: Float16.self, capacity: M * H)
        for i in 0 ..< (M * H) { hsp[i] = hp[i] }
        let normW = buf(H * 2)
        let np = normW.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { np[i] = 1.0 }
        let gateW = buf(E * H * 2)
        let gp = gateW.contents().bindMemory(to: Float16.self, capacity: E * H)
        for i in 0 ..< (E * H) { gp[i] = Float16((Float(i % 13) - 6.0) * 0.01) }
        let ugW = buf(E * 2 * I * packedH * 4)
        let dW = buf(E * H * packedI * 4)
        let u8 = ugW.contents().bindMemory(to: UInt8.self, capacity: E * 2 * I * packedH * 4)
        for i in 0 ..< (E * 2 * I * packedH * 4) { u8[i] = UInt8((i * 17 + 3) & 0xff) }
        let u8d = dW.contents().bindMemory(to: UInt8.self, capacity: E * H * packedI * 4)
        for i in 0 ..< (E * H * packedI * 4) { u8d[i] = UInt8((i * 13 + 7) & 0xff) }
        let ugS = buf(E * 2 * I * nGH * 2)
        let ugB = buf(E * 2 * I * nGH * 2)
        let dS = buf(E * H * nGI * 2)
        let dB = buf(E * H * nGI * 2)
        let ugSp = ugS.contents().bindMemory(to: Float16.self, capacity: E * 2 * I * nGH)
        let ugBp = ugB.contents().bindMemory(to: Float16.self, capacity: E * 2 * I * nGH)
        for i in 0 ..< (E * 2 * I * nGH) { ugSp[i] = 0.02; ugBp[i] = -0.02 }
        let dSp = dS.contents().bindMemory(to: Float16.self, capacity: E * H * nGI)
        let dBp = dB.contents().bindMemory(to: Float16.self, capacity: E * H * nGI)
        for i in 0 ..< (E * H * nGI) { dSp[i] = 0.02; dBp[i] = -0.02 }

        let xNorm = buf(M * H * 2)
        let logits = buf(M * E * 4)
        let inds = buf(M * Ktop * 4)
        let scores = buf(M * Ktop * 4)
        let ugOut = buf(M * Ktop * 2 * I * 2)
        let act = buf(M * Ktop * I * 2)
        let downOut = buf(M * Ktop * H * 2)
        let moeOut = buf(M * H * 2)

        try SeedlessMetal.moeBlockOneCB(
            h: h, normW: normW, gateW: gateW,
            upGateW: ugW, upGateS: ugS, upGateB: ugB,
            downW: dW, downS: dS, downB: dB,
            xNorm: xNorm, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: 1e-6, gs: gs, M: M)

        for m in 0 ..< M {
            let h1 = buf(H * 2)
            let hs = hSeq.contents().bindMemory(to: Float16.self, capacity: M * H)
            let hd = h1.contents().bindMemory(to: Float16.self, capacity: H)
            for i in 0 ..< H { hd[i] = hs[m * H + i] }
            let x1 = buf(H * 2), lg1 = buf(E * 4), i1 = buf(Ktop * 4), s1 = buf(Ktop * 4)
            let ug1 = buf(Ktop * 2 * I * 2), a1 = buf(Ktop * I * 2)
            let dn1 = buf(Ktop * H * 2), mo1 = buf(H * 2)
            try SeedlessMetal.moeBlockOneCB(
                h: h1, normW: normW, gateW: gateW,
                upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB,
                xNorm: x1, logits: lg1, inds: i1, scores: s1,
                ugOut: ug1, act: a1, downOut: dn1, moeOut: mo1,
                H: H, I: I, E: E, Ktop: Ktop, eps: 1e-6, gs: gs, M: 1)
            let ySrc = h1.contents().bindMemory(to: Float16.self, capacity: H)
            for i in 0 ..< H { hs[m * H + i] = ySrc[i] }
        }

        let a = h.contents().bindMemory(to: Float16.self, capacity: M * H)
        let bOut = hSeq.contents().bindMemory(to: Float16.self, capacity: M * H)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (M * H) {
            let dlt = abs(Float(a[i]) - Float(bOut[i]))
            maxAbs = max(maxAbs, dlt)
            num += Double(dlt * dlt)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "M-row MoE block must match sequential M=1; rel_l2=\(rel) maxAbs=\(maxAbs)")
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

    /// Two `maple_write_kv` dispatches ≡ one `maple_write_kv_pair`.
    func testWriteKVPairMatchesTwoWrites() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let KV = 4, D = 128, maxLen = 16, pos = 3, M = 2
        let kvDim = KV * D
        func buf(_ n: Int) -> MTLBuffer {
            device.makeBuffer(length: n * 2, options: .storageModeShared)!
        }
        func fill(_ b: MTLBuffer, _ n: Int, _ seed: Int) {
            let p = b.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = Float16(Float((i * seed) % 50) * 0.01 - 0.2) }
        }
        let kSrc = buf(M * kvDim); fill(kSrc, M * kvDim, 13)
        let vSrc = buf(M * kvDim); fill(vSrc, M * kvDim, 17)
        let kA = buf(KV * maxLen * D); let vA = buf(KV * maxLen * D)
        let kB = buf(KV * maxLen * D); let vB = buf(KV * maxLen * D)
        func zero(_ b: MTLBuffer) {
            let n = b.length / 2
            let p = b.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = 0 }
        }
        zero(kA); zero(vA); zero(kB); zero(vB)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeWriteKV(
            into: enc, src: kSrc, cache: kA, srcOffset: 0,
            KV: KV, D: D, maxLen: maxLen, pos: pos, M: M,
            srcHeadStride: D, srcSeqStride: kvDim)
        SeedlessMetal.encodeWriteKV(
            into: enc, src: vSrc, cache: vA, srcOffset: 0,
            KV: KV, D: D, maxLen: maxLen, pos: pos, M: M,
            srcHeadStride: D, srcSeqStride: kvDim)
        SeedlessMetal.encodeWriteKVPair(
            into: enc, kSrc: kSrc, vSrc: vSrc, kCache: kB, vCache: vB,
            kSrcOffset: 0, vSrcOffset: 0,
            KV: KV, D: D, maxLen: maxLen, pos: pos, M: M,
            kHeadStride: D, kSeqStride: kvDim,
            vHeadStride: D, vSeqStride: kvDim)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let n = KV * maxLen * D
        func rel(_ a: MTLBuffer, _ b: MTLBuffer) -> Double {
            let ap = a.contents().bindMemory(to: Float16.self, capacity: n)
            let bp = b.contents().bindMemory(to: Float16.self, capacity: n)
            var num = 0.0, den = 0.0
            for i in 0 ..< n {
                let d = Float(ap[i]) - Float(bp[i])
                num += Double(d * d)
                den += Double(Float(ap[i]) * Float(ap[i]))
            }
            return sqrt(num / max(den, 1e-30))
        }
        XCTAssertLessThan(rel(kA, kB), 1e-5)
        XCTAssertLessThan(rel(vA, vB), 1e-5)
    }

    /// Batched SDPA M=2 (shared KV) must match two sequential M=1 queries.
    func testSdpaMrowMatchesSequential() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let heads = 16, kv = 4, d = 128, M = 2, maxLen = 8, seqLen = 8
        func buf(_ n: Int, _ bpe: Int = 2) -> MTLBuffer {
            device.makeBuffer(length: n * bpe, options: .storageModeShared)!
        }
        let qM = buf(heads * M * d)
        let qp = qM.contents().bindMemory(to: Float16.self, capacity: heads * M * d)
        for i in 0 ..< (heads * M * d) { qp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let kCache = buf(kv * maxLen * d)
        let vCache = buf(kv * maxLen * d)
        let kp = kCache.contents().bindMemory(to: Float16.self, capacity: kv * maxLen * d)
        let vp = vCache.contents().bindMemory(to: Float16.self, capacity: kv * maxLen * d)
        for i in 0 ..< (kv * maxLen * d) {
            kp[i] = Float16((Float(i % 13) - 6.0) * 0.02)
            vp[i] = Float16((Float(i % 11) - 5.0) * 0.02)
        }
        let yM = buf(heads * M * d)
        let ySeq = buf(heads * M * d)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeSdpa(
            into: enc, queries: qM, kCache: kCache, vCache: vCache, out: yM,
            numHeads: heads, numKV: kv, headDim: d, maxLen: maxLen, seqLen: seqLen, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        for m in 0 ..< M {
            let q1 = buf(heads * d)
            let y1 = buf(heads * d)
            let qSrc = qM.contents().bindMemory(to: Float16.self, capacity: heads * M * d)
            let qDst = q1.contents().bindMemory(to: Float16.self, capacity: heads * d)
            for h in 0 ..< heads {
                for i in 0 ..< d { qDst[h * d + i] = qSrc[(m * heads + h) * d + i] }
            }
            let cb1 = q.makeCommandBuffer()!
            let enc1 = cb1.makeComputeCommandEncoder()!
            SeedlessMetal.encodeSdpa(
                into: enc1, queries: q1, kCache: kCache, vCache: vCache, out: y1,
                numHeads: heads, numKV: kv, headDim: d, maxLen: maxLen, seqLen: seqLen, M: 1)
            enc1.endEncoding()
            cb1.commit()
            cb1.waitUntilCompleted()
            let ySrc = y1.contents().bindMemory(to: Float16.self, capacity: heads * d)
            let yDst = ySeq.contents().bindMemory(to: Float16.self, capacity: heads * M * d)
            for h in 0 ..< heads {
                for i in 0 ..< d { yDst[(m * heads + h) * d + i] = ySrc[h * d + i] }
            }
        }
        let a = yM.contents().bindMemory(to: Float16.self, capacity: heads * M * d)
        let bOut = ySeq.contents().bindMemory(to: Float16.self, capacity: heads * M * d)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (heads * M * d) {
            let dlt = abs(Float(a[i]) - Float(bOut[i]))
            maxAbs = max(maxAbs, dlt)
            num += Double(dlt * dlt)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "M-row SDPA must match sequential M=1; rel_l2=\(rel) maxAbs=\(maxAbs)")
    }

    /// Batched encodeAttnBlock M=2 (causal SDPA, RoPE pos+m) must match two sequential M=1 steps.
    func testAttnBlockMrowMatchesSequential() throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device, let q = SeedlessMetal.queue else {
            throw XCTSkip("no Metal device")
        }
        let H = 512, heads = 4, kv = 2, d = 128, gs = 128, M = 2
        let qDim = heads * d, kvDim = kv * d, qkvN = qDim + 2 * kvDim
        let maxLen = 8, writePos = 2, ropePos = 2, seqLen = writePos + 1
        func buf(_ n: Int, _ bpe: Int = 2) -> MTLBuffer {
            device.makeBuffer(length: n * bpe, options: .storageModeShared)!
        }
        func fillU8(_ b: MTLBuffer) {
            let p = b.contents().bindMemory(to: UInt8.self, capacity: b.length)
            for i in 0 ..< b.length { p[i] = UInt8((i * 17 + 3) & 0xff) }
        }
        func fillH(_ b: MTLBuffer, _ v: Float16) {
            let n = b.length / 2
            let p = b.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = v }
        }
        // The M-row attn kernel reads `inds[mk]` per row; the buffer must hold
        // one entry per row (dense → all zeros).
        let dens = buf(M, 4)
        let densPtr = dens.contents().bindMemory(to: Int32.self, capacity: M)
        for i in 0 ..< M { densPtr[i] = 0 }
        let x = buf(M * H)
        let xp = x.contents().bindMemory(to: Float16.self, capacity: M * H)
        for i in 0 ..< (M * H) { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let packedH = H * 2 / 32
        let packedQ = qDim * 2 / 32
        let qkvW = buf(qkvN * packedH, 4); fillU8(qkvW)
        let qkvS = buf(qkvN * (H / gs)); fillH(qkvS, 0.02)
        let qkvB = buf(qkvN * (H / gs)); fillH(qkvB, -0.02)
        let oW = buf(H * packedQ, 4); fillU8(oW)
        let oS = buf(H * (qDim / gs)); fillH(oS, 0.02)
        let oB = buf(H * (qDim / gs)); fillH(oB, -0.02)
        let qkW = buf((heads + kv) * d); fillH(qkW, 1.0)
        let inv = buf(32, 4)
        let ip = inv.contents().bindMemory(to: Float.self, capacity: 32)
        for i in 0 ..< 32 { ip[i] = 0.1 / Float(i + 1) }
        let qkvOut = buf(M * qkvN)
        let qkOut = buf(M * (qDim + kvDim))
        let attnTmp = buf(M * qDim)
        let yM = buf(M * H)
        let ySeq = buf(M * H)
        let kCache = buf(kv * maxLen * d)
        let vCache = buf(kv * maxLen * d)
        fillH(kCache, 0.01)
        fillH(vCache, 0.02)

        let kSeq = buf(kv * maxLen * d)
        let vSeq = buf(kv * maxLen * d)
        let k0 = kCache.contents().bindMemory(to: Float16.self, capacity: kv * maxLen * d)
        let v0 = vCache.contents().bindMemory(to: Float16.self, capacity: kv * maxLen * d)
        let k1p = kSeq.contents().bindMemory(to: Float16.self, capacity: kv * maxLen * d)
        let v1p = vSeq.contents().bindMemory(to: Float16.self, capacity: kv * maxLen * d)
        for i in 0 ..< (kv * maxLen * d) { k1p[i] = k0[i]; v1p[i] = v0[i] }

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try SeedlessMetal.encodeAttnBlock(
            into: enc, xNorm: x,
            qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
            oW: oW, oS: oS, oB: oB,
            qkW: qkW, invFreq: inv, densInds: dens,
            qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: yM,
            kCache: kCache, vCache: vCache,
            H: H, numHeads: heads, numKV: kv, headDim: d,
            ropeDim: 64, ropePos: ropePos, writePos: writePos, maxLen: maxLen, seqLen: seqLen,
            eps: 1e-6, gs: gs, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        for m in 0 ..< M {
            let x1 = buf(H)
            let xSrc = x.contents().bindMemory(to: Float16.self, capacity: M * H)
            let xDst = x1.contents().bindMemory(to: Float16.self, capacity: H)
            for i in 0 ..< H { xDst[i] = xSrc[m * H + i] }
            let qkv1 = buf(qkvN), qk1 = buf(qDim + kvDim), tmp1 = buf(qDim), y1 = buf(H)
            let cb1 = q.makeCommandBuffer()!
            let enc1 = cb1.makeComputeCommandEncoder()!
            try SeedlessMetal.encodeAttnBlock(
                into: enc1, xNorm: x1,
                qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
                oW: oW, oS: oS, oB: oB,
                qkW: qkW, invFreq: inv, densInds: dens,
                qkvOut: qkv1, qkOut: qk1, attnTmp: tmp1, attnOut: y1,
                kCache: kSeq, vCache: vSeq,
                H: H, numHeads: heads, numKV: kv, headDim: d,
                ropeDim: 64, ropePos: ropePos + m, writePos: writePos + m, maxLen: maxLen,
                seqLen: seqLen + m, eps: 1e-6, gs: gs, M: 1)
            enc1.endEncoding()
            cb1.commit()
            cb1.waitUntilCompleted()
            let ySrc = y1.contents().bindMemory(to: Float16.self, capacity: H)
            let yDst = ySeq.contents().bindMemory(to: Float16.self, capacity: M * H)
            for i in 0 ..< H { yDst[m * H + i] = ySrc[i] }
        }
        let a = yM.contents().bindMemory(to: Float16.self, capacity: M * H)
        let bOut = ySeq.contents().bindMemory(to: Float16.self, capacity: M * H)
        var maxAbs: Float = 0
        var num = 0.0, den = 0.0
        for i in 0 ..< (M * H) {
            let dlt = abs(Float(a[i]) - Float(bOut[i]))
            maxAbs = max(maxAbs, dlt)
            num += Double(dlt * dlt)
            den += Double(Float(a[i]) * Float(a[i]))
        }
        let rel = sqrt(num / max(den, 1e-30))
        XCTAssertLessThan(rel, 1e-3, "M-row attn block must match sequential M=1; rel_l2=\(rel) maxAbs=\(maxAbs)")
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

final class OpenAIChatCompatTests: XCTestCase {
    func testIgnoresUnknownRequestFields() throws {
        let raw = """
        {"model":"maple","messages":[{"role":"user","content":"hi"}],\
        "tools":[{"type":"function","function":{"name":"ls","parameters":{"type":"object"}}}],\
        "tool_choice":"auto","structured_outputs":{"json":{"type":"object"}},\
        "response_format":{"type":"json_object"},"user":"u","metadata":{"k":1},\
        "max_tokens":16}
        """.data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertEqual(req.messages.count, 1)
        XCTAssertEqual(req.messages[0].content, "hi")
        XCTAssertEqual(req.maxTokens, 16)
        XCTAssertEqual(req.model, "maple")
        // P0: unknown fields ignored for parse success; P1: JSON schema captured.
        if case .jsonSchema(let schema)? = req.structuredConstraint {
            XCTAssertTrue(schema.contains("object"))
        } else {
            XCTFail("expected jsonSchema constraint, got \(String(describing: req.structuredConstraint))")
        }
    }

    func testParsesGuidedGrammarLiteralUnion() throws {
        let raw = """
        {"messages":[{"role":"user","content":"x"}],\
        "guided_grammar":"root ::= \\"ZQX\\"",\
        "max_tokens":8}
        """.data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertEqual(req.structuredConstraint, .choice(["ZQX"]))
    }

    func testParsesStructuredOutputsChoice() throws {
        let raw = """
        {"messages":[{"role":"user","content":"x"}],\
        "structured_outputs":{"choice":["red","green","blue"]}}
        """.data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertEqual(req.structuredConstraint, .choice(["red", "green", "blue"]))
    }

    func testTopLevelGrammarIgnored() throws {
        let raw = """
        {"messages":[{"role":"user","content":"x"}],"grammar":"root ::= \\"ZQX\\""}
        """.data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertNil(req.structuredConstraint)
    }

    func testNullContentIsEmptyString() throws {
        let raw = #"{"messages":[{"role":"assistant","content":null}]}"#.data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertEqual(req.messages[0].content, "")
    }

    func testDeveloperRoleMapsToSystem() throws {
        let raw = #"{"messages":[{"role":"developer","content":"sys"},{"role":"user","content":"u"}]}"#
            .data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertEqual(req.messages[0].role, "system")
        XCTAssertEqual(req.messages[1].role, "user")
    }

    func testMultimodalContentPartsFlattenToText() throws {
        let raw = """
        {"messages":[{"role":"user","content":[\
        {"type":"text","text":"<tools>"},{"type":"text","text":"</tools>"}]}]}
        """.data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertEqual(req.messages[0].content, "<tools></tools>")
    }

    func testMaxCompletionTokensPreferred() throws {
        let raw = #"{"messages":[{"role":"user","content":"x"}],"max_tokens":8,"max_completion_tokens":32}"#
            .data(using: .utf8)!
        let req = try OpenAIChatCompat.parseRequest(from: raw)
        XCTAssertEqual(req.maxTokens, 32)
    }

    func testMarkupRoundtripPreservedInMessages() throws {
        let system = "You have tools.\n<tools>\n[{\"name\":\"ls\"}]\n</tools>"
        let assistant = #"<tool_call>{"name":"ls","arguments":{"path":"."}}</tool_call>"#
        let user = "<tool_response>\nok\n</tool_response>"
        let payload: [String: Any] = [
            "messages": [
                ["role": "system", "content": system],
                ["role": "assistant", "content": assistant],
                ["role": "user", "content": user],
            ],
            "max_tokens": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let req = try OpenAIChatCompat.parseRequest(from: data)
        XCTAssertEqual(req.messages[0].content, system)
        XCTAssertEqual(req.messages[1].content, assistant)
        XCTAssertEqual(req.messages[2].content, user)
    }

    func testFinishReasonLengthWhenMaxFilled() {
        let tokens = Array(repeating: 1, count: 16)
        XCTAssertEqual(
            OpenAIChatCompat.finishReason(completionTokens: tokens, maxTokens: 16),
            "length")
    }

    func testFinishReasonStopOnEos() {
        let tokens = [10, 11, 151_645]
        XCTAssertEqual(
            OpenAIChatCompat.finishReason(completionTokens: tokens, maxTokens: 64),
            "stop")
        XCTAssertEqual(
            OpenAIChatCompat.contentTokenIds(tokens),
            [10, 11])
    }

    func testNonStreamResponseShape() throws {
        let data = try OpenAIChatCompat.nonStreamJSON(
            modelID: "maple-preview",
            content: "pong",
            finishReason: "stop",
            promptTokens: 3,
            completionTokens: 1,
            id: "chatcmpl-test",
            created: 1)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(obj["object"] as? String, "chat.completion")
        XCTAssertEqual(obj["model"] as? String, "maple-preview")
        let choices = obj["choices"] as! [[String: Any]]
        let msg = choices[0]["message"] as! [String: Any]
        XCTAssertEqual(msg["role"] as? String, "assistant")
        XCTAssertEqual(msg["content"] as? String, "pong")
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")
        let usage = obj["usage"] as! [String: Any]
        XCTAssertEqual(usage["total_tokens"] as? Int, 4)
    }
}

final class StructuredOutputsTests: XCTestCase {
    func testParseLiteralEBNF() {
        XCTAssertEqual(
            StructuredOutputs.parseLiteralUnionEBNF(#"root ::= "ZQX""#),
            .choice(["ZQX"]))
        XCTAssertEqual(
            StructuredOutputs.parseLiteralUnionEBNF(#"root ::= "red" | "green" | "blue""#),
            .choice(["red", "green", "blue"]))
    }

    func testFiniteStringGuideFrontier() throws {
        let guide = try FiniteStringGuide(
            strings: ["ab", "ac"],
            encode: { Array($0.unicodeScalars.map { Int($0.value) }) },
            eosTokenId: 999)
        XCTAssertEqual(guide.allowedNext(prefix: []), Set([Int(("a" as Character).asciiValue!)]))
        XCTAssertEqual(
            guide.allowedNext(prefix: [Int(("a" as Character).asciiValue!)]),
            Set([Int(("b" as Character).asciiValue!), Int(("c" as Character).asciiValue!)]))
        XCTAssertEqual(
            guide.allowedNext(prefix: [
                Int(("a" as Character).asciiValue!),
                Int(("b" as Character).asciiValue!),
            ]),
            Set([999]))
    }

    func testSeedlessConstrainedWalkEmitsLiteral() {
        let guide = try! FiniteStringGuide(
            strings: ["ZQX"],
            encode: { Array($0.utf8.map { Int($0) }) },
            eosTokenId: 7)
        let toks = SeedlessBackend.generateConstrained(
            maxTokens: 16,
            eos: 7,
            allowedNext: { guide.allowedNext(prefix: $0) })
        XCTAssertEqual(toks, Array("ZQX".utf8.map { Int($0) }) + [7])
    }

    /// Open JSON must not pick the lowest allowed id (`!` is often vocab 0-ish).
    func testConstrainedPickUsesLogitsNotLowestId() {
        var scores = [Float](repeating: 0, count: 16)
        scores[1] = 1   // "!"
        scores[5] = 9   // e.g. a letter
        scores[7] = 3
        let allowed: Set<Int> = [1, 5, 7]
        XCTAssertEqual(ConstrainedPick.argmax(allowed: allowed, scores: scores), 5)
        XCTAssertNotEqual(allowed.min(), 5)
    }
}

final class SeedlessContextTests: XCTestCase {
    private var modelDir: String {
        ProcessInfo.processInfo.environment["MOMIJ_MODEL"]
            ?? NSString("~/models/deepgrove/maple-preview-2bit-mlx").expandingTildeInPath
    }

    /// Oversized prompt must throw before Metal assert (encoder without endEncoding).
    func testGenerateOversizedPromptThrowsCleanly() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: 32)
        let prompt = Array(repeating: 100, count: 40)
        XCTAssertThrowsError(try eng.generate(prompt: prompt, maxTokens: 8, eos: nil))
    }

    /// Seedless sequential prefill must match pure-MLX batched prefill on greedy first token.
    /// Lengths straddle SWA wrap (512) so a ring/RoPE bug shows up as a cliff.
    func testFirstGeneratedTokenMatchesMLX() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 2048, enableFlashHead: false)
        var mismatches: [String] = []
        for len in [1, 4, 16, 64, 256, 512, 520] {
            let prompt = Array(repeating: 100, count: len)
            let mlxTok = mlx.generate(prompt: prompt, maxTokens: 1, eos: nil)
            let seedTok = try seed.generate(prompt: prompt, maxTokens: 1, eos: nil)
            if mlxTok != seedTok {
                mismatches.append("len=\(len) mlx=\(mlxTok) seedless=\(seedTok)")
            }
        }
        XCTAssertTrue(mismatches.isEmpty, mismatches.joined(separator: "; "))
    }

    /// Default serve loads FlashHead; greedy dummy-100 is outside its clusters.
    /// `MOMIJ_EXACT_HEAD` / `enableExactHead` keeps exact `lm_head` even with FlashHead loaded.
    func testGreedyFirstTokenIgnoresFlashHeadApprox() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let exact = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: false)
        let flashExact = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: true, enableExactHead: true)
        let prompt = Array(repeating: 100, count: 8)
        let a = try exact.generate(prompt: prompt, maxTokens: 1, eos: nil)
        let b = try flashExact.generate(prompt: prompt, maxTokens: 1, eos: nil)
        XCTAssertEqual(a, b, "exact-head with FlashHead loaded \(b) != exact lm_head \(a)")
    }

    /// Opt-in SuffixSpec sequential verify must stay on exact greedy (not FlashHead).
    func testSuffixSpecSequentialMatchesExactGenerate() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: true, enableExactHead: true)
        let prompt = Array(repeating: 100, count: 16)
        let greedy = try seed.generate(prompt: prompt, maxTokens: 8, eos: nil)
        let spec = try seed.generateSuffixSpec(prompt: prompt, maxTokens: 8, eos: nil)
        XCTAssertEqual(spec.tokens, greedy, "suffix-spec \(spec.tokens) != exact greedy \(greedy)")
        let banned = ChatTemplatePatch.bannedAssistantTokenIds
        let greedyBan = try seed.generate(
            prompt: prompt, maxTokens: 8, eos: nil, bannedTokenIds: banned)
        let specBan = try seed.generateSuffixSpec(
            prompt: prompt, maxTokens: 8, eos: nil, bannedTokenIds: banned)
        XCTAssertEqual(specBan.tokens, greedyBan)
    }

    /// Exact M-row 1-CB chain must match sequential skipFlash (lossless opt-out path).
    func testExactMrowChainMatchesSequential() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: true, enableExactHead: true)
        XCTAssertTrue(seed.greedyExactHead)
        func check(_ prompt: [Int]) throws {
            seed.reset()
            var last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count {
                    _ = try seed.step(prompt[i], skipFlash: true)
                } else {
                    last = try seed.step(prompt[i], skipFlash: true)
                }
            }
            let K = 4
            var feeds = [last]
            var expected: [Int] = []
            var cur = last
            for i in 0 ..< K {
                let q = try seed.step(cur, skipFlash: true)
                expected.append(q)
                cur = q
                if i + 1 < K { feeds.append(q) }
            }
            seed.reset()
            last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count {
                    _ = try seed.step(prompt[i], skipFlash: true)
                } else {
                    last = try seed.step(prompt[i], skipFlash: true)
                }
            }
            let chain = try seed.stepChainFeeds(feeds)
            XCTAssertEqual(chain, expected, "prompt=\(prompt.prefix(4))… chain=\(chain) seq=\(expected)")
        }
        try check(Array(repeating: 100, count: 16))
        try check([
            151_644, 872, 198, 20_841, 448, 6896, 25, 10_402, 151_645, 198,
            151_644, 77_091, 198,
        ])
    }

    /// After the first token, sequential Metal decode must stay on the MLX greedy path.
    func testGreedyContinuationMatchesMLX() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 1024, enableFlashHead: true, enableExactHead: true)
        var mismatches: [String] = []
        for len in [16, 64, 128] {
            let prompt = Array(repeating: 100, count: len)
            let mlxTok = mlx.generate(prompt: prompt, maxTokens: 8, eos: nil)
            let seedTok = try seed.generate(prompt: prompt, maxTokens: 8, eos: nil)
            if mlxTok != seedTok {
                mismatches.append("len=\(len) mlx=\(mlxTok) seedless=\(seedTok)")
            }
        }
        XCTAssertTrue(mismatches.isEmpty, mismatches.joined(separator: "; "))
    }

    /// Serve short chat (`Reply with exactly: OK`, patched template, 13 ids).
    func testServeShortChatPromptMatchesMLX() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        // <|im_start|>user\nReply with exactly: OK<|im_end|>\n<|im_start|>assistant\n
        let prompt = [
            151_644, 872, 198, 20_841, 448, 6896, 25, 10_402, 151_645, 198,
            151_644, 77_091, 198,
        ]
        let banned = ChatTemplatePatch.bannedAssistantTokenIds
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: true)
        let mlxTok = mlx.generate(
            prompt: prompt, maxTokens: 8, eos: 151_645, bannedTokenIds: banned)
        let seedTok = try seed.generate(
            prompt: prompt, maxTokens: 8, eos: 151_645, bannedTokenIds: banned)
        XCTAssertEqual(seedTok, mlxTok, "seedless \(seedTok) != mlx \(mlxTok)")
    }

    /// Dummy token-100 vs real chat/code-like prefixes: is FlashHead miss an artifact?
    func testFlashHeadVsExactOnDummyAndChat() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: true)
        XCTAssertTrue(seed.useFlashHead)

        let chat = [
            151_644, 872, 198, 20_841, 448, 6896, 25, 10_402, 151_645, 198,
            151_644, 77_091, 198,
        ]
        // Minimal "code" ids already in the chat template plus a fib-like tail of
        // distinct small-vocab tokens (not a single repeated dummy).
        let codeish = chat + [16, 13, 17, 13, 18, 13, 19]

        func probe(_ name: String, _ prompt: [Int]) throws -> (
            name: String, exact: Int, flash: Int, inTop: Bool, flashTop: [Int]
        ) {
            let r = try seed.firstTokenExactVsFlash(prompt)
            return (name, r.exact, r.flash, r.exactInFlashTop, r.flashTop)
        }

        let dummy8 = try probe("dummy8x100", Array(repeating: 100, count: 8))
        let dummy1 = try probe("dummy1x100", [100])
        let chat1 = try probe("chat13", chat)
        let code1 = try probe("codeish", codeish)

        let exactCont = try seed.generate(prompt: chat, maxTokens: 8, eos: nil)
        let flashCont = try seed.generateFlashHeadGreedy(prompt: chat, maxTokens: 8)

        let banned = ChatTemplatePatch.bannedAssistantTokenIds
        let exactBan = try seed.generate(
            prompt: chat, maxTokens: 1, eos: nil, bannedTokenIds: banned)
        let flashBan = chat1.flashTop.first { !banned.contains($0) }

        // Dummy token-100 is outside FlashHead's probed clusters (harsh, not chat).
        XCTAssertNotEqual(dummy8.exact, dummy8.flash, "dummy8: \(dummy8)")
        XCTAssertFalse(dummy8.inTop)
        XCTAssertNotEqual(dummy1.exact, dummy1.flash, "dummy1: \(dummy1)")
        XCTAssertFalse(dummy1.inTop)
        // Natural chat / code-like prefixes: FlashHead hits the exact argmax.
        XCTAssertEqual(chat1.exact, chat1.flash, "chat first \(chat1)")
        XCTAssertTrue(chat1.inTop)
        XCTAssertEqual(code1.exact, code1.flash, "codeish first \(code1)")
        XCTAssertEqual(exactCont, flashCont, "chat 8-tok exact=\(exactCont) flash=\(flashCont)")
        XCTAssertEqual(exactBan, [flashBan].compactMap { $0 }, "banned exact=\(exactBan) flashBan=\(String(describing: flashBan))")
    }

    /// Banning `<think>` must pick the same second-best as mlx (serve default).
    func testBannedPrefillArgmaxMatchesMLX() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let prompt = [
            151_644, 872, 198, 20_841, 448, 6896, 25, 10_402, 151_645, 198,
            151_644, 77_091, 198,
        ]
        let probes = [3925, 198, 151_645, 151_667, 151_668, 151_644]
        let banned = ChatTemplatePatch.bannedAssistantTokenIds
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: false)
        var mlxScores = ConstrainedPick.hostScores(mlx.forwardLastLogits(prompt, reset: true))
        var seedScores = try seed.greedyScoresAfterPrefill(prompt: prompt, skipFlash: true)
        XCTAssertGreaterThanOrEqual(seed.decodeMaxM, 2, "M-row prefill disabled")
        let mlxTop = ConstrainedPick.argmaxAll(mlxScores)
        let seedTop = ConstrainedPick.argmaxAll(seedScores)
        let probeNote = probes.map { id -> String in
            let ms = id < mlxScores.count ? mlxScores[id] : .nan
            let ss = id < seedScores.count ? seedScores[id] : .nan
            return "\(id) mlx=\(ms) seed=\(ss)"
        }.joined(separator: "; ")
        ConstrainedPick.applyBanned(&mlxScores, banned: banned)
        ConstrainedPick.applyBanned(&seedScores, banned: banned)
        let mlxBan = ConstrainedPick.argmaxAll(mlxScores)
        let seedBan = ConstrainedPick.argmaxAll(seedScores)
        XCTAssertEqual(
            seedTop, mlxTop,
            "unconstrained top seedless=\(seedTop) mlx=\(mlxTop) \(probeNote)")
        XCTAssertEqual(
            seedBan, mlxBan,
            "banned top seedless=\(seedBan) mlx=\(mlxBan) maxM=\(seed.decodeMaxM) chunks=\(seed.lastPrefillChunks) \(probeNote)")
    }

    /// Two-token chat prefix (`<|im_start|>` + `user`) last-token argmax vs batched MLX.
    func testTwoTokenChatPrefillMatchesMLX() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let prompt = [151_644, 872]
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: false)
        let mlxTop = ConstrainedPick.argmaxAll(
            ConstrainedPick.hostScores(mlx.forwardLastLogits(prompt, reset: true)))
        let seedTop = ConstrainedPick.argmaxAll(
            try seed.greedyScoresAfterPrefill(prompt: prompt, skipFlash: true))
        XCTAssertEqual(seedTop, mlxTop, "g=\(seed.layersPerCB) seedless=\(seedTop) mlx=\(mlxTop)")
    }

    /// Last-token residual after each layer matches batched MLX on the serve chat ids.
    func testChatPrefillHiddenMatchesMLX() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let prompt = [
            151_644, 872, 198, 20_841, 448, 6896, 25, 10_402, 151_645, 198,
            151_644, 77_091, 198,
        ]
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: false)
        let mlxH = mlx.lastTokenHiddenAfterEachLayer(prompt)
        let seedH = try seed.lastTokenHiddenAfterEachLayer(prompt: prompt)
        XCTAssertEqual(seedH.embed.count, mlxH.embed.count)
        XCTAssertEqual(seedH.layers.count, mlxH.layers.count)

        func cosine(_ a: [Float], _ b: [Float]) -> Float {
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< a.count {
                dot += a[i] * b[i]
                na += a[i] * a[i]
                nb += b[i] * b[i]
            }
            return dot / (sqrt(na) * sqrt(nb) + 1e-12)
        }
        func maxAbs(_ a: [Float], _ b: [Float]) -> Float {
            var m: Float = 0
            for i in 0 ..< a.count { m = max(m, abs(a[i] - b[i])) }
            return m
        }

        let embedCos = cosine(seedH.embed, mlxH.embed)
        let embedMax = maxAbs(seedH.embed, mlxH.embed)
        var notes = ["embed cos=\(embedCos) maxAbs=\(embedMax)"]
        XCTAssertGreaterThan(embedCos, 0.999, notes[0])

        // fp16 layer noise is small; 0.995 still leaves the OK-logit gap unexplained
        // if only tiny drift. Fail at the first layer that clearly leaves the mlx basin.
        let minCos: Float = 0.995
        var firstBad: Int?
        for (i, (s, m)) in zip(seedH.layers, mlxH.layers).enumerated() {
            let c = cosine(s, m)
            let d = maxAbs(s, m)
            notes.append("L\(i) sliding=\(store.config.isSliding(i)) cos=\(c) maxAbs=\(d)")
            if firstBad == nil, c < minCos { firstBad = i }
        }
        XCTAssertNil(
            firstBad,
            "first residual mismatch at layer \(firstBad!) " + notes.joined(separator: " | "))
    }

    /// Layer-0 attention vs mlx. Same-token pairs are exact; mixed ids expose SDPA mix bugs
    /// (identical V makes softmax errors invisible).
    func testChatPrefillLayer0AttnMatchesMLX() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let chat = [
            151_644, 872, 198, 20_841, 448, 6896, 25, 10_402, 151_645, 198,
            151_644, 77_091, 198,
        ]
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: false)

        func cosine(_ a: [Float], _ b: [Float]) -> Float {
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< a.count {
                dot += a[i] * b[i]
                na += a[i] * a[i]
                nb += b[i] * b[i]
            }
            return dot / (sqrt(na) * sqrt(nb) + 1e-12)
        }
        func maxAbs(_ a: [Float], _ b: [Float]) -> Float {
            var m: Float = 0
            for i in 0 ..< a.count { m = max(m, abs(a[i] - b[i])) }
            return m
        }

        var notes: [String] = []
        var firstBad: String?
        let cases: [(String, [Int])] = [
            ("dummy L=2", [100, 100]),
            ("special+special", [151_644, 151_644]),
            ("special+100", [151_644, 100]),
            ("chat L=2", Array(chat.prefix(2))),
        ]
        for (label, prompt) in cases {
            let m = mlx.lastTokenAfterFirstAttn(prompt)
            let s = try seed.lastTokenAfterFirstAttn(
                prompt: prompt, prefixFullStack: false)
            let c = cosine(s, m)
            let d = maxAbs(s, m)
            notes.append("\(label) cos=\(c) maxAbs=\(d)")
            if firstBad == nil, c < 0.995 { firstBad = label }
        }
        XCTAssertNil(firstBad, "layer0 attn mismatch at \(firstBad!) " + notes.joined(separator: " | "))
    }

    /// Metal SDPA output must match a host fp32 mix of the same Q/K/V (N=2 mixed ids).
    func testLayer0SdpaMatchesHostQKV() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let prompt = [151_644, 100]
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: false)
        _ = try seed.lastTokenAfterFirstAttn(prompt: prompt, prefixFullStack: false)
        let l0 = seed.stack.layers[0]
        let Hn = l0.numHeads, KV = l0.numKV, D = l0.headDim, maxLen = l0.maxLen
        let N = 2
        let qCount = Hn * D
        let qp = l0.qkOut.contents().bindMemory(to: Float16.self, capacity: qCount)
        let kp = l0.kCache.contents().bindMemory(to: Float16.self, capacity: KV * maxLen * D)
        let vp = l0.vCache.contents().bindMemory(to: Float16.self, capacity: KV * maxLen * D)
        let ap = l0.attnTmp.contents().bindMemory(to: Float16.self, capacity: qCount)
        let scale = Float(pow(Double(D), -0.5))
        var host = [Float](repeating: 0, count: qCount)
        for qh in 0 ..< Hn {
            let kvh = qh * KV / Hn
            var scores = [Float](repeating: 0, count: N)
            for t in 0 ..< N {
                var dot: Float = 0
                let qOff = qh * D
                let kOff = kvh * maxLen * D + t * D
                for d in 0 ..< D { dot += Float(qp[qOff + d]) * Float(kp[kOff + d]) }
                scores[t] = scale * dot
            }
            let mx = scores.max() ?? 0
            var z: Float = 0
            for t in 0 ..< N {
                scores[t] = exp(scores[t] - mx)
                z += scores[t]
            }
            for d in 0 ..< D {
                var acc: Float = 0
                for t in 0 ..< N {
                    let vOff = kvh * maxLen * D + t * D
                    acc += (scores[t] / z) * Float(vp[vOff + d])
                }
                host[qh * D + d] = acc
            }
        }
        var maxAbs: Float = 0
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< qCount {
            let a = host[i], b = Float(ap[i])
            maxAbs = max(maxAbs, abs(a - b))
            dot += a * b
            na += a * a
            nb += b * b
        }
        let cos = dot / (sqrt(na) * sqrt(nb) + 1e-12)
        XCTAssertGreaterThan(cos, 0.999, "Metal SDPA vs host QKV mix cos=\(cos) maxAbs=\(maxAbs)")
    }

    /// Split mixed-id L0 mismatch: QKV (gqmm2) vs Q after RoPE (qk-norm-rope vs MLXFast.RoPE).
    func testChatPrefillLayer0QSplitsQKVFromRoPE() throws {
        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("model not present")
        }
        let prompt = [151_644, 100]
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let mlx = MapleEngine(store: store, enableSeedlessMoE: false)
        let seed = try SeedlessDecodeEngine(
            store: store, fullMaxLen: 256, enableFlashHead: false)
        let metal = try seed.lastTokenLayer0QK(prompt: prompt)
        let qkvLast = mlx.lastTokenLayer0QKV(prompt, batched: false)
        let qkvBatched = mlx.lastTokenLayer0QKV(prompt, batched: true)
        let qFused = mlx.lastTokenLayer0Q(prompt, batched: false)
        let qBatched = mlx.lastTokenLayer0Q(prompt, batched: true)

        func cosine(_ a: [Float], _ b: [Float]) -> Float {
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            let n = min(a.count, b.count)
            for i in 0 ..< n {
                dot += a[i] * b[i]
                na += a[i] * a[i]
                nb += b[i] * b[i]
            }
            return dot / (sqrt(na) * sqrt(nb) + 1e-12)
        }
        func maxAbs(_ a: [Float], _ b: [Float]) -> Float {
            var m: Float = 0
            let n = min(a.count, b.count)
            for i in 0 ..< n { m = max(m, abs(a[i] - b[i])) }
            return m
        }
        func note(_ label: String, _ a: [Float], _ b: [Float]) -> String {
            "\(label) n=\(a.count)/\(b.count) cos=\(cosine(a, b)) maxAbs=\(maxAbs(a, b))"
        }
        let notes = [
            note("qkv metal vs mlx-last", metal.qkv, qkvLast),
            note("qkv metal vs mlx-batched", metal.qkv, qkvBatched),
            note("qkv mlx-last vs mlx-batched", qkvLast, qkvBatched),
            note("Q metal vs mlx-fused pos1", metal.q, qFused),
            note("Q metal vs mlx-batched pos1", metal.q, qBatched),
            note("Q mlx-fused vs mlx-batched", qFused, qBatched),
        ]
        var firstBad: String?
        let checks: [(String, [Float], [Float])] = [
            ("qkv metal vs mlx-last", metal.qkv, qkvLast),
            ("Q metal vs mlx-fused pos1", metal.q, qFused),
            ("Q metal vs mlx-batched pos1", metal.q, qBatched),
        ]
        for (label, a, b) in checks {
            if firstBad == nil, cosine(a, b) < 0.995 { firstBad = label }
        }
        XCTAssertNil(firstBad, "L0 split mismatch at \(firstBad!) " + notes.joined(separator: " | "))
    }
}

final class ChatTemplatePatchTests: XCTestCase {
    func testWithoutForcedThinkRemovesThinkOpener() {
        // Maple jinja uses escaped \n inside the quoted string.
        let fileStyle = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<think>\\n' }}\n{%- endif %}\n"
        let patched = ChatTemplatePatch.withoutForcedThink(fileStyle)
        XCTAssertFalse(patched.contains("<think>"))
        XCTAssertTrue(patched.contains("<|im_start|>assistant\\n' }}"))
    }

    func testStripThinkForContentKeepsAnswer() {
        let text = "<think>\nplan\n</think>\nhello"
        XCTAssertEqual(
            ChatTemplatePatch.stripThinkForContent(text, thinkingEnabled: false),
            "hello")
        XCTAssertEqual(
            ChatTemplatePatch.stripThinkForContent(text, thinkingEnabled: true),
            text)
    }

    func testBannedAssistantTokensWhenThinkingOff() {
        XCTAssertTrue(ChatTemplatePatch.bannedAssistantTokenIds.contains(151_667))
        XCTAssertTrue(ChatTemplatePatch.bannedAssistantTokenIds.contains(151_644))
    }

    func testStripThinkDropsUnclosedBlock() {
        XCTAssertEqual(
            ChatTemplatePatch.stripThinkForContent("<think>\nAvailable. Available.", thinkingEnabled: false),
            "")
        XCTAssertEqual(
            ChatTemplatePatch.stripThinkForContent("ok <think>secret", thinkingEnabled: false),
            "ok")
    }

    func testServeTemplatePatchesMapleJinja() throws {
        let modelDir = ProcessInfo.processInfo.environment["MOMIJ_MODEL"]
            ?? NSString("~/models/deepgrove/maple-preview-2bit-mlx").expandingTildeInPath
        guard let raw = ChatTemplatePatch.loadModelTemplate(modelDir: modelDir) else {
            throw XCTSkip("chat_template.jinja not present")
        }
        XCTAssertTrue(raw.contains("<think>"), "stock Maple template must force think")
        let patched = ChatTemplatePatch.withoutForcedThink(raw)
        XCTAssertFalse(
            patched.contains("<|im_start|>assistant\\n<think>\\n"),
            "generation prompt must not force think")
        XCTAssertTrue(patched.contains("<|im_start|>assistant\\n' }}"))
        // Historical reasoning_content path may still mention <think>; that is OK.
    }
}


