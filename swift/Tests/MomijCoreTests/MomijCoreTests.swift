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
        let dens = buf(1, 4)
        dens.contents().storeBytes(of: Int32(0), as: Int32.self)
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
