import Foundation
import Metal
import MLX

/// Raw Metal kernels for Maple ternary (2-bit affine) decode hot path.
///
/// `gqmm2_rows` is ported from Qwisp Seedless (MLX gather_qmv_fast bits=2 layout)
/// with `group_size` 64|128. Fused expert = up_gate gather → clamped SwiGLU →
/// down gather → score reduce, encoded on one command buffer.
public enum SeedlessMetal {
    nonisolated(unsafe) static var device: MTLDevice?
    nonisolated(unsafe) static var queue: MTLCommandQueue?
    nonisolated(unsafe) static var gqmm2Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var swigluPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var scoreReducePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var rmsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var residPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gateGemvPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var routeTop8Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var qkNormRopePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var writeKVPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var sdpaPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var stopBuf: MTLBuffer?
    nonisolated(unsafe) public static var ready = false

    public static func ensureCompiled() throws {
        if ready { return }
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { throw SeedlessError.noMetal }
        device = dev
        queue = q
        stopBuf = dev.makeBuffer(length: 4, options: .storageModeShared)
        stopBuf?.contents().storeBytes(of: Int32(0), as: Int32.self)

        let opts = mlxMatchCompileOpts()
        let lib = try dev.makeLibrary(source: metalSource, options: opts)
        func pipe(_ name: String) throws -> MTLComputePipelineState {
            try dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        }
        gqmm2Pipeline = try pipe("gqmm2_rows")
        swigluPipeline = try pipe("maple_clamped_swiglu")
        scoreReducePipeline = try pipe("maple_score_reduce")
        rmsPipeline = try pipe("maple_rms_norm")
        residPipeline = try pipe("maple_resid_add")
        gateGemvPipeline = try pipe("maple_gate_gemv")
        routeTop8Pipeline = try pipe("maple_route_top8")
        qkNormRopePipeline = try pipe("maple_qk_norm_rope")
        writeKVPipeline = try pipe("maple_write_kv")
        sdpaPipeline = try pipe("maple_sdpa_d128")
        ready = true
    }

    static func mlxMatchCompileOpts() -> MTLCompileOptions {
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) { opts.mathMode = .safe }
        else { opts.fastMathEnabled = false }
        return opts
    }

    static func mtlBuf(_ a: MLXArray, _ device: MTLDevice) -> MTLBuffer? {
        if let b = a.asMTLBuffer(device: device, noCopy: true) { return b }
        return a.asMTLBuffer(device: device, noCopy: false)
    }

    // MARK: - Public API

    /// 2-bit gather-qmv: x[1,K], w[E,N,K/16], scales/biases[E,N,K/gs], inds[Ktop] → y[Ktop,N]
    public static func gqmm2(
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, inds: MTLBuffer,
        out: MTLBuffer,
        Ktop: Int, K: Int, N: Int, gs: Int = 128, lhsPerExpert: Bool = false,
        into encoder: MTLComputeCommandEncoder? = nil,
        commandQueue: MTLCommandQueue? = nil
    ) throws {
        try ensureCompiled()
        guard let pipe = gqmm2Pipeline, let stop = stopBuf else { throw SeedlessError.notReady }
        guard N % 8 == 0, K % 512 == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: N, K: K, gs: gs)
        }

        let ownsCB = encoder == nil
        let q = commandQueue ?? queue!
        let cb = ownsCB ? q.makeCommandBuffer()! : nil
        let enc = encoder ?? cb!.makeComputeCommandEncoder()!

        enc.setComputePipelineState(pipe)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(inds, offset: 0, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 6)
        enc.setBytes(&nn, length: 4, index: 7)
        enc.setBytes(&kt, length: 4, index: 8)
        enc.setBuffer(stop, offset: 0, index: 9)
        var lp = UInt32(lhsPerExpert ? 1 : 0)
        enc.setBytes(&lp, length: 4, index: 10)
        var gsv = Int32(gs)
        enc.setBytes(&gsv, length: 4, index: 11)
        // grid: width=1, height=N/8, depth=M*Ktop (M=1)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: N / 8, depth: Ktop),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        if ownsCB {
            enc.endEncoding()
            cb!.commit()
            cb!.waitUntilCompleted()
        }
    }

    /// Encode fused expert block into an existing encoder (no commit/wait).
    public static func encodeFusedExpert(
        into enc: MTLComputeCommandEncoder,
        x: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, y: MTLBuffer,
        H: Int, I: Int, Ktop: Int, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let swiglu = swigluPipeline, let reduce = scoreReducePipeline, let stop = stopBuf
        else { throw SeedlessError.notReady }

        try gqmm2(x: x, w: upGateW, scales: upGateS, biases: upGateB, inds: inds, out: ugOut,
                  Ktop: Ktop, K: H, N: 2 * I, gs: gs, lhsPerExpert: false, into: enc)

        enc.setComputePipelineState(swiglu)
        enc.setBuffer(ugOut, offset: 0, index: 0)
        enc.setBuffer(act, offset: 0, index: 1)
        var i32 = Int32(I), kt32 = Int32(Ktop)
        enc.setBytes(&i32, length: 4, index: 2)
        enc.setBytes(&kt32, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: I, height: Ktop, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, I), height: 1, depth: 1))

        try gqmm2(x: act, w: downW, scales: downS, biases: downB, inds: inds, out: downOut,
                  Ktop: Ktop, K: I, N: H, gs: gs, lhsPerExpert: true, into: enc)

        enc.setComputePipelineState(reduce)
        enc.setBuffer(downOut, offset: 0, index: 0)
        enc.setBuffer(scores, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var h32 = Int32(H)
        enc.setBytes(&h32, length: 4, index: 3)
        enc.setBytes(&kt32, length: 4, index: 4)
        enc.setBuffer(stop, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    /// One-token fused MoE expert block on a single command buffer (wait once).
    public static func fusedExpertStep(
        x: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, y: MTLBuffer,
        H: Int, I: Int, Ktop: Int, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let q = queue else { throw SeedlessError.notReady }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try encodeFusedExpert(
            into: enc, x: x,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, y: y,
            H: H, I: I, Ktop: Ktop, gs: gs)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Encode one MoE block into an existing encoder (no commit/wait).
    public static func encodeMoEBlock(
        into enc: MTLComputeCommandEncoder,
        h: MTLBuffer, normW: MTLBuffer, gateW: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        xNorm: MTLBuffer, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, moeOut: MTLBuffer,
        H: Int, I: Int, E: Int, Ktop: Int, eps: Float, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let rms = rmsPipeline, let resid = residPipeline,
              let gemv = gateGemvPipeline, let route = routeTop8Pipeline,
              let stop = stopBuf
        else { throw SeedlessError.notReady }

        enc.setComputePipelineState(rms)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(normW, offset: 0, index: 1)
        enc.setBuffer(xNorm, offset: 0, index: 2)
        var epsV = eps, h32 = Int32(H)
        enc.setBytes(&epsV, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(gemv)
        enc.setBuffer(gateW, offset: 0, index: 0)
        enc.setBuffer(xNorm, offset: 0, index: 1)
        enc.setBuffer(logits, offset: 0, index: 2)
        var e32 = Int32(E)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: E, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(route)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(inds, offset: 0, index: 1)
        enc.setBuffer(scores, offset: 0, index: 2)
        enc.setBytes(&e32, length: 4, index: 3)
        var k32 = Int32(Ktop)
        enc.setBytes(&k32, length: 4, index: 4)
        enc.setBuffer(stop, offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        try encodeFusedExpert(
            into: enc, x: xNorm,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, y: moeOut,
            H: H, I: I, Ktop: Ktop, gs: gs)

        enc.setComputePipelineState(resid)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(moeOut, offset: 0, index: 1)
        enc.setBytes(&h32, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    public static func moeBlockOneCB(
        h: MTLBuffer, normW: MTLBuffer, gateW: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        xNorm: MTLBuffer, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, moeOut: MTLBuffer,
        H: Int, I: Int, E: Int, Ktop: Int, eps: Float, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let q = queue else { throw SeedlessError.notReady }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try encodeMoEBlock(
            into: enc, h: h, normW: normW, gateW: gateW,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xNorm: xNorm, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    public static func encodeResid(
        into enc: MTLComputeCommandEncoder, h: MTLBuffer, delta: MTLBuffer, H: Int
    ) {
        enc.setComputePipelineState(residPipeline!)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        var h32 = Int32(H)
        enc.setBytes(&h32, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    /// Decode attn body: xNorm[H] → attnOut[H]. Writes K/V at `pos` into caches.
    /// qkvW layout: single matrix as E=1 for gqmm2 (inds=[0]).
    public static func encodeAttnBlock(
        into enc: MTLComputeCommandEncoder,
        xNorm: MTLBuffer,
        qkvW: MTLBuffer, qkvS: MTLBuffer, qkvB: MTLBuffer,
        oW: MTLBuffer, oS: MTLBuffer, oB: MTLBuffer,
        qkW: MTLBuffer, invFreq: MTLBuffer, densInds: MTLBuffer,
        qkvOut: MTLBuffer, qkOut: MTLBuffer, attnTmp: MTLBuffer, attnOut: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        H: Int, numHeads: Int, numKV: Int, headDim: Int,
        ropeDim: Int, pos: Int, maxLen: Int, seqLen: Int, eps: Float, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let qkPipe = qkNormRopePipeline, let writeKV = writeKVPipeline, let sdpa = sdpaPipeline
        else { throw SeedlessError.notReady }

        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let qkvN = qDim + 2 * kvDim  // 3072

        try gqmm2(x: xNorm, w: qkvW, scales: qkvS, biases: qkvB, inds: densInds, out: qkvOut,
                  Ktop: 1, K: H, N: qkvN, gs: gs, lhsPerExpert: false, into: enc)

        // qk_norm_rope on first (numHeads+numKV) heads of qkv → qkOut
        enc.setComputePipelineState(qkPipe)
        enc.setBuffer(qkvOut, offset: 0, index: 0)
        enc.setBuffer(qkW, offset: 0, index: 1)
        enc.setBuffer(qkOut, offset: 0, index: 2)
        enc.setBuffer(invFreq, offset: 0, index: 3)
        var posF = Float(pos), epsV = eps, rope = Int32(ropeDim)
        enc.setBytes(&posF, length: 4, index: 4)
        enc.setBytes(&epsV, length: 4, index: 5)
        enc.setBytes(&rope, length: 4, index: 6)
        let nQK = numHeads + numKV
        enc.dispatchThreadgroups(MTLSize(width: 1, height: nQK, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // write K (qkOut after Q) and V (raw from qkvOut after qk)
        var kv32 = Int32(numKV), d32 = Int32(headDim), ml32 = Int32(maxLen), p32 = Int32(pos)
        enc.setComputePipelineState(writeKV)
        enc.setBuffer(qkOut, offset: qDim * 2, index: 0)  // K starts after Q
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBytes(&kv32, length: 4, index: 2)
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&ml32, length: 4, index: 4)
        enc.setBytes(&p32, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: kvDim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, kvDim), height: 1, depth: 1))

        enc.setComputePipelineState(writeKV)
        enc.setBuffer(qkvOut, offset: (qDim + kvDim) * 2, index: 0)  // V after Q+K
        enc.setBuffer(vCache, offset: 0, index: 1)
        enc.setBytes(&kv32, length: 4, index: 2)
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&ml32, length: 4, index: 4)
        enc.setBytes(&p32, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: kvDim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, kvDim), height: 1, depth: 1))

        // SDPA: Q from qkOut[0..<qDim], K/V caches, N=seqLen
        enc.setComputePipelineState(sdpa)
        enc.setBuffer(qkOut, offset: 0, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBuffer(vCache, offset: 0, index: 2)
        enc.setBuffer(attnTmp, offset: 0, index: 3)
        var gqa = Int32(numHeads / numKV), n32 = Int32(seqLen)
        var khs = Int32(maxLen * headDim), kss = Int32(headDim)
        var vhs = Int32(maxLen * headDim), vss = Int32(headDim)
        var sc = Float(pow(Double(headDim), -0.5))
        enc.setBytes(&gqa, length: 4, index: 4)
        enc.setBytes(&n32, length: 4, index: 5)
        enc.setBytes(&khs, length: 4, index: 6)
        enc.setBytes(&kss, length: 4, index: 7)
        enc.setBytes(&vhs, length: 4, index: 8)
        enc.setBytes(&vss, length: 4, index: 9)
        enc.setBytes(&sc, length: 4, index: 10)
        // 32 simdgroups × 32 lanes = 1024
        enc.dispatchThreadgroups(MTLSize(width: numHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))

        try gqmm2(x: attnTmp, w: oW, scales: oS, biases: oB, inds: densInds, out: attnOut,
                  Ktop: 1, K: qDim, N: H, gs: gs, lhsPerExpert: false, into: enc)
    }

    /// One full layer decode: input_rms → attn → resid → MoE block (post_rms+experts+resid).
    public static func encodeLayerBlock(
        into enc: MTLComputeCommandEncoder,
        h: MTLBuffer, inNorm: MTLBuffer, postNorm: MTLBuffer, gateW: MTLBuffer,
        qkvW: MTLBuffer, qkvS: MTLBuffer, qkvB: MTLBuffer,
        oW: MTLBuffer, oS: MTLBuffer, oB: MTLBuffer,
        qkW: MTLBuffer, invFreq: MTLBuffer, densInds: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        xAttn: MTLBuffer, qkvOut: MTLBuffer, qkOut: MTLBuffer, attnTmp: MTLBuffer, attnOut: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        xMoe: MTLBuffer, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, moeOut: MTLBuffer,
        H: Int, I: Int, E: Int, Ktop: Int, eps: Float, gs: Int,
        numHeads: Int, numKV: Int, headDim: Int, ropeDim: Int,
        pos: Int, maxLen: Int, seqLen: Int
    ) throws {
        encodeRms(into: enc, h: h, w: inNorm, out: xAttn, H: H, eps: eps)
        try encodeAttnBlock(
            into: enc, xNorm: xAttn,
            qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
            oW: oW, oS: oS, oB: oB,
            qkW: qkW, invFreq: invFreq, densInds: densInds,
            qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
            kCache: kCache, vCache: vCache,
            H: H, numHeads: numHeads, numKV: numKV, headDim: headDim,
            ropeDim: ropeDim, pos: pos, maxLen: maxLen, seqLen: seqLen, eps: eps, gs: gs)
        encodeResid(into: enc, h: h, delta: attnOut, H: H)
        try encodeMoEBlock(
            into: enc, h: h, normW: postNorm, gateW: gateW,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xNorm: xMoe, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs)
    }

    /// Wall + GPU time for one encoder body. GPU time is `gpuEndTime - gpuStartTime`.
    static func timeCB(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> (wallMs: Double, gpuMs: Double) {
        try ensureCompiled()
        guard let q = queue else { throw SeedlessError.notReady }
        let t0 = CFAbsoluteTimeGetCurrent()
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let wallMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        return (wallMs, gpuMs)
    }

    static func encodeRms(
        into enc: MTLComputeCommandEncoder, h: MTLBuffer, w: MTLBuffer, out: MTLBuffer,
        H: Int, eps: Float
    ) {
        enc.setComputePipelineState(rmsPipeline!)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(w, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var epsV = eps, h32 = Int32(H)
        enc.setBytes(&epsV, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    static func encodeGate(
        into enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
        E: Int, H: Int
    ) {
        enc.setComputePipelineState(gateGemvPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var e32 = Int32(E), h32 = Int32(H)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: E, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    static func encodeRoute(
        into enc: MTLComputeCommandEncoder, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        E: Int, Ktop: Int
    ) {
        enc.setComputePipelineState(routeTop8Pipeline!)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(inds, offset: 0, index: 1)
        enc.setBuffer(scores, offset: 0, index: 2)
        var e32 = Int32(E), k32 = Int32(Ktop)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&k32, length: 4, index: 4)
        enc.setBuffer(stopBuf, offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Benches

    public static func benchQmv2(H: Int = 2048, N: Int = 512, iters: Int = 200) throws -> Double {
        try ensureCompiled()
        guard let device, let queue else { throw SeedlessError.notReady }
        let gs = 128
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let E = 8, Ktop = 1

        let x = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        let w = device.makeBuffer(length: E * N * packedK * 4, options: .storageModeShared)!
        let s = device.makeBuffer(length: E * N * nGroups * 2, options: .storageModeShared)!
        let b = device.makeBuffer(length: E * N * nGroups * 2, options: .storageModeShared)!
        let inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        inds.contents().storeBytes(of: Int32(0), as: Int32.self)
        let y = device.makeBuffer(length: Ktop * N * 2, options: .storageModeShared)!

        for _ in 0 ..< 5 {
            try gqmm2(x: x, w: w, scales: s, biases: b, inds: inds, out: y,
                      Ktop: Ktop, K: H, N: N, gs: gs)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters {
            try gqmm2(x: x, w: w, scales: s, biases: b, inds: inds, out: y,
                      Ktop: Ktop, K: H, N: N, gs: gs)
        }
        _ = queue  // silence
        return Double(iters) / (CFAbsoluteTimeGetCurrent() - t0)
    }

    public static func benchFusedExpert(
        H: Int = 2048, I: Int = 512, E: Int = 256, Ktop: Int = 8, iters: Int = 100
    ) throws -> Double {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let gs = 128
        let packedH = H * 2 / 32
        let packedI = I * 2 / 32
        let nGroupsH = H / gs
        let nGroupsI = I / gs

        let x = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        let ugW = device.makeBuffer(length: E * 2 * I * packedH * 4, options: .storageModeShared)!
        let ugS = device.makeBuffer(length: E * 2 * I * nGroupsH * 2, options: .storageModeShared)!
        let ugB = device.makeBuffer(length: E * 2 * I * nGroupsH * 2, options: .storageModeShared)!
        let dW = device.makeBuffer(length: E * H * packedI * 4, options: .storageModeShared)!
        let dS = device.makeBuffer(length: E * H * nGroupsI * 2, options: .storageModeShared)!
        let dB = device.makeBuffer(length: E * H * nGroupsI * 2, options: .storageModeShared)!
        let inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let scores = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        let sp = scores.contents().bindMemory(to: Float.self, capacity: Ktop)
        for i in 0 ..< Ktop { ip[i] = Int32(i % E); sp[i] = 1.0 / Float(Ktop) }

        let ugOut = device.makeBuffer(length: Ktop * 2 * I * 2, options: .storageModeShared)!
        let act = device.makeBuffer(length: Ktop * I * 2, options: .storageModeShared)!
        let downOut = device.makeBuffer(length: Ktop * H * 2, options: .storageModeShared)!
        let y = device.makeBuffer(length: H * 2, options: .storageModeShared)!

        for _ in 0 ..< 3 {
            try fusedExpertStep(
                x: x, upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters {
            try fusedExpertStep(
                x: x, upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        return Double(iters) / (CFAbsoluteTimeGetCurrent() - t0)
    }

    // MARK: - Metal source

    private static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;
    #define SIMD_SIZE 32

    inline float ld16_b2(const device half* x, thread float* xt) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            xt[i]   = x[i];
            xt[i+1] = x[i+1] / 4.0f;
            xt[i+2] = x[i+2] / 16.0f;
            xt[i+3] = x[i+3] / 64.0f;
        }
        return sum;
    }
    inline float qd2(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
        float accum = 0.0f;
        for (int i = 0; i < 4; i++) {
            accum += (float)(w[i] & 0x03) * xt[4*i]
                   + (float)(w[i] & 0x0c) * xt[4*i+1]
                   + (float)(w[i] & 0x30) * xt[4*i+2]
                   + (float)(w[i] & 0xc0) * xt[4*i+3];
        }
        return scale * accum + sum * bias;
    }

    // Qwisp gqmm2_rows (bits=2 affine gather-qmv). gs=64|128.
    kernel void gqmm2_rows(
        device const uint32_t* w      [[buffer(0)]],
        device const half*     scales [[buffer(1)]],
        device const half*     biases [[buffer(2)]],
        device const half*     x      [[buffer(3)]],
        device const int*      inds   [[buffer(4)]],
        device half*           y      [[buffer(5)]],
        constant int& in_vec_size  [[buffer(6)]],
        constant int& out_vec_size [[buffer(7)]],
        constant int& ktop         [[buffer(8)]],
        device const int* stopFlag [[buffer(9)]],
        constant uint& lhsPer      [[buffer(10)]],
        constant int&  gsz         [[buffer(11)]],
        uint3 tid      [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]])
    {
        if (stopFlag[0] != 0) return;
        constexpr int packs_per_thread = 1, num_simdgroups = 2, results_per_simdgroup = 4;
        constexpr int pack_factor = 16, bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512;
        const int scale_step_per_thread = gsz / values_per_thread;
        const device uint8_t* ws = (const device uint8_t*)w;
        thread float x_thread[16];
        thread float result[4] = {0};
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        biases += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)mk * out_vec_size + out_row;
        for (int k = 0; k < in_vec_size; k += block_size) {
            float sum = ld16_b2(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device half* sl = scales + row * in_vec_size_g;
                const device half* bl = biases + row * in_vec_size_g;
                result[row] += qd2(wl, x_thread, sl[0], bl[0], sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / gsz; biases += block_size / gsz; x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }

    kernel void maple_clamped_swiglu(
        device const half* ug [[buffer(0)]],  // [Ktop, 2I] = up || gate
        device half* act       [[buffer(1)]],  // [Ktop, I]
        constant int& I        [[buffer(2)]],
        constant int& Ktop     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint j = gid.x, ki = gid.y;
        if (j >= (uint)I || ki >= (uint)Ktop) return;
        float up = ug[ki * (2 * I) + j];
        float gate = ug[ki * (2 * I) + I + j];
        gate = metal::min(gate, 7.0f);
        up = metal::clamp(up, -7.0f, 7.0f);
        float silu = gate / (1.0f + metal::exp(-gate));
        act[ki * I + j] = half(silu * up);
    }

    kernel void maple_score_reduce(
        device const half* down [[buffer(0)]],   // [Ktop, H]
        device const float* scores [[buffer(1)]], // [Ktop]
        device half* y [[buffer(2)]],             // [H]
        constant int& H [[buffer(3)]],
        constant int& Ktop [[buffer(4)]],
        device const int* stopFlag [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if (stopFlag[0] != 0) return;
        if (gid >= (uint)H) return;
        float acc = 0.0f;
        for (int ki = 0; ki < Ktop; ++ki) {
            acc += float(down[ki * H + gid]) * scores[ki];
        }
        y[gid] = half(acc);
    }

    // Milestone A helpers: RMSNorm / residual / dense gate / top-8 route.
    kernel void maple_rms_norm(
        device const half* x [[buffer(0)]],
        device const half* w [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant float& eps [[buffer(3)]],
        constant int& H [[buffer(4)]],
        uint lid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        threadgroup float red[256];
        float acc = 0.0f;
        for (uint i = lid; i < (uint)H; i += tgs) {
            float xi = float(x[i]);
            acc += xi * xi;
        }
        red[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (lid < s) red[lid] += red[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = precise::rsqrt(red[0] / float(H) + eps);
        for (uint i = lid; i < (uint)H; i += tgs) {
            out[i] = half(float(x[i]) * inv * float(w[i]));
        }
    }

    kernel void maple_resid_add(
        device half* h [[buffer(0)]],
        device const half* delta [[buffer(1)]],
        constant int& H [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= (uint)H) return;
        h[gid] = half(float(h[gid]) + float(delta[gid]));
    }

    // Dense gate: y[e] = sum_k W[e,k] * x[k]  (1 TG / expert, parallel reduce)
    kernel void maple_gate_gemv(
        device const half* W [[buffer(0)]],  // [E, H]
        device const half* x [[buffer(1)]],  // [H]
        device float* y [[buffer(2)]],       // [E]
        constant int& E [[buffer(3)]],
        constant int& H [[buffer(4)]],
        uint e [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        if (e >= (uint)E) return;
        threadgroup float red[256];
        const device half* row = W + (size_t)e * (size_t)H;
        float acc = 0.0f;
        for (uint k = tid; k < (uint)H; k += tgs) {
            acc += float(row[k]) * float(x[k]);
        }
        red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) y[e] = red[0];
    }

    // Softmax + top-K with renorm. E<=256, K<=8. scores are float (for score_reduce).
    kernel void maple_route_top8(
        device const float* logits [[buffer(0)]],
        device int* inds [[buffer(1)]],
        device float* scores [[buffer(2)]],
        constant int& E [[buffer(3)]],
        constant int& K [[buffer(4)]],
        device const int* stopFlag [[buffer(5)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        if (stopFlag[0] != 0) return;
        threadgroup float red[256];
        threadgroup int redi[256];
        threadgroup float gates[256];
        threadgroup float work[256];
        threadgroup float bcast[1];
        float lg = (tid < (uint)E) ? logits[tid] : -INFINITY;
        red[tid] = lg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = max(red[tid], red[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) bcast[0] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m = bcast[0];
        float e = (tid < (uint)E) ? precise::exp(lg - m) : 0.0f;
        red[tid] = e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) bcast[0] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float Z = bcast[0];
        if (tid < (uint)E) { gates[tid] = e / Z; work[tid] = lg; }
        else { work[tid] = -INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < K; k++) {
            red[tid] = work[tid];
            redi[tid] = (int)tid;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (red[tid + s] > red[tid]) {
                        red[tid] = red[tid + s];
                        redi[tid] = redi[tid + s];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                int bi = redi[0];
                inds[k] = bi;
                scores[k] = gates[bi];
                work[bi] = -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            float ss = 0.0f;
            for (int k = 0; k < K; k++) ss += scores[k];
            for (int k = 0; k < K; k++) scores[k] = scores[k] / ss;
        }
    }

    // Milestone B: decode attention (HEAD_DIM=128, GQA).
    kernel void maple_qk_norm_rope(
        device const half* x [[buffer(0)]],      // [nHeads, 128]
        device const half* w [[buffer(1)]],
        device half* out [[buffer(2)]],
        device const float* inv_freq [[buffer(3)]],
        constant float& pos [[buffer(4)]],
        constant float& eps [[buffer(5)]],
        constant int& ropeDim [[buffer(6)]],
        uint2 tid [[thread_position_in_grid]])
    {
        constexpr int HEAD_DIM = 128;
        constexpr int per_lane = HEAD_DIM / 32;
        uint head = tid.y;
        uint lane = tid.x;
        const device half* xh = x + head * HEAD_DIM;
        const device half* wh = w + head * HEAD_DIM;
        device half* oh = out + head * HEAD_DIM;
        float ss = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            float v = float(xh[lane * per_lane + i]);
            ss += v * v;
        }
        ss = simd_sum(ss);
        float scale = precise::rsqrt(ss / float(HEAD_DIM) + eps);
        for (int i = 0; i < per_lane; ++i) {
            int j = int(lane * per_lane + i);
            float v = float(xh[j]) * scale * float(wh[j]);
            if (ropeDim > 0 && j < ropeDim) {
                int rhalf = ropeDim / 2;
                int p = j < rhalf ? j : j - rhalf;
                float theta = pos * inv_freq[p];
                float c = metal::cos(theta);
                float s = metal::sin(theta);
                int j2 = j < rhalf ? j + rhalf : j - rhalf;
                float u = float(xh[j2]) * scale * float(wh[j2]);
                v = j < rhalf ? (v * c - u * s) : (v * c + u * s);
            }
            oh[j] = half(v);
        }
    }

    kernel void maple_write_kv(
        device const half* src [[buffer(0)]],
        device half* cache [[buffer(1)]],
        constant int& KV [[buffer(2)]],
        constant int& D [[buffer(3)]],
        constant int& maxLen [[buffer(4)]],
        constant int& pos [[buffer(5)]],
        uint i [[thread_position_in_grid]])
    {
        if (i >= (uint)(KV * D)) return;
        uint h = i / (uint)D, d = i % (uint)D;
        cache[(size_t)h * (size_t)maxLen * (size_t)D + (size_t)pos * (size_t)D + d] = src[h * D + d];
    }

    // Online-softmax SDPA decode, D=V=128 (Qwisp sdpa templated down from 256).
    kernel void maple_sdpa_d128(
        device const half* queries [[buffer(0)]],   // [H, 128]
        device const half* keys    [[buffer(1)]],   // [KV, maxLen, 128]
        device const half* values  [[buffer(2)]],
        device half* out           [[buffer(3)]],   // [H, 128]
        constant int& gqa_factor   [[buffer(4)]],
        constant int& N            [[buffer(5)]],
        constant int& k_head_stride[[buffer(6)]],
        constant int& k_seq_stride [[buffer(7)]],
        constant int& v_head_stride[[buffer(8)]],
        constant int& v_seq_stride [[buffer(9)]],
        constant float& scale      [[buffer(10)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint3 tpg [[threadgroups_per_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]])
    {
        constexpr int BN = 32, BD = 32, D = 128, V = 128;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        int inner_k_stride = BN * k_seq_stride;
        int inner_v_stride = BN * v_seq_stride;
        typedef float U;
        thread U q[qk_per_thread]; thread U k[qk_per_thread]; thread U o[v_per_thread];
        threadgroup U outputs[BN * BD];
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];
        const int q_batch_head_idx = tid.x;
        const int q_seq_idx = tid.y;
        const int kv_head_idx = q_batch_head_idx / gqa_factor;
        const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
        queries += o_offset * D + simd_lid * qk_per_thread;
        keys   += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
        values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * v_per_thread;
        out += o_offset * V + simd_gid * v_per_thread;
        for (int i = 0; i < qk_per_thread; i++) q[i] = (U)scale * (U)queries[i];
        for (int i = 0; i < v_per_thread; i++) o[i] = 0;
        U max_score = -INFINITY;
        U sum_exp_score = 0;
        for (int i = simd_gid; i < N; i += BN) {
            for (int j = 0; j < qk_per_thread; j++) k[j] = (U)keys[j];
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * (U)values[j];
            keys += inner_k_stride;
            values += inner_v_stride;
        }
        if (simd_lid == 0) { max_scores[simd_gid] = max_score; sum_exp_scores[simd_gid] = sum_exp_score; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_score = max_scores[simd_lid];
        U new_max = simd_max(max_score);
        U factor = fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
        for (int i = 0; i < v_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (simd_lid == 0) { for (int i = 0; i < v_per_thread; i++) out[i] = half(o[i]); }
    }
    """
}

public enum SeedlessError: Error, CustomStringConvertible {
    case noMetal, notReady
    case unsupportedShape(N: Int, K: Int, gs: Int)
    public var description: String {
        switch self {
        case .noMetal: return "no Metal device"
        case .notReady: return "SeedlessMetal not compiled"
        case let .unsupportedShape(N, K, gs): return "unsupported shape N=\(N) K=\(K) gs=\(gs)"
        }
    }
}

/// Bind Maple layer-0 expert weights into Seedless fused path and microbench.
public enum SeedlessEngine {
    public static func benchRealExpert(store: WeightStore, iters: Int = 50) throws -> Double {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        let cfg = store.config
        let H = cfg.hiddenSize, I = cfg.moeIntermediateSize, Ktop = cfg.numExpertsPerTok
        let gs = cfg.expertGroupSize

        let ugW = store.req("model.layers.0.mlp.switch_mlp.up_gate_proj.weight")
        let ugS = store.req("model.layers.0.mlp.switch_mlp.up_gate_proj.scales")
        let ugB = store.req("model.layers.0.mlp.switch_mlp.up_gate_proj.biases")
        let dW = store.req("model.layers.0.mlp.switch_mlp.down_proj.weight")
        let dS = store.req("model.layers.0.mlp.switch_mlp.down_proj.scales")
        let dB = store.req("model.layers.0.mlp.switch_mlp.down_proj.biases")
        MLX.eval([ugW, ugS, ugB, dW, dS, dB])

        guard let bugW = SeedlessMetal.mtlBuf(ugW, device),
              let bugS = SeedlessMetal.mtlBuf(ugS.asType(.float16), device),
              let bugB = SeedlessMetal.mtlBuf(ugB.asType(.float16), device),
              let bdW = SeedlessMetal.mtlBuf(dW, device),
              let bdS = SeedlessMetal.mtlBuf(dS.asType(.float16), device),
              let bdB = SeedlessMetal.mtlBuf(dB.asType(.float16), device)
        else { throw SeedlessError.notReady }

        let x = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        // fill x with ones
        let xp = x.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { xp[i] = 0.01 }

        let inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let scores = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        let sp = scores.contents().bindMemory(to: Float.self, capacity: Ktop)
        for i in 0 ..< Ktop { ip[i] = Int32(i); sp[i] = 1.0 / Float(Ktop) }

        let ugOut = device.makeBuffer(length: Ktop * 2 * I * 2, options: .storageModeShared)!
        let act = device.makeBuffer(length: Ktop * I * 2, options: .storageModeShared)!
        let downOut = device.makeBuffer(length: Ktop * H * 2, options: .storageModeShared)!
        let y = device.makeBuffer(length: H * 2, options: .storageModeShared)!

        for _ in 0 ..< 3 {
            try SeedlessMetal.fusedExpertStep(
                x: x, upGateW: bugW, upGateS: bugS, upGateB: bugB,
                downW: bdW, downS: bdS, downB: bdB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters {
            try SeedlessMetal.fusedExpertStep(
                x: x, upGateW: bugW, upGateS: bugS, upGateB: bugB,
                downW: bdW, downS: bdS, downB: bdB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        return Double(iters) / (CFAbsoluteTimeGetCurrent() - t0)
    }
}
