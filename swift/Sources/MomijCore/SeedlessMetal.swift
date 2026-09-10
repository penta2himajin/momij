import Foundation
import Metal
import MLX

/// Raw Metal kernels for Maple ternary (2-bit affine) decode hot path.
///
/// `gqmm2_rows` is ported from Qwisp Seedless (MLX gather_qmv_fast bits=2 layout)
/// with `group_size` 64|128. Expert path = up_gate gather → clamped SwiGLU →
/// down gather → score reduce on one command buffer. Optional `gqmm2_up_swiglu`
/// (`MOMIJ_FUSE_UP_SWIGLU=1`) fuses the first two; default keeps them separate (faster e2e).
/// Experimental (default off): `MOMIJ_GQMM2_SPLITK=1` (micro↑ e2e↓), `MOMIJ_GQMM2_W16=1` (mild).
/// `MOMIJ_GQMM2_TERNARY=1` Maple signed-trit path (separate metallib; packed/e2e regress).
/// Default gqmm2: Maple fold (hoisted row α, bias=-α, stock qd2). `MOMIJ_GQMM2_FOLD=0` restores per-group affine.
/// `MOMIJ_GQMM2_DEFER_A=1` applies α after simd_sum (separate metallib; opt-in).
/// `gqmm2_rows_vec` (vectorized loads) and `gqmm2_rows_pf` measured packed-CB regress — not shipped.
public enum SeedlessMetal {
    nonisolated(unsafe) static var device: MTLDevice?
    nonisolated(unsafe) static var queue: MTLCommandQueue?
    nonisolated(unsafe) static var gqmm2Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2W16Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2TernaryPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2FoldPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2DeferAPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2SplitKPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2ReduceSKPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2UpSwigluPipeline: MTLComputePipelineState?
    /// Scratch for split-K partials: [nChunks, Ktop, N] half. Grown on demand.
    nonisolated(unsafe) static var gqmm2SKPartials: MTLBuffer?
    nonisolated(unsafe) static var gqmm2SKPartialsBytes: Int = 0
    nonisolated(unsafe) static var swigluPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var scoreReducePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var rmsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var residPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gateGemvPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var batchedGemvPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var qmm4GatherPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var flashTopKChunkPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var flashTopKMergePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var routeTop8Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var qkNormRopePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var writeKVPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var shiftKVPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var sdpaPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var embedTokenPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var flashArgmaxTokenPipeline: MTLComputePipelineState?
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
        gqmm2W16Pipeline = try pipe("gqmm2_rows_w16")
        gqmm2SplitKPipeline = try pipe("gqmm2_rows_sk")
        gqmm2ReduceSKPipeline = try pipe("gqmm2_reduce_sk")
        gqmm2UpSwigluPipeline = try pipe("gqmm2_up_swiglu")
        swigluPipeline = try pipe("maple_clamped_swiglu")
        scoreReducePipeline = try pipe("maple_score_reduce")
        rmsPipeline = try pipe("maple_rms_norm")
        residPipeline = try pipe("maple_resid_add")
        gateGemvPipeline = try pipe("maple_gate_gemv")
        batchedGemvPipeline = try pipe("maple_batched_gemv")
        qmm4GatherPipeline = try pipe("maple_qmm4_gather")
        flashTopKChunkPipeline = try pipe("maple_flash_topk_chunk")
        flashTopKMergePipeline = try pipe("maple_flash_topk_merge")
        routeTop8Pipeline = try pipe("maple_route_top8")
        qkNormRopePipeline = try pipe("maple_qk_norm_rope")
        writeKVPipeline = try pipe("maple_write_kv")
        shiftKVPipeline = try pipe("maple_shift_kv")
        sdpaPipeline = try pipe("maple_sdpa_d128")
        embedTokenPipeline = try pipe("maple_embed_token")
        flashArgmaxTokenPipeline = try pipe("maple_flash_argmax_token")
        ready = true
        if useFold {
            let foldLib = try dev.makeLibrary(source: foldMetalSource, options: opts)
            gqmm2FoldPipeline = try dev.makeComputePipelineState(
                function: foldLib.makeFunction(name: "gqmm2_rows_fold")!)
        }
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

    /// When true, up gather + clamped SwiGLU run as one kernel.
    /// Default off: e2e favors the two-dispatch path; set `MOMIJ_FUSE_UP_SWIGLU=1` to enable.
    public static var fuseUpSwiglu: Bool {
        guard let raw = getenv("MOMIJ_FUSE_UP_SWIGLU") else { return false }
        return String(cString: raw) == "1"
    }

    /// Split-K gqmm2 (more TGs along K). Default off — micro↑ packed-CB↓.
    /// Env: `MOMIJ_GQMM2_SPLITK=1`.
    public static var useSplitK: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_SPLITK") else { return false }
        return String(cString: raw) == "1"
    }

    /// Wider TG: 16 output rows / TG (4 simdgroups). Env: `MOMIJ_GQMM2_W16=1`.
    public static var useW16: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_W16") else { return false }
        return String(cString: raw) == "1"
    }

    /// Maple ternary: `y = α · Σ(q−1)·x` (codes {0,1,2}, bias unused). Env: `MOMIJ_GQMM2_TERNARY=1`.
    /// Separate metallib (not in stock compile) — measured packed/e2e regress; opt-in only.
    public static var useTernary: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_TERNARY") else { return false }
        return String(cString: raw) == "1"
    }

    private static func ensureTernaryCompiled() throws {
        if gqmm2TernaryPipeline != nil { return }
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let lib = try device.makeLibrary(source: ternaryMetalSource, options: mlxMatchCompileOpts())
        gqmm2TernaryPipeline = try device.makeComputePipelineState(
            function: lib.makeFunction(name: "gqmm2_rows_ternary")!)
    }

    /// Maple fold: stock qd2/ld16_b2, hoisted row α, bias=-α. Default on.
    /// `MOMIJ_GQMM2_FOLD=0` restores per-group affine `gqmm2_rows`.
    public static var useFold: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_FOLD") else { return true }
        return String(cString: raw) != "0"
    }

    private static func ensureFoldCompiled() throws {
        if gqmm2FoldPipeline != nil { return }
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let lib = try device.makeLibrary(source: foldMetalSource, options: mlxMatchCompileOpts())
        gqmm2FoldPipeline = try device.makeComputePipelineState(
            function: lib.makeFunction(name: "gqmm2_rows_fold")!)
    }

    /// Stock ld16_b2 / masked accum; α after simd_sum. Env: `MOMIJ_GQMM2_DEFER_A=1`.
    public static var useDeferA: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_DEFER_A") else { return false }
        return String(cString: raw) == "1"
    }

    private static func ensureDeferACompiled() throws {
        if gqmm2DeferAPipeline != nil { return }
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let lib = try device.makeLibrary(source: deferAMetalSource, options: mlxMatchCompileOpts())
        gqmm2DeferAPipeline = try device.makeComputePipelineState(
            function: lib.makeFunction(name: "gqmm2_rows_defer_a")!)
    }

    private static let gqmm2BlockSize = 512

    private static func ensureSKPartials(bytes: Int) throws -> MTLBuffer {
        if let b = gqmm2SKPartials, gqmm2SKPartialsBytes >= bytes { return b }
        guard let device else { throw SeedlessError.notReady }
        let buf = device.makeBuffer(length: bytes, options: .storageModeShared)!
        gqmm2SKPartials = buf
        gqmm2SKPartialsBytes = bytes
        return buf
    }

    /// 2-bit gather-qmv: x[1,K], w[E,N,K/16], scales/biases[E,N,K/gs], inds[Ktop] → y[Ktop,N]
    public static func gqmm2(
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, inds: MTLBuffer,
        out: MTLBuffer,
        Ktop: Int, K: Int, N: Int, gs: Int = 128, lhsPerExpert: Bool = false,
        into encoder: MTLComputeCommandEncoder? = nil,
        commandQueue: MTLCommandQueue? = nil,
        splitK: Bool? = nil,
        w16: Bool? = nil,
        ternary: Bool? = nil,
        fold: Bool? = nil,
        deferA: Bool? = nil
    ) throws {
        try ensureCompiled()
        guard let stop = stopBuf else { throw SeedlessError.notReady }
        let doSK = splitK ?? useSplitK
        let doTernary = !doSK && (ternary ?? useTernary)
        let doDeferA = !doSK && !doTernary && (deferA ?? useDeferA)
        let doFold = !doSK && !doTernary && !doDeferA && (fold ?? useFold)
        let doW16 = !doSK && !doTernary && !doDeferA && !doFold && (w16 ?? useW16)
        let rowsPerTG = doW16 ? 16 : 8
        guard N % rowsPerTG == 0, K % gqmm2BlockSize == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: N, K: K, gs: gs)
        }
        if doTernary { try ensureTernaryCompiled() }
        if doDeferA { try ensureDeferACompiled() }
        if doFold { try ensureFoldCompiled() }

        let ownsCB = encoder == nil
        let q = commandQueue ?? queue!
        let cb = ownsCB ? q.makeCommandBuffer()! : nil
        let enc = encoder ?? cb!.makeComputeCommandEncoder()!

        if doSK {
            guard let sk = gqmm2SplitKPipeline, let red = gqmm2ReduceSKPipeline else {
                throw SeedlessError.notReady
            }
            let nChunks = K / gqmm2BlockSize
            let partials = try ensureSKPartials(bytes: nChunks * Ktop * N * 2)
            enc.setComputePipelineState(sk)
            enc.setBuffer(w, offset: 0, index: 0)
            enc.setBuffer(scales, offset: 0, index: 1)
            enc.setBuffer(biases, offset: 0, index: 2)
            enc.setBuffer(x, offset: 0, index: 3)
            enc.setBuffer(inds, offset: 0, index: 4)
            enc.setBuffer(partials, offset: 0, index: 5)
            var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
            enc.setBytes(&kk, length: 4, index: 6)
            enc.setBytes(&nn, length: 4, index: 7)
            enc.setBytes(&kt, length: 4, index: 8)
            enc.setBuffer(stop, offset: 0, index: 9)
            var lp = UInt32(lhsPerExpert ? 1 : 0)
            enc.setBytes(&lp, length: 4, index: 10)
            var gsv = Int32(gs)
            enc.setBytes(&gsv, length: 4, index: 11)
            // width = K-chunks → more TGs, one block each (no large TG cache).
            enc.dispatchThreadgroups(
                MTLSize(width: nChunks, height: N / 8, depth: Ktop),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

            enc.setComputePipelineState(red)
            enc.setBuffer(partials, offset: 0, index: 0)
            enc.setBuffer(out, offset: 0, index: 1)
            var nc = Int32(nChunks)
            enc.setBytes(&nn, length: 4, index: 2)
            enc.setBytes(&kt, length: 4, index: 3)
            enc.setBytes(&nc, length: 4, index: 4)
            enc.setBuffer(stop, offset: 0, index: 5)
            enc.dispatchThreads(
                MTLSize(width: N, height: Ktop, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, N), height: 1, depth: 1))
        } else {
            let pipe: MTLComputePipelineState
            if doTernary {
                guard let t = gqmm2TernaryPipeline else { throw SeedlessError.notReady }
                pipe = t
            } else if doDeferA {
                guard let d = gqmm2DeferAPipeline else { throw SeedlessError.notReady }
                pipe = d
            } else if doFold {
                guard let f = gqmm2FoldPipeline else { throw SeedlessError.notReady }
                pipe = f
            } else if doW16 {
                guard let w = gqmm2W16Pipeline else { throw SeedlessError.notReady }
                pipe = w
            } else {
                guard let p = gqmm2Pipeline else { throw SeedlessError.notReady }
                pipe = p
            }
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
            let tptg = doW16 ? 128 : 64
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: N / rowsPerTG, depth: Ktop),
                threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
        }

        if ownsCB {
            enc.endEncoding()
            cb!.commit()
            cb!.waitUntilCompleted()
        }
    }

    /// Fused up||gate gather-qmv + clamped SwiGLU → act[Ktop, I]. Weight rows = 2I (up||gate).
    public static func gqmm2UpSwiglu(
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, inds: MTLBuffer,
        out: MTLBuffer,
        Ktop: Int, K: Int, I: Int, gs: Int = 128,
        into encoder: MTLComputeCommandEncoder? = nil,
        commandQueue: MTLCommandQueue? = nil
    ) throws {
        try ensureCompiled()
        guard let pipe = gqmm2UpSwigluPipeline, let stop = stopBuf else { throw SeedlessError.notReady }
        guard I % 8 == 0, K % 512 == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: I, K: K, gs: gs)
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
        var kk = Int32(K), ii = Int32(I), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 6)
        enc.setBytes(&ii, length: 4, index: 7)
        enc.setBytes(&kt, length: 4, index: 8)
        enc.setBuffer(stop, offset: 0, index: 9)
        var gsv = Int32(gs)
        enc.setBytes(&gsv, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: I / 8, depth: Ktop),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        if ownsCB {
            enc.endEncoding()
            cb!.commit()
            cb!.waitUntilCompleted()
        }
    }

    public static func encodeClampedSwiglu(
        into enc: MTLComputeCommandEncoder,
        ug: MTLBuffer, act: MTLBuffer, I: Int, Ktop: Int
    ) {
        enc.setComputePipelineState(swigluPipeline!)
        enc.setBuffer(ug, offset: 0, index: 0)
        enc.setBuffer(act, offset: 0, index: 1)
        var i32 = Int32(I), kt32 = Int32(Ktop)
        enc.setBytes(&i32, length: 4, index: 2)
        enc.setBytes(&kt32, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: I, height: Ktop, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, I), height: 1, depth: 1))
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
        guard let reduce = scoreReducePipeline, let stop = stopBuf
        else { throw SeedlessError.notReady }

        if fuseUpSwiglu {
            try gqmm2UpSwiglu(x: x, w: upGateW, scales: upGateS, biases: upGateB, inds: inds,
                             out: act, Ktop: Ktop, K: H, I: I, gs: gs, into: enc)
        } else {
            try gqmm2(x: x, w: upGateW, scales: upGateS, biases: upGateB, inds: inds, out: ugOut,
                      Ktop: Ktop, K: H, N: 2 * I, gs: gs, lhsPerExpert: false, into: enc)
            encodeClampedSwiglu(into: enc, ug: ugOut, act: act, I: I, Ktop: Ktop)
        }

        try gqmm2(x: act, w: downW, scales: downS, biases: downB, inds: inds, out: downOut,
                  Ktop: Ktop, K: I, N: H, gs: gs, lhsPerExpert: true, into: enc)

        enc.setComputePipelineState(reduce)
        enc.setBuffer(downOut, offset: 0, index: 0)
        enc.setBuffer(scores, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var h32 = Int32(H)
        var kt32 = Int32(Ktop)
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

    /// Decode attn body. `ropePos` = absolute RoPE index; `writePos` = cache slot; `seqLen` = SDPA N.
    public static func encodeAttnBlock(
        into enc: MTLComputeCommandEncoder,
        xNorm: MTLBuffer,
        qkvW: MTLBuffer, qkvS: MTLBuffer, qkvB: MTLBuffer,
        oW: MTLBuffer, oS: MTLBuffer, oB: MTLBuffer,
        qkW: MTLBuffer, invFreq: MTLBuffer, densInds: MTLBuffer,
        qkvOut: MTLBuffer, qkOut: MTLBuffer, attnTmp: MTLBuffer, attnOut: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        H: Int, numHeads: Int, numKV: Int, headDim: Int,
        ropeDim: Int, ropePos: Int, writePos: Int, maxLen: Int, seqLen: Int,
        eps: Float, gs: Int = 128, rotateFirst: Bool = false
    ) throws {
        try ensureCompiled()
        guard let qkPipe = qkNormRopePipeline, let writeKV = writeKVPipeline, let sdpa = sdpaPipeline
        else { throw SeedlessError.notReady }

        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let qkvN = qDim + 2 * kvDim

        if rotateFirst {
            guard let shift = shiftKVPipeline else { throw SeedlessError.notReady }
            var kv32 = Int32(numKV), d32 = Int32(headDim), ml32 = Int32(maxLen)
            for cache in [kCache, vCache] {
                enc.setComputePipelineState(shift)
                enc.setBuffer(cache, offset: 0, index: 0)
                enc.setBytes(&kv32, length: 4, index: 1)
                enc.setBytes(&d32, length: 4, index: 2)
                enc.setBytes(&ml32, length: 4, index: 3)
                enc.dispatchThreads(
                    MTLSize(width: headDim, height: max(maxLen - 1, 1), depth: numKV),
                    threadsPerThreadgroup: MTLSize(width: min(32, headDim), height: 1, depth: 1))
            }
        }

        try gqmm2(x: xNorm, w: qkvW, scales: qkvS, biases: qkvB, inds: densInds, out: qkvOut,
                  Ktop: 1, K: H, N: qkvN, gs: gs, lhsPerExpert: false, into: enc)

        enc.setComputePipelineState(qkPipe)
        enc.setBuffer(qkvOut, offset: 0, index: 0)
        enc.setBuffer(qkW, offset: 0, index: 1)
        enc.setBuffer(qkOut, offset: 0, index: 2)
        enc.setBuffer(invFreq, offset: 0, index: 3)
        var posF = Float(ropePos), epsV = eps, rope = Int32(ropeDim)
        enc.setBytes(&posF, length: 4, index: 4)
        enc.setBytes(&epsV, length: 4, index: 5)
        enc.setBytes(&rope, length: 4, index: 6)
        let nQK = numHeads + numKV
        enc.dispatchThreadgroups(MTLSize(width: 1, height: nQK, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        var kv32 = Int32(numKV), d32 = Int32(headDim), ml32 = Int32(maxLen), p32 = Int32(writePos)
        enc.setComputePipelineState(writeKV)
        enc.setBuffer(qkOut, offset: qDim * 2, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBytes(&kv32, length: 4, index: 2)
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&ml32, length: 4, index: 4)
        enc.setBytes(&p32, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: kvDim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, kvDim), height: 1, depth: 1))

        enc.setComputePipelineState(writeKV)
        enc.setBuffer(qkvOut, offset: (qDim + kvDim) * 2, index: 0)
        enc.setBuffer(vCache, offset: 0, index: 1)
        enc.setBytes(&kv32, length: 4, index: 2)
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&ml32, length: 4, index: 4)
        enc.setBytes(&p32, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: kvDim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, kvDim), height: 1, depth: 1))

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
        enc.dispatchThreadgroups(MTLSize(width: numHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))

        try gqmm2(x: attnTmp, w: oW, scales: oS, biases: oB, inds: densInds, out: attnOut,
                  Ktop: 1, K: qDim, N: H, gs: gs, lhsPerExpert: false, into: enc)
    }

    /// One full layer decode: input_rms → attn → resid → MoE block.
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
        ropePos: Int, writePos: Int, maxLen: Int, seqLen: Int, rotateFirst: Bool
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
            ropeDim: ropeDim, ropePos: ropePos, writePos: writePos, maxLen: maxLen, seqLen: seqLen,
            eps: eps, gs: gs, rotateFirst: rotateFirst)
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

    /// `out[H] = embTable[ids[row], :]` (GPU gather; safe inside a multi-step CB).
    static func encodeEmbedToken(
        into enc: MTLComputeCommandEncoder,
        table: MTLBuffer, ids: MTLBuffer, out: MTLBuffer,
        H: Int, row: Int
    ) {
        enc.setComputePipelineState(embedTokenPipeline!)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(ids, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var h32 = Int32(H), r32 = Int32(row)
        enc.setBytes(&h32, length: 4, index: 3)
        enc.setBytes(&r32, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    /// Argmax over FlashHead gathered logits → token id via `tokenMap[inds[p]*C + row]`.
    /// Optional force-token rows (same rule as CPU `argmaxWithForce`).
    static func encodeFlashArgmaxToken(
        into enc: MTLComputeCommandEncoder,
        logits: MTLBuffer, inds: MTLBuffer, tokenMap: MTLBuffer, ids: MTLBuffer,
        nProbes: Int, clusterSize: Int, outRow: Int,
        h: MTLBuffer? = nil,
        forceRows: MTLBuffer? = nil, forceIds: MTLBuffer? = nil,
        nForce: Int = 0, H: Int = 0
    ) {
        enc.setComputePipelineState(flashArgmaxTokenPipeline!)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(inds, offset: 0, index: 1)
        enc.setBuffer(tokenMap, offset: 0, index: 2)
        enc.setBuffer(ids, offset: 0, index: 3)
        var np = Int32(nProbes), cs = Int32(clusterSize), or = Int32(outRow)
        enc.setBytes(&np, length: 4, index: 4)
        enc.setBytes(&cs, length: 4, index: 5)
        enc.setBytes(&or, length: 4, index: 6)
        enc.setBuffer(stopBuf, offset: 0, index: 7)
        // Optional force path buffers (dummy when nForce==0 — kernel ignores).
        enc.setBuffer(h ?? ids, offset: 0, index: 8)
        enc.setBuffer(forceRows ?? ids, offset: 0, index: 9)
        enc.setBuffer(forceIds ?? ids, offset: 0, index: 10)
        var nf = Int32(nForce), h32 = Int32(H)
        enc.setBytes(&nf, length: 4, index: 11)
        enc.setBytes(&h32, length: 4, index: 12)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Dense MoE gate gemv. Default: one simdgroup / row (`maple_batched_gemv`).
    /// `MOMIJ_GATE_SIMD=0`: legacy 256-thread TG reduce (`maple_gate_gemv`).
    public static var useGateSimd: Bool {
        guard let raw = getenv("MOMIJ_GATE_SIMD") else { return true }
        return String(cString: raw) != "0"
    }

    static func encodeGate(
        into enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
        E: Int, H: Int, threadsPerTG: Int = 256
    ) {
        if useGateSimd {
            encodeBatchedGemv(into: enc, w: w, x: x, y: y, E: E, H: H)
            return
        }
        enc.setComputePipelineState(gateGemvPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var e32 = Int32(E), h32 = Int32(H)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        let tpt = max(32, min(256, threadsPerTG))
        enc.dispatchThreadgroups(MTLSize(width: E, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tpt, height: 1, depth: 1))
    }

    /// Force a specific gate path for tests (`simd: true` → batched; `false` → TG reduce).
    static func encodeGate(
        into enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
        E: Int, H: Int, simd: Bool
    ) {
        if simd {
            encodeBatchedGemv(into: enc, w: w, x: x, y: y, E: E, H: H)
        } else {
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
    }

    /// Dense gemv y[E]=W[E,H]@x[H]. One simdgroup (32 threads) per row — no TG barriers.
    static func encodeBatchedGemv(
        into enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
        E: Int, H: Int, threadgroups: Int = 256, threadsPerTG: Int = 64
    ) {
        _ = threadgroups; _ = threadsPerTG
        enc.setComputePipelineState(batchedGemvPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var e32 = Int32(E), h32 = Int32(H)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: E, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// 4-bit affine gather-qmv: for each probe cluster `inds[p]`, score all N rows → y[p*N + r].
    static func encodeQmm4Gather(
        into enc: MTLComputeCommandEncoder,
        w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, x: MTLBuffer,
        inds: MTLBuffer, y: MTLBuffer,
        nProbes: Int, N: Int, K: Int, gs: Int
    ) {
        enc.setComputePipelineState(qmm4GatherPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(inds, offset: 0, index: 4)
        enc.setBuffer(y, offset: 0, index: 5)
        var n32 = Int32(N), k32 = Int32(K), gs32 = Int32(gs), np32 = Int32(nProbes)
        enc.setBytes(&n32, length: 4, index: 6)
        enc.setBytes(&k32, length: 4, index: 7)
        enc.setBytes(&gs32, length: 4, index: 8)
        enc.setBytes(&np32, length: 4, index: 9)
        // One TG per probe; N simdgroups × 32 threads (clusterSize=32 → 1024 threads).
        enc.dispatchThreadgroups(
            MTLSize(width: nProbes, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: N * 32, height: 1, depth: 1))
    }

    /// Hierarchical FlashHead top-k: chunk local tops → merge.
    /// `localTop` must be ≥ K for worst-case correctness (all top-K in one chunk).
    static let flashTopKChunk = 256

    static func encodeFlashTopK(
        into enc: MTLComputeCommandEncoder,
        scores: MTLBuffer, inds: MTLBuffer,
        candScores: MTLBuffer, candInds: MTLBuffer,
        E: Int, K: Int
    ) {
        let localTop = min(K, flashTopKChunk)
        let nChunks = (E + flashTopKChunk - 1) / flashTopKChunk
        var e32 = Int32(E), k32 = Int32(K)
        var chunk = Int32(flashTopKChunk), local = Int32(localTop)
        enc.setComputePipelineState(flashTopKChunkPipeline!)
        enc.setBuffer(scores, offset: 0, index: 0)
        enc.setBuffer(candScores, offset: 0, index: 1)
        enc.setBuffer(candInds, offset: 0, index: 2)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&chunk, length: 4, index: 4)
        enc.setBytes(&local, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: nChunks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: flashTopKChunk, height: 1, depth: 1))

        var nCand = Int32(nChunks * localTop)
        enc.setComputePipelineState(flashTopKMergePipeline!)
        enc.setBuffer(candScores, offset: 0, index: 0)
        enc.setBuffer(candInds, offset: 0, index: 1)
        enc.setBuffer(inds, offset: 0, index: 2)
        enc.setBytes(&nCand, length: 4, index: 3)
        enc.setBytes(&k32, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
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

    // Same math as gqmm2_rows; 4 simdgroups → 16 rows/TG (more x reuse, fewer TGs).
    kernel void gqmm2_rows_w16(
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
        constexpr int packs_per_thread = 1, num_simdgroups = 4, results_per_simdgroup = 4;
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

    // Split-K: one K-block (512) per tid.x → partials[chunk, mk, N]. No TG cache.
    kernel void gqmm2_rows_sk(
        device const uint32_t* w      [[buffer(0)]],
        device const half*     scales [[buffer(1)]],
        device const half*     biases [[buffer(2)]],
        device const half*     x      [[buffer(3)]],
        device const int*      inds   [[buffer(4)]],
        device half*           partials [[buffer(5)]],
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
        const int nChunks = in_vec_size / block_size;
        uint chunk = tid.x;
        if ((int)chunk >= nChunks) return;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        biases += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        const int k0 = (int)chunk * block_size;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack
                + k0 * bytes_per_pack / pack_factor;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread + k0 / gsz;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread + k0 / gsz;
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size
           + simd_lid * values_per_thread + k0;
        float sum = ld16_b2(x, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device half* sl = scales + row * in_vec_size_g;
            const device half* bl = biases + row * in_vec_size_g;
            result[row] += qd2(wl, x_thread, sl[0], bl[0], sum);
        }
        device half* y = partials
            + ((size_t)chunk * (size_t)ktop + (size_t)mk) * (size_t)out_vec_size
            + out_row;
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }

    kernel void gqmm2_reduce_sk(
        device const half* partials [[buffer(0)]],
        device half*       y        [[buffer(1)]],
        constant int& out_vec_size  [[buffer(2)]],
        constant int& ktop          [[buffer(3)]],
        constant int& nChunks       [[buffer(4)]],
        device const int* stopFlag  [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (stopFlag[0] != 0) return;
        uint col = gid.x;
        uint mk = gid.y;
        if ((int)col >= out_vec_size || (int)mk >= ktop) return;
        float acc = 0;
        for (int c = 0; c < nChunks; c++) {
            acc += (float)partials[((size_t)c * (size_t)ktop + (size_t)mk) * (size_t)out_vec_size + col];
        }
        y[(size_t)mk * (size_t)out_vec_size + col] = (half)acc;
    }

    // Up||gate gather-qmv fused with clamped SwiGLU. out_vec_size = I; weight has 2I rows.
    kernel void gqmm2_up_swiglu(
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
        constant int&  gsz         [[buffer(10)]],
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
        thread float result_up[4] = {0};
        thread float result_gate[4] = {0};
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        const int weight_rows = out_vec_size * 2;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * weight_rows * in_vec_size_w;
        scales += (size_t)e * weight_rows * in_vec_size_g;
        biases += (size_t)e * weight_rows * in_vec_size_g;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        const device uint8_t* ws_up = ws + out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        const device uint8_t* ws_gate = ws + (out_row + out_vec_size) * in_vec_size_w
                                      + simd_lid * packs_per_thread * bytes_per_pack;
        const device half* sc_up = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        const device half* sc_gate = scales + (out_row + out_vec_size) * in_vec_size_g
                                    + simd_lid / scale_step_per_thread;
        const device half* bi_up = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        const device half* bi_gate = biases + (out_row + out_vec_size) * in_vec_size_g
                                    + simd_lid / scale_step_per_thread;
        x += (size_t)(mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)mk * out_vec_size + out_row;
        for (int k = 0; k < in_vec_size; k += block_size) {
            float sum = ld16_b2(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl_u = (const device uint8_t*)(ws_up + row * in_vec_size_w);
                auto wl_g = (const device uint8_t*)(ws_gate + row * in_vec_size_w);
                const device half* slu = sc_up + row * in_vec_size_g;
                const device half* slg = sc_gate + row * in_vec_size_g;
                const device half* blu = bi_up + row * in_vec_size_g;
                const device half* blg = bi_gate + row * in_vec_size_g;
                result_up[row]   += qd2(wl_u, x_thread, slu[0], blu[0], sum);
                result_gate[row] += qd2(wl_g, x_thread, slg[0], blg[0], sum);
            }
            ws_up += block_size * bytes_per_pack / pack_factor;
            ws_gate += block_size * bytes_per_pack / pack_factor;
            sc_up += block_size / gsz; sc_gate += block_size / gsz;
            bi_up += block_size / gsz; bi_gate += block_size / gsz;
            x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            float up = simd_sum(result_up[row]);
            float gate = simd_sum(result_gate[row]);
            if (simd_lid == 0) {
                gate = metal::min(gate, 7.0f);
                up = metal::clamp(up, -7.0f, 7.0f);
                float silu = gate / (1.0f + metal::exp(-gate));
                y[row] = half(silu * up);
            }
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

    // One simdgroup per output row — simd_sum, no threadgroup barriers.
    kernel void maple_batched_gemv(
        device const half* W [[buffer(0)]],
        device const half* x [[buffer(1)]],
        device float* y [[buffer(2)]],
        constant int& E [[buffer(3)]],
        constant int& H [[buffer(4)]],
        uint e [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]])
    {
        if (e >= (uint)E) return;
        const device half* row = W + (size_t)e * (size_t)H;
        float acc = 0.0f;
        for (uint k = tid; k < (uint)H; k += 32) {
            acc += float(row[k]) * float(x[k]);
        }
        acc = simd_sum(acc);
        if (tid == 0) y[e] = acc;
    }

    // 4-bit affine gather-qmv: one TG per probe; one simdgroup per row (tgs = N*32).
    kernel void maple_qmm4_gather(
        device const uint* w [[buffer(0)]],
        device const half* scales [[buffer(1)]],
        device const half* biases [[buffer(2)]],
        device const half* x [[buffer(3)]],
        device const int* inds [[buffer(4)]],
        device float* y [[buffer(5)]],
        constant int& N [[buffer(6)]],
        constant int& K [[buffer(7)]],
        constant int& gs [[buffer(8)]],
        constant int& nProbes [[buffer(9)]],
        uint probe [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]])
    {
        if (probe >= (uint)nProbes) return;
        uint row = simd_gid;
        if (row >= (uint)N) return;
        uint e = (uint)inds[probe];
        uint packs = (uint)K / 8;
        uint groups = (uint)K / (uint)gs;
        const device uint* wr = w + ((size_t)e * (size_t)N + row) * packs;
        const device half* sr = scales + ((size_t)e * (size_t)N + row) * groups;
        const device half* br = biases + ((size_t)e * (size_t)N + row) * groups;
        float acc = 0.0f;
        for (uint k = simd_lid; k < (uint)K; k += 32) {
            uint pack = wr[k >> 3];
            uint nib = (pack >> ((k & 7) * 4)) & 0xfu;
            uint g = k / (uint)gs;
            float wd = float(nib) * float(sr[g]) + float(br[g]);
            acc += wd * float(x[k]);
        }
        acc = simd_sum(acc);
        if (simd_lid == 0) y[probe * (uint)N + row] = acc;
    }

    // FlashHead top-k stage 1: each chunk (256) emits LOCAL local maxima (masked).
    // Serial rounds only over the chunk — not over full E — so cost is O(LOCAL * chunk).
    kernel void maple_flash_topk_chunk(
        device const float* scores [[buffer(0)]],
        device float* candScores [[buffer(1)]],
        device int* candInds [[buffer(2)]],
        constant int& E [[buffer(3)]],
        constant int& chunk [[buffer(4)]],
        constant int& localTop [[buffer(5)]],
        uint cid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        threadgroup float red[256];
        threadgroup int redi[256];
        threadgroup float work[256];
        int base = int(cid) * chunk;
        int idx = base + int(tid);
        float v = (idx < E) ? scores[idx] : -INFINITY;
        work[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = 0; p < localTop; ++p) {
            red[tid] = work[tid];
            redi[tid] = (idx < E) ? idx : -1;
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
                candScores[cid * (uint)localTop + (uint)p] = red[0];
                candInds[cid * (uint)localTop + (uint)p] = redi[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (redi[0] == idx) work[tid] = -INFINITY;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // FlashHead top-k stage 2: merge candidates → K inds (serial K over tiny nCand).
    kernel void maple_flash_topk_merge(
        device float* candScores [[buffer(0)]],
        device int* candInds [[buffer(1)]],
        device int* inds [[buffer(2)]],
        constant int& nCand [[buffer(3)]],
        constant int& K [[buffer(4)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        threadgroup float red[256];
        threadgroup int redi[256];
        for (int p = 0; p < K; ++p) {
            float best = -INFINITY;
            int besti = -1;
            for (uint i = tid; i < (uint)nCand; i += tgs) {
                float v = candScores[i];
                if (v > best) { best = v; besti = candInds[i]; }
            }
            red[tid] = best;
            redi[tid] = besti;
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
            if (tid == 0) inds[p] = redi[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < (uint)nCand; i += tgs) {
                if (candInds[i] == redi[0]) candScores[i] = -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
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

    // Drop oldest token: cache[h, s, :] <- cache[h, s+1, :] for s in 0..maxLen-2.
    kernel void maple_shift_kv(
        device half* cache [[buffer(0)]],
        constant int& KV [[buffer(1)]],
        constant int& D [[buffer(2)]],
        constant int& maxLen [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]])
    {
        uint d = gid.x, seq = gid.y, h = gid.z;
        if (h >= (uint)KV || d >= (uint)D || seq + 1 >= (uint)maxLen) return;
        size_t base = (size_t)h * (size_t)maxLen * (size_t)D;
        cache[base + (size_t)seq * (size_t)D + d] = cache[base + (size_t)(seq + 1) * (size_t)D + d];
    }

    // Online-softmax SDPA decode, D=V=128 (Qwisp sdpa templated down from 256).
    // Requires 32 simdgroups (1024 threads): V is striped across simd_gid.
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

    // Gather one embedding row: out[H] = table[token, :]
    kernel void maple_embed_token(
        device const half* table [[buffer(0)]],
        device const int* ids [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant int& H [[buffer(3)]],
        constant int& row [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= (uint)H) return;
        int tok = ids[row];
        out[gid] = table[(size_t)tok * (size_t)H + gid];
    }

    // FlashHead gather logits → token id + optional force-token override. One TG.
    kernel void maple_flash_argmax_token(
        device const float* logits [[buffer(0)]],
        device const int* inds [[buffer(1)]],
        device const int* tokenMap [[buffer(2)]],
        device int* ids [[buffer(3)]],
        constant int& nProbes [[buffer(4)]],
        constant int& clusterSize [[buffer(5)]],
        constant int& outRow [[buffer(6)]],
        device const int* stopFlag [[buffer(7)]],
        device const half* h [[buffer(8)]],
        device const half* forceRows [[buffer(9)]],
        device const int* forceIds [[buffer(10)]],
        constant int& nForce [[buffer(11)]],
        constant int& H [[buffer(12)]],
        uint lid [[thread_index_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]])
    {
        if (stopFlag[0] != 0) return;
        int n = nProbes * clusterSize;
        threadgroup float tgScore[256];
        threadgroup int tgIdx[256];
        float best = -INFINITY;
        int besti = 0;
        for (int i = (int)lid; i < n; i += (int)tptg) {
            float v = logits[i];
            if (v > best) { best = v; besti = i; }
        }
        tgScore[lid] = best;
        tgIdx[lid] = besti;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                if (tgScore[lid + stride] > tgScore[lid]) {
                    tgScore[lid] = tgScore[lid + stride];
                    tgIdx[lid] = tgIdx[lid + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) {
            int local = tgIdx[0];
            float bestScore = tgScore[0];
            int probe = local / clusterSize;
            int row = local - probe * clusterSize;
            int cluster = inds[probe];
            int bestId = tokenMap[(size_t)cluster * (size_t)clusterSize + (size_t)row];
            for (int fi = 0; fi < nForce; fi++) {
                float dot = 0;
                size_t base = (size_t)fi * (size_t)H;
                for (int k = 0; k < H; k++) {
                    dot += (float)h[k] * (float)forceRows[base + (size_t)k];
                }
                if (dot > bestScore) {
                    bestScore = dot;
                    bestId = forceIds[fi];
                }
            }
            ids[outRow] = bestId;
        }
    }
    """

    /// Opt-in Maple ternary gather (separate metallib — avoids stock-path bloat).
    private static let ternaryMetalSource = """
    #include <metal_stdlib>
    using namespace metal;
    #define SIMD_SIZE 32

    kernel void gqmm2_rows_ternary(
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
        (void)biases;
        constexpr int packs_per_thread = 1, num_simdgroups = 2, results_per_simdgroup = 4;
        constexpr int pack_factor = 16, bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512;
        const device uint8_t* ws = (const device uint8_t*)w;
        thread float x_thread[16];
        thread float result[4] = {0};
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g;
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)mk * out_vec_size + out_row;
        for (int k = 0; k < in_vec_size; k += block_size) {
            for (int i = 0; i < 16; i++) x_thread[i] = (float)x[i];
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                float accum = 0.0f;
                for (int i = 0; i < 4; i++) {
                    accum += ((float)( wl[i]        & 0x03) - 1.0f) * x_thread[4*i]
                           + ((float)((wl[i] >> 2) & 0x03) - 1.0f) * x_thread[4*i+1]
                           + ((float)((wl[i] >> 4) & 0x03) - 1.0f) * x_thread[4*i+2]
                           + ((float)((wl[i] >> 6) & 0x03) - 1.0f) * x_thread[4*i+3];
                }
                result[row] += accum;
            }
            ws += block_size * bytes_per_pack / pack_factor;
            x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            float alpha = (float)scales[row * in_vec_size_g];
            result[row] = simd_sum(result[row]) * alpha;
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }
    """

    /// Stock ld16_b2/qd2 with Maple row-α hoist: load α once, bias = -α, no per-group advance.
    private static let foldMetalSource = """
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

    kernel void gqmm2_rows_fold(
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
        (void)biases;
        constexpr int packs_per_thread = 1, num_simdgroups = 2, results_per_simdgroup = 4;
        constexpr int pack_factor = 16, bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512;
        const device uint8_t* ws = (const device uint8_t*)w;
        thread float x_thread[16];
        thread float result[4] = {0};
        thread float alpha[4];
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g;
        for (int row = 0; row < results_per_simdgroup; row++) {
            alpha[row] = (float)scales[row * in_vec_size_g];
        }
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)mk * out_vec_size + out_row;
        for (int k = 0; k < in_vec_size; k += block_size) {
            float sum = ld16_b2(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                result[row] += qd2(wl, x_thread, alpha[row], -alpha[row], sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }
    """

    /// Stock ld16_b2 / masked accum; α applied once after simd_sum. Inner 16-way unchanged.
    private static let deferAMetalSource = """
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
    inline float qd2_acc(const device uint8_t* w, const thread float* xt) {
        float accum = 0.0f;
        for (int i = 0; i < 4; i++) {
            accum += (float)(w[i] & 0x03) * xt[4*i]
                   + (float)(w[i] & 0x0c) * xt[4*i+1]
                   + (float)(w[i] & 0x30) * xt[4*i+2]
                   + (float)(w[i] & 0xc0) * xt[4*i+3];
        }
        return accum;
    }

    kernel void gqmm2_rows_defer_a(
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
        (void)biases;
        constexpr int packs_per_thread = 1, num_simdgroups = 2, results_per_simdgroup = 4;
        constexpr int pack_factor = 16, bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512;
        const device uint8_t* ws = (const device uint8_t*)w;
        thread float x_thread[16];
        thread float result[4] = {0};
        thread float alpha[4];
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g;
        for (int row = 0; row < results_per_simdgroup; row++) {
            alpha[row] = (float)scales[row * in_vec_size_g];
        }
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)mk * out_vec_size + out_row;
        float xsum = 0.0f;
        for (int k = 0; k < in_vec_size; k += block_size) {
            float sum = ld16_b2(x, x_thread);
            xsum += sum;
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                result[row] += qd2_acc(wl, x_thread);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            x += block_size;
        }
        float sx = simd_sum(xsum);
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)(alpha[row] * (result[row] - sx));
        }
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
