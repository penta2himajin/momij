import Foundation
import Metal
import MLX

/// Raw Metal kernels for Maple ternary (2-bit affine) decode hot path.
///
/// `gqmm2_rows` is ported from Qwisp Seedless (MLX gather_qmv_fast bits=2 layout)
/// with `group_size` 64|128. Expert path = up_gate gather → clamped SwiGLU →
/// down gather → score reduce (or fused `maple_score_resid`) on one command buffer. Optional `gqmm2_up_swiglu`
/// (`MOMIJ_FUSE_UP_SWIGLU=1`) fuses the first two; default keeps them separate (faster e2e).
/// Experimental (default off): `MOMIJ_GQMM2_SPLITK=1` (micro↑ e2e↓), `MOMIJ_GQMM2_W16=1` (mild).
/// `MOMIJ_GQMM2_TERNARY=1` Maple signed-trit path (separate metallib; packed/e2e regress).
/// Default gqmm2: Maple fold (hoisted row α, bias=-α, stock qd2). `MOMIJ_GQMM2_FOLD=0` restores per-group affine.
/// `MOMIJ_GQMM2_TG2D=1` uses Qwisp TG shape 32×2 (same 2 simdgroups; opt-in until e2e wins).
/// `MOMIJ_GQMM2_DEFER_A=1` applies α after simd_sum (separate metallib; opt-in).
/// `MOMIJ_GQMM2_FOLD_A=1` fold epilogue `α·(accum−sum)` (separate metallib; opt-in).
/// `MOMIJ_FUSE_DOWN_SCORE=0` restores down gather + `score_resid` (Ktop loop in-TG is default).
/// `MOMIJ_FUSE_UP_KTOP=1` loops Ktop in the fold up+SwiGLU TG (packed e2e regress; opt-in).
/// `MOMIJ_FUSE_O_RESID=0` restores o-proj write + `resid_add` (fold-add into `h` is default).
/// `MOMIJ_FUSE_WRITEKV=0` restores two `maple_write_kv` dispatches (K then V).
/// `gqmm2_rows_vec` (vectorized loads) and `gqmm2_rows_pf` measured packed-CB regress — not shipped.
public enum SeedlessMetal {
    nonisolated(unsafe) static var device: MTLDevice?
    nonisolated(unsafe) static var queue: MTLCommandQueue?
    nonisolated(unsafe) static var gqmm2Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2W16Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2TernaryPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2FoldPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2FoldScorePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2FoldUpPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2FoldAddPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2DeferAPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2FoldAPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2SplitKPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2ReduceSKPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gqmm2UpSwigluPipeline: MTLComputePipelineState?
    /// Scratch for split-K partials: [nChunks, Ktop, N] half. Grown on demand.
    nonisolated(unsafe) static var gqmm2SKPartials: MTLBuffer?
    nonisolated(unsafe) static var gqmm2SKPartialsBytes: Int = 0
    nonisolated(unsafe) static var swigluPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var scoreReducePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var scoreResidPipeline: MTLComputePipelineState?
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
    nonisolated(unsafe) static var writeKVPairPipeline: MTLComputePipelineState?
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
        scoreResidPipeline = try pipe("maple_score_resid")
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
        writeKVPairPipeline = try pipe("maple_write_kv_pair")
        shiftKVPipeline = try pipe("maple_shift_kv")
        sdpaPipeline = try pipe("maple_sdpa_d128")
        embedTokenPipeline = try pipe("maple_embed_token")
        flashArgmaxTokenPipeline = try pipe("maple_flash_argmax_token")
        ready = true
        if useFold { try compileFoldPipelines(dev) }
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

    /// Block until MLX's GPU stream completes.
    ///
    /// Weight / embedding / norm buffers are noCopy-aliased out of the MLX
    /// allocator and then read on the SeedlessMetal command queue (a different
    /// queue). MLX `eval` is asynchronous: when an engine is created while the
    /// MLX stream is still draining earlier work, the seedless kernels can
    /// read the aliases before the evals land. Measured effects: all-zero
    /// logits (argmax 0) and deterministically flipped tokens, order-dependent
    /// on prior process allocations. Init paths that alias MLX arrays call
    /// this once before returning.
    static func syncMLXStream() {
        Stream.defaultStream(Device.gpu).synchronize()
    }

    /// Owned copy via host bytes. Does not alias MLX storage.
    static func mtlBufHostCopy(_ a: MLXArray, _ device: MTLDevice) -> MTLBuffer? {
        a.eval()
        let data = a.asData(access: .copy)
        let n = data.data.count
        guard n > 0, let buf = device.makeBuffer(length: n, options: .storageModeShared) else {
            return nil
        }
        data.data.withUnsafeBytes { raw in
            guard let p = raw.baseAddress else { return }
            buf.contents().copyMemory(from: p, byteCount: n)
        }
        return buf
    }

    /// Owned copy. Use at init when the MLXArray will not outlive the buffer.
    static func mtlBufCopy(_ a: MLXArray, _ device: MTLDevice) -> MTLBuffer? {
        a.asMTLBuffer(device: device, noCopy: false)
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

    /// Fold `score_reduce` into residual add (Qwisp combine→S2 analog). Default on.
    /// `MOMIJ_FUSE_SCORE_RESID=0` restores two dispatches + `moeOut` roundtrip.
    public static var fuseScoreResid: Bool {
        guard let raw = getenv("MOMIJ_FUSE_SCORE_RESID") else { return true }
        return String(cString: raw) != "0"
    }

    /// Fold down-gqmm2 epilogue into `h += Σ score·down` (drops `downOut` + `score_resid`).
    /// Default on. `MOMIJ_FUSE_DOWN_SCORE=0` restores down gather + `score_resid`.
    public static var fuseDownScore: Bool {
        guard let raw = getenv("MOMIJ_FUSE_DOWN_SCORE") else { return true }
        return String(cString: raw) != "0"
    }

    /// Ktop-in-TG fold up+SwiGLU (shared x). Default off — packed e2e regress.
    /// Env: `MOMIJ_FUSE_UP_KTOP=1`.
    public static var fuseUpKtop: Bool {
        guard let raw = getenv("MOMIJ_FUSE_UP_KTOP") else { return false }
        return String(cString: raw) == "1"
    }

    /// Fold o-proj into residual add (`h += o`). Default on.
    /// `MOMIJ_FUSE_O_RESID=0` restores o-proj write + `resid_add`.
    public static var fuseOResid: Bool {
        guard let raw = getenv("MOMIJ_FUSE_O_RESID") else { return true }
        return String(cString: raw) != "0"
    }

    /// Write K and V caches in one kernel. Default on.
    /// `MOMIJ_FUSE_WRITEKV=0` restores two `maple_write_kv` dispatches.
    public static var fuseWriteKV: Bool {
        guard let raw = getenv("MOMIJ_FUSE_WRITEKV") else { return true }
        return String(cString: raw) != "0"
    }

    /// Wider TG: 16 output rows / TG (4 simdgroups). Env: `MOMIJ_GQMM2_W16=1`.
    public static var useW16: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_W16") else { return false }
        return String(cString: raw) == "1"
    }

    /// Qwisp TG shape: 32×2 instead of 64×1 (same 2 simdgroups). Env: `MOMIJ_GQMM2_TG2D=1`.
    public static var useTg2d: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_TG2D") else { return false }
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
        if gqmm2FoldPipeline != nil, gqmm2FoldScorePipeline != nil,
           gqmm2FoldUpPipeline != nil, gqmm2FoldAddPipeline != nil { return }
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        try compileFoldPipelines(device)
    }

    private static func compileFoldPipelines(_ device: MTLDevice) throws {
        if gqmm2FoldPipeline != nil, gqmm2FoldScorePipeline != nil,
           gqmm2FoldUpPipeline != nil, gqmm2FoldAddPipeline != nil { return }
        let lib = try device.makeLibrary(source: foldMetalSource, options: mlxMatchCompileOpts())
        if gqmm2FoldPipeline == nil {
            gqmm2FoldPipeline = try device.makeComputePipelineState(
                function: lib.makeFunction(name: "gqmm2_rows_fold")!)
        }
        if gqmm2FoldScorePipeline == nil {
            gqmm2FoldScorePipeline = try device.makeComputePipelineState(
                function: lib.makeFunction(name: "gqmm2_rows_fold_score")!)
        }
        if gqmm2FoldUpPipeline == nil {
            gqmm2FoldUpPipeline = try device.makeComputePipelineState(
                function: lib.makeFunction(name: "gqmm2_rows_fold_up_swiglu")!)
        }
        if gqmm2FoldAddPipeline == nil {
            gqmm2FoldAddPipeline = try device.makeComputePipelineState(
                function: lib.makeFunction(name: "gqmm2_rows_fold_add")!)
        }
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

    /// Fold with epilogue `α·(accum−sum)` instead of `α·accum+(−α)·sum`. Env: `MOMIJ_GQMM2_FOLD_A=1`.
    public static var useFoldA: Bool {
        guard let raw = getenv("MOMIJ_GQMM2_FOLD_A") else { return false }
        return String(cString: raw) == "1"
    }

    private static func ensureFoldACompiled() throws {
        if gqmm2FoldAPipeline != nil { return }
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let lib = try device.makeLibrary(source: foldAMetalSource, options: mlxMatchCompileOpts())
        gqmm2FoldAPipeline = try device.makeComputePipelineState(
            function: lib.makeFunction(name: "gqmm2_rows_fold_a")!)
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

    /// 2-bit gather-qmv: x[M,K] (or [M·Ktop,K] if lhsPerExpert), w[E,N,K/16],
    /// scales/biases[E,N,K/gs], inds[M·Ktop] → y[M·Ktop,N]. Kernel `ktop` stays per-token;
    /// grid depth is `M * Ktop` (Qwisp gqmm2Rows).
    public static func gqmm2(
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, inds: MTLBuffer,
        out: MTLBuffer,
        Ktop: Int, K: Int, N: Int, gs: Int = 128, lhsPerExpert: Bool = false,
        M: Int = 1,
        into encoder: MTLComputeCommandEncoder? = nil,
        commandQueue: MTLCommandQueue? = nil,
        splitK: Bool? = nil,
        w16: Bool? = nil,
        ternary: Bool? = nil,
        fold: Bool? = nil,
        deferA: Bool? = nil,
        foldA: Bool? = nil,
        tg2d: Bool? = nil
    ) throws {
        try ensureCompiled()
        guard let stop = stopBuf else { throw SeedlessError.notReady }
        let doSK = M == 1 && (splitK ?? useSplitK)
        let doTernary = !doSK && (ternary ?? useTernary)
        let doDeferA = !doSK && !doTernary && (deferA ?? useDeferA)
        let doFoldA = !doSK && !doTernary && !doDeferA && (foldA ?? useFoldA)
        let doFold = !doSK && !doTernary && !doDeferA && !doFoldA && (fold ?? useFold)
        let doW16 = !doSK && !doTernary && !doDeferA && !doFoldA && !doFold && (w16 ?? useW16)
        let rowsPerTG = doW16 ? 16 : 8
        guard M >= 1, N % rowsPerTG == 0, K % gqmm2BlockSize == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: N, K: K, gs: gs)
        }
        if doTernary { try ensureTernaryCompiled() }
        if doDeferA { try ensureDeferACompiled() }
        if doFoldA { try ensureFoldACompiled() }
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
            } else if doFoldA {
                guard let a = gqmm2FoldAPipeline else { throw SeedlessError.notReady }
                pipe = a
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
            let tg2 = !doW16 && (tg2d ?? useTg2d)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: N / rowsPerTG, depth: M * Ktop),
                threadsPerThreadgroup: tg2
                    ? MTLSize(width: 32, height: 2, depth: 1)
                    : MTLSize(width: tptg, height: 1, depth: 1))
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
        Ktop: Int, K: Int, I: Int, gs: Int = 128, M: Int = 1,
        into encoder: MTLComputeCommandEncoder? = nil,
        commandQueue: MTLCommandQueue? = nil
    ) throws {
        try ensureCompiled()
        guard let pipe = gqmm2UpSwigluPipeline, let stop = stopBuf else { throw SeedlessError.notReady }
        guard M >= 1, I % 8 == 0, K % 512 == 0, gs == 64 || gs == 128 else {
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
            MTLSize(width: 1, height: I / 8, depth: M * Ktop),
            threadsPerThreadgroup: useTg2d
                ? MTLSize(width: 32, height: 2, depth: 1)
                : MTLSize(width: 64, height: 1, depth: 1))

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
    /// `M>1`: x[M,H], inds/scores[M·Ktop], ug/act/down stacked as [M·Ktop, …], y[M,H].
    /// `residInto`: fold score-reduce into `h += Σ score·down` (skips `y`).
    public static func encodeFusedExpert(
        into enc: MTLComputeCommandEncoder,
        x: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, y: MTLBuffer,
        H: Int, I: Int, Ktop: Int, gs: Int = 128, M: Int = 1,
        residInto: MTLBuffer? = nil
    ) throws {
        try ensureCompiled()
        guard scoreReducePipeline != nil, scoreResidPipeline != nil, stopBuf != nil
        else { throw SeedlessError.notReady }
        guard M >= 1 else { throw SeedlessError.unsupportedShape(N: H, K: I, gs: gs) }

        if fuseUpKtop {
            try encodeGqmm2UpKtopSwiglu(
                into: enc, x: x, w: upGateW, scales: upGateS, biases: upGateB, inds: inds, out: act,
                Ktop: Ktop, K: H, I: I, gs: gs, M: M)
        } else if fuseUpSwiglu {
            try gqmm2UpSwiglu(x: x, w: upGateW, scales: upGateS, biases: upGateB, inds: inds,
                             out: act, Ktop: Ktop, K: H, I: I, gs: gs, M: M, into: enc)
        } else {
            try gqmm2(x: x, w: upGateW, scales: upGateS, biases: upGateB, inds: inds, out: ugOut,
                      Ktop: Ktop, K: H, N: 2 * I, gs: gs, lhsPerExpert: false, M: M, into: enc)
            encodeClampedSwiglu(into: enc, ug: ugOut, act: act, I: I, Ktop: M * Ktop)
        }

        if let residH = residInto, fuseDownScore {
            try encodeGqmm2DownScoreResid(
                into: enc, x: act, w: downW, scales: downS, biases: downB,
                inds: inds, scores: scores, h: residH,
                Ktop: Ktop, K: I, N: H, gs: gs, M: M)
        } else {
            try gqmm2(x: act, w: downW, scales: downS, biases: downB, inds: inds, out: downOut,
                      Ktop: Ktop, K: I, N: H, gs: gs, lhsPerExpert: true, M: M, into: enc)
            if let residH = residInto {
                encodeScoreResid(
                    into: enc, down: downOut, scores: scores, h: residH, H: H, Ktop: Ktop, M: M)
            } else {
                encodeScoreReduce(
                    into: enc, down: downOut, scores: scores, y: y, H: H, Ktop: Ktop, M: M)
            }
        }
    }

    /// Fused MoE expert block on a single command buffer (wait once). `M=1` is one token.
    public static func fusedExpertStep(
        x: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, y: MTLBuffer,
        H: Int, I: Int, Ktop: Int, gs: Int = 128, M: Int = 1
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
            H: H, I: I, Ktop: Ktop, gs: gs, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Encode one MoE block into an existing encoder (no commit/wait).
    /// `M>1`: h/xNorm/moeOut `[M,H]`, logits `[M,E]`, inds/scores `[M·Ktop]`.
    public static func encodeMoEBlock(
        into enc: MTLComputeCommandEncoder,
        h: MTLBuffer, normW: MTLBuffer, gateW: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        xNorm: MTLBuffer, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, moeOut: MTLBuffer,
        H: Int, I: Int, E: Int, Ktop: Int, eps: Float, gs: Int = 128, M: Int = 1
    ) throws {
        try ensureCompiled()
        guard rmsPipeline != nil, residPipeline != nil,
              gateGemvPipeline != nil, routeTop8Pipeline != nil,
              stopBuf != nil
        else { throw SeedlessError.notReady }
        guard M >= 1 else { throw SeedlessError.unsupportedShape(N: H, K: I, gs: gs) }

        // rms then gate stay separate: xNorm is also the up-gqmm2 LHS, so a fused
        // rms+gate kernel would still write H and/or recompute RMS in each of E TGs.
        encodeRms(into: enc, h: h, w: normW, out: xNorm, H: H, eps: eps, M: M)
        encodeGate(into: enc, w: gateW, x: xNorm, y: logits, E: E, H: H, M: M)
        encodeRoute(into: enc, logits: logits, inds: inds, scores: scores, E: E, Ktop: Ktop, M: M)

        try encodeFusedExpert(
            into: enc, x: xNorm,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, y: moeOut,
            H: H, I: I, Ktop: Ktop, gs: gs, M: M,
            residInto: fuseScoreResid ? h : nil)
        if !fuseScoreResid {
            encodeResid(into: enc, h: h, delta: moeOut, H: H, M: M)
        }
    }

    public static func moeBlockOneCB(
        h: MTLBuffer, normW: MTLBuffer, gateW: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        xNorm: MTLBuffer, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, moeOut: MTLBuffer,
        H: Int, I: Int, E: Int, Ktop: Int, eps: Float, gs: Int = 128, M: Int = 1
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
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs, M: M)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    public static func encodeResid(
        into enc: MTLComputeCommandEncoder, h: MTLBuffer, delta: MTLBuffer, H: Int, M: Int = 1
    ) {
        enc.setComputePipelineState(residPipeline!)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        var h32 = Int32(H), m32 = Int32(M)
        enc.setBytes(&h32, length: 4, index: 2)
        enc.setBytes(&m32, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: H, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    static func encodeScoreReduce(
        into enc: MTLComputeCommandEncoder,
        down: MTLBuffer, scores: MTLBuffer, y: MTLBuffer,
        H: Int, Ktop: Int, M: Int
    ) {
        enc.setComputePipelineState(scoreReducePipeline!)
        enc.setBuffer(down, offset: 0, index: 0)
        enc.setBuffer(scores, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var h32 = Int32(H), kt32 = Int32(Ktop), m32 = Int32(M)
        enc.setBytes(&h32, length: 4, index: 3)
        enc.setBytes(&kt32, length: 4, index: 4)
        enc.setBuffer(stopBuf, offset: 0, index: 5)
        enc.setBytes(&m32, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: H, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    static func encodeScoreResid(
        into enc: MTLComputeCommandEncoder,
        down: MTLBuffer, scores: MTLBuffer, h: MTLBuffer,
        H: Int, Ktop: Int, M: Int
    ) {
        enc.setComputePipelineState(scoreResidPipeline!)
        enc.setBuffer(down, offset: 0, index: 0)
        enc.setBuffer(scores, offset: 0, index: 1)
        enc.setBuffer(h, offset: 0, index: 2)
        var h32 = Int32(H), kt32 = Int32(Ktop), m32 = Int32(M)
        enc.setBytes(&h32, length: 4, index: 3)
        enc.setBytes(&kt32, length: 4, index: 4)
        enc.setBuffer(stopBuf, offset: 0, index: 5)
        enc.setBytes(&m32, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: H, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    /// Down gather with Ktop loop in-TG: `h += half(Σ_k half(down_k)·score_k)`.
    /// Grid depth is `M` (not `M·Ktop`). lhs is per-expert `x[M·Ktop, K]`.
    static func encodeGqmm2DownScoreResid(
        into enc: MTLComputeCommandEncoder,
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
        inds: MTLBuffer, scores: MTLBuffer, h: MTLBuffer,
        Ktop: Int, K: Int, N: Int, gs: Int = 128, M: Int = 1
    ) throws {
        try ensureFoldCompiled()
        guard let pipe = gqmm2FoldScorePipeline, let stop = stopBuf else {
            throw SeedlessError.notReady
        }
        guard M >= 1, N % 8 == 0, K % gqmm2BlockSize == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: N, K: K, gs: gs)
        }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(inds, offset: 0, index: 4)
        enc.setBuffer(scores, offset: 0, index: 5)
        enc.setBuffer(h, offset: 0, index: 6)
        var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 7)
        enc.setBytes(&nn, length: 4, index: 8)
        enc.setBytes(&kt, length: 4, index: 9)
        enc.setBuffer(stop, offset: 0, index: 10)
        var gsv = Int32(gs)
        enc.setBytes(&gsv, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: N / 8, depth: M),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Fold up||gate gather with Ktop loop in-TG + clamped SwiGLU → `act[M·Ktop, I]`.
    /// Shared `x[M, K]`. Two-kernel rounding: SwiGLU from `half(up)` / `half(gate)`.
    static func encodeGqmm2UpKtopSwiglu(
        into enc: MTLComputeCommandEncoder,
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
        inds: MTLBuffer, out: MTLBuffer,
        Ktop: Int, K: Int, I: Int, gs: Int = 128, M: Int = 1
    ) throws {
        try ensureFoldCompiled()
        guard let pipe = gqmm2FoldUpPipeline, let stop = stopBuf else {
            throw SeedlessError.notReady
        }
        guard M >= 1, I % 8 == 0, K % gqmm2BlockSize == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: I, K: K, gs: gs)
        }
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
            MTLSize(width: 1, height: I / 8, depth: M),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Fold gather-qmv epilogue into `h += half(y)`. Same grid as `gqmm2` (`depth = M·Ktop`).
    static func encodeGqmm2FoldAdd(
        into enc: MTLComputeCommandEncoder,
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
        inds: MTLBuffer, h: MTLBuffer,
        Ktop: Int, K: Int, N: Int, gs: Int = 128, M: Int = 1, lhsPerExpert: Bool = false
    ) throws {
        try ensureFoldCompiled()
        guard let pipe = gqmm2FoldAddPipeline, let stop = stopBuf else {
            throw SeedlessError.notReady
        }
        guard M >= 1, N % 8 == 0, K % gqmm2BlockSize == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: N, K: K, gs: gs)
        }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(inds, offset: 0, index: 4)
        enc.setBuffer(h, offset: 0, index: 5)
        var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 6)
        enc.setBytes(&nn, length: 4, index: 7)
        enc.setBytes(&kt, length: 4, index: 8)
        enc.setBuffer(stop, offset: 0, index: 9)
        var lp = UInt32(lhsPerExpert ? 1 : 0)
        enc.setBytes(&lp, length: 4, index: 10)
        var gsv = Int32(gs)
        enc.setBytes(&gsv, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: N / 8, depth: M * Ktop),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Decode SDPA. Queries laid out `[M, numHeads, headDim]`; `tid.y` is the query row.
    /// Same KV cache for all M. `causalM`: query m uses `N = nUseBase + m` (draft-causal).
    /// `ring != 0`: sliding-window cache read cyclically — row m reads the
    /// `nUseBase + m` slots ending at slot `(endSlot + m) % ring`.
    public static func encodeSdpa(
        into enc: MTLComputeCommandEncoder,
        queries: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer, out: MTLBuffer,
        numHeads: Int, numKV: Int, headDim: Int,
        maxLen: Int, nUseBase: Int, ring: Int, endSlot: Int, M: Int = 1, causalM: Bool = false
    ) {
        enc.setComputePipelineState(sdpaPipeline!)
        enc.setBuffer(queries, offset: 0, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBuffer(vCache, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var gqa = Int32(numHeads / numKV), n32 = Int32(nUseBase)
        var khs = Int32(maxLen * headDim), kss = Int32(headDim)
        var vhs = Int32(maxLen * headDim), vss = Int32(headDim)
        var sc = Float(pow(Double(headDim), -0.5))
        var causal = Int32(causalM ? 1 : 0)
        var ring32 = Int32(ring), end32 = Int32(endSlot)
        enc.setBytes(&gqa, length: 4, index: 4)
        enc.setBytes(&n32, length: 4, index: 5)
        enc.setBytes(&khs, length: 4, index: 6)
        enc.setBytes(&kss, length: 4, index: 7)
        enc.setBytes(&vhs, length: 4, index: 8)
        enc.setBytes(&vss, length: 4, index: 9)
        enc.setBytes(&sc, length: 4, index: 10)
        enc.setBytes(&causal, length: 4, index: 11)
        enc.setBytes(&ring32, length: 4, index: 12)
        enc.setBytes(&end32, length: 4, index: 13)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: M, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Copy one KV cache slot from `src` (token-major head layout).
    static func encodeWriteKV(
        into enc: MTLComputeCommandEncoder,
        src: MTLBuffer, cache: MTLBuffer, srcOffset: Int,
        KV: Int, D: Int, maxLen: Int, pos: Int, M: Int,
        srcHeadStride: Int, srcSeqStride: Int, ring: Int = 0
    ) {
        enc.setComputePipelineState(writeKVPipeline!)
        enc.setBuffer(src, offset: srcOffset, index: 0)
        enc.setBuffer(cache, offset: 0, index: 1)
        var kv32 = Int32(KV), d32 = Int32(D), ml32 = Int32(maxLen), p32 = Int32(pos)
        var m32 = Int32(M), hs = Int32(srcHeadStride), ss = Int32(srcSeqStride)
        var r32 = Int32(ring)
        enc.setBytes(&kv32, length: 4, index: 2)
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&ml32, length: 4, index: 4)
        enc.setBytes(&p32, length: 4, index: 5)
        enc.setBytes(&m32, length: 4, index: 6)
        enc.setBytes(&hs, length: 4, index: 7)
        enc.setBytes(&ss, length: 4, index: 8)
        enc.setBytes(&r32, length: 4, index: 9)
        let n = KV * D
        enc.dispatchThreads(MTLSize(width: n, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, n), height: 1, depth: 1))
    }

    /// Copy K and V cache slots in one dispatch.
    static func encodeWriteKVPair(
        into enc: MTLComputeCommandEncoder,
        kSrc: MTLBuffer, vSrc: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer,
        kSrcOffset: Int, vSrcOffset: Int,
        KV: Int, D: Int, maxLen: Int, pos: Int, M: Int,
        kHeadStride: Int, kSeqStride: Int,
        vHeadStride: Int, vSeqStride: Int, ring: Int = 0
    ) {
        enc.setComputePipelineState(writeKVPairPipeline!)
        enc.setBuffer(kSrc, offset: kSrcOffset, index: 0)
        enc.setBuffer(vSrc, offset: vSrcOffset, index: 1)
        enc.setBuffer(kCache, offset: 0, index: 2)
        enc.setBuffer(vCache, offset: 0, index: 3)
        var kv32 = Int32(KV), d32 = Int32(D), ml32 = Int32(maxLen), p32 = Int32(pos)
        var m32 = Int32(M)
        var khs = Int32(kHeadStride), kss = Int32(kSeqStride)
        var vhs = Int32(vHeadStride), vss = Int32(vSeqStride)
        var r32 = Int32(ring)
        enc.setBytes(&kv32, length: 4, index: 4)
        enc.setBytes(&d32, length: 4, index: 5)
        enc.setBytes(&ml32, length: 4, index: 6)
        enc.setBytes(&p32, length: 4, index: 7)
        enc.setBytes(&m32, length: 4, index: 8)
        enc.setBytes(&khs, length: 4, index: 9)
        enc.setBytes(&kss, length: 4, index: 10)
        enc.setBytes(&vhs, length: 4, index: 11)
        enc.setBytes(&vss, length: 4, index: 12)
        enc.setBytes(&r32, length: 4, index: 13)
        let n = KV * D
        enc.dispatchThreads(MTLSize(width: n, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, n), height: 1, depth: 1))
    }

    /// Decode attn body. `ropePos` / `writePos` / `seqLen` are for token 0;
    /// `M>1` uses RoPE `pos+m`, writes cache slots `writePos+m`, and causal SDPA
    /// (`N = seqLen + m`). Decode still calls this with `M=1`. `rotateFirst` is M=1 only.
    public static func encodeAttnBlock(
        into enc: MTLComputeCommandEncoder,
        xNorm: MTLBuffer,
        qkvW: MTLBuffer, qkvS: MTLBuffer, qkvB: MTLBuffer,
        oW: MTLBuffer, oS: MTLBuffer, oB: MTLBuffer,
        qkW: MTLBuffer, invFreq: MTLBuffer, densInds: MTLBuffer,
        qkvOut: MTLBuffer, qkOut: MTLBuffer, attnTmp: MTLBuffer, attnOut: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        H: Int, numHeads: Int, numKV: Int, headDim: Int,
        ropeDim: Int, ropePos: Int, writePos: Int, maxLen: Int, nUseBase: Int,
        ring: Int,
        eps: Float, gs: Int = 128, M: Int = 1,
        residInto: MTLBuffer? = nil
    ) throws {
        try ensureCompiled()
        guard qkNormRopePipeline != nil, writeKVPipeline != nil, writeKVPairPipeline != nil,
              sdpaPipeline != nil
        else { throw SeedlessError.notReady }
        let qkPipe = qkNormRopePipeline!
        guard M >= 1 else { throw SeedlessError.unsupportedShape(N: H, K: H, gs: gs) }

        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let qkvN = qDim + 2 * kvDim

        try gqmm2(x: xNorm, w: qkvW, scales: qkvS, biases: qkvB, inds: densInds, out: qkvOut,
                  Ktop: 1, K: H, N: qkvN, gs: gs, lhsPerExpert: false, M: M, into: enc)

        enc.setComputePipelineState(qkPipe)
        enc.setBuffer(qkvOut, offset: 0, index: 0)
        enc.setBuffer(qkW, offset: 0, index: 1)
        enc.setBuffer(qkOut, offset: 0, index: 2)
        enc.setBuffer(invFreq, offset: 0, index: 3)
        var posF = Float(ropePos), epsV = eps, rope = Int32(ropeDim)
        var m32 = Int32(M), inRow = Int32(qkvN), nQ = Int32(numHeads), nK = Int32(numKV)
        enc.setBytes(&posF, length: 4, index: 4)
        enc.setBytes(&epsV, length: 4, index: 5)
        enc.setBytes(&rope, length: 4, index: 6)
        enc.setBytes(&m32, length: 4, index: 7)
        enc.setBytes(&inRow, length: 4, index: 8)
        enc.setBytes(&nQ, length: 4, index: 9)
        enc.setBytes(&nK, length: 4, index: 10)
        let nQK = numHeads + numKV
        enc.dispatchThreadgroups(MTLSize(width: 1, height: nQK, depth: M),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        if fuseWriteKV {
            encodeWriteKVPair(
                into: enc,
                kSrc: qkOut, vSrc: qkvOut, kCache: kCache, vCache: vCache,
                kSrcOffset: qDim * M * 2, vSrcOffset: (qDim + kvDim) * 2,
                KV: numKV, D: headDim, maxLen: maxLen, pos: writePos, M: M,
                kHeadStride: headDim, kSeqStride: kvDim,
                vHeadStride: headDim, vSeqStride: qkvN, ring: ring)
        } else {
            encodeWriteKV(
                into: enc, src: qkOut, cache: kCache, srcOffset: qDim * M * 2,
                KV: numKV, D: headDim, maxLen: maxLen, pos: writePos, M: M,
                srcHeadStride: headDim, srcSeqStride: kvDim, ring: ring)
            encodeWriteKV(
                into: enc, src: qkvOut, cache: vCache, srcOffset: (qDim + kvDim) * 2,
                KV: numKV, D: headDim, maxLen: maxLen, pos: writePos, M: M,
                srcHeadStride: headDim, srcSeqStride: qkvN, ring: ring)
        }

        encodeSdpa(
            into: enc, queries: qkOut, kCache: kCache, vCache: vCache, out: attnTmp,
            numHeads: numHeads, numKV: numKV, headDim: headDim,
            maxLen: maxLen, nUseBase: nUseBase, ring: ring, endSlot: writePos,
            M: M, causalM: M > 1)

        if let residH = residInto {
            try encodeGqmm2FoldAdd(
                into: enc, x: attnTmp, w: oW, scales: oS, biases: oB, inds: densInds, h: residH,
                Ktop: 1, K: qDim, N: H, gs: gs, M: M)
        } else {
            try gqmm2(x: attnTmp, w: oW, scales: oS, biases: oB, inds: densInds, out: attnOut,
                      Ktop: 1, K: qDim, N: H, gs: gs, lhsPerExpert: false, M: M, into: enc)
        }
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
        ropePos: Int, writePos: Int, maxLen: Int, nUseBase: Int, ring: Int,
        M: Int = 1
    ) throws {
        encodeRms(into: enc, h: h, w: inNorm, out: xAttn, H: H, eps: eps, M: M)
        try encodeAttnBlock(
            into: enc, xNorm: xAttn,
            qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
            oW: oW, oS: oS, oB: oB,
            qkW: qkW, invFreq: invFreq, densInds: densInds,
            qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
            kCache: kCache, vCache: vCache,
            H: H, numHeads: numHeads, numKV: numKV, headDim: headDim,
            ropeDim: ropeDim, ropePos: ropePos, writePos: writePos, maxLen: maxLen, nUseBase: nUseBase,
            ring: ring,
            eps: eps, gs: gs, M: M,
            residInto: fuseOResid ? h : nil)
        if !fuseOResid {
            encodeResid(into: enc, h: h, delta: attnOut, H: H, M: M)
        }
        try encodeMoEBlock(
            into: enc, h: h, normW: postNorm, gateW: gateW,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xNorm: xMoe, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs, M: M)
    }

    /// Wall + GPU time for one encoder body. GPU time is `gpuEndTime - gpuStartTime`.
    static func timeCB(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> (wallMs: Double, gpuMs: Double) {
        try ensureCompiled()
        guard let q = queue else { throw SeedlessError.notReady }
        let t0 = CFAbsoluteTimeGetCurrent()
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        do {
            try body(enc)
            enc.endEncoding()
        } catch {
            enc.endEncoding()
            throw error
        }
        cb.commit()
        cb.waitUntilCompleted()
        let wallMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        return (wallMs, gpuMs)
    }

    static func encodeRms(
        into enc: MTLComputeCommandEncoder, h: MTLBuffer, w: MTLBuffer, out: MTLBuffer,
        H: Int, eps: Float, M: Int = 1
    ) {
        enc.setComputePipelineState(rmsPipeline!)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(w, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var epsV = eps, h32 = Int32(H), m32 = Int32(M)
        enc.setBytes(&epsV, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.setBytes(&m32, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// `out[outRow, :] = embTable[ids[idRow], :]` (GPU gather; safe inside a multi-step CB).
    static func encodeEmbedToken(
        into enc: MTLComputeCommandEncoder,
        table: MTLBuffer, ids: MTLBuffer, out: MTLBuffer,
        H: Int, row: Int, outRow: Int = 0
    ) {
        enc.setComputePipelineState(embedTokenPipeline!)
        enc.setBuffer(table, offset: 0, index: 0)
        enc.setBuffer(ids, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var h32 = Int32(H), r32 = Int32(row), or32 = Int32(outRow)
        enc.setBytes(&h32, length: 4, index: 3)
        enc.setBytes(&r32, length: 4, index: 4)
        enc.setBytes(&or32, length: 4, index: 5)
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
        E: Int, H: Int, threadsPerTG: Int = 256, M: Int = 1
    ) {
        if useGateSimd {
            encodeBatchedGemv(into: enc, w: w, x: x, y: y, E: E, H: H, M: M)
            return
        }
        enc.setComputePipelineState(gateGemvPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var e32 = Int32(E), h32 = Int32(H), m32 = Int32(M)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.setBytes(&m32, length: 4, index: 5)
        let tpt = max(32, min(256, threadsPerTG))
        enc.dispatchThreadgroups(MTLSize(width: E, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tpt, height: 1, depth: 1))
    }

    /// Force a specific gate path for tests (`simd: true` → batched; `false` → TG reduce).
    static func encodeGate(
        into enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
        E: Int, H: Int, simd: Bool, M: Int = 1
    ) {
        if simd {
            encodeBatchedGemv(into: enc, w: w, x: x, y: y, E: E, H: H, M: M)
        } else {
            enc.setComputePipelineState(gateGemvPipeline!)
            enc.setBuffer(w, offset: 0, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            var e32 = Int32(E), h32 = Int32(H), m32 = Int32(M)
            enc.setBytes(&e32, length: 4, index: 3)
            enc.setBytes(&h32, length: 4, index: 4)
            enc.setBytes(&m32, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: E, height: M, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// Dense gemv y[M,E]=W[E,H]@x[M,H]. One simdgroup (32 threads) per (e,m) — no TG barriers.
    static func encodeBatchedGemv(
        into enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
        E: Int, H: Int, threadgroups: Int = 256, threadsPerTG: Int = 64, M: Int = 1,
        xByteOffset: Int = 0, yByteOffset: Int = 0
    ) {
        _ = threadgroups; _ = threadsPerTG
        enc.setComputePipelineState(batchedGemvPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: xByteOffset, index: 1)
        enc.setBuffer(y, offset: yByteOffset, index: 2)
        var e32 = Int32(E), h32 = Int32(H), m32 = Int32(M)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.setBytes(&m32, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: E, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// 4-bit affine gather-qmv: for each probe cluster `inds[p]`, score all N rows → y[p*N + r].
    static func encodeQmm4Gather(
        into enc: MTLComputeCommandEncoder,
        w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, x: MTLBuffer,
        inds: MTLBuffer, y: MTLBuffer,
        nProbes: Int, N: Int, K: Int, gs: Int, xByteOffset: Int = 0
    ) {
        enc.setComputePipelineState(qmm4GatherPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: xByteOffset, index: 3)
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
        E: Int, Ktop: Int, M: Int = 1
    ) {
        enc.setComputePipelineState(routeTop8Pipeline!)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(inds, offset: 0, index: 1)
        enc.setBuffer(scores, offset: 0, index: 2)
        var e32 = Int32(E), k32 = Int32(Ktop), m32 = Int32(M)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&k32, length: 4, index: 4)
        enc.setBuffer(stopBuf, offset: 0, index: 5)
        enc.setBytes(&m32, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
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

    /// GPU-timed M-row gqmm2 sweep. `shared` = all tokens reuse the same Ktop experts;
    /// `disjoint` = token m uses experts `[m·Ktop ..)`. Reports ms/token vs M=1.
    public static func benchGqmm2Mrow(
        K: Int = 2048, N: Int = 1024, Ktop: Int = 8, E: Int = 256, gs: Int = 128,
        maxM: Int = 8, iters: Int = 40, lhsPerExpert: Bool = false
    ) throws -> String {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let packedK = K * 2 / 32
        let nGroups = K / gs
        let xRows = lhsPerExpert ? maxM * Ktop : maxM
        let x = device.makeBuffer(length: xRows * K * 2, options: .storageModeShared)!
        let xp = x.contents().bindMemory(to: Float16.self, capacity: xRows * K)
        for i in 0 ..< (xRows * K) { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let w = device.makeBuffer(length: E * N * packedK * 4, options: .storageModeShared)!
        let s = device.makeBuffer(length: E * N * nGroups * 2, options: .storageModeShared)!
        let b = device.makeBuffer(length: E * N * nGroups * 2, options: .storageModeShared)!
        let sp = s.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        let bp = b.contents().bindMemory(to: Float16.self, capacity: E * N * nGroups)
        for i in 0 ..< (E * N * nGroups) {
            let alpha = Float16(0.02)
            sp[i] = alpha
            bp[i] = -alpha
        }
        let inds = device.makeBuffer(length: maxM * Ktop * 4, options: .storageModeShared)!
        let y = device.makeBuffer(length: maxM * Ktop * N * 2, options: .storageModeShared)!

        func fillInds(M: Int, shared: Bool) {
            let p = inds.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
            for m in 0 ..< M {
                for k in 0 ..< Ktop {
                    let e = shared ? k : (m * Ktop + k)
                    p[m * Ktop + k] = Int32(e % E)
                }
            }
        }

        func time(M: Int, shared: Bool) throws -> (wall: Double, gpu: Double) {
            fillInds(M: M, shared: shared)
            var wall = 0.0, gpu = 0.0
            for _ in 0 ..< iters {
                let t = try timeCB { enc in
                    try gqmm2(x: x, w: w, scales: s, biases: b, inds: inds, out: y,
                              Ktop: Ktop, K: K, N: N, gs: gs, lhsPerExpert: lhsPerExpert, M: M,
                              into: enc, splitK: false)
                }
                wall += t.wallMs
                gpu += t.gpuMs
            }
            return (wall / Double(iters), gpu / Double(iters))
        }

        // Warm the GPU / caches on the largest grid before the sweep.
        fillInds(M: maxM, shared: true)
        for _ in 0 ..< 8 {
            try gqmm2(x: x, w: w, scales: s, biases: b, inds: inds, out: y,
                      Ktop: Ktop, K: K, N: N, gs: gs, lhsPerExpert: lhsPerExpert, M: maxM,
                      splitK: false)
        }

        var lines: [String] = []
        let tag = lhsPerExpert ? "down-like lhsPer" : "up-like shared-x"
        lines.append("gqmm2 M-row (\(tag) K=\(K) N=\(N) Ktop=\(Ktop) E=\(E), \(iters) iters, warm)")
        lines.append("  mode        M   gpu_ms   ms/tok     vsM1    tok/s")
        for shared in [true, false] {
            let mode = shared ? "shared" : "disjoint"
            var acc: [Int: (gpu: Double, n: Int)] = [:]
            for M in [1, 2, 4, 8, 8, 4, 2, 1] where M <= maxM {
                let t = try time(M: M, shared: shared)
                let a = acc[M] ?? (0, 0)
                acc[M] = (a.gpu + t.gpu, a.n + 1)
            }
            let gpu1 = (acc[1] ?? (0, 1)).gpu / Double((acc[1] ?? (0, 1)).n)
            for M in [1, 2, 4, 8] where M <= maxM {
                let a = acc[M]!
                let gpu = a.gpu / Double(a.n)
                let per = gpu / Double(M)
                let vs = gpu1 > 0 ? per / gpu1 : 0
                let tps = per > 0 ? 1000.0 / per : 0
                lines.append(
                    "  \(mode.padding(toLength: 8, withPad: " ", startingAt: 0)) \(String(format: "%4d %8.3f %8.3f %8.3f %8.0f", M, gpu, per, vs, tps))"
                )
            }
        }
        return lines.joined(separator: "\n")
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

    /// GPU-timed M-row fused expert (up + SwiGLU + down + score reduce). Same shared/disjoint
    /// expert layout as `benchGqmm2Mrow`. Warm interleaved `M=1,2,4,8,8,4,2,1`.
    public static func benchFusedExpertMrow(
        H: Int = 2048, I: Int = 512, E: Int = 256, Ktop: Int = 8,
        maxM: Int = 8, iters: Int = 30
    ) throws -> String {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let gs = 128
        let packedH = H * 2 / 32
        let packedI = I * 2 / 32
        let nGroupsH = H / gs
        let nGroupsI = I / gs
        let x = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!
        let xp = x.contents().bindMemory(to: Float16.self, capacity: maxM * H)
        for i in 0 ..< (maxM * H) { xp[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        let ugW = device.makeBuffer(length: E * 2 * I * packedH * 4, options: .storageModeShared)!
        let ugS = device.makeBuffer(length: E * 2 * I * nGroupsH * 2, options: .storageModeShared)!
        let ugB = device.makeBuffer(length: E * 2 * I * nGroupsH * 2, options: .storageModeShared)!
        let dW = device.makeBuffer(length: E * H * packedI * 4, options: .storageModeShared)!
        let dS = device.makeBuffer(length: E * H * nGroupsI * 2, options: .storageModeShared)!
        let dB = device.makeBuffer(length: E * H * nGroupsI * 2, options: .storageModeShared)!
        let nUG = E * 2 * I * nGroupsH
        let nDN = E * H * nGroupsI
        let ugSp = ugS.contents().bindMemory(to: Float16.self, capacity: nUG)
        let ugBp = ugB.contents().bindMemory(to: Float16.self, capacity: nUG)
        for i in 0 ..< nUG { ugSp[i] = 0.02; ugBp[i] = -0.02 }
        let dSp = dS.contents().bindMemory(to: Float16.self, capacity: nDN)
        let dBp = dB.contents().bindMemory(to: Float16.self, capacity: nDN)
        for i in 0 ..< nDN { dSp[i] = 0.02; dBp[i] = -0.02 }
        let inds = device.makeBuffer(length: maxM * Ktop * 4, options: .storageModeShared)!
        let scores = device.makeBuffer(length: maxM * Ktop * 4, options: .storageModeShared)!
        let ugOut = device.makeBuffer(length: maxM * Ktop * 2 * I * 2, options: .storageModeShared)!
        let act = device.makeBuffer(length: maxM * Ktop * I * 2, options: .storageModeShared)!
        let downOut = device.makeBuffer(length: maxM * Ktop * H * 2, options: .storageModeShared)!
        let y = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!

        func fill(M: Int, shared: Bool) {
            let ip = inds.contents().bindMemory(to: Int32.self, capacity: M * Ktop)
            let sp = scores.contents().bindMemory(to: Float.self, capacity: M * Ktop)
            for m in 0 ..< M {
                for k in 0 ..< Ktop {
                    let e = shared ? k : (m * Ktop + k)
                    ip[m * Ktop + k] = Int32(e % E)
                    sp[m * Ktop + k] = 1.0 / Float(Ktop)
                }
            }
        }

        func time(M: Int, shared: Bool) throws -> Double {
            fill(M: M, shared: shared)
            var gpu = 0.0
            for _ in 0 ..< iters {
                let t = try timeCB { enc in
                    try encodeFusedExpert(
                        into: enc, x: x,
                        upGateW: ugW, upGateS: ugS, upGateB: ugB,
                        downW: dW, downS: dS, downB: dB,
                        inds: inds, scores: scores,
                        ugOut: ugOut, act: act, downOut: downOut, y: y,
                        H: H, I: I, Ktop: Ktop, gs: gs, M: M)
                }
                gpu += t.gpuMs
            }
            return gpu / Double(iters)
        }

        fill(M: maxM, shared: true)
        for _ in 0 ..< 8 {
            try fusedExpertStep(
                x: x, upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs, M: maxM)
        }

        var lines: [String] = []
        lines.append("fused-expert M-row (H=\(H) I=\(I) Ktop=\(Ktop) E=\(E), \(iters) iters, warm)")
        lines.append("  mode        M   gpu_ms   ms/tok     vsM1    tok/s")
        for shared in [true, false] {
            let mode = shared ? "shared" : "disjoint"
            var acc: [Int: (gpu: Double, n: Int)] = [:]
            for M in [1, 2, 4, 8, 8, 4, 2, 1] where M <= maxM {
                let gpu = try time(M: M, shared: shared)
                let a = acc[M] ?? (0, 0)
                acc[M] = (a.gpu + gpu, a.n + 1)
            }
            let gpu1 = (acc[1] ?? (0, 1)).gpu / Double((acc[1] ?? (0, 1)).n)
            for M in [1, 2, 4, 8] where M <= maxM {
                let a = acc[M]!
                let gpu = a.gpu / Double(a.n)
                let per = gpu / Double(M)
                let vs = gpu1 > 0 ? per / gpu1 : 0
                let tps = per > 0 ? 1000.0 / per : 0
                lines.append(
                    "  \(mode.padding(toLength: 8, withPad: " ", startingAt: 0)) \(String(format: "%4d %8.3f %8.3f %8.3f %8.0f", M, gpu, per, vs, tps))"
                )
            }
        }
        return lines.joined(separator: "\n")
    }

    /// GPU-timed attn M-row: dense QKV/O `gqmm2` (Ktop=1, weight reuse) and SDPA
    /// (`tid.y` query row, shared KV). Warm interleaved `M=1,2,4,8,8,4,2,1`.
    public static func benchAttnMrow(
        H: Int = 2048, numHeads: Int = 16, numKV: Int = 4, headDim: Int = 128,
        maxM: Int = 8, iters: Int = 20, seqLens: [Int] = [128, 512]
    ) throws -> String {
        try ensureCompiled()
        guard let device, sdpaPipeline != nil else { throw SeedlessError.notReady }
        let gs = 128
        let qDim = numHeads * headDim
        let qkvN = qDim + 2 * numKV * headDim
        let packedH = H * 2 / 32
        let packedQ = qDim * 2 / 32
        let nGH = H / gs
        let nGQ = qDim / gs
        let maxLen = seqLens.max() ?? 128

        func fillAlpha(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = 0.02 }
        }
        func fillNeg(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = -0.02 }
        }
        func fillX(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        }

        let dens = device.makeBuffer(length: 4, options: .storageModeShared)!
        dens.contents().storeBytes(of: Int32(0), as: Int32.self)

        let xQ = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!
        fillX(xQ, n: maxM * H)
        let qkvW = device.makeBuffer(length: qkvN * packedH * 4, options: .storageModeShared)!
        let qkvS = device.makeBuffer(length: qkvN * nGH * 2, options: .storageModeShared)!
        let qkvB = device.makeBuffer(length: qkvN * nGH * 2, options: .storageModeShared)!
        fillAlpha(qkvS, n: qkvN * nGH)
        fillNeg(qkvB, n: qkvN * nGH)
        let yQ = device.makeBuffer(length: maxM * qkvN * 2, options: .storageModeShared)!

        let xO = device.makeBuffer(length: maxM * qDim * 2, options: .storageModeShared)!
        fillX(xO, n: maxM * qDim)
        let oW = device.makeBuffer(length: H * packedQ * 4, options: .storageModeShared)!
        let oS = device.makeBuffer(length: H * nGQ * 2, options: .storageModeShared)!
        let oB = device.makeBuffer(length: H * nGQ * 2, options: .storageModeShared)!
        fillAlpha(oS, n: H * nGQ)
        fillNeg(oB, n: H * nGQ)
        let yO = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!

        let q = device.makeBuffer(length: numHeads * maxM * headDim * 2, options: .storageModeShared)!
        fillX(q, n: numHeads * maxM * headDim)
        let kCache = device.makeBuffer(length: numKV * maxLen * headDim * 2, options: .storageModeShared)!
        let vCache = device.makeBuffer(length: numKV * maxLen * headDim * 2, options: .storageModeShared)!
        fillX(kCache, n: numKV * maxLen * headDim)
        fillX(vCache, n: numKV * maxLen * headDim)
        let yS = device.makeBuffer(length: numHeads * maxM * headDim * 2, options: .storageModeShared)!

        func sweep(_ title: String, time: (Int) throws -> Double) throws -> [String] {
            var acc: [Int: (gpu: Double, n: Int)] = [:]
            for M in [1, 2, 4, 8, 8, 4, 2, 1] where M <= maxM {
                let gpu = try time(M)
                let a = acc[M] ?? (0, 0)
                acc[M] = (a.gpu + gpu, a.n + 1)
            }
            let gpu1 = (acc[1] ?? (0, 1)).gpu / Double((acc[1] ?? (0, 1)).n)
            var lines = [title, "  mode        M   gpu_ms   ms/tok     vsM1    tok/s"]
            for M in [1, 2, 4, 8] where M <= maxM {
                let a = acc[M]!
                let gpu = a.gpu / Double(a.n)
                let per = gpu / Double(M)
                let vs = gpu1 > 0 ? per / gpu1 : 0
                let tps = per > 0 ? 1000.0 / per : 0
                lines.append(
                    "  dense    \(String(format: "%4d %8.3f %8.3f %8.3f %8.0f", M, gpu, per, vs, tps))"
                )
            }
            return lines
        }

        for _ in 0 ..< 8 {
            try gqmm2(x: xQ, w: qkvW, scales: qkvS, biases: qkvB, inds: dens, out: yQ,
                      Ktop: 1, K: H, N: qkvN, gs: gs, M: maxM, splitK: false)
            try gqmm2(x: xO, w: oW, scales: oS, biases: oB, inds: dens, out: yO,
                      Ktop: 1, K: qDim, N: H, gs: gs, M: maxM, splitK: false)
            _ = try timeCB { enc in
                encodeSdpa(into: enc, queries: q, kCache: kCache, vCache: vCache, out: yS,
                           numHeads: numHeads, numKV: numKV, headDim: headDim,
                           maxLen: maxLen, nUseBase: maxLen, ring: 0, endSlot: maxLen - 1, M: maxM)
            }
        }

        var out: [String] = []
        out += try sweep("attn M-row qkv (K=\(H) N=\(qkvN) Ktop=1, \(iters) iters, warm)") { M in
            var gpu = 0.0
            for _ in 0 ..< iters {
                gpu += try timeCB { enc in
                    try gqmm2(x: xQ, w: qkvW, scales: qkvS, biases: qkvB, inds: dens, out: yQ,
                              Ktop: 1, K: H, N: qkvN, gs: gs, M: M, into: enc, splitK: false)
                }.gpuMs
            }
            return gpu / Double(iters)
        }
        out += try sweep("attn M-row o-proj (K=\(qDim) N=\(H) Ktop=1, \(iters) iters, warm)") { M in
            var gpu = 0.0
            for _ in 0 ..< iters {
                gpu += try timeCB { enc in
                    try gqmm2(x: xO, w: oW, scales: oS, biases: oB, inds: dens, out: yO,
                              Ktop: 1, K: qDim, N: H, gs: gs, M: M, into: enc, splitK: false)
                }.gpuMs
            }
            return gpu / Double(iters)
        }
        for N in seqLens {
            out += try sweep("attn M-row sdpa (H=\(numHeads) KV=\(numKV) D=\(headDim) N=\(N), \(iters) iters, warm)") { M in
                var gpu = 0.0
                for _ in 0 ..< iters {
                    gpu += try timeCB { enc in
                        encodeSdpa(into: enc, queries: q, kCache: kCache, vCache: vCache, out: yS,
                                   numHeads: numHeads, numKV: numKV, headDim: headDim,
                                   maxLen: maxLen, nUseBase: N, ring: 0, endSlot: N - 1, M: M)
                    }.gpuMs
                }
                return gpu / Double(iters)
            }
        }
        return out.joined(separator: "\n")
    }

    /// GPU-timed full `encodeAttnBlock` M-row (qkv + qk-norm/RoPE + writeKV + causal SDPA + o).
    public static func benchAttnBlockMrow(
        H: Int = 2048, numHeads: Int = 16, numKV: Int = 4, headDim: Int = 128,
        maxM: Int = 8, iters: Int = 20, writePos: Int = 32, maxLen: Int = 128
    ) throws -> String {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let gs = 128
        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let qkvN = qDim + 2 * kvDim
        let packedH = H * 2 / 32
        let packedQ = qDim * 2 / 32
        let nGH = H / gs
        let nGQ = qDim / gs
        let seqLen = writePos + 1

        func fillAlpha(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = 0.02 }
        }
        func fillNeg(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = -0.02 }
        }
        func fillX(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        }

        let dens = device.makeBuffer(length: 4, options: .storageModeShared)!
        dens.contents().storeBytes(of: Int32(0), as: Int32.self)
        let x = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!
        fillX(x, n: maxM * H)
        let qkvW = device.makeBuffer(length: qkvN * packedH * 4, options: .storageModeShared)!
        let qkvS = device.makeBuffer(length: qkvN * nGH * 2, options: .storageModeShared)!
        let qkvB = device.makeBuffer(length: qkvN * nGH * 2, options: .storageModeShared)!
        fillAlpha(qkvS, n: qkvN * nGH)
        fillNeg(qkvB, n: qkvN * nGH)
        let oW = device.makeBuffer(length: H * packedQ * 4, options: .storageModeShared)!
        let oS = device.makeBuffer(length: H * nGQ * 2, options: .storageModeShared)!
        let oB = device.makeBuffer(length: H * nGQ * 2, options: .storageModeShared)!
        fillAlpha(oS, n: H * nGQ)
        fillNeg(oB, n: H * nGQ)
        let qkW = device.makeBuffer(length: (numHeads + numKV) * headDim * 2, options: .storageModeShared)!
        fillAlpha(qkW, n: (numHeads + numKV) * headDim)
        let inv = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
        let ip = inv.contents().bindMemory(to: Float.self, capacity: 32)
        for i in 0 ..< 32 { ip[i] = 0.1 / Float(i + 1) }
        let qkvOut = device.makeBuffer(length: maxM * qkvN * 2, options: .storageModeShared)!
        let qkOut = device.makeBuffer(length: maxM * (qDim + kvDim) * 2, options: .storageModeShared)!
        let attnTmp = device.makeBuffer(length: maxM * qDim * 2, options: .storageModeShared)!
        let attnOut = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!
        let kCache = device.makeBuffer(length: numKV * maxLen * headDim * 2, options: .storageModeShared)!
        let vCache = device.makeBuffer(length: numKV * maxLen * headDim * 2, options: .storageModeShared)!
        fillX(kCache, n: numKV * maxLen * headDim)
        fillX(vCache, n: numKV * maxLen * headDim)

        func time(M: Int) throws -> Double {
            var gpu = 0.0
            for _ in 0 ..< iters {
                gpu += try timeCB { enc in
                    try encodeAttnBlock(
                        into: enc, xNorm: x,
                        qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
                        oW: oW, oS: oS, oB: oB,
                        qkW: qkW, invFreq: inv, densInds: dens,
                        qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
                        kCache: kCache, vCache: vCache,
                        H: H, numHeads: numHeads, numKV: numKV, headDim: headDim,
                        ropeDim: 64, ropePos: writePos, writePos: writePos, maxLen: maxLen, nUseBase: seqLen, ring: 0,
                        eps: 1e-6, gs: gs, M: M)
                }.gpuMs
            }
            return gpu / Double(iters)
        }

        for _ in 0 ..< 8 {
            _ = try timeCB { enc in
                try encodeAttnBlock(
                    into: enc, xNorm: x,
                    qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
                    oW: oW, oS: oS, oB: oB,
                    qkW: qkW, invFreq: inv, densInds: dens,
                    qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
                    kCache: kCache, vCache: vCache,
                    H: H, numHeads: numHeads, numKV: numKV, headDim: headDim,
                    ropeDim: 64, ropePos: writePos, writePos: writePos, maxLen: maxLen, nUseBase: seqLen, ring: 0,
                    eps: 1e-6, gs: gs, M: maxM)
            }
        }

        var acc: [Int: (gpu: Double, n: Int)] = [:]
        for M in [1, 2, 4, 8, 8, 4, 2, 1] where M <= maxM {
            let gpu = try time(M: M)
            let a = acc[M] ?? (0, 0)
            acc[M] = (a.gpu + gpu, a.n + 1)
        }
        let gpu1 = (acc[1] ?? (0, 1)).gpu / Double((acc[1] ?? (0, 1)).n)
        var lines = [
            "attn-block M-row (H=\(H) 16h/4kv N=\(seqLen), \(iters) iters, warm)",
            "  mode        M   gpu_ms   ms/tok     vsM1    tok/s"
        ]
        for M in [1, 2, 4, 8] where M <= maxM {
            let a = acc[M]!
            let gpu = a.gpu / Double(a.n)
            let per = gpu / Double(M)
            let vs = gpu1 > 0 ? per / gpu1 : 0
            let tps = per > 0 ? 1000.0 / per : 0
            lines.append(
                "  dense    \(String(format: "%4d %8.3f %8.3f %8.3f %8.0f", M, gpu, per, vs, tps))"
            )
        }
        return lines.joined(separator: "\n")
    }

    /// GPU-timed K/V cache write: two `maple_write_kv` vs one `maple_write_kv_pair`.
    public static func benchWriteKV(
        KV: Int = 4, D: Int = 128, maxLen: Int = 512, M: Int = 1, pos: Int = 32, iters: Int = 200
    ) throws -> String {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let kvDim = KV * D
        func fillX(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        }
        let kSrc = device.makeBuffer(length: M * kvDim * 2, options: .storageModeShared)!
        let vSrc = device.makeBuffer(length: M * kvDim * 2, options: .storageModeShared)!
        fillX(kSrc, n: M * kvDim)
        fillX(vSrc, n: M * kvDim)
        let kA = device.makeBuffer(length: KV * maxLen * D * 2, options: .storageModeShared)!
        let vA = device.makeBuffer(length: KV * maxLen * D * 2, options: .storageModeShared)!
        let kB = device.makeBuffer(length: KV * maxLen * D * 2, options: .storageModeShared)!
        let vB = device.makeBuffer(length: KV * maxLen * D * 2, options: .storageModeShared)!
        func avg(_ body: (MTLComputeCommandEncoder) -> Void) throws -> Double {
            var gpu = 0.0
            for _ in 0 ..< 8 {
                _ = try timeCB { enc in body(enc) }
            }
            for _ in 0 ..< iters {
                gpu += try timeCB { enc in body(enc) }.gpuMs
            }
            return gpu / Double(iters)
        }
        let two = try avg { enc in
            encodeWriteKV(
                into: enc, src: kSrc, cache: kA, srcOffset: 0,
                KV: KV, D: D, maxLen: maxLen, pos: pos, M: M,
                srcHeadStride: D, srcSeqStride: kvDim)
            encodeWriteKV(
                into: enc, src: vSrc, cache: vA, srcOffset: 0,
                KV: KV, D: D, maxLen: maxLen, pos: pos, M: M,
                srcHeadStride: D, srcSeqStride: kvDim)
        }
        let pair = try avg { enc in
            encodeWriteKVPair(
                into: enc, kSrc: kSrc, vSrc: vSrc, kCache: kB, vCache: vB,
                kSrcOffset: 0, vSrcOffset: 0,
                KV: KV, D: D, maxLen: maxLen, pos: pos, M: M,
                kHeadStride: D, kSeqStride: kvDim,
                vHeadStride: D, vSeqStride: kvDim)
        }
        return String(
            format: "writeKV two-dispatch=%.4f ms  pair=%.4f ms  (KV=%d D=%d M=%d)",
            two, pair, KV, D, M)
    }

    /// GPU-timed full MoE block M-row (rms + gate + route + fused expert + resid).
    public static func benchMoEBlockMrow(
        H: Int = 2048, I: Int = 512, E: Int = 256, Ktop: Int = 8,
        maxM: Int = 8, iters: Int = 20
    ) throws -> String {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let gs = 128
        let packedH = H * 2 / 32
        let packedI = I * 2 / 32
        let nGH = H / gs
        let nGI = I / gs

        func fillX(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        }
        func fillA(_ buf: MTLBuffer, n: Int, v: Float16) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = v }
        }

        let h = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!
        fillX(h, n: maxM * H)
        let normW = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        fillA(normW, n: H, v: 1.0)
        let gateW = device.makeBuffer(length: E * H * 2, options: .storageModeShared)!
        fillX(gateW, n: E * H)
        let ugW = device.makeBuffer(length: E * 2 * I * packedH * 4, options: .storageModeShared)!
        let ugS = device.makeBuffer(length: E * 2 * I * nGH * 2, options: .storageModeShared)!
        let ugB = device.makeBuffer(length: E * 2 * I * nGH * 2, options: .storageModeShared)!
        fillA(ugS, n: E * 2 * I * nGH, v: 0.02)
        fillA(ugB, n: E * 2 * I * nGH, v: -0.02)
        let dW = device.makeBuffer(length: E * H * packedI * 4, options: .storageModeShared)!
        let dS = device.makeBuffer(length: E * H * nGI * 2, options: .storageModeShared)!
        let dB = device.makeBuffer(length: E * H * nGI * 2, options: .storageModeShared)!
        fillA(dS, n: E * H * nGI, v: 0.02)
        fillA(dB, n: E * H * nGI, v: -0.02)
        let xNorm = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!
        let logits = device.makeBuffer(length: maxM * E * 4, options: .storageModeShared)!
        let inds = device.makeBuffer(length: maxM * Ktop * 4, options: .storageModeShared)!
        let scores = device.makeBuffer(length: maxM * Ktop * 4, options: .storageModeShared)!
        let ugOut = device.makeBuffer(length: maxM * Ktop * 2 * I * 2, options: .storageModeShared)!
        let act = device.makeBuffer(length: maxM * Ktop * I * 2, options: .storageModeShared)!
        let downOut = device.makeBuffer(length: maxM * Ktop * H * 2, options: .storageModeShared)!
        let moeOut = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared)!

        func time(M: Int) throws -> Double {
            var gpu = 0.0
            for _ in 0 ..< iters {
                gpu += try timeCB { enc in
                    try encodeMoEBlock(
                        into: enc, h: h, normW: normW, gateW: gateW,
                        upGateW: ugW, upGateS: ugS, upGateB: ugB,
                        downW: dW, downS: dS, downB: dB,
                        xNorm: xNorm, logits: logits, inds: inds, scores: scores,
                        ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
                        H: H, I: I, E: E, Ktop: Ktop, eps: 1e-6, gs: gs, M: M)
                }.gpuMs
            }
            return gpu / Double(iters)
        }

        for _ in 0 ..< 8 {
            _ = try timeCB { enc in
                try encodeMoEBlock(
                    into: enc, h: h, normW: normW, gateW: gateW,
                    upGateW: ugW, upGateS: ugS, upGateB: ugB,
                    downW: dW, downS: dS, downB: dB,
                    xNorm: xNorm, logits: logits, inds: inds, scores: scores,
                    ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
                    H: H, I: I, E: E, Ktop: Ktop, eps: 1e-6, gs: gs, M: maxM)
            }
        }

        var acc: [Int: (gpu: Double, n: Int)] = [:]
        for M in [1, 2, 4, 8, 8, 4, 2, 1] where M <= maxM {
            let gpu = try time(M: M)
            let a = acc[M] ?? (0, 0)
            acc[M] = (a.gpu + gpu, a.n + 1)
        }
        let gpu1 = (acc[1] ?? (0, 1)).gpu / Double((acc[1] ?? (0, 1)).n)
        var lines = [
            "moe-block M-row (H=\(H) I=\(I) E=\(E) K=\(Ktop), \(iters) iters, warm)",
            "  mode        M   gpu_ms   ms/tok     vsM1    tok/s"
        ]
        for M in [1, 2, 4, 8] where M <= maxM {
            let a = acc[M]!
            let gpu = a.gpu / Double(a.n)
            let per = gpu / Double(M)
            let vs = gpu1 > 0 ? per / gpu1 : 0
            let tps = per > 0 ? 1000.0 / per : 0
            lines.append(
                "  full    \(String(format: "%4d %8.3f %8.3f %8.3f %8.0f", M, gpu, per, vs, tps))"
            )
        }
        return lines.joined(separator: "\n")
    }

    /// Layer-level M-row config sweep for picking the fastest M≥2 packing.
    /// Compares: `full` (attn+moe M-row), `seq` (M× M=1 in one CB), `moe` (attn seq + moe M-row).
    public static func benchLayerMrowConfigs(
        H: Int = 2048, I: Int = 512, E: Int = 256, Ktop: Int = 8,
        numHeads: Int = 16, numKV: Int = 4, headDim: Int = 128,
        maxM: Int = 4, iters: Int = 12, writePos: Int = 32, maxLen: Int = 128
    ) throws -> String {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let gs = 128, eps: Float = 1e-6
        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let qkvN = qDim + 2 * kvDim
        let packedH = H * 2 / 32
        let packedQ = qDim * 2 / 32
        let packedI = I * 2 / 32
        let nGH = H / gs
        let nGQ = qDim / gs
        let nGI = I / gs
        let seqLen = writePos + 1

        func fillX(_ buf: MTLBuffer, n: Int) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = Float16((Float(i % 17) - 8.0) * 0.01) }
        }
        func fillA(_ buf: MTLBuffer, n: Int, v: Float16) {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0 ..< n { p[i] = v }
        }
        func buf(_ bytes: Int) -> MTLBuffer {
            device.makeBuffer(length: bytes, options: .storageModeShared)!
        }

        let inNorm = buf(H * 2); fillA(inNorm, n: H, v: 1.0)
        let postNorm = buf(H * 2); fillA(postNorm, n: H, v: 1.0)
        let gateW = buf(E * H * 2); fillX(gateW, n: E * H)
        let dens = buf(4); dens.contents().storeBytes(of: Int32(0), as: Int32.self)
        let qkvW = buf(qkvN * packedH * 4)
        let qkvS = buf(qkvN * nGH * 2); fillA(qkvS, n: qkvN * nGH, v: 0.02)
        let qkvB = buf(qkvN * nGH * 2); fillA(qkvB, n: qkvN * nGH, v: -0.02)
        let oW = buf(H * packedQ * 4)
        let oS = buf(H * nGQ * 2); fillA(oS, n: H * nGQ, v: 0.02)
        let oB = buf(H * nGQ * 2); fillA(oB, n: H * nGQ, v: -0.02)
        let qkW = buf((numHeads + numKV) * headDim * 2)
        fillA(qkW, n: (numHeads + numKV) * headDim, v: 1.0)
        let inv = buf(32 * 4)
        let ip = inv.contents().bindMemory(to: Float.self, capacity: 32)
        for i in 0 ..< 32 { ip[i] = 0.1 / Float(i + 1) }
        let ugW = buf(E * 2 * I * packedH * 4)
        let ugS = buf(E * 2 * I * nGH * 2); fillA(ugS, n: E * 2 * I * nGH, v: 0.02)
        let ugB = buf(E * 2 * I * nGH * 2); fillA(ugB, n: E * 2 * I * nGH, v: -0.02)
        let dW = buf(E * H * packedI * 4)
        let dS = buf(E * H * nGI * 2); fillA(dS, n: E * H * nGI, v: 0.02)
        let dB = buf(E * H * nGI * 2); fillA(dB, n: E * H * nGI, v: -0.02)
        let kCache = buf(numKV * maxLen * headDim * 2); fillX(kCache, n: numKV * maxLen * headDim)
        let vCache = buf(numKV * maxLen * headDim * 2); fillX(vCache, n: numKV * maxLen * headDim)

        let h = buf(maxM * H * 2); fillX(h, n: maxM * H)
        let xAttn = buf(maxM * H * 2)
        let qkvOut = buf(maxM * qkvN * 2)
        let qkOut = buf(maxM * (qDim + kvDim) * 2)
        let attnTmp = buf(maxM * qDim * 2)
        let attnOut = buf(maxM * H * 2)
        let xMoe = buf(maxM * H * 2)
        let logits = buf(maxM * E * 4)
        let inds = buf(maxM * Ktop * 4)
        let scores = buf(maxM * Ktop * 4)
        let ugOut = buf(maxM * Ktop * 2 * I * 2)
        let act = buf(maxM * Ktop * I * 2)
        let downOut = buf(maxM * Ktop * H * 2)
        let moeOut = buf(maxM * H * 2)

        var h1 = [MTLBuffer](); var xA1 = [MTLBuffer](); var qkv1 = [MTLBuffer]()
        var qk1 = [MTLBuffer](); var tmp1 = [MTLBuffer](); var aOut1 = [MTLBuffer]()
        var xM1 = [MTLBuffer](); var lg1 = [MTLBuffer](); var ind1 = [MTLBuffer]()
        var sc1 = [MTLBuffer](); var ug1 = [MTLBuffer](); var ac1 = [MTLBuffer]()
        var dn1 = [MTLBuffer](); var mo1 = [MTLBuffer]()
        for m in 0 ..< maxM {
            let hb = buf(H * 2)
            let src = h.contents().bindMemory(to: Float16.self, capacity: maxM * H)
            let dst = hb.contents().bindMemory(to: Float16.self, capacity: H)
            for i in 0 ..< H { dst[i] = src[m * H + i] }
            h1.append(hb)
            xA1.append(buf(H * 2)); qkv1.append(buf(qkvN * 2)); qk1.append(buf((qDim + kvDim) * 2))
            tmp1.append(buf(qDim * 2)); aOut1.append(buf(H * 2)); xM1.append(buf(H * 2))
            lg1.append(buf(E * 4)); ind1.append(buf(Ktop * 4)); sc1.append(buf(Ktop * 4))
            ug1.append(buf(Ktop * 2 * I * 2)); ac1.append(buf(Ktop * I * 2))
            dn1.append(buf(Ktop * H * 2)); mo1.append(buf(H * 2))
        }

        func encodeFull(_ enc: MTLComputeCommandEncoder, M: Int) throws {
            try encodeLayerBlock(
                into: enc, h: h, inNorm: inNorm, postNorm: postNorm, gateW: gateW,
                qkvW: qkvW, qkvS: qkvS, qkvB: qkvB, oW: oW, oS: oS, oB: oB,
                qkW: qkW, invFreq: inv, densInds: dens,
                upGateW: ugW, upGateS: ugS, upGateB: ugB, downW: dW, downS: dS, downB: dB,
                xAttn: xAttn, qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
                kCache: kCache, vCache: vCache,
                xMoe: xMoe, logits: logits, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
                H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs,
                numHeads: numHeads, numKV: numKV, headDim: headDim, ropeDim: 64,
                ropePos: writePos, writePos: writePos, maxLen: maxLen, nUseBase: seqLen, ring: 0,
                M: M)
        }

        func encodeSeq(_ enc: MTLComputeCommandEncoder, M: Int) throws {
            for m in 0 ..< M {
                try encodeLayerBlock(
                    into: enc, h: h1[m], inNorm: inNorm, postNorm: postNorm, gateW: gateW,
                    qkvW: qkvW, qkvS: qkvS, qkvB: qkvB, oW: oW, oS: oS, oB: oB,
                    qkW: qkW, invFreq: inv, densInds: dens,
                    upGateW: ugW, upGateS: ugS, upGateB: ugB, downW: dW, downS: dS, downB: dB,
                    xAttn: xA1[m], qkvOut: qkv1[m], qkOut: qk1[m], attnTmp: tmp1[m], attnOut: aOut1[m],
                    kCache: kCache, vCache: vCache,
                    xMoe: xM1[m], logits: lg1[m], inds: ind1[m], scores: sc1[m],
                    ugOut: ug1[m], act: ac1[m], downOut: dn1[m], moeOut: mo1[m],
                    H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs,
                    numHeads: numHeads, numKV: numKV, headDim: headDim, ropeDim: 64,
                    ropePos: writePos + m, writePos: writePos + m, maxLen: maxLen,
                    nUseBase: seqLen + m, ring: 0, M: 1)
            }
        }

        func encodeMoeAttn(_ enc: MTLComputeCommandEncoder, M: Int) throws {
            for m in 0 ..< M {
                encodeRms(into: enc, h: h1[m], w: inNorm, out: xA1[m], H: H, eps: eps, M: 1)
                try encodeAttnBlock(
                    into: enc, xNorm: xA1[m],
                    qkvW: qkvW, qkvS: qkvS, qkvB: qkvB, oW: oW, oS: oS, oB: oB,
                    qkW: qkW, invFreq: inv, densInds: dens,
                    qkvOut: qkv1[m], qkOut: qk1[m], attnTmp: tmp1[m], attnOut: aOut1[m],
                    kCache: kCache, vCache: vCache,
                    H: H, numHeads: numHeads, numKV: numKV, headDim: headDim,
                    ropeDim: 64, ropePos: writePos + m, writePos: writePos + m,
                    maxLen: maxLen, nUseBase: seqLen + m, ring: 0, eps: eps, gs: gs, M: 1)
                encodeResid(into: enc, h: h1[m], delta: aOut1[m], H: H, M: 1)
            }
        }

        func packH(M: Int) {
            let dst = h.contents().bindMemory(to: Float16.self, capacity: maxM * H)
            for m in 0 ..< M {
                let src = h1[m].contents().bindMemory(to: Float16.self, capacity: H)
                for i in 0 ..< H { dst[m * H + i] = src[i] }
            }
        }

        func encodeMoeOnly(_ enc: MTLComputeCommandEncoder, M: Int) throws {
            try encodeMoEBlock(
                into: enc, h: h, normW: postNorm, gateW: gateW,
                upGateW: ugW, upGateS: ugS, upGateB: ugB, downW: dW, downS: dS, downB: dB,
                xNorm: xMoe, logits: logits, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
                H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs, M: M)
        }

        for _ in 0 ..< 4 {
            _ = try timeCB { enc in try encodeFull(enc, M: maxM) }
        }

        var lines = [
            "layer M-row configs (H=\(H) Maple-like, N=\(seqLen), \(iters) iters, warm)",
            "  cfg         M   gpu_ms   ms/tok     vsSeq    tok/s"
        ]
        for M in [1, 2, 4] where M <= maxM {
            var seqG = 0.0, fullG = 0.0, moeG = 0.0
            for _ in 0 ..< iters {
                seqG += try timeCB { enc in try encodeSeq(enc, M: M) }.gpuMs
                fullG += try timeCB { enc in try encodeFull(enc, M: M) }.gpuMs
                let a = try timeCB { enc in try encodeMoeAttn(enc, M: M) }.gpuMs
                packH(M: M)
                let b = try timeCB { enc in try encodeMoeOnly(enc, M: M) }.gpuMs
                moeG += a + b
            }
            let seq = seqG / Double(iters)
            let full = fullG / Double(iters)
            let moe = moeG / Double(iters)
            let seqPer = seq / Double(M)
            for (name, gpu) in [("seq", seq), ("full", full), ("moe", moe)] {
                let per = gpu / Double(M)
                let vs = seqPer > 0 ? per / seqPer : 0
                let tps = per > 0 ? 1000.0 / per : 0
                lines.append(
                    "  \(name.padding(toLength: 8, withPad: " ", startingAt: 0)) \(String(format: "%4d %8.3f %8.3f %8.3f %8.0f", M, gpu, per, vs, tps))"
                )
            }
        }
        return lines.joined(separator: "\n")
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
        device const half* down [[buffer(0)]],   // [M·Ktop, H]
        device const float* scores [[buffer(1)]], // [M·Ktop]
        device half* y [[buffer(2)]],             // [M, H]
        constant int& H [[buffer(3)]],
        constant int& Ktop [[buffer(4)]],
        device const int* stopFlag [[buffer(5)]],
        constant int& M [[buffer(6)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (stopFlag[0] != 0) return;
        uint h = gid.x, m = gid.y;
        if (h >= (uint)H || m >= (uint)M) return;
        float acc = 0.0f;
        for (int ki = 0; ki < Ktop; ++ki) {
            acc += float(down[(m * (uint)Ktop + (uint)ki) * (uint)H + h]) * scores[m * (uint)Ktop + (uint)ki];
        }
        y[m * (uint)H + h] = half(acc);
    }

    // Fold score_reduce + resid_add: h += half(Σ down·score). Two-kernel rounding:
    // h = half(float(h) + float(half(acc))), Ktop loop ki=0..<Ktop.
    kernel void maple_score_resid(
        device const half* down [[buffer(0)]],   // [M·Ktop, H]
        device const float* scores [[buffer(1)]], // [M·Ktop]
        device half* h [[buffer(2)]],             // [M, H]
        constant int& H [[buffer(3)]],
        constant int& Ktop [[buffer(4)]],
        device const int* stopFlag [[buffer(5)]],
        constant int& M [[buffer(6)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (stopFlag[0] != 0) return;
        uint hx = gid.x, m = gid.y;
        if (hx >= (uint)H || m >= (uint)M) return;
        float acc = 0.0f;
        for (int ki = 0; ki < Ktop; ++ki) {
            acc += float(down[(m * (uint)Ktop + (uint)ki) * (uint)H + hx]) * scores[m * (uint)Ktop + (uint)ki];
        }
        size_t off = (size_t)m * (size_t)H + hx;
        h[off] = half(float(h[off]) + float(half(acc)));
    }

    // Milestone A helpers: RMSNorm / residual / dense gate / top-8 route.
    kernel void maple_rms_norm(
        device const half* x [[buffer(0)]],
        device const half* w [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant float& eps [[buffer(3)]],
        constant int& H [[buffer(4)]],
        constant int& M [[buffer(5)]],
        uint lid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]],
        uint m [[threadgroup_position_in_grid]])
    {
        if (m >= (uint)M) return;
        const device half* xm = x + (size_t)m * (size_t)H;
        device half* om = out + (size_t)m * (size_t)H;
        threadgroup float red[256];
        float acc = 0.0f;
        for (uint i = lid; i < (uint)H; i += tgs) {
            float xi = float(xm[i]);
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
            om[i] = half(float(xm[i]) * inv * float(w[i]));
        }
    }

    kernel void maple_resid_add(
        device half* h [[buffer(0)]],
        device const half* delta [[buffer(1)]],
        constant int& H [[buffer(2)]],
        constant int& M [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint i = gid.x, m = gid.y;
        if (i >= (uint)H || m >= (uint)M) return;
        size_t off = (size_t)m * (size_t)H + i;
        h[off] = half(float(h[off]) + float(delta[off]));
    }

    // Dense gate: y[e] = sum_k W[e,k] * x[k]  (1 TG / expert, parallel reduce)
    kernel void maple_gate_gemv(
        device const half* W [[buffer(0)]],  // [E, H]
        device const half* x [[buffer(1)]],  // [M, H]
        device float* y [[buffer(2)]],       // [M, E]
        constant int& E [[buffer(3)]],
        constant int& H [[buffer(4)]],
        constant int& M [[buffer(5)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        uint2 lid [[thread_position_in_threadgroup]],
        uint2 tgsv [[threads_per_threadgroup]])
    {
        uint e = tgp.x, m = tgp.y;
        uint tid = lid.x, tgs = tgsv.x;
        if (e >= (uint)E || m >= (uint)M) return;
        threadgroup float red[256];
        const device half* row = W + (size_t)e * (size_t)H;
        const device half* xm = x + (size_t)m * (size_t)H;
        float acc = 0.0f;
        for (uint k = tid; k < (uint)H; k += tgs) {
            acc += float(row[k]) * float(xm[k]);
        }
        red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) y[(size_t)m * (size_t)E + e] = red[0];
    }

    // One simdgroup per output row — simd_sum, no threadgroup barriers.
    // Grid: (E, M). y[m,e] = W[e,:] · x[m,:].
    kernel void maple_batched_gemv(
        device const half* W [[buffer(0)]],
        device const half* x [[buffer(1)]],
        device float* y [[buffer(2)]],
        constant int& E [[buffer(3)]],
        constant int& H [[buffer(4)]],
        constant int& M [[buffer(5)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        uint2 lid [[thread_position_in_threadgroup]])
    {
        uint e = tgp.x, m = tgp.y;
        uint tid = lid.x;
        if (e >= (uint)E || m >= (uint)M) return;
        const device half* row = W + (size_t)e * (size_t)H;
        const device half* xm = x + (size_t)m * (size_t)H;
        float acc = 0.0f;
        for (uint k = tid; k < (uint)H; k += 32) {
            acc += float(row[k]) * float(xm[k]);
        }
        acc = simd_sum(acc);
        if (tid == 0) y[(size_t)m * (size_t)E + e] = acc;
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
    // One TG per token; logits[M,E] → inds/scores[M,K].
    kernel void maple_route_top8(
        device const float* logits [[buffer(0)]],
        device int* inds [[buffer(1)]],
        device float* scores [[buffer(2)]],
        constant int& E [[buffer(3)]],
        constant int& K [[buffer(4)]],
        device const int* stopFlag [[buffer(5)]],
        constant int& M [[buffer(6)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]],
        uint m [[threadgroup_position_in_grid]])
    {
        if (stopFlag[0] != 0) return;
        if (m >= (uint)M) return;
        const device float* lgIn = logits + (size_t)m * (size_t)E;
        device int* indsM = inds + (size_t)m * (size_t)K;
        device float* scoresM = scores + (size_t)m * (size_t)K;
        threadgroup float red[256];
        threadgroup int redi[256];
        threadgroup float gates[256];
        threadgroup float work[256];
        threadgroup float bcast[1];
        float lg = (tid < (uint)E) ? lgIn[tid] : -INFINITY;
        red[tid] = lg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = max(red[tid], red[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) bcast[0] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mx = bcast[0];
        float e = (tid < (uint)E) ? precise::exp(lg - mx) : 0.0f;
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
                indsM[k] = bi;
                scoresM[k] = gates[bi];
                work[bi] = -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            float ss = 0.0f;
            for (int k = 0; k < K; k++) ss += scoresM[k];
            for (int k = 0; k < K; k++) scoresM[k] = scoresM[k] / ss;
        }
    }

    // Milestone B: decode attention (HEAD_DIM=128, GQA).
    kernel void maple_qk_norm_rope(
        device const half* x [[buffer(0)]],      // [M, qkvN] token-major; Q|K at start of each row
        device const half* w [[buffer(1)]],      // [nQ+nK, 128]
        device half* out [[buffer(2)]],          // Q [M, nQ, 128] then K [M, nK, 128]
        device const float* inv_freq [[buffer(3)]],
        constant float& pos [[buffer(4)]],
        constant float& eps [[buffer(5)]],
        constant int& ropeDim [[buffer(6)]],
        constant int& M [[buffer(7)]],
        constant int& inRow [[buffer(8)]],
        constant int& nQ [[buffer(9)]],
        constant int& nK [[buffer(10)]],
        uint3 tid [[thread_position_in_grid]])
    {
        constexpr int HEAD_DIM = 128;
        constexpr int per_lane = HEAD_DIM / 32;
        uint lane = tid.x;
        uint head = tid.y;
        uint m = tid.z;
        if (m >= (uint)M) return;
        const device half* xh = x + (size_t)m * (size_t)inRow + (size_t)head * HEAD_DIM;
        const device half* wh = w + head * HEAD_DIM;
        device half* oh;
        if ((int)head < nQ) {
            oh = out + ((size_t)m * (size_t)nQ + (size_t)head) * HEAD_DIM;
        } else {
            uint hk = head - (uint)nQ;
            oh = out + (size_t)nQ * (size_t)M * HEAD_DIM
               + ((size_t)m * (size_t)nK + (size_t)hk) * HEAD_DIM;
        }
        float pos_m = pos + float(m);
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
                float theta = pos_m * inv_freq[p];
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
        constant int& M [[buffer(6)]],
        constant int& srcHeadStride [[buffer(7)]],
        constant int& srcSeqStride [[buffer(8)]],
        constant int& ring [[buffer(9)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint i = gid.x, m = gid.y;
        if (i >= (uint)(KV * D) || m >= (uint)M) return;
        uint h = i / (uint)D, d = i % (uint)D;
        half v = src[(size_t)h * (size_t)srcHeadStride + (size_t)m * (size_t)srcSeqStride + d];
        int slot = pos + (int)m;
        if (ring != 0) { slot = slot % ring; }
        cache[(size_t)h * (size_t)maxLen * (size_t)D + (size_t)slot * (size_t)D + d] = v;
    }

    kernel void maple_write_kv_pair(
        device const half* ksrc [[buffer(0)]],
        device const half* vsrc [[buffer(1)]],
        device half* kcache     [[buffer(2)]],
        device half* vcache     [[buffer(3)]],
        constant int& KV        [[buffer(4)]],
        constant int& D         [[buffer(5)]],
        constant int& maxLen    [[buffer(6)]],
        constant int& pos       [[buffer(7)]],
        constant int& M         [[buffer(8)]],
        constant int& kHeadStride [[buffer(9)]],
        constant int& kSeqStride  [[buffer(10)]],
        constant int& vHeadStride [[buffer(11)]],
        constant int& vSeqStride  [[buffer(12)]],
        constant int& ring        [[buffer(13)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint i = gid.x, m = gid.y;
        if (i >= (uint)(KV * D) || m >= (uint)M) return;
        uint h = i / (uint)D, d = i % (uint)D;
        half kv = ksrc[(size_t)h * (size_t)kHeadStride + (size_t)m * (size_t)kSeqStride + d];
        half vv = vsrc[(size_t)h * (size_t)vHeadStride + (size_t)m * (size_t)vSeqStride + d];
        // ring != 0: sliding-window cache — wrap the slot cyclically so an
        // M-row chunk may cross the ring boundary (FIFO eviction is unchanged).
        int slot = pos + (int)m;
        if (ring != 0) { slot = slot % ring; }
        size_t dst = (size_t)h * (size_t)maxLen * (size_t)D
                   + (size_t)slot * (size_t)D + d;
        kcache[dst] = kv;
        vcache[dst] = vv;
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
        constant int& causal       [[buffer(11)]],
        constant int& ring         [[buffer(12)]],
        constant int& endSlot      [[buffer(13)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint3 tpg [[threadgroups_per_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]])
    {
        constexpr int BN = 32, BD = 32, D = 128, V = 128;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        typedef float U;
        thread U q[qk_per_thread]; thread U k[qk_per_thread]; thread U o[v_per_thread];
        threadgroup U outputs[BN * BD];
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];
        const int q_batch_head_idx = tid.x;
        const int q_seq_idx = tid.y;
        const int kv_head_idx = q_batch_head_idx / gqa_factor;
        // Token-major: [M, numHeads, D]. M=1 is identical to the old [heads, D].
        const int o_offset = q_seq_idx * tpg.x + q_batch_head_idx;
        const int nUse = N + (causal ? q_seq_idx : 0);
        // Row m's window ends at slot `endSlot + m` (cyclic when ring != 0) and
        // spans nUse slots backward: slot(i) = end - nUse + 1 + i (mod ring).
        const int end = endSlot + (causal ? (int)q_seq_idx : 0);
        const int slot0 = end - nUse + 1;
        queries += o_offset * D + simd_lid * qk_per_thread;
        const device half* keysBase = keys
            + (size_t)kv_head_idx * (size_t)k_head_stride + (size_t)simd_lid * qk_per_thread;
        const device half* valuesBase = values
            + (size_t)kv_head_idx * (size_t)v_head_stride + (size_t)simd_lid * v_per_thread;
        out += o_offset * V + simd_gid * v_per_thread;
        for (int i = 0; i < qk_per_thread; i++) q[i] = (U)scale * (U)queries[i];
        for (int i = 0; i < v_per_thread; i++) o[i] = 0;
        U max_score = -INFINITY;
        U sum_exp_score = 0;
        for (int i = simd_gid; i < nUse; i += BN) {
            int slot = slot0 + i;
            if (ring != 0) { slot %= ring; if (slot < 0) slot += ring; }
            const device half* kp = keysBase + (size_t)slot * (size_t)k_seq_stride;
            const device half* vp = valuesBase + (size_t)slot * (size_t)v_seq_stride;
            for (int j = 0; j < qk_per_thread; j++) k[j] = (U)kp[j];
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * (U)vp[j];
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

    // Gather one embedding row: out[outRow, :] = table[ids[row], :]
    kernel void maple_embed_token(
        device const half* table [[buffer(0)]],
        device const int* ids [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant int& H [[buffer(3)]],
        constant int& row [[buffer(4)]],
        constant int& outRow [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= (uint)H) return;
        int tok = ids[row];
        out[(size_t)outRow * (size_t)H + gid] = table[(size_t)tok * (size_t)H + gid];
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

    kernel void gqmm2_rows_fold_score(
        device const uint32_t* w      [[buffer(0)]],
        device const half*     scales [[buffer(1)]],
        device const half*     biases [[buffer(2)]],
        device const half*     x      [[buffer(3)]],
        device const int*      inds   [[buffer(4)]],
        device const float*    scores [[buffer(5)]],
        device half*           h      [[buffer(6)]],
        constant int& in_vec_size  [[buffer(7)]],
        constant int& out_vec_size [[buffer(8)]],
        constant int& ktop         [[buffer(9)]],
        device const int* stopFlag [[buffer(10)]],
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
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        uint m = tid.z;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        thread float acc[4] = {0};
        thread float x_thread[16];
        const device uint8_t* w0 = (const device uint8_t*)w;
        const device half* scales0 = scales;
        const device half* x0 = x;
        for (int ki = 0; ki < ktop; ++ki) {
            uint mk = m * (uint)ktop + (uint)ki;
            uint e = (uint)inds[mk];
            const device uint8_t* ws = w0
                + (size_t)e * out_vec_size * in_vec_size_w
                + (size_t)out_row * in_vec_size_w
                + simd_lid * packs_per_thread * bytes_per_pack;
            const device half* scp = scales0
                + (size_t)e * out_vec_size * in_vec_size_g
                + (size_t)out_row * in_vec_size_g;
            thread float alpha[4];
            thread float result[4] = {0};
            for (int row = 0; row < results_per_simdgroup; row++) {
                alpha[row] = (float)scp[row * in_vec_size_g];
            }
            const device half* xk = x0 + (size_t)mk * in_vec_size + simd_lid * values_per_thread;
            for (int k = 0; k < in_vec_size; k += block_size) {
                float sum = ld16_b2(xk, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    result[row] += qd2(wl, x_thread, alpha[row], -alpha[row], sum);
                }
                ws += block_size * bytes_per_pack / pack_factor;
                xk += block_size;
            }
            float sc = scores[mk];
            for (int row = 0; row < results_per_simdgroup; row++) {
                result[row] = simd_sum(result[row]);
                acc[row] += float(half(result[row])) * sc;
            }
        }
        if (simd_lid == 0) {
            device half* ho = h + (size_t)m * out_vec_size + out_row;
            for (int row = 0; row < results_per_simdgroup; row++) {
                ho[row] = half(float(ho[row]) + float(half(acc[row])));
            }
        }
    }

    kernel void gqmm2_rows_fold_up_swiglu(
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
        (void)biases;
        constexpr int packs_per_thread = 1, num_simdgroups = 2, results_per_simdgroup = 4;
        constexpr int pack_factor = 16, bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512;
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        const int weight_rows = out_vec_size * 2;
        uint m = tid.z;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        thread float x_thread[16];
        const device uint8_t* w0 = (const device uint8_t*)w;
        const device half* scales0 = scales;
        const device half* x0 = x + (size_t)m * in_vec_size + simd_lid * values_per_thread;
        for (int ki = 0; ki < ktop; ++ki) {
            uint mk = m * (uint)ktop + (uint)ki;
            uint e = (uint)inds[mk];
            const device uint8_t* ws = w0 + (size_t)e * weight_rows * in_vec_size_w;
            const device half* scp = scales0 + (size_t)e * weight_rows * in_vec_size_g;
            const device uint8_t* ws_up = ws + (size_t)out_row * in_vec_size_w
                + simd_lid * packs_per_thread * bytes_per_pack;
            const device uint8_t* ws_gate = ws + (size_t)(out_row + out_vec_size) * in_vec_size_w
                + simd_lid * packs_per_thread * bytes_per_pack;
            const device half* sc_up = scp + (size_t)out_row * in_vec_size_g;
            const device half* sc_gate = scp + (size_t)(out_row + out_vec_size) * in_vec_size_g;
            thread float alpha_up[4];
            thread float alpha_gate[4];
            thread float result_up[4] = {0};
            thread float result_gate[4] = {0};
            for (int row = 0; row < results_per_simdgroup; row++) {
                alpha_up[row] = (float)sc_up[row * in_vec_size_g];
                alpha_gate[row] = (float)sc_gate[row * in_vec_size_g];
            }
            const device half* xk = x0;
            for (int k = 0; k < in_vec_size; k += block_size) {
                float sum = ld16_b2(xk, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl_u = (const device uint8_t*)(ws_up + row * in_vec_size_w);
                    auto wl_g = (const device uint8_t*)(ws_gate + row * in_vec_size_w);
                    result_up[row] += qd2(wl_u, x_thread, alpha_up[row], -alpha_up[row], sum);
                    result_gate[row] += qd2(wl_g, x_thread, alpha_gate[row], -alpha_gate[row], sum);
                }
                ws_up += block_size * bytes_per_pack / pack_factor;
                ws_gate += block_size * bytes_per_pack / pack_factor;
                xk += block_size;
            }
            device half* yk = y + (size_t)mk * out_vec_size + out_row;
            for (int row = 0; row < results_per_simdgroup; row++) {
                float up = simd_sum(result_up[row]);
                float gate = simd_sum(result_gate[row]);
                if (simd_lid == 0) {
                    float uf = float(half(up));
                    float gf = float(half(gate));
                    gf = metal::min(gf, 7.0f);
                    uf = metal::clamp(uf, -7.0f, 7.0f);
                    float silu = gf / (1.0f + metal::exp(-gf));
                    yk[row] = half(silu * uf);
                }
            }
        }
    }

    kernel void gqmm2_rows_fold_add(
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
            if (simd_lid == 0) {
                y[row] = half(float(y[row]) + float(half(result[row])));
            }
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

    /// Fold with epilogue `α·(accum−sum)`; inner ld16_b2 / masked 16-way unchanged.
    private static let foldAMetalSource = """
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

    kernel void gqmm2_rows_fold_a(
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
                result[row] += alpha[row] * (qd2_acc(wl, x_thread) - sum);
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
