import Foundation
import Metal
import MLX

/// One Maple layer (attn + MoE) on Seedless Metal with resident KV cache.
/// Tracks absolute `offset` for RoPE; SWA writes a ring (`offset % maxLen`) after
/// the window fills. Attention is permutation-invariant over the last `maxLen` slots.
public final class SeedlessLayerBlock {
    public let layer: Int
    public let H: Int
    public let I: Int
    public let E: Int
    public let Ktop: Int
    public let gs: Int
    public let eps: Float
    public let numHeads: Int
    public let numKV: Int
    public let headDim: Int
    public let ropeDim: Int
    public let maxLen: Int
    public let isSliding: Bool
    /// Absolute tokens written (RoPE index for next write).
    public private(set) var offset: Int = 0
    /// Scratch capacity for True M-row (env-driven via stack).
    public let maxM: Int

    let hBuf: MTLBuffer
    let inNorm: MTLBuffer
    let postNorm: MTLBuffer
    let gateW: MTLBuffer
    let qkvW: MTLBuffer
    let qkvS: MTLBuffer
    let qkvB: MTLBuffer
    let oW: MTLBuffer
    let oS: MTLBuffer
    let oB: MTLBuffer
    let qkW: MTLBuffer
    let invFreq: MTLBuffer
    let densInds: MTLBuffer

    let upGateW: MTLBuffer
    let upGateS: MTLBuffer
    let upGateB: MTLBuffer
    let downW: MTLBuffer
    let downS: MTLBuffer
    let downB: MTLBuffer

    let xAttn: MTLBuffer
    let qkvOut: MTLBuffer
    let qkOut: MTLBuffer
    let attnTmp: MTLBuffer
    let attnOut: MTLBuffer
    let kCache: MTLBuffer
    let vCache: MTLBuffer

    let xMoe: MTLBuffer
    let logits: MTLBuffer
    let inds: MTLBuffer
    let scores: MTLBuffer
    let ugOut: MTLBuffer
    let act: MTLBuffer
    let downOut: MTLBuffer
    let moeOut: MTLBuffer

    public init(
        store: WeightStore, layer: Int, device: MTLDevice,
        sharedH: MTLBuffer? = nil, maxLen: Int? = nil, maxM: Int = 1
    ) throws {
        self.layer = layer
        self.maxM = max(1, maxM)
        let cfg = store.config
        H = cfg.hiddenSize
        I = cfg.moeIntermediateSize
        E = cfg.numExperts
        Ktop = cfg.numExpertsPerTok
        gs = cfg.expertGroupSize
        eps = cfg.rmsNormEps
        numHeads = cfg.numAttentionHeads
        numKV = cfg.numKeyValueHeads
        headDim = cfg.headDim
        isSliding = cfg.isSliding(layer)
        ropeDim = isSliding ? cfg.ropeDim : 0
        self.maxLen = maxLen ?? (isSliding ? cfg.slidingWindow : 2048)

        let p = "model.layers.\(layer)"
        let attn = "\(p).self_attn"
        let moe = "\(p).mlp.switch_mlp"

        let inW = store.req("\(p).input_layernorm.weight").asType(.float16)
        let postW = store.req("\(p).post_attention_layernorm.weight").asType(.float16)
        let gW = store.req("\(p).mlp.gate.weight").asType(.float16)
        let qkvW0 = store.req("\(attn).qkv_proj.weight")
        let qkvS0 = store.req("\(attn).qkv_proj.scales").asType(.float16)
        let qkvB0 = store.req("\(attn).qkv_proj.biases").asType(.float16)
        let oW0 = store.req("\(attn).o_proj.weight")
        let oS0 = store.req("\(attn).o_proj.scales").asType(.float16)
        let oB0 = store.req("\(attn).o_proj.biases").asType(.float16)
        let qNorm = store.req("\(attn).q_norm.weight").asType(.float16)
        let kNorm = store.req("\(attn).k_norm.weight").asType(.float16)
        let ugW = store.req("\(moe).up_gate_proj.weight")
        let ugS = store.req("\(moe).up_gate_proj.scales").asType(.float16)
        let ugB = store.req("\(moe).up_gate_proj.biases").asType(.float16)
        let dW = store.req("\(moe).down_proj.weight")
        let dS = store.req("\(moe).down_proj.scales").asType(.float16)
        let dB = store.req("\(moe).down_proj.biases").asType(.float16)

        let qPart = MLX.broadcast(qNorm.reshaped([1, headDim]), to: [numHeads, headDim])
        let kPart = MLX.broadcast(kNorm.reshaped([1, headDim]), to: [numKV, headDim])
        let qkWArr = MLX.contiguous(MLX.concatenated([qPart, kPart], axis: 0)).asType(.float16)
        let half = max(ropeDim / 2, 1)
        var freqs = [Float](repeating: 1, count: half)
        if ropeDim > 0 {
            let base = cfg.ropeTheta
            for i in 0 ..< half {
                freqs[i] = pow(base, -Float(i) / Float(half))
            }
        }
        MLX.eval([inW, postW, gW, qkvW0, qkvS0, qkvB0, oW0, oS0, oB0, qkWArr,
                  ugW, ugS, ugB, dW, dS, dB])

        func mtl(_ a: MLXArray) throws -> MTLBuffer {
            guard let b = SeedlessMetal.mtlBuf(a, device) else { throw SeedlessError.notReady }
            return b
        }
        // Derived locals die at end of init; noCopy aliases would clobber inv_freq
        // (pos=0 RoPE is identity, so the dangling table was invisible).
        func owned(_ a: MLXArray) throws -> MTLBuffer {
            guard let b = SeedlessMetal.mtlBufHostCopy(a, device) else {
                throw SeedlessError.notReady
            }
            return b
        }
        inNorm = try mtl(inW)
        postNorm = try mtl(postW)
        gateW = try mtl(gW)
        qkvW = try mtl(qkvW0)
        qkvS = try mtl(qkvS0)
        qkvB = try mtl(qkvB0)
        oW = try mtl(oW0)
        oS = try mtl(oS0)
        oB = try mtl(oB0)
        qkW = try owned(qkWArr)
        let invBytes = freqs.count * MemoryLayout<Float>.size
        guard let invBuf = device.makeBuffer(
            length: invBytes, options: .storageModeShared)
        else { throw SeedlessError.notReady }
        freqs.withUnsafeBytes { raw in
            invBuf.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        invFreq = invBuf
        upGateW = try mtl(ugW)
        upGateS = try mtl(ugS)
        upGateB = try mtl(ugB)
        downW = try mtl(dW)
        downS = try mtl(dS)
        downB = try mtl(dB)

        hBuf = sharedH ?? device.makeBuffer(length: self.maxM * H * 2, options: .storageModeShared)!
        densInds = device.makeBuffer(length: 4, options: .storageModeShared)!
        densInds.contents().storeBytes(of: Int32(0), as: Int32.self)

        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let qkvN = qDim + 2 * kvDim
        let m = self.maxM
        xAttn = device.makeBuffer(length: m * H * 2, options: .storageModeShared)!
        qkvOut = device.makeBuffer(length: m * qkvN * 2, options: .storageModeShared)!
        qkOut = device.makeBuffer(length: m * (qDim + kvDim) * 2, options: .storageModeShared)!
        attnTmp = device.makeBuffer(length: m * qDim * 2, options: .storageModeShared)!
        attnOut = device.makeBuffer(length: m * H * 2, options: .storageModeShared)!
        kCache = device.makeBuffer(length: numKV * self.maxLen * headDim * 2, options: .storageModeShared)!
        vCache = device.makeBuffer(length: numKV * self.maxLen * headDim * 2, options: .storageModeShared)!
        memset(kCache.contents(), 0, numKV * self.maxLen * headDim * 2)
        memset(vCache.contents(), 0, numKV * self.maxLen * headDim * 2)

        xMoe = device.makeBuffer(length: m * H * 2, options: .storageModeShared)!
        logits = device.makeBuffer(length: m * E * 4, options: .storageModeShared)!
        inds = device.makeBuffer(length: m * Ktop * 4, options: .storageModeShared)!
        scores = device.makeBuffer(length: m * Ktop * 4, options: .storageModeShared)!
        ugOut = device.makeBuffer(length: m * Ktop * 2 * I * 2, options: .storageModeShared)!
        act = device.makeBuffer(length: m * Ktop * I * 2, options: .storageModeShared)!
        downOut = device.makeBuffer(length: m * Ktop * H * 2, options: .storageModeShared)!
        moeOut = device.makeBuffer(length: m * H * 2, options: .storageModeShared)!
    }

    public func resetCache() {
        offset = 0
        memset(kCache.contents(), 0, numKV * maxLen * headDim * 2)
        memset(vCache.contents(), 0, numKV * maxLen * headDim * 2)
    }

    /// Filled KV timesteps that decode actually reads (`offset`, or full SWA window).
    public static func liveSeq(offset: Int, maxLen: Int, isSliding: Bool) -> Int {
        if isSliding && offset >= maxLen { return maxLen }
        return min(max(0, offset), maxLen)
    }

    /// Ring slot for the token about to be written. Sliding layers wrap after
    /// `maxLen`; full-attn layers append until capacity.
    public static func kvWritePos(offset: Int, maxLen: Int, isSliding: Bool) -> Int {
        guard isSliding, maxLen > 0, offset >= maxLen else { return offset }
        return offset % maxLen
    }

    /// SDPA `N` while writing the token at `offset` (before increment).
    public static func kvSeqLen(offset: Int, maxLen: Int, isSliding: Bool) -> Int {
        if isSliding && offset >= maxLen { return maxLen }
        return offset + 1
    }

    public static func liveBytes(seq: Int, numKV: Int, headDim: Int) -> Int {
        numKV * seq * headDim * 2
    }

    /// True when `steps` append-only KV writes will not `maple_shift_kv`.
    /// Then reject rollback is `offset` rewind; dirty tail is unread.
    public static func canRewind(isSliding: Bool, offset: Int, maxLen: Int, steps: Int) -> Bool {
        guard steps >= 0 else { return false }
        if !isSliding { return true }
        return offset + steps <= maxLen
    }

    var liveSeq: Int { Self.liveSeq(offset: offset, maxLen: maxLen, isSliding: isSliding) }

    /// Per-head live prefix (layout `[numKV][maxLen][headDim]`). Full allocation is not copied.
    func snapshotCache() -> (offset: Int, k: Data, v: Data) {
        let seq = liveSeq
        let perHead = seq * headDim * 2
        let stride = maxLen * headDim * 2
        var k = Data(count: numKV * perHead)
        var v = Data(count: numKV * perHead)
        guard perHead > 0 else { return (offset, k, v) }
        k.withUnsafeMutableBytes { kd in
            v.withUnsafeMutableBytes { vd in
                guard let kDst = kd.baseAddress, let vDst = vd.baseAddress else { return }
                let kSrc = kCache.contents()
                let vSrc = vCache.contents()
                for h in 0 ..< numKV {
                    memcpy(kDst.advanced(by: h * perHead), kSrc.advanced(by: h * stride), perHead)
                    memcpy(vDst.advanced(by: h * perHead), vSrc.advanced(by: h * stride), perHead)
                }
            }
        }
        return (offset, k, v)
    }

    func restoreCache(offset: Int, k: Data, v: Data) {
        self.offset = offset
        let seq = Self.liveSeq(offset: offset, maxLen: maxLen, isSliding: isSliding)
        let perHead = seq * headDim * 2
        let expect = numKV * perHead
        precondition(k.count == expect && v.count == expect)
        guard perHead > 0 else { return }
        let stride = maxLen * headDim * 2
        k.withUnsafeBytes { kd in
            v.withUnsafeBytes { vd in
                guard let kSrc = kd.baseAddress, let vSrc = vd.baseAddress else { return }
                let kDst = kCache.contents()
                let vDst = vCache.contents()
                for h in 0 ..< numKV {
                    memcpy(kDst.advanced(by: h * stride), kSrc.advanced(by: h * perHead), perHead)
                    memcpy(vDst.advanced(by: h * stride), vSrc.advanced(by: h * perHead), perHead)
                }
            }
        }
    }

    /// Encode `M` decode tokens at current `offset`, then advance offset by M.
    /// `M>1` requires True M-row scratch (`maxM`) and no SWA rotate in the window.
    public func encodeStep(into enc: MTLComputeCommandEncoder, M: Int = 1) throws {
        precondition(M >= 1 && M <= maxM)
        if !isSliding && offset + M > maxLen {
            throw SeedlessError.unsupportedShape(N: offset + M, K: maxLen, gs: 0)
        }
        // SWA wrap is M=1 only (ring overwrite). M-row stays append-only.
        if isSliding && M > 1 && offset + M > maxLen {
            throw SeedlessError.unsupportedShape(N: offset + M, K: maxLen, gs: 0)
        }
        let writePos = Self.kvWritePos(offset: offset, maxLen: maxLen, isSliding: isSliding)
        let seqLen = Self.kvSeqLen(offset: offset, maxLen: maxLen, isSliding: isSliding)
        let ropePos = offset
        try SeedlessMetal.encodeLayerBlock(
            into: enc, h: hBuf, inNorm: inNorm, postNorm: postNorm, gateW: gateW,
            qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
            oW: oW, oS: oS, oB: oB,
            qkW: qkW, invFreq: invFreq, densInds: densInds,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xAttn: xAttn, qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
            kCache: kCache, vCache: vCache,
            xMoe: xMoe, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs,
            numHeads: numHeads, numKV: numKV, headDim: headDim, ropeDim: ropeDim,
            ropePos: ropePos, writePos: writePos, maxLen: maxLen, seqLen: seqLen,
            rotateFirst: false, M: M)
        offset += M
    }

    func rewindOffset(_ newOffset: Int) {
        precondition(newOffset >= 0 && newOffset <= offset)
        offset = newOffset
    }

    /// True when this layer can pack `M` tokens without SWA rotate / overflow.
    public func canEncodeMrow(M: Int) -> Bool {
        guard M >= 1, M <= maxM else { return false }
        if !isSliding { return offset + M <= maxLen }
        if offset >= maxLen { return M == 1 }  // rotateFirst is M=1 only
        return offset + M <= maxLen
    }

    /// Encode without advancing offset (microbench helper).
    public func encode(into enc: MTLComputeCommandEncoder, at pos: Int) throws {
        let writePos = Self.kvWritePos(offset: pos, maxLen: maxLen, isSliding: isSliding)
        let seqLen = Self.kvSeqLen(offset: pos, maxLen: maxLen, isSliding: isSliding)
        try SeedlessMetal.encodeLayerBlock(
            into: enc, h: hBuf, inNorm: inNorm, postNorm: postNorm, gateW: gateW,
            qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
            oW: oW, oS: oS, oB: oB,
            qkW: qkW, invFreq: invFreq, densInds: densInds,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xAttn: xAttn, qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
            kCache: kCache, vCache: vCache,
            xMoe: xMoe, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs,
            numHeads: numHeads, numKV: numKV, headDim: headDim, ropeDim: ropeDim,
            ropePos: pos, writePos: writePos, maxLen: maxLen, seqLen: seqLen, rotateFirst: false)
    }

    public func encodeAttnOnly(into enc: MTLComputeCommandEncoder, at pos: Int) throws {
        let writePos = Self.kvWritePos(offset: pos, maxLen: maxLen, isSliding: isSliding)
        let seqLen = Self.kvSeqLen(offset: pos, maxLen: maxLen, isSliding: isSliding)
        SeedlessMetal.encodeRms(into: enc, h: hBuf, w: inNorm, out: xAttn, H: H, eps: eps)
        try SeedlessMetal.encodeAttnBlock(
            into: enc, xNorm: xAttn,
            qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
            oW: oW, oS: oS, oB: oB,
            qkW: qkW, invFreq: invFreq, densInds: densInds,
            qkvOut: qkvOut, qkOut: qkOut, attnTmp: attnTmp, attnOut: attnOut,
            kCache: kCache, vCache: vCache,
            H: H, numHeads: numHeads, numKV: numKV, headDim: headDim,
            ropeDim: ropeDim, ropePos: pos, writePos: writePos, maxLen: maxLen, seqLen: seqLen,
            eps: eps, gs: gs, rotateFirst: false)
        SeedlessMetal.encodeResid(into: enc, h: hBuf, delta: attnOut, H: H)
    }
}

/// Full decoder stack of attn+MoE layers.
public final class SeedlessLayerStack {
    public let layers: [SeedlessLayerBlock]
    public let hBuf: MTLBuffer
    public let H: Int
    public let maxM: Int

    public init(store: WeightStore, device: MTLDevice, fullMaxLen: Int = 2048, maxM: Int = 1) throws {
        let n = store.config.numHiddenLayers
        H = store.config.hiddenSize
        self.maxM = max(1, maxM)
        hBuf = device.makeBuffer(length: self.maxM * H * 2, options: .storageModeShared)!
        var built: [SeedlessLayerBlock] = []
        built.reserveCapacity(n)
        for i in 0 ..< n {
            let ml = store.config.isSliding(i) ? store.config.slidingWindow : fullMaxLen
            built.append(try SeedlessLayerBlock(
                store: store, layer: i, device: device, sharedH: hBuf, maxLen: ml, maxM: self.maxM))
        }
        layers = built
    }

    public func canEncodeMrow(M: Int) -> Bool {
        layers.allSatisfy { $0.canEncodeMrow(M: M) }
    }

    /// Append-only chain of `n` tokens: offset rewind is a correct KV rollback.
    public func canRewindAfter(steps n: Int) -> Bool {
        layers.allSatisfy {
            SeedlessLayerBlock.canRewind(
                isSliding: $0.isSliding, offset: $0.offset, maxLen: $0.maxLen, steps: n)
        }
    }

    public func fillH(_ v: Float16) {
        let p = hBuf.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { p[i] = v }
    }

    public func copyH(from src: UnsafePointer<Float16>) {
        hBuf.contents().copyMemory(from: src, byteCount: H * 2)
    }

    public func resetCaches() {
        for l in layers { l.resetCache() }
    }

    /// Commit layer CBs in groups of `layersPerCB`, wait once on last.
    /// Optionally encode `tail` into the last CB. `M>1` is causal M-row prefill/draft.
    public func stepCommitWait(
        layersPerCB: Int = 1, M: Int = 1, tail: ((MTLComputeCommandEncoder) -> Void)? = nil
    ) throws {
        let last = try stepCommit(layersPerCB: layersPerCB, M: M, tail: tail)
        last?.waitUntilCompleted()
    }

    /// Like `stepCommitWait` but does not wait — caller waits on the returned buffer.
    @discardableResult
    public func stepCommit(
        layersPerCB: Int = 1, M: Int = 1, tail: ((MTLComputeCommandEncoder) -> Void)? = nil
    ) throws -> MTLCommandBuffer? {
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        let g = max(1, layersPerCB)
        var last: MTLCommandBuffer?
        var i = 0
        while i < layers.count {
            let cb = q.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            let end = min(i + g, layers.count)
            do {
                for j in i ..< end {
                    try layers[j].encodeStep(into: enc, M: M)
                    enc.memoryBarrier(scope: .buffers)
                }
                if end == layers.count, let tail { tail(enc) }
                enc.endEncoding()
            } catch {
                // Metal asserts if an encoder is released without endEncoding.
                enc.endEncoding()
                throw error
            }
            cb.commit()
            last = cb
            i = end
        }
        return last
    }

    /// CPU encode vs GPU fill for `layersPerCB` command buffers (ICB gate).
    public func profileEncodeVsGpu(layersPerCB: Int = 4, iters: Int = 8, pos: Int = 32) throws -> String {
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        let g = max(1, layersPerCB)
        // Warm: fill the GPU so later timings are not first-dispatch noise.
        for _ in 0 ..< 2 {
            fillH(0.01)
            resetCaches()
            _ = try SeedlessMetal.timeCB { enc in
                try layers[0].encode(into: enc, at: pos)
            }
        }

        func run(waitEach: Bool) throws -> (encode: Double, gpu: Double, span: Double, wall: Double) {
            var encodeSum = 0.0, gpuSum = 0.0, spanSum = 0.0, wallSum = 0.0
            for _ in 0 ..< iters {
                fillH(0.01)
                resetCaches()
                let t0 = CFAbsoluteTimeGetCurrent()
                var cbs: [MTLCommandBuffer] = []
                var encodeMs = 0.0
                var i = 0
                while i < layers.count {
                    let te = CFAbsoluteTimeGetCurrent()
                    let cb = q.makeCommandBuffer()!
                    let enc = cb.makeComputeCommandEncoder()!
                    let end = min(i + g, layers.count)
                    do {
                        for j in i ..< end {
                            try layers[j].encode(into: enc, at: pos)
                        }
                        enc.endEncoding()
                    } catch {
                        enc.endEncoding()
                        throw error
                    }
                    encodeMs += (CFAbsoluteTimeGetCurrent() - te) * 1000
                    cb.commit()
                    if waitEach { cb.waitUntilCompleted() }
                    cbs.append(cb)
                    i = end
                }
                cbs.last?.waitUntilCompleted()
                wallSum += (CFAbsoluteTimeGetCurrent() - t0) * 1000
                encodeSum += encodeMs
                gpuSum += cbs.reduce(0.0) { $0 + max(0, ($1.gpuEndTime - $1.gpuStartTime) * 1000) }
                if let f = cbs.first, let l = cbs.last, f.gpuStartTime > 0, l.gpuEndTime > f.gpuStartTime {
                    spanSum += (l.gpuEndTime - f.gpuStartTime) * 1000
                }
            }
            let n = Double(iters)
            return (encodeSum / n, gpuSum / n, spanSum / n, wallSum / n)
        }

        let piped = try run(waitEach: false)
        let isolated = try run(waitEach: true)
        let nCB = (layers.count + g - 1) / g
        func row(_ name: String, _ t: (encode: Double, gpu: Double, span: Double, wall: Double)) -> String {
            String(
                format: "  %-10@ encode=%.3f ms  gpu_sum=%.3f  gpu_span=%.3f  wall=%.3f  encode/gpu=%.2f",
                name as NSString, t.encode, t.gpu, t.span, t.wall, t.encode / max(t.gpu, 1e-9))
        }
        return """
        seedless encode vs GPU (layersPerCB=\(g), \(nCB) CBs, pos=\(pos), \(iters) iters)
        \(row("piped", piped))
        \(row("wait-each", isolated))
        note: ICB only helps if CPU encode is a large fraction of GPU fill (encode/gpu ≳ 0.2 piped).
        """
    }

    public static func profileEncodeVsGpu(
        store: WeightStore, layersPerCB: Int = 4, iters: Int = 8, pos: Int = 32
    ) throws -> String {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        fputs("[momij] encode-vs-gpu profile: loading layers…\n", stderr)
        let stack = try SeedlessLayerStack(store: store, device: device)
        return try stack.profileEncodeVsGpu(layersPerCB: layersPerCB, iters: iters, pos: pos)
    }

    /// Host-side KV + offset snapshot for SuffixSpec reject rollback.
    /// `offsetOnly`: rewind `offset` without copying caches (append-only chains).
    public struct CacheSnapshot {
        public let offsets: [Int]
        public let kData: [Data]
        public let vData: [Data]
        public let offsetOnly: Bool
    }

    public func snapshotOffsets() -> CacheSnapshot {
        CacheSnapshot(
            offsets: layers.map(\.offset),
            kData: Array(repeating: Data(), count: layers.count),
            vData: Array(repeating: Data(), count: layers.count),
            offsetOnly: true)
    }

    public func snapshotForChain(steps: Int) -> CacheSnapshot {
        canRewindAfter(steps: steps) ? snapshotOffsets() : snapshotCaches()
    }

    public func snapshotCaches() -> CacheSnapshot {
        var offsets: [Int] = []
        var kData: [Data] = []
        var vData: [Data] = []
        offsets.reserveCapacity(layers.count)
        kData.reserveCapacity(layers.count)
        vData.reserveCapacity(layers.count)
        for l in layers {
            offsets.append(l.offset)
            let snap = l.snapshotCache()
            kData.append(snap.k)
            vData.append(snap.v)
        }
        return CacheSnapshot(offsets: offsets, kData: kData, vData: vData, offsetOnly: false)
    }

    public func restoreCaches(_ snap: CacheSnapshot) {
        precondition(snap.offsets.count == layers.count)
        if snap.offsetOnly {
            for (i, l) in layers.enumerated() {
                l.rewindOffset(snap.offsets[i])
            }
            return
        }
        for (i, l) in layers.enumerated() {
            l.restoreCache(offset: snap.offsets[i], k: snap.kData[i], v: snap.vData[i])
        }
    }

    public func encodeAll(into enc: MTLComputeCommandEncoder, pos: Int) throws {
        for layer in layers {
            try layer.encode(into: enc, at: pos)
        }
    }
}

extension SeedlessEngine {
    /// Milestone B profile: attn-only, layer(attn+moe), 24L real 1-CB. Fixed pos=0 (cold cache).
    public static func profileLayerStack(store: WeightStore, iters: Int = 12) throws -> String {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        fputs("[momij] loading \(store.config.numHiddenLayers) attn+MoE layers…\n", stderr)
        let stack = try SeedlessLayerStack(store: store, device: device)
        let n = stack.layers.count
        let l0 = stack.layers[0]

        stack.fillH(0.01)
        stack.resetCaches()
        for _ in 0 ..< 3 {
            _ = try SeedlessMetal.timeCB { enc in try l0.encode(into: enc, at: 0) }
        }

        func avg(_ count: Int, reset: Bool = true, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> (wall: Double, gpu: Double) {
            var w = 0.0, g = 0.0
            for _ in 0 ..< count {
                stack.fillH(0.01)
                if reset { stack.resetCaches() }
                let t = try SeedlessMetal.timeCB(body)
                w += t.wallMs; g += t.gpuMs
            }
            return (w / Double(count), g / Double(count))
        }

        let moeOnly = try avg(iters) { enc in
            try SeedlessMetal.encodeMoEBlock(
                into: enc, h: l0.hBuf, normW: l0.postNorm, gateW: l0.gateW,
                upGateW: l0.upGateW, upGateS: l0.upGateS, upGateB: l0.upGateB,
                downW: l0.downW, downS: l0.downS, downB: l0.downB,
                xNorm: l0.xMoe, logits: l0.logits, inds: l0.inds, scores: l0.scores,
                ugOut: l0.ugOut, act: l0.act, downOut: l0.downOut, moeOut: l0.moeOut,
                H: l0.H, I: l0.I, E: l0.E, Ktop: l0.Ktop, eps: l0.eps, gs: l0.gs)
        }
        let attnOnly = try avg(iters) { enc in
            try l0.encodeAttnOnly(into: enc, at: 0)
        }
        let layer1 = try avg(iters) { enc in
            try l0.encode(into: enc, at: 0)
        }
        let real24 = try avg(max(6, iters / 2)) { enc in
            try stack.encodeAll(into: enc, pos: 0)
        }
        let commitWait: (wall: Double, gpu: Double) = try {
            var w = 0.0, g = 0.0
            let count = max(6, iters / 2)
            guard let q = SeedlessMetal.queue else { return (0.0, 0.0) }
            for _ in 0 ..< count {
                stack.fillH(0.01)
                stack.resetCaches()
                let t0 = CFAbsoluteTimeGetCurrent()
                var cbs: [MTLCommandBuffer] = []
                cbs.reserveCapacity(n)
                for layer in stack.layers {
                    let cb = q.makeCommandBuffer()!
                    let enc = cb.makeComputeCommandEncoder()!
                    try layer.encode(into: enc, at: 0)
                    enc.endEncoding()
                    cb.commit()
                    cbs.append(cb)
                }
                cbs.last!.waitUntilCompleted()
                w += (CFAbsoluteTimeGetCurrent() - t0) * 1000
                if let f = cbs.first, let l = cbs.last, f.gpuStartTime > 0, l.gpuEndTime > f.gpuStartTime {
                    g += (l.gpuEndTime - f.gpuStartTime) * 1000
                } else {
                    g += cbs.reduce(0.0) { $0 + max(0, ($1.gpuEndTime - $1.gpuStartTime) * 1000) }
                }
            }
            return (w / Double(count), g / Double(count))
        }()
        var waitWall = 0.0, waitGpu = 0.0
        let waitCount = max(4, iters / 3)
        for _ in 0 ..< waitCount {
            stack.fillH(0.01)
            stack.resetCaches()
            let t0 = CFAbsoluteTimeGetCurrent()
            var gpuSum = 0.0
            for layer in stack.layers {
                let t = try SeedlessMetal.timeCB { enc in try layer.encode(into: enc, at: 0) }
                gpuSum += t.gpuMs
            }
            waitWall += (CFAbsoluteTimeGetCurrent() - t0) * 1000
            waitGpu += gpuSum
        }
        let wait24 = (wall: waitWall / Double(waitCount), gpu: waitGpu / Double(waitCount))

        func line(_ name: String, _ t: (wall: Double, gpu: Double)) -> String {
            let busy = t.gpu / max(t.wall, 1e-9) * 100
            let tpsW = 1000.0 / t.wall
            let tpsG = 1000.0 / t.gpu
            return String(format: "  %-14@ wall=%.3f ms  gpu=%.3f ms  busy=%.0f%%  → tok/s wall=%.1f gpu=%.1f",
                          name as NSString, t.wall, t.gpu, busy, tpsW, tpsG)
        }

        let hp = stack.hBuf.contents().bindMemory(to: Float16.self, capacity: stack.H)
        let finite = (0 ..< min(8, stack.H)).allSatisfy { hp[$0].isFinite }

        return """
        seedless layer (attn+MoE) Milestone B profile — pos=0 cold cache, \(iters) iters
        \(line("moe-only L0", moeOnly))
        \(line("attn+resid L0", attnOnly))
        \(line("layer L0", layer1))
        \(line("24L real 1cb", real24))
        \(line("24L commit→1w", commitWait))
        \(line("24L ×wait", wait24))
        h finite: \(finite)  (no SWA rotate / no embed-lm_head; pos=0 N=1 best-case attn)
        note: tok/s floors are decode-layer estimate at fixed pos=0; growing KV will add cost.
        """
    }

    /// Per-op GPU floor inside one decode layer (attn + MoE), at cold and warm KV.
    public static func profileDecodeFloor(store: WeightStore, iters: Int = 20) throws -> String {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        fputs("[momij] decode-floor profile: loading layers…\n", stderr)
        let stack = try SeedlessLayerStack(store: store, device: device)
        let swa = stack.layers[0]
        let full = stack.layers.first(where: { !$0.isSliding }) ?? stack.layers[min(3, stack.layers.count - 1)]
        let warmPos = 127

        func fillKV(_ layer: SeedlessLayerBlock, upTo pos: Int) throws {
            layer.resetCache()
            stack.fillH(0.01)
            for p in 0 ..< pos {
                _ = try SeedlessMetal.timeCB { enc in try layer.encode(into: enc, at: p) }
            }
        }

        func avg(_ n: Int, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
            var g = 0.0
            for _ in 0 ..< n {
                let t = try SeedlessMetal.timeCB(body)
                g += t.gpuMs
            }
            return g / Double(n)
        }

        func breakdown(_ layer: SeedlessLayerBlock, pos: Int, label: String) throws -> String {
            let H = layer.H, I = layer.I, E = layer.E, Ktop = layer.Ktop, gs = layer.gs, eps = layer.eps
            let qDim = layer.numHeads * layer.headDim
            let kvDim = layer.numKV * layer.headDim
            let qkvN = qDim + 2 * kvDim
            let rotate = layer.isSliding && pos >= layer.maxLen
            let writePos = rotate ? layer.maxLen - 1 : pos
            let seqLen = rotate ? layer.maxLen : pos + 1

            stack.fillH(0.01)
            // Warm pipelines
            for _ in 0 ..< 3 {
                _ = try SeedlessMetal.timeCB { enc in try layer.encode(into: enc, at: pos) }
            }

            let inRms = try avg(iters) { enc in
                SeedlessMetal.encodeRms(into: enc, h: layer.hBuf, w: layer.inNorm, out: layer.xAttn, H: H, eps: eps)
            }
            let qkv = try avg(iters) { enc in
                try SeedlessMetal.gqmm2(
                    x: layer.xAttn, w: layer.qkvW, scales: layer.qkvS, biases: layer.qkvB,
                    inds: layer.densInds, out: layer.qkvOut,
                    Ktop: 1, K: H, N: qkvN, gs: gs, into: enc)
            }
            let oProj = try avg(iters) { enc in
                try SeedlessMetal.gqmm2(
                    x: layer.attnTmp, w: layer.oW, scales: layer.oS, biases: layer.oB,
                    inds: layer.densInds, out: layer.attnOut,
                    Ktop: 1, K: qDim, N: H, gs: gs, into: enc)
            }
            let attnFull = try avg(iters) { enc in
                try layer.encodeAttnOnly(into: enc, at: pos)
            }
            let moeFull = try avg(iters) { enc in
                try SeedlessMetal.encodeMoEBlock(
                    into: enc, h: layer.hBuf, normW: layer.postNorm, gateW: layer.gateW,
                    upGateW: layer.upGateW, upGateS: layer.upGateS, upGateB: layer.upGateB,
                    downW: layer.downW, downS: layer.downS, downB: layer.downB,
                    xNorm: layer.xMoe, logits: layer.logits, inds: layer.inds, scores: layer.scores,
                    ugOut: layer.ugOut, act: layer.act, downOut: layer.downOut, moeOut: layer.moeOut,
                    H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs)
            }
            let moeRms = try avg(iters) { enc in
                SeedlessMetal.encodeRms(into: enc, h: layer.hBuf, w: layer.postNorm, out: layer.xMoe, H: H, eps: eps)
            }
            let gate = try avg(iters) { enc in
                SeedlessMetal.encodeGate(into: enc, w: layer.gateW, x: layer.xMoe, y: layer.logits, E: E, H: H)
            }
            let route = try avg(iters) { enc in
                SeedlessMetal.encodeRoute(into: enc, logits: layer.logits, inds: layer.inds,
                                          scores: layer.scores, E: E, Ktop: Ktop)
            }
            let up = try avg(iters) { enc in
                try SeedlessMetal.gqmm2(
                    x: layer.xMoe, w: layer.upGateW, scales: layer.upGateS, biases: layer.upGateB,
                    inds: layer.inds, out: layer.ugOut,
                    Ktop: Ktop, K: H, N: 2 * I, gs: gs, into: enc)
            }
            let swiglu = try avg(iters) { enc in
                SeedlessMetal.encodeClampedSwiglu(into: enc, ug: layer.ugOut, act: layer.act, I: I, Ktop: Ktop)
            }
            let down = try avg(iters) { enc in
                try SeedlessMetal.gqmm2(
                    x: layer.act, w: layer.downW, scales: layer.downS, biases: layer.downB,
                    inds: layer.inds, out: layer.downOut,
                    Ktop: Ktop, K: I, N: H, gs: gs, lhsPerExpert: true, into: enc)
            }
            let layerFull = try avg(iters) { enc in
                try layer.encode(into: enc, at: pos)
            }

            // encodeAttnOnly = inRms + attn body + resid; residual ≈ sdpa+qk+kv+resid
            let sdpaish = max(0, attnFull - inRms - qkv - oProj)
            _ = kvDim; _ = writePos; _ = rotate

            func pct(_ x: Double, _ tot: Double) -> Double { 100 * x / max(tot, 1e-9) }
            let sumParts = inRms + qkv + sdpaish + oProj + moeRms + gate + route + up + swiglu + down
            return String(format: """
              [%@] pos=%d seqLen=%d layer_gpu=%.3f ms
                attn: in_rms=%.3f  qkv=%.3f  sdpa+qk+kv≈%.3f  o_proj=%.3f  attn_full=%.3f
                moe:  rms=%.3f  gate=%.3f  route=%.3f  gqmm2_up=%.3f  swiglu=%.3f  gqmm2_dn=%.3f  moe_full=%.3f
                share%% of sum_parts(%.3f): qkv=%.0f o=%.0f sdpaish=%.0f up=%.0f dn=%.0f gate=%.0f
              """,
                label as NSString, pos, seqLen, layerFull,
                inRms, qkv, sdpaish, oProj, attnFull,
                moeRms, gate, route, up, swiglu, down, moeFull,
                sumParts,
                pct(qkv, sumParts), pct(oProj, sumParts), pct(sdpaish, sumParts),
                pct(up, sumParts), pct(down, sumParts), pct(gate, sumParts))
        }

        var lines: [String] = ["seedless decode-floor per-op GPU (own CB each; launch tax included)"]
        try fillKV(swa, upTo: 0)
        lines.append(try breakdown(swa, pos: 0, label: "SWA L0 cold"))
        try fillKV(swa, upTo: warmPos)
        lines.append(try breakdown(swa, pos: warmPos, label: "SWA L0 warm"))
        try fillKV(full, upTo: 0)
        lines.append(try breakdown(full, pos: 0, label: "full-attn cold"))
        try fillKV(full, upTo: warmPos)
        lines.append(try breakdown(full, pos: warmPos, label: "full-attn warm"))

        // Packed 24L at warm-ish: advance all layers once then measure commit→1w
        stack.resetCaches()
        stack.fillH(0.01)
        for _ in 0 ..< min(32, warmPos) {
            try stack.stepCommitWait(layersPerCB: 4)
            stack.fillH(0.01)
        }
        var packW = 0.0, packG = 0.0
        let packN = 8
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        for _ in 0 ..< packN {
            stack.fillH(0.01)
            let t0 = CFAbsoluteTimeGetCurrent()
            var cbs: [MTLCommandBuffer] = []
            let g = 4
            var i = 0
            while i < stack.layers.count {
                let cb = q.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                let end = min(i + g, stack.layers.count)
                for j in i ..< end { try stack.layers[j].encodeStep(into: enc) }
                enc.endEncoding()
                cb.commit()
                cbs.append(cb)
                i = end
            }
            cbs.last!.waitUntilCompleted()
            packW += (CFAbsoluteTimeGetCurrent() - t0) * 1000
            if let f = cbs.first, let l = cbs.last, f.gpuStartTime > 0, l.gpuEndTime > f.gpuStartTime {
                packG += (l.gpuEndTime - f.gpuStartTime) * 1000
            }
        }
        lines.append(String(format: "  24L commit→1w (after ~32 steps) wall=%.3f ms gpu=%.3f ms → ~%.0f tok/s gpu",
                            packW / Double(packN), packG / Double(packN),
                            1000.0 / max(packG / Double(packN), 1e-9)))
        lines.append("note: per-op times are solo-CB (overstate launch); use shares to rank kernel work.")
        return lines.joined(separator: "\n")
    }
}
