import Foundation
import Metal
import MLX

/// One Maple layer (attn + MoE) on Seedless Metal with resident KV cache.
/// Tracks absolute `offset` for RoPE; SWA rotates when offset >= maxLen.
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

    public init(store: WeightStore, layer: Int, device: MTLDevice, sharedH: MTLBuffer? = nil, maxLen: Int? = nil) throws {
        self.layer = layer
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
        let invArr = MLXArray(freqs)
        MLX.eval([inW, postW, gW, qkvW0, qkvS0, qkvB0, oW0, oS0, oB0, qkWArr, invArr,
                  ugW, ugS, ugB, dW, dS, dB])

        func mtl(_ a: MLXArray) throws -> MTLBuffer {
            guard let b = SeedlessMetal.mtlBuf(a, device) else { throw SeedlessError.notReady }
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
        qkW = try mtl(qkWArr)
        invFreq = try mtl(invArr)
        upGateW = try mtl(ugW)
        upGateS = try mtl(ugS)
        upGateB = try mtl(ugB)
        downW = try mtl(dW)
        downS = try mtl(dS)
        downB = try mtl(dB)

        hBuf = sharedH ?? device.makeBuffer(length: H * 2, options: .storageModeShared)!
        densInds = device.makeBuffer(length: 4, options: .storageModeShared)!
        densInds.contents().storeBytes(of: Int32(0), as: Int32.self)

        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let qkvN = qDim + 2 * kvDim
        xAttn = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        qkvOut = device.makeBuffer(length: qkvN * 2, options: .storageModeShared)!
        qkOut = device.makeBuffer(length: (qDim + kvDim) * 2, options: .storageModeShared)!
        attnTmp = device.makeBuffer(length: qDim * 2, options: .storageModeShared)!
        attnOut = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        kCache = device.makeBuffer(length: numKV * self.maxLen * headDim * 2, options: .storageModeShared)!
        vCache = device.makeBuffer(length: numKV * self.maxLen * headDim * 2, options: .storageModeShared)!
        memset(kCache.contents(), 0, numKV * self.maxLen * headDim * 2)
        memset(vCache.contents(), 0, numKV * self.maxLen * headDim * 2)

        xMoe = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        logits = device.makeBuffer(length: E * 4, options: .storageModeShared)!
        inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        scores = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        ugOut = device.makeBuffer(length: Ktop * 2 * I * 2, options: .storageModeShared)!
        act = device.makeBuffer(length: Ktop * I * 2, options: .storageModeShared)!
        downOut = device.makeBuffer(length: Ktop * H * 2, options: .storageModeShared)!
        moeOut = device.makeBuffer(length: H * 2, options: .storageModeShared)!
    }

    public func resetCache() {
        offset = 0
        memset(kCache.contents(), 0, numKV * maxLen * headDim * 2)
        memset(vCache.contents(), 0, numKV * maxLen * headDim * 2)
    }

    /// Encode one decode step at current `offset`, then advance offset.
    public func encodeStep(into enc: MTLComputeCommandEncoder) throws {
        if !isSliding && offset >= maxLen {
            throw SeedlessError.unsupportedShape(N: offset, K: maxLen, gs: 0)
        }
        let rotate = isSliding && offset >= maxLen
        let writePos = rotate ? maxLen - 1 : offset
        let seqLen = rotate ? maxLen : offset + 1
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
            ropePos: ropePos, writePos: writePos, maxLen: maxLen, seqLen: seqLen, rotateFirst: rotate)
        offset += 1
    }

    /// Encode without advancing offset (microbench helper).
    public func encode(into enc: MTLComputeCommandEncoder, at pos: Int) throws {
        let rotate = isSliding && pos >= maxLen
        let writePos = rotate ? maxLen - 1 : pos
        let seqLen = rotate ? maxLen : pos + 1
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
            ropePos: pos, writePos: writePos, maxLen: maxLen, seqLen: seqLen, rotateFirst: rotate)
    }

    public func encodeAttnOnly(into enc: MTLComputeCommandEncoder, at pos: Int) throws {
        let rotate = isSliding && pos >= maxLen
        let writePos = rotate ? maxLen - 1 : pos
        let seqLen = rotate ? maxLen : pos + 1
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
            eps: eps, gs: gs, rotateFirst: rotate)
        SeedlessMetal.encodeResid(into: enc, h: hBuf, delta: attnOut, H: H)
    }
}

/// Full decoder stack of attn+MoE layers.
public final class SeedlessLayerStack {
    public let layers: [SeedlessLayerBlock]
    public let hBuf: MTLBuffer
    public let H: Int

    public init(store: WeightStore, device: MTLDevice, fullMaxLen: Int = 2048) throws {
        let n = store.config.numHiddenLayers
        H = store.config.hiddenSize
        hBuf = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        var built: [SeedlessLayerBlock] = []
        built.reserveCapacity(n)
        for i in 0 ..< n {
            let ml = store.config.isSliding(i) ? store.config.slidingWindow : fullMaxLen
            built.append(try SeedlessLayerBlock(store: store, layer: i, device: device, sharedH: hBuf, maxLen: ml))
        }
        layers = built
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
    /// Optionally encode `tail` into the last CB. `layersPerCB` balances CPU encode vs GPU fill.
    public func stepCommitWait(layersPerCB: Int = 1, tail: ((MTLComputeCommandEncoder) -> Void)? = nil) throws {
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        let g = max(1, layersPerCB)
        var last: MTLCommandBuffer?
        var i = 0
        while i < layers.count {
            let cb = q.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            let end = min(i + g, layers.count)
            for j in i ..< end {
                try layers[j].encodeStep(into: enc)
            }
            if end == layers.count, let tail { tail(enc) }
            enc.endEncoding()
            cb.commit()
            last = cb
            i = end
        }
        last?.waitUntilCompleted()
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
}
