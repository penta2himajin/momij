import Foundation
import Metal
import MLX
import MLXFast

/// End-to-end Seedless decode: MLX embed + Metal layers (commit→1wait) + MLX final norm/lm_head.
/// KV grows with `offset`; SWA layers rotate via `maple_shift_kv`.
public final class SeedlessDecodeEngine: @unchecked Sendable {
    public let store: WeightStore
    public let config: MapleConfig
    public let stack: SeedlessLayerStack
    /// Layers encoded per Metal command buffer (env `MOMIJ_LAYERS_PER_CB`, default 4).
    public var layersPerCB: Int
    private let embTable: MLXArray
    private let embBuf: MTLBuffer
    private let vocab: Int
    private let normW: MLXArray
    private let normWBuf: MTLBuffer
    private let lmHead: QuantProj
    private let flashHead: SeedlessFlashHead?
    public let useFlashHead: Bool
    /// FlashHead cluster probes (env `MOMIJ_FLASH_PROBES`, default 64). 0 if FlashHead off.
    public var flashProbes: Int { flashHead?.nProbes ?? 0 }
    /// True when `MOMIJ_FLASH_FUSE=1` (topk+gather in layer CB).
    public var flashFused: Bool { flashHead?.fuseIntoLayerCB ?? false }
    private let finalNormBuf: MTLBuffer
    private let H: Int

    public struct PhaseMs: Sendable {
        public var embed = 0.0
        public var layers = 0.0
        public var head = 0.0
        public var sample = 0.0
        public var steps = 0
    }

    public private(set) var lastPhase = PhaseMs()

    public init(store: WeightStore, fullMaxLen: Int = 2048) throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        self.store = store
        self.config = store.config
        self.H = store.config.hiddenSize
        self.stack = try SeedlessLayerStack(store: store, device: device, fullMaxLen: fullMaxLen)
        if let s = ProcessInfo.processInfo.environment["MOMIJ_LAYERS_PER_CB"], let v = Int(s), v > 0 {
            layersPerCB = v
        } else {
            // Sweep sweet spot on M1 Max: g=4 under cooling.
            layersPerCB = 4
        }

        let embW = store.req("model.word_embeddings.weight")
        let embS = store.req("model.word_embeddings.scales")
        let embB = store.req("model.word_embeddings.biases")
        self.embTable = MLX.dequantized(
            embW, scales: embS, biases: embB,
            groupSize: 64, bits: 4, mode: .affine
        ).asType(.float16)
        self.normW = store.req("model.norm.weight").asType(.float16)
        self.lmHead = .quantized(
            weight: store.req("lm_head.weight"),
            scales: store.req("lm_head.scales"),
            biases: store.req("lm_head.biases"),
            bits: config.headBits,
            groupSize: config.headGroupSize)
        let flashEnv = ProcessInfo.processInfo.environment["MOMIJ_FLASH_HEAD"]
        let wantFlash = flashEnv != "0"
        flashHead = wantFlash ? SeedlessFlashHead(store: store, device: device) : nil
        useFlashHead = flashHead != nil
        if wantFlash && flashHead == nil {
            fputs("[momij] FlashHead weights missing; using exact lm_head\n", stderr)
        }
        MLX.eval(embTable, normW)
        vocab = embTable.dim(0)
        guard let ebuf = SeedlessMetal.mtlBuf(embTable, device),
              let nbuf = SeedlessMetal.mtlBuf(normW, device)
        else { throw SeedlessError.notReady }
        embBuf = ebuf
        normWBuf = nbuf
        finalNormBuf = device.makeBuffer(length: H * 2, options: .storageModeShared)!
    }

    public func reset() { stack.resetCaches() }

    private func embedToken(_ id: Int) {
        precondition(id >= 0 && id < vocab)
        let src = embBuf.contents().advanced(by: id * H * 2)
        stack.hBuf.contents().copyMemory(from: src, byteCount: H * 2)
    }

    private func nextToken() -> Int {
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        if useFlashHead, let fh = flashHead {
            if fh.fuseIntoLayerCB {
                return fh.greedyAfterFusedGather(hostH: ptr)
            }
            return fh.greedyAfterCentroids(hBuf: finalNormBuf, hostH: ptr)
        }
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let logits = lmHead.apply(h.reshaped([1, 1, H]))
        let y = MLX.argMax(logits.reshaped([-1]), axis: -1)
        MLX.eval(y)
        return y.item(Int.self)
    }

    /// One token: embed → Metal layers (+final RMS + FlashHead centroids) → gather/sample.
    @discardableResult
    public func step(_ token: Int, profile: Bool = false) throws -> Int {
        var ph = PhaseMs()
        let t0 = CFAbsoluteTimeGetCurrent()
        embedToken(token)
        let t1 = CFAbsoluteTimeGetCurrent()
        try stack.stepCommitWait(layersPerCB: layersPerCB) { [self] enc in
            SeedlessMetal.encodeRms(
                into: enc, h: stack.hBuf, w: normWBuf,
                out: finalNormBuf, H: H, eps: config.rmsNormEps)
            if let fh = flashHead {
                fh.encodeCentroids(into: enc, h: finalNormBuf)
                if fh.fuseIntoLayerCB {
                    fh.encodeFusedAfterCentroids(into: enc, h: finalNormBuf)
                }
            }
        }
        let t2 = CFAbsoluteTimeGetCurrent()
        let next = nextToken()
        let t3 = CFAbsoluteTimeGetCurrent()
        if profile {
            ph.embed = (t1 - t0) * 1000
            ph.layers = (t2 - t1) * 1000
            ph.head = (t3 - t2) * 1000
            ph.sample = 0
            ph.steps = 1
            lastPhase = ph
        }
        return next
    }

    public func generate(prompt: [Int], maxTokens: Int, eos: Int? = 151_645) throws -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        reset()
        var last = prompt[0]
        for i in 0 ..< prompt.count {
            let id = prompt[i]
            if i + 1 < prompt.count {
                _ = try step(id)
            } else {
                last = try step(id)
            }
        }
        var out: [Int] = []
        var y = last
        for _ in 0 ..< maxTokens {
            out.append(y)
            if let eos, y == eos { break }
            y = try step(y)
        }
        return out
    }

    public func benchmark(promptTokens: Int, genTokens: Int, trials: Int, profile: Bool = true)
        throws -> (promptTps: Double, genTps: Double, phase: PhaseMs)
    {
        let prompt = Array(repeating: 100, count: promptTokens)
        // Warmup
        _ = try generate(prompt: Array(prompt.prefix(min(32, promptTokens))), maxTokens: 4, eos: nil)

        var pTps: [Double] = [], gTps: [Double] = []
        var accum = PhaseMs()
        for _ in 0 ..< trials {
            reset()
            let t0 = CFAbsoluteTimeGetCurrent()
            var last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count {
                    _ = try step(prompt[i], profile: false)
                } else {
                    last = try step(prompt[i], profile: false)
                }
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            var y = last
            for _ in 0 ..< genTokens {
                y = try step(y, profile: profile)
                if profile {
                    accum.embed += lastPhase.embed
                    accum.layers += lastPhase.layers
                    accum.head += lastPhase.head
                    accum.steps += 1
                }
            }
            let t2 = CFAbsoluteTimeGetCurrent()
            pTps.append(Double(promptTokens) / max(t1 - t0, 1e-9))
            gTps.append(Double(genTokens) / max(t2 - t1, 1e-9))
        }
        if accum.steps > 0 {
            let n = Double(accum.steps)
            accum.embed /= n; accum.layers /= n; accum.head /= n; accum.sample /= n
        }
        lastPhase = accum
        let p = pTps.reduce(0, +) / Double(pTps.count)
        let g = gTps.reduce(0, +) / Double(gTps.count)
        return (p, g, accum)
    }

    /// L0 hidden vs MapleEngine MLX (decode step after same prompt token). Rel L2.
    public static func parityL0(store: WeightStore) throws -> Double {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        let token = 100
        _ = MapleEngine(store: store, enableSeedlessMoE: false) // ensure weights sanitize path OK
        let block = try SeedlessLayerBlock(store: store, layer: 0, device: device)
        let embW = store.req("model.word_embeddings.weight")
        let embS = store.req("model.word_embeddings.scales")
        let embB = store.req("model.word_embeddings.biases")
        let emb = MLX.dequantized(embW, scales: embS, biases: embB, groupSize: 64, bits: 4, mode: .affine)
            .asType(.float16)
        let row = MLX.take(emb, MLXArray(Int32(token)), axis: 0)
        MLX.eval(row)
        let arr = row.asArray(Float16.self)
        arr.withUnsafeBufferPointer {
            block.hBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: block.H * 2)
        }
        let cb = SeedlessMetal.queue!.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try block.encodeStep(into: enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        var out = [Float16](repeating: 0, count: block.H)
        out.withUnsafeMutableBytes {
            $0.copyMemory(from: UnsafeRawBufferPointer(start: block.hBuf.contents(), count: block.H * 2))
        }
        // MLX one layer
        let cfg = store.config
        let p = "model.layers.0"
        func q(_ name: String) -> QuantProj {
            .quantized(weight: store.req("\(name).weight"), scales: store.req("\(name).scales"),
                       biases: store.req("\(name).biases"), bits: cfg.expertBits, groupSize: cfg.expertGroupSize)
        }
        let attn = MapleAttention(
            numHeads: cfg.numAttentionHeads, numKVHeads: cfg.numKeyValueHeads, headDim: cfg.headDim,
            ropeDim: cfg.ropeDim, ropeBase: cfg.ropeTheta, eps: cfg.rmsNormEps, useRope: cfg.isSliding(0),
            qkvProj: q("\(p).self_attn.qkv_proj"), oProj: q("\(p).self_attn.o_proj"),
            qNorm: store.req("\(p).self_attn.q_norm.weight"), kNorm: store.req("\(p).self_attn.k_norm.weight"))
        let moe = MapleMoE(
            topK: cfg.numExpertsPerTok, numExperts: cfg.numExperts,
            bits: cfg.expertBits, groupSize: cfg.expertGroupSize,
            gateW: store.req("\(p).mlp.gate.weight"),
            upGateW: store.req("\(p).mlp.switch_mlp.up_gate_proj.weight"),
            upGateS: store.req("\(p).mlp.switch_mlp.up_gate_proj.scales"),
            upGateB: store.req("\(p).mlp.switch_mlp.up_gate_proj.biases"),
            downW: store.req("\(p).mlp.switch_mlp.down_proj.weight"),
            downS: store.req("\(p).mlp.switch_mlp.down_proj.scales"),
            downB: store.req("\(p).mlp.switch_mlp.down_proj.biases"))
        let cache = KVCache(maxSize: cfg.isSliding(0) ? cfg.slidingWindow : nil)
        var h = row.reshaped([1, 1, cfg.hiddenSize])
        let inN = store.req("\(p).input_layernorm.weight")
        let postN = store.req("\(p).post_attention_layernorm.weight")
        let hn = MLXFast.rmsNorm(h.asType(.float32), weight: inN.asType(.float32), eps: cfg.rmsNormEps).asType(.float16)
        let r = attn(hn, cache: cache)
        h = h + r
        let hn2 = MLXFast.rmsNorm(h.asType(.float32), weight: postN.asType(.float32), eps: cfg.rmsNormEps).asType(.float16)
        h = h + moe(hn2)
        MLX.eval(h)
        let ref = h.reshaped([-1]).asArray(Float16.self)
        var num: Float = 0, den: Float = 0
        for i in 0 ..< cfg.hiddenSize {
            let a = Float(out[i]), b = Float(ref[i])
            num += (a - b) * (a - b)
            den += b * b
        }
        return sqrt(Double(num / max(den, 1e-12)))
    }

    /// Sweep `layersPerCB` candidates on a warm engine.
    public static func sweepLayersPerCB(store: WeightStore, prompt: Int = 64, gen: Int = 64) throws -> String {
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: max(prompt + gen + 64, 2048))
        var lines: [String] = []
        var bestTps = 0.0
        var bestG = 1
        for g in [1, 2, 3, 4, 6, 8, 12, 24] {
            eng.layersPerCB = g
            _ = try eng.generate(prompt: Array(repeating: 100, count: 16), maxTokens: 4, eos: nil)
            let r = try eng.benchmark(promptTokens: prompt, genTokens: gen, trials: 2, profile: true)
            let ph = eng.lastPhase
            lines.append(String(format: "  g=%2d  gen=%.1f tok/s  layers=%.3f head=%.3f ms",
                                g, r.genTps, ph.layers, ph.head))
            if r.genTps > bestTps { bestTps = r.genTps; bestG = g }
        }
        return "seedless layers_per_cb sweep (p\(prompt)/g\(gen)):\n" + lines.joined(separator: "\n")
            + String(format: "\nbest g=%d @ %.1f tok/s", bestG, bestTps)
    }
}
