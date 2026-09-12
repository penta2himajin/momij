import Foundation
import MLX
import MLXFast
import MLXNN
import Metal

/// MLX-path Maple engine (exact greedy). Correctness baseline alongside Seedless.
public final class MapleEngine: @unchecked Sendable {
    public let store: WeightStore
    public let config: MapleConfig
    private var layers: [Layer] = []
    private let embTable: MLXArray  // dequantized, checkpoint dtype (bf16)
    private let normW: MLXArray
    private let lmHead: QuantProj
    private var caches: [KVCache] = []
    /// Cached zeros for decode fused residual (oracle `MapleModel._zero`).
    private var decodeZero: MLXArray?
    private struct Layer {
        let attn: MapleAttention
        let moe: MapleMoEHybrid
        let inNorm: MLXArray
        let postNorm: MLXArray
    }

    public init(store: WeightStore, enableSeedlessMoE: Bool = MapleMoEHybrid.enabled) {
        self.store = store
        self.config = store.config
        let embW = store.req("model.word_embeddings.weight")
        let embS = store.req("model.word_embeddings.scales")
        let embB = store.req("model.word_embeddings.biases")
        self.embTable = MLX.dequantized(
            embW, scales: embS, biases: embB,
            groupSize: 64, bits: 4, mode: .affine
        ).asType(.float16)
        MLX.eval(self.embTable)
        self.normW = store.req("model.norm.weight")
        self.lmHead = .quantized(
            weight: store.req("lm_head.weight"),
            scales: store.req("lm_head.scales"),
            biases: store.req("lm_head.biases"),
            bits: config.headBits,
            groupSize: config.headGroupSize)

        var metalDevice: MTLDevice? = nil
        if enableSeedlessMoE {
            do {
                try SeedlessMetal.ensureCompiled()
                metalDevice = SeedlessMetal.device
            } catch {
                fputs("[momij] seedless init failed: \(error)\n", stderr)
            }
        }

        for i in 0 ..< config.numHiddenLayers {
            let p = "model.layers.\(i)"
            let bits = config.expertBits
            let gs = config.expertGroupSize
            func q(_ name: String) -> QuantProj {
                .quantized(
                    weight: store.req("\(name).weight"),
                    scales: store.req("\(name).scales"),
                    biases: store.req("\(name).biases"),
                    bits: bits, groupSize: gs)
            }
            let attn = MapleAttention(
                numHeads: config.numAttentionHeads,
                numKVHeads: config.numKeyValueHeads,
                headDim: config.headDim,
                ropeDim: config.ropeDim,
                ropeBase: config.ropeTheta,
                eps: config.rmsNormEps,
                useRope: config.isSliding(i),
                qkvProj: q("\(p).self_attn.qkv_proj"),
                oProj: q("\(p).self_attn.o_proj"),
                qNorm: store.req("\(p).self_attn.q_norm.weight"),
                kNorm: store.req("\(p).self_attn.k_norm.weight"))
            let moePrefix = "\(p).mlp.switch_mlp"
            let upGate = "\(moePrefix).up_gate_proj"
            let gateW = store.req("\(p).mlp.gate.weight")
            let mlxMoE = MapleMoE(
                topK: config.numExpertsPerTok,
                numExperts: config.numExperts,
                bits: bits, groupSize: gs,
                gateW: gateW,
                upGateW: store.req("\(upGate).weight"),
                upGateS: store.req("\(upGate).scales"),
                upGateB: store.req("\(upGate).biases"),
                downW: store.req("\(moePrefix).down_proj.weight"),
                downS: store.req("\(moePrefix).down_proj.scales"),
                downB: store.req("\(moePrefix).down_proj.biases"))
            var metalLayer: SeedlessMoELayer? = nil
            if let device = metalDevice {
                metalLayer = try? SeedlessMoELayer(store: store, layer: i, device: device)
            }
            let moe = MapleMoEHybrid(
                mlx: mlxMoE, metal: metalLayer, gateW: gateW,
                topK: config.numExpertsPerTok, numExperts: config.numExperts)
            layers.append(Layer(
                attn: attn, moe: moe,
                inNorm: store.req("\(p).input_layernorm.weight"),
                postNorm: store.req("\(p).post_attention_layernorm.weight")))
            caches.append(KVCache(maxSize: config.isSliding(i) ? config.slidingWindow : nil))
        }
    }

    public func resetCaches() {
        for c in caches { c.reset() }
    }

    private func embed(_ ids: MLXArray) -> MLXArray {
        // ids: [1, T] int → [1, T, H]
        MLX.take(embTable, ids.reshaped([-1]), axis: 0)
            .reshaped([1, ids.dim(1), config.hiddenSize])
    }

    private func rms(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: w.asType(.float32), eps: config.rmsNormEps)
            .asType(x.dtype)
    }

    func forwardLastLogits(_ ids: [Int], reset: Bool) -> MLXArray {
        if reset { resetCaches() }
        let promptArr = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        var h = embed(promptArr)
        for (layer, cache) in zip(layers, caches) {
            let r = layer.attn(rms(h, layer.inNorm), cache: cache)
            h = h + r
            let r2 = layer.moe(rms(h, layer.postNorm))
            h = h + r2
        }
        h = rms(h, normW)
        return lmHead.apply(h[0..., (h.dim(1) - 1) ..< h.dim(1), 0...])
    }

    /// Last-token residual after embed and after each layer (batched prefill).
    func lastTokenHiddenAfterEachLayer(_ ids: [Int]) -> (embed: [Float], layers: [[Float]]) {
        resetCaches()
        let promptArr = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        var h = embed(promptArr)
        MLX.eval(h)
        let embedRow = Self.hostLastRow(h)
        var layersOut: [[Float]] = []
        layersOut.reserveCapacity(layers.count)
        for (layer, cache) in zip(layers, caches) {
            let r = layer.attn(rms(h, layer.inNorm), cache: cache)
            h = h + r
            let r2 = layer.moe(rms(h, layer.postNorm))
            h = h + r2
            MLX.eval(h)
            layersOut.append(Self.hostLastRow(h))
        }
        return (embedRow, layersOut)
    }

    /// Last-token residual after layer-0 attention only (batched prefill).
    func lastTokenAfterFirstAttn(_ ids: [Int]) -> [Float] {
        resetCaches()
        let promptArr = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        var h = embed(promptArr)
        let layer = layers[0]
        let r = layer.attn(rms(h, layer.inNorm), cache: caches[0])
        h = h + r
        MLX.eval(h)
        return Self.hostLastRow(h)
    }

    /// Last-token L0 QKV after RMS (no RoPE). `batched` uses the full prompt; else last id only.
    func lastTokenLayer0QKV(_ ids: [Int], batched: Bool) -> [Float] {
        let attn = layers[0].attn
        let x: MLXArray
        if batched {
            let promptArr = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
            x = rms(embed(promptArr), layers[0].inNorm)
        } else {
            let tok = MLXArray([Int32(ids.last!)]).reshaped([1, 1])
            x = rms(embed(tok), layers[0].inNorm)
        }
        let qkv = attn.qkvProj.apply(x)
        MLX.eval(qkv)
        return Self.hostLastRow(qkv)
    }

    /// Last-token L0 Q after qk-norm + RoPE. Batched = MLXFast.RoPE; else decode fused kernel.
    func lastTokenLayer0Q(_ ids: [Int], batched: Bool) -> [Float] {
        let attn = layers[0].attn
        if batched {
            let promptArr = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
            let x = rms(embed(promptArr), layers[0].inNorm)
            let qkv = attn.qkvProj.apply(x)
            let B = 1, L = ids.count
            let qEnd = attn.numHeads * attn.headDim
            var q = qkv[0..., 0..., 0 ..< qEnd].reshaped([B, L, attn.numHeads, attn.headDim])
            q = MLXFast.rmsNorm(q.asType(.float32), weight: attn.qNorm.asType(.float32), eps: attn.eps)
                .asType(qkv.dtype)
                .transposed(0, 2, 1, 3)
            if attn.useRope {
                q = MLXFast.RoPE(
                    q, dimensions: attn.ropeDim, traditional: false,
                    base: attn.ropeBase, scale: 1.0, offset: 0)
            }
            let last = q[0..., 0..., L - 1, 0...].reshaped([-1]).asType(.float32)
            MLX.eval(last)
            return Self.hostVec(last)
        }
        let tok = MLXArray([Int32(ids.last!)]).reshaped([1, 1])
        let x = rms(embed(tok), layers[0].inNorm)
        let qkv = attn.qkvProj.apply(x)
        let flat = qkv.reshaped([-1])
        let qk = flat[0 ..< attn.qkSize].reshaped([attn.numHeads + attn.numKVHeads, attn.headDim])
        let out = MapleFused.qkNormRope(
            qk: qk, w: attn.qkW.asType(qk.dtype), invFreq: attn.invFreq,
            offset: ids.count - 1, eps: attn.eps,
            ropeDim: attn.useRope ? attn.ropeDim : 0)
        let q = out[0 ..< attn.numHeads].reshaped([-1]).asType(.float32)
        MLX.eval(q)
        return Self.hostVec(q)
    }

    private static func hostVec(_ a: MLXArray) -> [Float] {
        let x = a.asType(.float32)
        MLX.eval(x)
        let n = x.dim(0)
        var out = [Float](repeating: 0, count: n)
        x.asData(access: .copy).data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float.self)
            let m = min(n, raw.count / MemoryLayout<Float>.size)
            for i in 0 ..< m { out[i] = src[i] }
        }
        return out
    }

    private static func hostLastRow(_ h: MLXArray) -> [Float] {
        let last = h[0, h.dim(1) - 1, 0...].asType(.float32)
        MLX.eval(last)
        let n = last.dim(0)
        var out = [Float](repeating: 0, count: n)
        last.asData(access: .copy).data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float.self)
            let m = min(n, raw.count / MemoryLayout<Float>.size)
            for i in 0 ..< m { out[i] = src[i] }
        }
        return out
    }

    private func stepLogits(_ token: MLXArray) -> MLXArray {
        // token: scalar or [1]/int — keep on-device (oracle generate_step style)
        let tok = token.asType(.int32).reshaped([1, 1])
        var h = embed(tok)
        // Oracle `_decode_fused`: carry residual `r`, fold add+RMSNorm into one dispatch.
        if decodeZero == nil || decodeZero!.dim(0) != h.dim(0) {
            let z = MLXArray.zeros(like: h)
            MLX.eval(z)
            decodeZero = z
        }
        var r = decodeZero!
        for (layer, cache) in zip(layers, caches) {
            let hn: MLXArray
            (h, hn) = MapleFused.addRmsNorm(h, r, weight: layer.inNorm, eps: config.rmsNormEps)
            r = layer.attn(hn, cache: cache)
            let hn2: MLXArray
            (h, hn2) = MapleFused.addRmsNorm(h, r, weight: layer.postNorm, eps: config.rmsNormEps)
            r = layer.moe(hn2)
        }
        let hnFinal: MLXArray
        (_, hnFinal) = MapleFused.addRmsNorm(h, r, weight: normW, eps: config.rmsNormEps)
        return lmHead.apply(hnFinal)
    }

    private func stepLogits(_ token: Int) -> MLXArray {
        stepLogits(MLXArray(Int32(token)))
    }

    /// Prefill + decode. Returns generated token ids (not including prompt).
    ///
    /// When `allowedNext` is set, each step's argmax is restricted to that set
    /// (finite-string / grammar frontier). Empty set → stop.
    public func generate(
        prompt: [Int],
        maxTokens: Int,
        eos: Int? = 151_645,
        allowedNext: (([Int]) -> Set<Int>)? = nil,
        bannedTokenIds: [Int] = []
    ) -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        let logits0 = forwardLastLogits(prompt, reset: true)
        var y = pickToken(
            logits0, prefix: [], allowedNext: allowedNext, banned: bannedTokenIds)
        asyncEval(y)
        var out: [Int] = []
        for i in 0 ..< maxTokens {
            var nextY: MLXArray? = nil
            let next = y.item(Int.self)
            out.append(next)
            if let eos, next == eos { break }
            if i + 1 < maxTokens {
                let logits = stepLogits(y)
                nextY = pickToken(
                    logits, prefix: out, allowedNext: allowedNext, banned: bannedTokenIds)
                asyncEval(nextY!)
            }
            guard let n = nextY else { break }
            y = n
        }
        return out
    }

    private func pickToken(
        _ logits: MLXArray,
        prefix: [Int],
        allowedNext: (([Int]) -> Set<Int>)?,
        banned: [Int]
    ) -> MLXArray {
        if let allowedNext {
            var allowed = allowedNext(prefix)
            for b in banned { allowed.remove(b) }
            guard !allowed.isEmpty else {
                return MLXArray(Int32(0))
            }
            if allowed.count == 1, let only = allowed.first {
                return MLXArray(Int32(only))
            }
            let scores = ConstrainedPick.hostScores(logits[0, 0])
            let id = ConstrainedPick.argmax(allowed: allowed, scores: scores) ?? allowed.first!
            return MLXArray(Int32(id))
        }
        if banned.isEmpty {
            return MLX.argMax(logits[0, 0], axis: -1)
        }
        var scores = ConstrainedPick.hostScores(logits[0, 0])
        ConstrainedPick.applyBanned(&scores, banned: banned)
        return MLXArray(Int32(ConstrainedPick.argmaxAll(scores)))
    }

    public func benchmark(promptTokens: Int, genTokens: Int, trials: Int)
        -> (promptTps: Double, genTps: Double, peakGB: Double)
    {
        store.residentAll()
        _ = generate(prompt: Array(repeating: 100, count: min(32, promptTokens)), maxTokens: 8, eos: nil)

        var pTps: [Double] = [], gTps: [Double] = []
        for _ in 0 ..< trials {
            let prompt = Array(repeating: 100, count: promptTokens)
            let t0 = CFAbsoluteTimeGetCurrent()
            let logits0 = forwardLastLogits(prompt, reset: true)
            var y = MLX.argMax(logits0[0, 0], axis: -1)
            asyncEval(y)
            // Prefill complete once first token id is ready (matches oracle first-yield).
            _ = y.item(Int.self)
            let t1 = CFAbsoluteTimeGetCurrent()
            Stream.withNewDefaultStream {
                for _ in 0 ..< genTokens {
                    let logits = stepLogits(y)
                    let nextY = MLX.argMax(logits[0, 0], axis: -1)
                    asyncEval(nextY)
                    _ = y.item(Int.self)
                    y = nextY
                }
                _ = y.item(Int.self)
            }
            let t2 = CFAbsoluteTimeGetCurrent()
            pTps.append(Double(promptTokens) / max(t1 - t0, 1e-9))
            gTps.append(Double(genTokens) / max(t2 - t1, 1e-9))
        }
        let peak = Double(MLX.Memory.snapshot().activeMemory) / 1e9
        return (
            pTps.reduce(0, +) / Double(pTps.count),
            gTps.reduce(0, +) / Double(gTps.count),
            peak
        )
    }
}
