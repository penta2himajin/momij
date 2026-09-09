import Foundation
import MLX
import MLXFast
import MLXNN

/// MLX-path Maple engine (exact greedy). Correctness baseline alongside Seedless.
public final class MapleEngine: @unchecked Sendable {
    public let store: WeightStore
    public let config: MapleConfig
    private var layers: [Layer] = []
    private let embTable: MLXArray  // [V, H] f16, dequantized once
    private let normW: MLXArray
    private let lmHead: QuantProj
    private var caches: [KVCache] = []

    private struct Layer {
        let attn: MapleAttention
        let moe: MapleMoE
        let inNorm: MLXArray
        let postNorm: MLXArray
    }

    public init(store: WeightStore) {
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
                qProj: q("\(p).self_attn.q_proj"),
                kProj: q("\(p).self_attn.k_proj"),
                vProj: q("\(p).self_attn.v_proj"),
                oProj: q("\(p).self_attn.o_proj"),
                qNorm: store.req("\(p).self_attn.q_norm.weight"),
                kNorm: store.req("\(p).self_attn.k_norm.weight"))
            let moePrefix = "\(p).mlp.switch_mlp"
            let upGate = "\(moePrefix).up_gate_proj"
            let moe = MapleMoE(
                topK: config.numExpertsPerTok,
                numExperts: config.numExperts,
                bits: bits, groupSize: gs,
                gateW: store.req("\(p).mlp.gate.weight"),
                upGateW: store.req("\(upGate).weight"),
                upGateS: store.req("\(upGate).scales"),
                upGateB: store.req("\(upGate).biases"),
                downW: store.req("\(moePrefix).down_proj.weight"),
                downS: store.req("\(moePrefix).down_proj.scales"),
                downB: store.req("\(moePrefix).down_proj.biases"))
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

    private func forwardLastLogits(_ ids: [Int], reset: Bool) -> MLXArray {
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

    private func stepLogits(_ token: Int) -> MLXArray {
        let tok = MLXArray(Int32(token)).reshaped([1, 1])
        var h = embed(tok)
        for (layer, cache) in zip(layers, caches) {
            let r = layer.attn(rms(h, layer.inNorm), cache: cache)
            h = h + r
            let r2 = layer.moe(rms(h, layer.postNorm))
            h = h + r2
        }
        h = rms(h, normW)
        return lmHead.apply(h)
    }

    /// Prefill + decode. Returns generated token ids (not including prompt).
    public func generate(prompt: [Int], maxTokens: Int, eos: Int? = 151_645) -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        var logits = forwardLastLogits(prompt, reset: true)
        MLX.eval(logits)
        var out: [Int] = []
        var next = MLX.argMax(logits[0, 0], axis: -1).item(Int.self)
        out.append(next)
        if let eos, next == eos { return out }
        for _ in 1 ..< maxTokens {
            logits = stepLogits(next)
            MLX.eval(logits)
            next = MLX.argMax(logits[0, 0], axis: -1).item(Int.self)
            out.append(next)
            if let eos, next == eos { break }
        }
        return out
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
            var logits = forwardLastLogits(prompt, reset: true)
            MLX.eval(logits)
            let t1 = CFAbsoluteTimeGetCurrent()
            var next = MLX.argMax(logits[0, 0], axis: -1).item(Int.self)
            for _ in 0 ..< genTokens {
                logits = stepLogits(next)
                MLX.eval(logits)
                next = MLX.argMax(logits[0, 0], axis: -1).item(Int.self)
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
