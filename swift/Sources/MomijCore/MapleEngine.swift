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
    private let embTable: MLXArray  // [V, H] f16, dequantized once
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
                qProj: q("\(p).self_attn.q_proj"),
                kProj: q("\(p).self_attn.k_proj"),
                vProj: q("\(p).self_attn.v_proj"),
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
    public func generate(prompt: [Int], maxTokens: Int, eos: Int? = 151_645) -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        let logits0 = forwardLastLogits(prompt, reset: true)
        var y = MLX.argMax(logits0[0, 0], axis: -1)
        asyncEval(y)
        var out: [Int] = []
        for i in 0 ..< maxTokens {
            // Overlap next forward with reading current token (mlx_lm.generate_step).
            var nextY: MLXArray? = nil
            if i + 1 < maxTokens {
                let logits = stepLogits(y)
                nextY = MLX.argMax(logits[0, 0], axis: -1)
                asyncEval(nextY!)
            }
            let next = y.item(Int.self)
            out.append(next)
            if let eos, next == eos { break }
            guard let n = nextY else { break }
            y = n
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
