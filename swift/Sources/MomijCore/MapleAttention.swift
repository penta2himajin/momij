import Foundation
import MLX
import MLXFast

/// Maple attention aligned with mlx-lm-deepgrove: fused qkv + decode fused qk-norm(+RoPE).
public struct MapleAttention {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeDim: Int
    let ropeBase: Float
    let eps: Float
    let useRope: Bool

    let qkvProj: QuantProj
    let oProj: QuantProj
    let qNorm: MLXArray
    let kNorm: MLXArray
    let qkW: MLXArray
    let invFreq: MLXArray

    var scale: Float { Float(pow(Double(headDim), -0.5)) }
    var qkSize: Int { (numHeads + numKVHeads) * headDim }

    public init(
        numHeads: Int, numKVHeads: Int, headDim: Int, ropeDim: Int, ropeBase: Float,
        eps: Float, useRope: Bool,
        qkvProj: QuantProj, oProj: QuantProj,
        qNorm: MLXArray, kNorm: MLXArray
    ) {
        self.numHeads = numHeads; self.numKVHeads = numKVHeads; self.headDim = headDim
        self.ropeDim = ropeDim; self.ropeBase = ropeBase; self.eps = eps; self.useRope = useRope
        self.qkvProj = qkvProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm

        let qPart = MLX.broadcast(qNorm.reshaped([1, headDim]), to: [numHeads, headDim])
        let kPart = MLX.broadcast(kNorm.reshaped([1, headDim]), to: [numKVHeads, headDim])
        let qw = MLX.contiguous(MLX.concatenated([qPart, kPart], axis: 0))
        let half = max(useRope ? ropeDim / 2 : 1, 1)
        var freqs = [Float](repeating: 1, count: half)
        if useRope {
            for i in 0 ..< half {
                freqs[i] = pow(ropeBase, -Float(i) / Float(half))
            }
        }
        let inv = MLXArray(freqs)
        MLX.eval(qw, inv)
        self.qkW = qw
        self.invFreq = inv
    }

    /// MapleRMSNorm: fp32 rms_norm then cast back (oracle).
    private func mapleRMS(_ x: MLXArray, weight w: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: w.asType(.float32), eps: eps)
            .asType(x.dtype)
    }

    public func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let qkv = qkvProj.apply(x)

        let queries: MLXArray
        let keys: MLXArray
        let values: MLXArray

        if B == 1, L == 1 {
            // Oracle: one qkv matmul + fused qk norm/rope metal_kernel.
            let flat = qkv.reshaped([-1])
            let qk = flat[0 ..< qkSize].reshaped([numHeads + numKVHeads, headDim])
            let out = MapleFused.qkNormRope(
                qk: qk, w: qkW.asType(qk.dtype), invFreq: invFreq,
                offset: cache.offset, eps: eps,
                ropeDim: useRope ? ropeDim : 0)
            queries = out[0 ..< numHeads].reshaped([1, numHeads, 1, headDim])
            keys = out[numHeads...].reshaped([1, numKVHeads, 1, headDim])
            values = flat[qkSize...].reshaped([1, numKVHeads, 1, headDim])
        } else {
            let qEnd = numHeads * headDim
            let kEnd = qEnd + numKVHeads * headDim
            var q = qkv[0..., 0..., 0 ..< qEnd].reshaped([B, L, numHeads, headDim])
            var k = qkv[0..., 0..., qEnd ..< kEnd].reshaped([B, L, numKVHeads, headDim])
            let v = qkv[0..., 0..., kEnd...].reshaped([B, L, numKVHeads, headDim])
            q = mapleRMS(q, weight: qNorm).transposed(0, 2, 1, 3)
            k = mapleRMS(k, weight: kNorm).transposed(0, 2, 1, 3)
            if useRope {
                let off = cache.offset
                q = MLXFast.RoPE(q, dimensions: ropeDim, traditional: false,
                                 base: ropeBase, scale: 1.0, offset: off)
                k = MLXFast.RoPE(k, dimensions: ropeDim, traditional: false,
                                 base: ropeBase, scale: 1.0, offset: off)
            }
            queries = q
            keys = k
            values = v.transposed(0, 2, 1, 3)
        }

        let (allK, allV) = cache.update(keys, values)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = L > 1 ? .causal : .none
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allK, values: allV,
            scale: scale, mask: mask
        )
        let flat = out.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])
        return oProj.apply(flat)
    }
}
