import Foundation
import MLX
import MLXFast

/// Maple attention: GQA + q/k RMSNorm + partial RoPE on SWA layers (NoPE on full).
public struct MapleAttention {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeDim: Int
    let ropeBase: Float
    let eps: Float
    let useRope: Bool

    let qProj: QuantProj
    let kProj: QuantProj
    let vProj: QuantProj
    let oProj: QuantProj
    let qNorm: MLXArray
    let kNorm: MLXArray

    var scale: Float { Float(pow(Double(headDim), -0.5)) }

    public init(
        numHeads: Int, numKVHeads: Int, headDim: Int, ropeDim: Int, ropeBase: Float,
        eps: Float, useRope: Bool,
        qProj: QuantProj, kProj: QuantProj, vProj: QuantProj, oProj: QuantProj,
        qNorm: MLXArray, kNorm: MLXArray
    ) {
        self.numHeads = numHeads; self.numKVHeads = numKVHeads; self.headDim = headDim
        self.ropeDim = ropeDim; self.ropeBase = ropeBase; self.eps = eps; self.useRope = useRope
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
    }

    public func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var queries = qProj.apply(x).reshaped([B, L, numHeads, headDim])
        var keys = kProj.apply(x).reshaped([B, L, numKVHeads, headDim])
        var values = vProj.apply(x).reshaped([B, L, numKVHeads, headDim])

        queries = MLXFast.rmsNorm(queries, weight: qNorm, eps: eps).transposed(0, 2, 1, 3)
        keys = MLXFast.rmsNorm(keys, weight: kNorm, eps: eps).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        if useRope {
            let off = cache.offset
            queries = MLXFast.RoPE(queries, dimensions: ropeDim, traditional: false,
                                   base: ropeBase, scale: 1.0, offset: off)
            keys = MLXFast.RoPE(keys, dimensions: ropeDim, traditional: false,
                                base: ropeBase, scale: 1.0, offset: off)
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
