import Foundation
import MLX

private let mlpClamp: Float = 7.0

/// Maple MoE: dense fp16 router + 2-bit switch experts (fused up_gate when present).
public struct MapleMoE {
    let topK: Int
    let numExperts: Int
    let bits: Int
    let groupSize: Int

    let gateW: MLXArray  // [E, H] dense
    let upGateW: MLXArray, upGateS: MLXArray, upGateB: MLXArray  // [E, 2I, ...]
    let downW: MLXArray, downS: MLXArray, downB: MLXArray

    public init(
        topK: Int, numExperts: Int, bits: Int, groupSize: Int,
        gateW: MLXArray,
        upGateW: MLXArray, upGateS: MLXArray, upGateB: MLXArray,
        downW: MLXArray, downS: MLXArray, downB: MLXArray
    ) {
        self.topK = topK; self.numExperts = numExperts
        self.bits = bits; self.groupSize = groupSize
        self.gateW = gateW
        self.upGateW = upGateW; self.upGateS = upGateS; self.upGateB = upGateB
        self.downW = downW; self.downS = downS; self.downB = downB
    }

    private func clampedSwiglu(gate: MLXArray, up: MLXArray) -> MLXArray {
        let g = MLX.minimum(gate, MLXArray(mlpClamp))
        let u = MLX.clip(up, min: -mlpClamp, max: mlpClamp)
        return (g * MLX.sigmoid(g)) * u
    }

    /// x: [B, L, H] → [B, L, H]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1), H = x.dim(2)
        let flat = x.reshaped([B * L, H])
        // dense router in fp32 for stability (matches maple fused_router spirit)
        let logits = MLX.matmul(flat.asType(.float32), gateW.transposed(1, 0).asType(.float32))
        let gates = MLX.softmax(logits, axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        let inds = order[0..., (numExperts - topK)...].asType(.uint32)
        var scores = MLX.takeAlong(gates, inds.asType(.int32), axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)

        let xe = flat.expandedDimensions(axes: [-2, -3])  // [T,1,1,H]
        let ug = MLX.gatherQuantizedMM(
            xe, upGateW, scales: upGateS, biases: upGateB, rhsIndices: inds,
            transpose: true, groupSize: groupSize, bits: bits, mode: .affine,
            sortedIndices: false)  // [T, K, 1, 2I]
        // maple.py: x_up, x_gate = split(up_gate, 2, axis=-1)
        let parts = ug.split(parts: 2, axis: -1)
        let h = clampedSwiglu(gate: parts[1], up: parts[0])
        let d = MLX.gatherQuantizedMM(
            h, downW, scales: downS, biases: downB, rhsIndices: inds,
            transpose: true, groupSize: groupSize, bits: bits, mode: .affine,
            sortedIndices: false
        ).squeezed(axis: -2)  // [T,K,H]
        let y = (d.asType(.float32) * scores.expandedDimensions(axis: -1))
            .sum(axis: -2)
            .asType(flat.dtype)
        return y.reshaped([B, L, H])
    }
}
