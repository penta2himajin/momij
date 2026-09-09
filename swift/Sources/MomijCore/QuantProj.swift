import Foundation
import MLX

public enum QuantProj {
    case quantized(weight: MLXArray, scales: MLXArray, biases: MLXArray, bits: Int, groupSize: Int)
    case dense(MLXArray)

    public func apply(_ x: MLXArray) -> MLXArray {
        switch self {
        case let .quantized(w, s, b, bits, gs):
            return MLX.quantizedMM(
                x, w, scales: s, biases: b,
                transpose: true, groupSize: gs, bits: bits, mode: .affine)
        case let .dense(w):
            return MLX.matmul(x, w.transposed(1, 0))
        }
    }
}

public final class KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var offset: Int = 0
    public let maxSize: Int?  // nil = full (chunked grow), else SWA capacity
    private let step = 256

    public init(maxSize: Int? = nil) { self.maxSize = maxSize }

    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let L = k.dim(2)
        let prev = offset

        if let maxSize {
            // Rotating / bounded window: preallocate once to maxSize, write in-place.
            if keys == nil {
                let B = k.dim(0), H = k.dim(1), D = k.dim(3)
                keys = MLXArray.zeros([B, H, maxSize, D], dtype: k.dtype)
                values = MLXArray.zeros([B, H, maxSize, v.dim(3)], dtype: v.dtype)
                MLX.eval(keys!, values!)
            }
            if prev + L <= maxSize {
                keys![0..., 0..., prev ..< (prev + L), 0...] = k
                values![0..., 0..., prev ..< (prev + L), 0...] = v
                offset = prev + L
                return (keys![0..., 0..., 0 ..< offset, 0...],
                        values![0..., 0..., 0 ..< offset, 0...])
            }
            // Simple rotate: drop oldest L tokens, shift, append (concat once).
            let keep = maxSize - L
            if keep <= 0 {
                keys![0..., 0..., 0 ..< maxSize, 0...] = k[0..., 0..., (L - maxSize)..., 0...]
                values![0..., 0..., 0 ..< maxSize, 0...] = v[0..., 0..., (v.dim(2) - maxSize)..., 0...]
            } else {
                let start = offset - keep
                let oldK = keys![0..., 0..., start ..< offset, 0...]
                let oldV = values![0..., 0..., start ..< offset, 0...]
                keys![0..., 0..., 0 ..< keep, 0...] = oldK
                values![0..., 0..., 0 ..< keep, 0...] = oldV
                keys![0..., 0..., keep ..< maxSize, 0...] = k
                values![0..., 0..., keep ..< maxSize, 0...] = v
            }
            offset = maxSize
            return (keys!, values!)
        }

        // Full attention: chunk-grow like mlx_lm.models.cache.KVCache (step=256).
        if keys == nil || (prev + L) > keys!.dim(2) {
            let B = k.dim(0), nKV = k.dim(1), kDim = k.dim(3), vDim = v.dim(3)
            let nSteps = (step + L - 1) / step
            let newK = MLXArray.zeros([B, nKV, nSteps * step, kDim], dtype: k.dtype)
            let newV = MLXArray.zeros([B, nKV, nSteps * step, vDim], dtype: v.dtype)
            if let ok = keys, let ov = values {
                var kk = ok
                var vv = ov
                if prev % step != 0 {
                    kk = ok[0..., 0..., 0 ..< prev, 0...]
                    vv = ov[0..., 0..., 0 ..< prev, 0...]
                }
                keys = MLX.concatenated([kk, newK], axis: 2)
                values = MLX.concatenated([vv, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }
        offset = prev + L
        keys![0..., 0..., prev ..< offset, 0...] = k
        values![0..., 0..., prev ..< offset, 0...] = v
        return (keys![0..., 0..., 0 ..< offset, 0...],
                values![0..., 0..., 0 ..< offset, 0...])
    }

    public func reset() {
        keys = nil; values = nil; offset = 0
    }
}
