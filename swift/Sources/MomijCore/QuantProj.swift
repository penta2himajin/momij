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

/// KV cache: unbounded grow (full attention) or oracle-aligned `RotatingKVCache` (SWA).
///
/// For SWA (`maxSize != nil`), `offset` is the **absolute** token position used for RoPE —
/// it must keep increasing after the window fills. The circular write head is `idx`.
public final class KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    /// Absolute sequence position (RoPE); never clamped to `maxSize`.
    public private(set) var offset: Int = 0
    public let maxSize: Int?  // nil = full (chunked grow), else SWA capacity
    /// Circular write head for in-place SWA updates (oracle `_idx`).
    private var idx: Int = 0
    private let keep: Int = 0
    private let step = 256

    public init(maxSize: Int? = nil) { self.maxSize = maxSize }

    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if maxSize != nil {
            if k.dim(2) == 1 {
                return updateInPlace(k, v)
            }
            return updateConcat(k, v)
        }
        return updateFull(k, v)
    }

    // MARK: - Full attention (mlx_lm KVCache)

    private func updateFull(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let L = k.dim(2)
        let prev = offset
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

    // MARK: - Rotating (mlx_lm RotatingKVCache)

    private func trim(_ trimSize: Int, _ v: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var parts: [MLXArray] = []
        if trimSize > 0 {
            parts.append(v[0..., 0..., 0 ..< keep, 0...])
            parts.append(v[0..., 0..., (trimSize + keep)..., 0...])
        } else {
            parts.append(v)
        }
        if let append { parts.append(append) }
        return MLX.concatenated(parts, axis: 2)
    }

    private func temporalOrder(_ v: MLXArray) -> MLXArray {
        if idx == v.dim(2) {
            return v
        } else if idx < offset {
            return MLX.concatenated([
                v[0..., 0..., 0 ..< keep, 0...],
                v[0..., 0..., idx..., 0...],
                v[0..., 0..., keep ..< idx, 0...],
            ], axis: 2)
        } else {
            return v[0..., 0..., 0 ..< idx, 0...]
        }
    }

    private func updateConcat(_ keysIn: MLXArray, _ valuesIn: MLXArray) -> (MLXArray, MLXArray) {
        let maxSize = self.maxSize!
        if keys == nil {
            keys = keysIn
            values = valuesIn
        } else {
            keys = temporalOrder(keys!)
            values = temporalOrder(values!)
            idx = keys!.dim(2)
            let trimSize = idx - maxSize + 1
            keys = trim(trimSize, keys!, append: keysIn)
            values = trim(trimSize, values!, append: valuesIn)
        }
        offset += keysIn.dim(2)
        idx = keys!.dim(2)
        return (keys!, values!)
    }

    private func updateInPlace(_ keysIn: MLXArray, _ valuesIn: MLXArray) -> (MLXArray, MLXArray) {
        let maxSize = self.maxSize!
        let S = keysIn.dim(2)
        let B = keysIn.dim(0), nKV = keysIn.dim(1), kDim = keysIn.dim(3), vDim = valuesIn.dim(3)
        let prev = offset
        if keys == nil || (prev >= keys!.dim(2) && keys!.dim(2) < maxSize) {
            let newSize = min(step, maxSize - prev)
            let newK = MLXArray.zeros([B, nKV, newSize, kDim], dtype: keysIn.dtype)
            let newV = MLXArray.zeros([B, nKV, newSize, vDim], dtype: valuesIn.dtype)
            if let ok = keys, let ov = values {
                keys = MLX.concatenated([ok, newK], axis: 2)
                values = MLX.concatenated([ov, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
            idx = prev
        }

        let trimSize = keys!.dim(2) - maxSize
        if trimSize > 0 {
            keys = trim(trimSize, keys!)
            values = trim(trimSize, values!)
            idx = maxSize
        }

        if idx == maxSize {
            idx = keep
        }

        keys![0..., 0..., idx ..< (idx + S), 0...] = keysIn
        values![0..., 0..., idx ..< (idx + S), 0...] = valuesIn
        offset += S
        idx += S

        if offset < maxSize {
            return (keys![0..., 0..., 0 ..< offset, 0...],
                    values![0..., 0..., 0 ..< offset, 0...])
        }
        return (keys!, values!)
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
        idx = 0
    }
}
