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
            return MLX.matmul(x, w.transposed(0, 1))
        }
    }
}

public final class KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var offset: Int = 0
    public let maxSize: Int?  // nil = full, else rotating SWA

    public init(maxSize: Int? = nil) { self.maxSize = maxSize }

    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let L = k.dim(2)
        if let maxSize {
            if keys == nil {
                keys = k; values = v
            } else if keys!.dim(2) + L <= maxSize {
                keys = MLX.concatenated([keys!, k], axis: 2)
                values = MLX.concatenated([values!, v], axis: 2)
            } else {
                // rotate: keep last maxSize-L, append new
                let keep = max(0, maxSize - L)
                if keep == 0 {
                    keys = k; values = v
                } else {
                    let start = keys!.dim(2) - keep
                    keys = MLX.concatenated([keys![0..., 0..., start..., 0...], k], axis: 2)
                    values = MLX.concatenated([values![0..., 0..., start..., 0...], v], axis: 2)
                }
            }
            offset += L
            return (keys!, values!)
        } else {
            if keys == nil {
                keys = k; values = v
            } else {
                keys = MLX.concatenated([keys!, k], axis: 2)
                values = MLX.concatenated([values!, v], axis: 2)
            }
            offset += L
            return (keys!, values!)
        }
    }

    public func reset() {
        keys = nil; values = nil; offset = 0
    }
}
