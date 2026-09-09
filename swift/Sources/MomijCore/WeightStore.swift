import Foundation
import MLX

/// mmap-load Maple safetensors shards and expand `row_alpha` → scales/biases.
public final class WeightStore: @unchecked Sendable {
    public private(set) var arrays: [String: MLXArray] = [:]
    public let config: MapleConfig

    public init(modelDir: String, config: MapleConfig? = nil) throws {
        self.config = try config ?? MapleConfig.load(modelDir: modelDir)
        let dir = URL(fileURLWithPath: modelDir)
        let idxURL = dir.appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: idxURL)
        let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let wm = (top["weight_map"] as? [String: String]) ?? [:]
        for shard in Set(wm.values).sorted() {
            let m = try loadArrays(url: dir.appendingPathComponent(shard))
            for (k, v) in m {
                // M1 Max: f16 path measures faster than bf16 for momij MLX (~130 vs ~110).
                // Seedless Metal also prefers f16 scales. Oracle stays bf16 in Python.
                arrays[k] = Self.metalF16(v)
            }
        }
        sanitize()
    }

    /// Convert bf16 → f16 only when a raw Metal kernel needs f16 buffers.
    public static func metalF16(_ a: MLXArray) -> MLXArray {
        a.dtype == .bfloat16 ? a.asType(.float16) : a
    }

    public func get(_ name: String) -> MLXArray? { arrays[name] }
    public func req(_ name: String) -> MLXArray {
        guard let a = arrays[name] else {
            fatalError("missing weight: \(name)")
        }
        return a
    }

    public func residentAll() { MLX.eval(Array(arrays.values)) }

    public func residentNonExperts() {
        let xs = arrays.filter { !$0.key.contains(".switch_mlp.") }.map(\.value)
        MLX.eval(xs)
    }

    /// Mirror `Model.sanitize` in mlx-lm-deepgrove maple.py (row_alpha + up/gate fuse).
    private func sanitize() {
        let groupSize = config.expertGroupSize
        let alphaKeys = arrays.keys.filter { $0.hasSuffix(".row_alpha") }
        for key in alphaKeys {
            guard let alpha = arrays.removeValue(forKey: key) else { continue }
            let prefix = String(key.dropLast(".row_alpha".count))
            guard let packed = arrays["\(prefix).weight"] else { continue }
            // 2-bit: 16 codes / uint32 — repeat row α across groups (bias = -scale).
            let nGroups = (packed.dim(-1) * 16) / groupSize
            // alpha[..., None] * ones(nGroups) via tiled concat
            let expanded = alpha.expandedDimensions(axis: -1)  // [..., 1]
            let ones = MLXArray.ones([nGroups], dtype: expanded.dtype)
            let scales = expanded * ones  // broadcast multiply
            let cont = MLX.contiguous(scales)
            arrays["\(prefix).scales"] = cont
            arrays["\(prefix).biases"] = cont * MLXArray(-1, dtype: cont.dtype)
        }

        for l in 0 ..< config.numHiddenLayers {
            let attn = "model.layers.\(l).self_attn"
            for suffix in ["weight", "scales", "biases"] {
                let q = "\(attn).q_proj.\(suffix)"
                let k = "\(attn).k_proj.\(suffix)"
                let v = "\(attn).v_proj.\(suffix)"
                if arrays[q] != nil, arrays[k] != nil, arrays[v] != nil {
                    arrays["\(attn).qkv_proj.\(suffix)"] =
                        MLX.concatenated([
                            arrays.removeValue(forKey: q)!,
                            arrays.removeValue(forKey: k)!,
                            arrays.removeValue(forKey: v)!,
                        ], axis: 0)
                }
            }
            let p = "model.layers.\(l).mlp.switch_mlp"
            for suffix in ["weight", "scales", "biases"] {
                let up = "\(p).up_proj.\(suffix)"
                let gate = "\(p).gate_proj.\(suffix)"
                if arrays[up] != nil, arrays[gate] != nil {
                    arrays["\(p).up_gate_proj.\(suffix)"] =
                        MLX.concatenated([arrays.removeValue(forKey: up)!,
                                          arrays.removeValue(forKey: gate)!], axis: 1)
                }
            }
        }
    }
}
