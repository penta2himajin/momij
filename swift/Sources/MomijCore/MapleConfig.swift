import Foundation

/// Maple-Preview architecture constants loaded from `config.json`.
public struct MapleConfig: Codable, Sendable {
    public var modelType: String = "maple"
    public var hiddenSize: Int = 2048
    public var intermediateSize: Int = 5120
    public var moeIntermediateSize: Int = 512
    public var numHiddenLayers: Int = 24
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 4
    public var headDim: Int = 128
    public var vocabSize: Int = 151_936
    public var numExperts: Int = 256
    public var numExpertsPerTok: Int = 8
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 10_000
    public var slidingWindow: Int = 512
    public var partialRotaryFactor: Float = 0.5
    public var tieWordEmbeddings: Bool = false
    public var layerTypes: [String] = []
    public var quantization: QuantConfig?
    public var flashHead: FlashHeadMeta?

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "maple"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_936
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        quantization = try c.decodeIfPresent(QuantConfig.self, forKey: .quantization)
        flashHead = try c.decodeIfPresent(FlashHeadMeta.self, forKey: .flashHead)
    }

    public struct QuantConfig: Codable, Sendable {
        public var bits: Int = 2
        public var groupSize: Int = 128
        public var mode: String = "affine"
        public var lmHead: NestedBits?
        public struct NestedBits: Codable, Sendable {
            public var bits: Int = 4
            public var groupSize: Int = 64
            enum CodingKeys: String, CodingKey {
                case bits
                case groupSize = "group_size"
            }
            public init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                bits = try c.decodeIfPresent(Int.self, forKey: .bits) ?? 4
                groupSize = try c.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
            }
        }
        enum CodingKeys: String, CodingKey {
            case bits, mode
            case groupSize = "group_size"
            case lmHead = "lm_head"
        }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            bits = try c.decodeIfPresent(Int.self, forKey: .bits) ?? 2
            groupSize = try c.decodeIfPresent(Int.self, forKey: .groupSize) ?? 128
            mode = try c.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
            lmHead = try c.decodeIfPresent(NestedBits.self, forKey: .lmHead)
        }
    }

    public struct FlashHeadMeta: Codable, Sendable {
        public var bits: Int = 4
        public var clusterSize: Int = 32
        public var nClusters: Int = 4748
        public var nProbes: Int = 512
        public var groupSize: Int = 64
        public var headBits: Int = 4
        public var headGroupSize: Int = 64
        public var forceTokens: [Int] = []
        public var scaledCentroids: Bool = true
        enum CodingKeys: String, CodingKey {
            case bits
            case clusterSize = "cluster_size"
            case nClusters = "n_clusters"
            case nProbes = "n_probes"
            case groupSize = "group_size"
            case headBits = "head_bits"
            case headGroupSize = "head_group_size"
            case forceTokens = "force_tokens"
            case scaledCentroids = "scaled_centroids"
        }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            bits = try c.decodeIfPresent(Int.self, forKey: .bits) ?? 4
            clusterSize = try c.decodeIfPresent(Int.self, forKey: .clusterSize) ?? 32
            nClusters = try c.decodeIfPresent(Int.self, forKey: .nClusters) ?? 4748
            nProbes = try c.decodeIfPresent(Int.self, forKey: .nProbes) ?? 512
            groupSize = try c.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
            headBits = try c.decodeIfPresent(Int.self, forKey: .headBits) ?? 4
            headGroupSize = try c.decodeIfPresent(Int.self, forKey: .headGroupSize) ?? 64
            forceTokens = try c.decodeIfPresent([Int].self, forKey: .forceTokens) ?? []
            scaledCentroids = try c.decodeIfPresent(Bool.self, forKey: .scaledCentroids) ?? true
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case quantization
        case flashHead = "flash_head"
    }

    public var expertBits: Int { quantization?.bits ?? 2 }
    public var expertGroupSize: Int { quantization?.groupSize ?? 128 }
    public var headBits: Int { quantization?.lmHead?.bits ?? 4 }
    public var headGroupSize: Int { quantization?.lmHead?.groupSize ?? 64 }
    public var ropeDim: Int { Int(Float(headDim) * partialRotaryFactor) }

    public func isSliding(_ layer: Int) -> Bool {
        guard layer < layerTypes.count else { return (layer + 1) % 4 != 0 }
        return layerTypes[layer] == "sliding_attention"
    }

    public static func load(modelDir: String) throws -> MapleConfig {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        return try dec.decode(MapleConfig.self, from: data)
    }
}
