import XCTest
@testable import MomijCore

final class SuffixSpecTests: XCTestCase {
    func testFindsRepeatedSuffixDraft() {
        // history ends with [1,2,3] and earlier contains [1,2,3,9,8]
        let history = [7, 1, 2, 3, 9, 8, 1, 2, 3]
        let draft = SuffixSpec.suffixDraft(history: history, k: 2)
        XCTAssertEqual(draft, [9, 8])
    }

    func testSpecAcceptsMatchingDraft() throws {
        // Deterministic "model": always emits next = last+1
        var seq = 10
        let out = try SuffixSpec.run(
            prompt: [1, 2, 3, 1, 2, 3],
            maxTokens: 4,
            draftK: 2,
            eos: nil,
            step: { ids in
                _ = ids
                let t = seq
                seq += 1
                return t
            },
            multiStep: { ids, k in
                _ = ids
                var r: [Int] = []
                for _ in 0 ..< k {
                    r.append(seq)
                    seq += 1
                }
                return r
            }
        )
        XCTAssertEqual(out.count, 4)
    }
}

final class ConfigTests: XCTestCase {
    func testConfigCodingKeys() throws {
        let json = """
        {"model_type":"maple","hidden_size":2048,"num_hidden_layers":24,
         "num_attention_heads":16,"num_key_value_heads":4,"head_dim":128,
         "vocab_size":10,"num_experts":256,"num_experts_per_tok":8,
         "moe_intermediate_size":512,"rms_norm_eps":1e-6,"rope_theta":10000,
         "sliding_window":512,"partial_rotary_factor":0.5,"tie_word_embeddings":false,
         "layer_types":["sliding_attention","full_attention"],
         "quantization":{"bits":2,"group_size":128,"mode":"affine"}}
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MapleConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 2048)
        XCTAssertEqual(cfg.expertBits, 2)
        XCTAssertTrue(cfg.isSliding(0))
        XCTAssertFalse(cfg.isSliding(1))
    }
}
