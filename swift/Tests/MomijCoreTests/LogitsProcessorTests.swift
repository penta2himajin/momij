import XCTest
@testable import MomijCore

final class LogitsProcessorTests: XCTestCase {
    func testTempZeroIsArgmax() {
        let p = LogitsProcessor(temperature: 0, topP: 1)
        var rng = SplitMix64(seed: 1)
        let id = p.sample(tokenIds: [10, 20, 30], logits: [1.0, 5.0, 2.0], rng: &rng)
        XCTAssertEqual(id, 20)
    }

    func testPresencePenaltyLowersSeenTokens() {
        var logits: [Float] = [3.0, 3.0, 3.0]
        let ids = [1, 2, 3]
        LogitsProcessor.applyPresencePenalty(
            logits: &logits, tokenIds: ids, seen: [2], penalty: 2.0)
        XCTAssertEqual(logits[0], 3.0)
        XCTAssertEqual(logits[1], 1.0)
        XCTAssertEqual(logits[2], 3.0)
    }

    func testFrequencyPenaltyScalesWithCount() {
        var logits: [Float] = [0, 0, 0]
        let ids = [7, 8, 9]
        LogitsProcessor.applyFrequencyPenalty(
            logits: &logits, tokenIds: ids, counts: [8: 3], penalty: 0.5)
        XCTAssertEqual(logits[1], -1.5)
    }

    func testRepetitionPenaltyShrinksPositiveLogits() {
        var logits: [Float] = [4.0, -2.0]
        LogitsProcessor.applyRepetitionPenalty(
            logits: &logits, tokenIds: [1, 2], seen: [1, 2], penalty: 2.0)
        XCTAssertEqual(logits[0], 2.0)
        XCTAssertEqual(logits[1], -4.0)
    }

    func testTopPDropsLowMassTail() {
        // probs after softmax roughly proportional; with extreme scores only top survives
        let p = LogitsProcessor(temperature: 1, topP: 0.5)
        var rng = SplitMix64(seed: 42)
        var hits = [Int: Int]()
        for _ in 0 ..< 200 {
            let id = p.sample(
                tokenIds: [1, 2, 3],
                logits: [10.0, 0.0, -10.0],
                rng: &rng)
            hits[id, default: 0] += 1
        }
        XCTAssertGreaterThan(hits[1, default: 0], 180)
        XCTAssertEqual(hits[3, default: 0], 0)
    }

    func testGreedyCompatibleRequiresDefaultSampling() {
        XCTAssertTrue(GenerateOptions().isGreedyCompatible)
        XCTAssertFalse(GenerateOptions(temperature: 0.7).isGreedyCompatible)
        XCTAssertFalse(GenerateOptions(topP: 0.9).isGreedyCompatible)
        XCTAssertFalse(GenerateOptions(presencePenalty: 0.1).isGreedyCompatible)
        XCTAssertFalse(GenerateOptions(frequencyPenalty: 0.1).isGreedyCompatible)
        XCTAssertFalse(GenerateOptions(repetitionPenalty: 1.1).isGreedyCompatible)
    }
}
