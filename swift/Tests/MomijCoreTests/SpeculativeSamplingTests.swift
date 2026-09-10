import XCTest
@testable import MomijCore

final class SpeculativeSamplingTests: XCTestCase {
    func testAcceptsWhenPDominatesUniformQ() {
        // Single candidate → q=1, p=1 → always accept
        var rng = SplitMix64(seed: 1)
        let proc = LogitsProcessor(temperature: 1, topP: 1)
        let d = SpeculativeSampling.tryAccept(
            draftToken: 7, ids: [7], logits: [0], processor: proc, rng: &rng)
        XCTAssertTrue(d.accepted)
        XCTAssertEqual(d.token, 7)
    }

    func testRejectResamplesFromTarget() {
        // draft token missing from support → q=0 → reject + sample
        var rng = SplitMix64(seed: 99)
        let proc = LogitsProcessor(temperature: 0, topP: 1)
        let d = SpeculativeSampling.tryAccept(
            draftToken: 99, ids: [1, 2, 3], logits: [0, 5, 0], processor: proc, rng: &rng)
        XCTAssertFalse(d.accepted)
        XCTAssertEqual(d.token, 2)
    }

    func testRoutingGreedyUsesSuffixFlagOnlyWhenCompatible() {
        var opts = GenerateOptions(temperature: 0.8, useSuffixSpec: true)
        XCTAssertFalse(opts.isGreedyCompatible)
        opts = GenerateOptions(temperature: 0, useSuffixSpec: true)
        XCTAssertTrue(opts.isGreedyCompatible)
    }
}
