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

    /// Deterministic draft (q=1): low-p proposal is rejected; residual is the argmax.
    func testDeterministicQRejectsLowProbDraft() {
        var rng = SplitMix64(seed: 1)
        let proc = LogitsProcessor(temperature: 0, topP: 1)
        let d = SpeculativeSampling.tryAccept(
            draftToken: 1, ids: [1, 2, 3], logits: [0, 9, 0],
            processor: proc, rng: &rng, draftQ: 1)
        XCTAssertFalse(d.accepted)
        XCTAssertEqual(d.token, 2)
    }

    /// Tree-peaked q < p → always accept (p/q ≥ 1).
    func testTreeQAcceptsWhenPAtLeastQ() {
        var rng = SplitMix64(seed: 1)
        let proc = LogitsProcessor(temperature: 0, topP: 1)
        let d = SpeculativeSampling.tryAccept(
            draftToken: 2, ids: [1, 2, 3], logits: [0, 5, 0],
            processor: proc, rng: &rng, draftQ: 0.4)
        XCTAssertTrue(d.accepted)
        XCTAssertEqual(d.token, 2)
    }

    func testQAlongFollowsChildFrequency() {
        let idx = SuffixDraftIndex(maxDepth: 16)
        idx.insert([1, 2, 3, 4])
        idx.insert([1, 2, 3, 4])
        idx.insert([1, 2, 3, 9])
        let q = idx.qAlong(history: [1, 2, 3], draft: [4])
        XCTAssertEqual(q.count, 1)
        XCTAssertEqual(q[0], 2 / 3, accuracy: 1e-5)
    }

    func testRoutingGreedyUsesSuffixFlagOnlyWhenCompatible() {
        var opts = GenerateOptions(temperature: 0.8, useSuffixSpec: true)
        XCTAssertFalse(opts.isGreedyCompatible)
        opts = GenerateOptions(temperature: 0, useSuffixSpec: true)
        XCTAssertTrue(opts.isGreedyCompatible)
    }

    /// temp=0 + draft matches per-row argmax → full accept, bonus from last row.
    func testWalkDraftGreedyFullAccept() {
        var rng = SplitMix64(seed: 1)
        let proc = LogitsProcessor(temperature: 0, topP: 1)
        let rows: [(ids: [Int], logits: [Float])] = [
            ([10, 20, 30], [0, 5, 1]),   // argmax 20
            ([10, 21, 30], [0, 9, 1]),   // argmax 21
            ([11, 22, 30], [0, 1, 8]),   // bonus argmax 30
        ]
        let r = SpeculativeSampling.walkDraft(
            draft: [20, 21], rows: rows, processor: proc,
            seen: [], counts: [:], rng: &rng)
        XCTAssertEqual(r.accepted, 2)
        XCTAssertEqual(r.next, 30)
    }

    func testWalkDraftRejectsAtFirstMismatch() {
        var rng = SplitMix64(seed: 1)
        let proc = LogitsProcessor(temperature: 0, topP: 1)
        let rows: [(ids: [Int], logits: [Float])] = [
            ([10, 20, 30], [0, 5, 1]),
            ([10, 21, 30], [0, 9, 1]),
            ([11, 22, 30], [8, 1, 0]),
        ]
        let r = SpeculativeSampling.walkDraft(
            draft: [99, 21], rows: rows, processor: proc,
            seen: [], counts: [:], rng: &rng)
        XCTAssertEqual(r.accepted, 0)
        XCTAssertEqual(r.next, 20)
    }

    func testWalkDraftPartialAcceptThenReject() {
        var rng = SplitMix64(seed: 1)
        let proc = LogitsProcessor(temperature: 0, topP: 1)
        let rows: [(ids: [Int], logits: [Float])] = [
            ([10, 20, 30], [0, 5, 1]),
            ([10, 21, 30], [0, 9, 1]),
            ([11, 22, 30], [8, 1, 0]),
        ]
        let r = SpeculativeSampling.walkDraft(
            draft: [20, 99], rows: rows, processor: proc,
            seen: [], counts: [:], rng: &rng)
        XCTAssertEqual(r.accepted, 1)
        XCTAssertEqual(r.next, 21)
    }
}
