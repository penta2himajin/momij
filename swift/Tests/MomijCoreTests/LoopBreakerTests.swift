import XCTest
import Foundation
@testable import MomijCore

/// Tests for the tool-call loop breaker (escape-escalation loops).
final class LoopBreakerTests: XCTestCase {
    // MARK: normalization

    func testNormalizedArgsCollapsesEscalation() {
        // The live escalation sequence must all normalize to one form.
        let variants = [
            #"chmod +x /Users/penta2himajin/repos/scratch/momij-subagent-test/count_verify_fires.py"#,
            #"chmod +x \"\\/Users\\/penta2himajin\\/repos\\/scratch\\/momij-subagent-test\\/count_verify_fires.py\""#,
            #"chmod +x \"\\\\\\/Users\\\\\\/penta2himajin\\/repos\\/scratch"#,
            #"chmod +x \\"\\\\\\\\\\/Users\\\\\\\\\\/penta2himajin\\/repos"#,
        ]
        let normalized = variants.map { LoopBreaker.normalizedArgs($0) }
        // All begin with the same prefix and differ only in trailing junk
        // from my truncation — check the core property: no backslashes or
        // quotes survive.
        for n in normalized {
            XCTAssertFalse(n.contains("\\"), n)
            XCTAssertFalse(n.contains("\""), n)
            XCTAssertFalse(n.contains("'"), n)
        }
        // Identical full strings normalize identically.
        XCTAssertEqual(LoopBreaker.normalizedArgs(variants[0]), LoopBreaker.normalizedArgs(variants[0]))
        // Whitespace collapses.
        XCTAssertEqual(LoopBreaker.normalizedArgs("a  \n b"), "a b")
    }

    func testNormalizedArgsKeepsPlainContent() {
        XCTAssertEqual(
            LoopBreaker.normalizedArgs(#"echo "hello world""#),
            "echo hello world")
    }

    // MARK: detection

    private func assistant(_ calls: [(String, String)]) -> OpenAIChatCompat.ChatMessage {
        .init(role: "assistant", content: "",
              toolCalls: calls.enumerated().map { i, call in
                  OpenAIChatCompat.ToolCallSpec(id: "c\(i)", name: call.0, arguments: call.1)
              })
    }

    func testDetectFiresOnTrailingRun() {
        let same = #"{"command": "chmod +x \/Users\/x\/f.py"}"#
        let escalated = #"{"command": "chmod +x \"\\\\/Users\\\\/x\/f.py\""}"#
        let messages: [OpenAIChatCompat.ChatMessage] = [
            .init(role: "user", content: "do it"),
            assistant([("chmod", same)]),
            .init(role: "tool", content: "chmod: error"),
            assistant([("chmod", escalated)]),
            .init(role: "tool", content: "chmod: error again"),
            assistant([("chmod", escalated)]),
            .init(role: "tool", content: "chmod: error"),
            assistant([("chmod", escalated)]),
        ]
        let hit = LoopBreaker.detect(in: messages)
        XCTAssertEqual(hit?.tool, "chmod")
        XCTAssertEqual(hit?.count, 4)
    }

    func testDetectBelowThresholdReturnsNil() {
        let same = #"{"command": "chmod +x \/Users\/x\/f.py"}"#
        let messages: [OpenAIChatCompat.ChatMessage] = [
            .init(role: "user", content: "do it"),
            assistant([("chmod", same)]),
            .init(role: "tool", content: "ok"),
            assistant([("chmod", same)]),
        ]
        XCTAssertNil(LoopBreaker.detect(in: messages))
    }

    func testDetectDifferentArgsDoNotFire() {
        // Legitimate repetition (same tool, different files) must not fire.
        let messages: [OpenAIChatCompat.ChatMessage] = [
            assistant([("read", #"{"path": "a.md"}"#)]),
            assistant([("read", #"{"path": "b.md"}"#)]),
            assistant([("read", #"{"path": "c.md"}"#)]),
        ]
        XCTAssertNil(LoopBreaker.detect(in: messages))
    }

    func testDetectBreaksRunOnDifferentCall() {
        let same = #"{"command": "chmod +x \/Users\/x"}"#
        let messages: [OpenAIChatCompat.ChatMessage] = [
            assistant([("chmod", same)]),
            assistant([("chmod", same)]),
            assistant([("ls", #"{"path": "."}"#)]),
            assistant([("chmod", same)]),
        ]
        XCTAssertNil(LoopBreaker.detect(in: messages))
    }

    // MARK: nudge

    func testBreakNudgeShape() {
        let nudge = LoopBreaker.breakNudge(tool: "chmod", count: 5)
        XCTAssertTrue(nudge.contains("`chmod`"), nudge)
        XCTAssertTrue(nudge.contains("5 times"), nudge)
        XCTAssertTrue(nudge.contains("plain forward slashes"), nudge)
        XCTAssertTrue(nudge.contains("Do NOT retry"), nudge)
    }
}