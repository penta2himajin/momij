import XCTest
import Foundation
@testable import MomijCore

/// Contract tests for the native Maple tools markup port.
final class ToolMarkupTests: XCTestCase {
    private let tcOpen = "<" + "tool_call" + ">"
    private let tcClose = "<" + "/" + "tool_call" + ">"

    private let lsToolLine = "{"
        + "\"type\": \"function\", \"function\": {\"name\": \"ls\", "
        + "\"description\": \"List directory contents\"}}"

    private func callBlock(_ name: String, _ args: String) -> String {
        tcOpen + "\n{\"name\": " + jsonQuote(name)
            + ", \"arguments\": " + args + "}\n" + tcClose
    }

    private func jsonQuote(_ s: String) -> String {
        ToolMarkup.jsonDumpString(s)
    }

    /// The tools instruction bytes must match the compositor contract.
    func testSystemSuffixBytes() {
        let suffix = ToolMarkup.systemSuffix(toolLines: [lsToolLine])
        XCTAssertTrue(suffix.hasPrefix("# Tools\n\nYou may call one or more functions"), suffix)
        XCTAssertTrue(suffix.contains("within " + ToolMarkup.callOpen + " XML tags:"), suffix)
        XCTAssertTrue(suffix.contains("List directory contents"), suffix)
        XCTAssertTrue(suffix.hasSuffix("tags).\n"), suffix)
        XCTAssertTrue(suffix.contains("</" + "tools" + ">"), suffix)
        // The example payload line exists.
        XCTAssertTrue(suffix.contains("<function-name>"), suffix)
    }

    func testJsonDumpsSpacing() {
        let obj: [String: Any] = [
            "name": "ls",
            "arguments": ["path": "/tmp"] as [String: Any],
        ]
        let dumped = ToolMarkup.jsonDumps(obj)
        XCTAssertTrue(dumped.contains("name") && dumped.contains("ls"), dumped)
        XCTAssertEqual(ToolMarkup.jsonDumps([1, 2, 3]), "[1, 2, 3]")
        XCTAssertEqual(ToolMarkup.jsonDumps(NSNull()), "null")
    }

    func testJsonDumpStringEscapes() {
        let out = ToolMarkup.jsonDumpString("a\nb")
        XCTAssertFalse(out.contains("\n"), out)
    }

    func testParseSingleToolCall() {
        let content = "I will list.\n\n" + callBlock("ls", "{}")
        let parsed = ToolMarkup.parsePseudoToolCalls(content)
        XCTAssertEqual(parsed.calls.count, 1)
        XCTAssertEqual(parsed.calls[0].name, "ls")
        XCTAssertEqual(parsed.calls[0].arguments, "{}")
        XCTAssertEqual(parsed.cleanedContent, "I will list.")
    }

    func testParseMultipleToolCalls() {
        let one = callBlock("ls", "{}")
        let two = callBlock("grep", "{\"pattern\": \"Overview\"}")
        let parsed = ToolMarkup.parsePseudoToolCalls("go.\n\n" + one + "\n\n" + two)
        XCTAssertEqual(parsed.calls.map { $0.name }, ["ls", "grep"])
        XCTAssertEqual(parsed.cleanedContent, "go.")
    }

    func testParseRobustness() {
        // Unterminated block with truncated JSON: no recovery possible —
        // bare open tag stripped, body kept as prose, no phantom call.
        let p1 = ToolMarkup.parsePseudoToolCalls("oops " + tcOpen + "\n{\"name\"")
        XCTAssertTrue(p1.calls.isEmpty)
        XCTAssertFalse(p1.cleanedContent.contains(tcOpen), p1.cleanedContent)
        XCTAssertTrue(p1.cleanedContent.contains("{\"name\""), p1.cleanedContent)
        // Garbage inside: no phantom call, prose kept.
        let p2 = ToolMarkup.parsePseudoToolCalls("not json " + tcOpen + " ??? " + tcClose + " done")
        XCTAssertTrue(p2.calls.isEmpty)
        XCTAssertEqual(p2.cleanedContent, "not json  done")
        // Empty content.
        let p3 = ToolMarkup.parsePseudoToolCalls("")
        XCTAssertTrue(p3.calls.isEmpty)
        XCTAssertEqual(p3.cleanedContent, "")
    }

    func testUnterminatedBlockRecovery() {
        // JSON complete, only the close tag missing (EOS cut right after
        // the object): recover the real call, consume the body.
        let body = "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}"
        let p = ToolMarkup.parsePseudoToolCalls("listing:\n" + tcOpen + "\n" + body)
        XCTAssertEqual(p.calls.count, 1)
        XCTAssertEqual(p.calls[0].name, "bash")
        XCTAssertTrue(p.calls[0].arguments.contains("ls -la"))
        XCTAssertFalse(p.cleanedContent.contains(tcOpen), p.cleanedContent)
        XCTAssertFalse(p.cleanedContent.contains("ls -la"), p.cleanedContent)
    }

    func testUnterminatedBlockRecoversFirstOfConcatenatedObjects() {
        // Live shape (2026-09-13 subagent session): a complete call object
        // followed by a stray second object (the model trying to add a
        // description field). Recover the FIRST complete object only.
        let body = "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la x\"}}{\n  \"description\": \"List test file\"}\n}"
        let p = ToolMarkup.parsePseudoToolCalls(tcOpen + "\n" + body)
        XCTAssertEqual(p.calls.count, 1)
        XCTAssertEqual(p.calls[0].name, "bash")
        XCTAssertTrue(p.calls[0].arguments.contains("ls -la x"))
        XCTAssertFalse(p.cleanedContent.contains("description"), p.cleanedContent)
    }

    func testUnterminatedBlockStringAwareScanner() {
        // A brace inside a string must not count for balancing; the object
        // never completes (string left open) -> prose fallback.
        let body = "{\"name\": \"bash\", \"command\": \"echo \\\"}\""
        let p = ToolMarkup.parsePseudoToolCalls("note " + tcOpen + body)
        XCTAssertTrue(p.calls.isEmpty)
        XCTAssertFalse(p.cleanedContent.contains(tcOpen), p.cleanedContent)
    }

    func testStrayCloseTagStripped() {
        // Close tag without a matching open: bare-tag garbage, stripped.
        let p = ToolMarkup.parsePseudoToolCalls("text " + tcClose + " more")
        XCTAssertTrue(p.calls.isEmpty)
        XCTAssertFalse(p.cleanedContent.contains(tcClose), p.cleanedContent)
        XCTAssertEqual(p.cleanedContent, "text  more")
    }

    func testFirstCompleteJSONObjectScanner() {
        XCTAssertEqual(
            ToolMarkup.firstCompleteJSONObject(in: "x {\"a\": \"}\"} tail"),
            "{\"a\": \"}\"}")
        XCTAssertEqual(ToolMarkup.firstCompleteJSONObject(in: "{\"broken {"), "")
        XCTAssertEqual(ToolMarkup.firstCompleteJSONObject(in: "no braces"), "")
    }

    func testRewriteMessages() {
        let calls: [OpenAIChatCompat.ToolCallSpec] = [
            OpenAIChatCompat.ToolCallSpec(id: "c1", name: "ls", arguments: "{}"),
        ]
        let messages: [OpenAIChatCompat.ChatMessage] = [
            OpenAIChatCompat.ChatMessage(role: "system", content: "sys"),
            OpenAIChatCompat.ChatMessage(role: "user", content: "list files"),
            OpenAIChatCompat.ChatMessage(role: "assistant", content: "listing", toolCalls: calls),
            OpenAIChatCompat.ChatMessage(role: "tool", content: "a.txt\nb.txt"),
        ]
        let out = ToolMarkup.rewriteMessages(messages)
        XCTAssertTrue(out[2].content.contains(ToolMarkup.callOpen) && out[2].content.contains("ls"), out[2].content)
        XCTAssertEqual(out[3].role, "user")
        XCTAssertEqual(out[3].content, "<tool_result>\na.txt\nb.txt\n</tool_result>")
        XCTAssertEqual(out[0].content, "sys")
        XCTAssertEqual(out[1].content, "list files")
    }
}


    func testLenientJSONWithLiteralNewlines() {
        // Live failure (2026-09-14): Maple emitted a bash tool call whose
        // command value contained RAW newlines (multiline python -c) —
        // invalid strict JSON, so the call was dropped and the body leaked
        // as prose. The lenient retry must recover it.
        let open = ToolMarkup.callOpen
        let close = ToolMarkup.callClose
        let inner = "{\"name\": \"bash\", \"arguments\": {\"command\": \"python3 -c \"\nimport json\nprint(1)\n\", \"description\": \"run\"}}"
        let p = ToolMarkup.parsePseudoToolCalls("trying:\n" + open + "\n" + inner + "\n" + close)
        XCTAssertEqual(p.calls.count, 1, "multiline command must parse as a call")
        XCTAssertEqual(p.calls[0].name, "bash")
        XCTAssertTrue(p.calls[0].arguments.contains("import json"))
        XCTAssertTrue(p.cleanedContent.contains("trying:"), p.cleanedContent)
        XCTAssertFalse(p.cleanedContent.contains(open), p.cleanedContent)
    }
