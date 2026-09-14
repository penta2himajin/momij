import XCTest
import Foundation
@testable import MomijCore

/// Tests for protocol-tool interception (the ending fumble absorb): after
/// long tool-call runs the model signals "I'm done" by calling a protocol
/// tool — often with unusable arguments — and the harness rejection loops
/// the turn. momij absorbs the intent into a clean stop.
final class ProtocolAbsorbTests: XCTestCase {
    private func call(_ name: String, _ args: String = "{}") -> OpenAIChatCompat.ToolCallSpec {
        .init(id: "call-1", name: name, arguments: args)
    }

    // MARK: tool names (MOMIJ_PROTOCOL_TOOLS)

    func testToolNamesDefault() {
        XCTAssertEqual(
            ProtocolAbsorb.toolNames(env: nil),
            Set(["submit", "ask_user_question"]))
    }

    func testToolNamesOverrideParsesCommaList() {
        XCTAssertEqual(
            ProtocolAbsorb.toolNames(env: " submit , report_done "),
            Set(["submit", "report_done"]))
    }

    func testToolNamesEmptyDisables() {
        XCTAssertEqual(ProtocolAbsorb.toolNames(env: "  "), Set())
        XCTAssertEqual(ProtocolAbsorb.toolNames(env: ","), Set())
    }

    // MARK: absorb detection

    func testAbsorbCallFiresOnProtocolTool() {
        let names = ProtocolAbsorb.toolNames(env: nil)
        XCTAssertEqual(
            ProtocolAbsorb.absorbCall(
                in: [call("write", #"{"path": "a.md"}"#), call("submit")],
                names: names
            )?.name, "submit")
        XCTAssertEqual(
            ProtocolAbsorb.absorbCall(in: [call("ask_user_question")], names: names)?
                .name, "ask_user_question")
    }

    func testAbsorbCallNilWhenNoProtocolTool() {
        let names = ProtocolAbsorb.toolNames(env: nil)
        XCTAssertNil(ProtocolAbsorb.absorbCall(
            in: [call("write", "{}"), call("bash", "{}")], names: names))
        // Empty names can never match a protocol tool.
        XCTAssertNil(ProtocolAbsorb.absorbCall(in: [call("")], names: names))
    }

    func testAbsorbDisabledByEmptyEnv() {
        // Empty MOMIJ_PROTOCOL_TOOLS disables interception entirely.
        XCTAssertNil(ProtocolAbsorb.absorbCall(
            in: [call("submit")], names: ProtocolAbsorb.toolNames(env: "")))
    }

    // MARK: final content

    func testFinalContentPrefersDescription() {
        XCTAssertEqual(
            ProtocolAbsorb.finalContent(for: call(
                "submit",
                #"{"description": "Wrote the script and verified its output."}"#)),
            "Wrote the script and verified its output.")
    }

    func testFinalContentPlaceholderWhenDescriptionUnusable() {
        XCTAssertEqual(
            ProtocolAbsorb.finalContent(for: call("submit", "{}")),
            "Task complete.")
        XCTAssertEqual(
            ProtocolAbsorb.finalContent(
                for: call("ask_user_question", #"{"description": "   "}"#)),
            "Task complete.")
    }
}