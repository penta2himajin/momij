import XCTest
import Foundation
@testable import MomijCore

/// Tests for targeted field repair of degenerate tool-call arguments.
final class ArgRepairTests: XCTestCase {
    // MARK: fieldsToFix

    func testFieldsToFixMapping() {
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "path_not_filelike"]), ["path"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "content_equals_path"]), ["content"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "cargo_toml_too_thin"]), ["content"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "source_content_too_short"]), ["content"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "empty_edits"]), ["edits"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "empty_edit_texts"]), ["edits"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "noop_edit"]), ["edits"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "invented_old_text"]), ["edits"])
        XCTAssertEqual(ArgRepair.fieldsToFix(detail: ["reason": "unknown_thing"]), [])
    }

    // MARK: originalUserText (evprtr parity)

    func testOriginalUserTextPrefersLongestEarlyMessage() {
        let messages: [OpenAIChatCompat.ChatMessage] = [
            .init(role: "system", content: "sys"),
            .init(role: "user", content: "short steer"),
            .init(role: "assistant", content: "ok"),
            .init(role: "tool", content: "result"),
            .init(role: "user", content: String(repeating: "primary task text. ", count: 20)),
        ]
        let out = ArgRepair.originalUserText(messages)
        XCTAssertTrue(out.hasPrefix("primary task text."), out)
    }

    func testOriginalUserTextSkipsPlaceholders() {
        let messages: [OpenAIChatCompat.ChatMessage] = [
            .init(role: "user", content: "agent"),
            .init(role: "user", content: "(no user text)"),
        ]
        XCTAssertEqual(ArgRepair.originalUserText(messages), "(no user text)")
    }

    func testOriginalUserTextSkipsHarnessReminderBlocks() {
        // DSH wraps harness context (workspace instructions incl. repo paths
        // like .github/PULL_REQUEST_TEMPLATE.md) as <system-reminder> user
        // turns. They are LONGER than the task, so the "longest early"
        // heuristic selects them and path extraction reads the wrong text
        // (measured live: adopted path github/PULL_REQUEST_TEMPLATE.md).
        let reminder = "<system-reminder>\nThe following workspace instructions "
            + String(repeating: "may be relevant to your work. ", count: 30)
            + ".github/PULL_REQUEST_TEMPLATE.md\n</system-reminder>"
        let task = String(repeating: "write scratch/momij-subagent-test/x.py ", count: 6)
            .trimmingCharacters(in: .whitespaces)
        let messages: [OpenAIChatCompat.ChatMessage] = [
            .init(role: "system", content: "sys"),
            .init(role: "user", content: reminder),
            .init(role: "user", content: task),
        ]
        XCTAssertEqual(ArgRepair.originalUserText(messages), task)
    }

    func testOriginalUserTextSkipsToolResultTurns() {
        // ToolMarkup.rewriteMessages turns role "tool" into role "user" with a
        // <tool_result> wrapper. Those turns are long, so the "longest early"
        // heuristic selected one as the task and extraction read a trace-file
        // path out of it (measured live: task_head started with
        // "<tool_result>\ntotal 2504\n...").
        let toolResult = "<tool_result>\n"
            + String(repeating: "drwxr-xr-x  315 staff  10080 Sep 14 14:05 .\n", count: 30)
            + "</tool_result>"
        let task = "Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py."
        let messages: [OpenAIChatCompat.ChatMessage] = [
            .init(role: "user", content: task),
            .init(role: "assistant", content: "listing the directory"),
            .init(role: "user", content: toolResult),
        ]
        XCTAssertEqual(ArgRepair.originalUserText(messages), task)
    }

    func testOriginalUserTextSkipsRuntimeContextBlocks() {
        // DSH subagent requests carry the runtime/policy preamble as its own
        // user turn ("Current runtime context. This snapshot supersedes…"),
        // longer than the task, so the length heuristic selected it and
        // extraction returned no path (measured live: reason=no_task_path,
        // task_head="Current runtime context. …").
        let runtime = "Current runtime context. This snapshot supersedes earlier "
            + "runtime-context snapshots.\n\nCurrent DSH file policy: workspace-write. "
            + String(repeating: "Any available operation enforced by the DSH file sandbox. ", count: 15)
        let task = "Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py."
        let messages: [OpenAIChatCompat.ChatMessage] = [
            .init(role: "system", content: "sys"),
            .init(role: "user", content: runtime),
            .init(role: "user", content: task),
        ]
        XCTAssertEqual(ArgRepair.originalUserText(messages), task)
    }

    func testNonTaskUserTurnMarkers() {
        // The scaffolding markers observed live; a task quoting one mid-text
        // stays a candidate.
        XCTAssertTrue(ArgRepair.isNonTaskUserTurn("<system-reminder>\nx"))
        XCTAssertTrue(ArgRepair.isNonTaskUserTurn("<tool_result>\nx"))
        XCTAssertTrue(ArgRepair.isNonTaskUserTurn("Current runtime context. This snapshot …"))
        XCTAssertFalse(ArgRepair.isNonTaskUserTurn("Write a script per <system-reminder> notes"))
    }

    // MARK: schema construction

    private let toolLines = [
        #"{"type":"function","function":{"name":"write","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string"}},"required":["path","content"]}}}"#,
        #"{"type":"function","function":{"name":"ls","description":"List directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}"#,
    ]

    func testConstrainedObjectSchemaUsesToolProperties() {
        let schema = ArgRepair.constrainedObjectSchema(
            toolLines: toolLines, tool: "write", fields: ["path"])
        XCTAssertEqual(schema?["type"] as? String, "object")
        XCTAssertEqual(schema?["required"] as? [String], ["path"])
        let props = schema?["properties"] as? [String: Any]
        let path = props?["path"] as? [String: Any]
        XCTAssertEqual(path?["type"] as? String, "string")
        XCTAssertEqual(path?["description"] as? String, "File path")
        XCTAssertNil(props?["content"], "only requested fields are included")
    }

    func testConstrainedObjectSchemaFallsBackToString() {
        let schema = ArgRepair.constrainedObjectSchema(
            toolLines: toolLines, tool: "nonexistent", fields: ["edits"])
        let props = schema?["properties"] as? [String: Any]
        XCTAssertEqual((props?["edits"] as? [String: Any])?["type"] as? String, "string")
    }

    // MARK: prompt

    func testFieldRepairPromptShape() {
        let prompt = ArgRepair.fieldRepairPrompt(
            task: "Create notes.md",
            tool: "write",
            argsJSON: "{\"content\": \"hello\"}",
            fields: ["path"],
            reasons: ["path": "path_not_filelike"])
        XCTAssertTrue(prompt.contains("ONLY a JSON object"), prompt)
        XCTAssertTrue(prompt.contains("Create notes.md"), prompt)
        XCTAssertTrue(prompt.contains("`write`"), prompt)
        XCTAssertTrue(prompt.contains("path_not_filelike"), prompt)
        XCTAssertTrue(prompt.hasSuffix("No prose, no markdown."), prompt)
    }

    // MARK: merge + parse

    func testMergeFieldsKeepsExisting() {
        let args: [String: Any] = ["path": "", "content": "long content body"]
        let merged = ArgRepair.mergeFields(
            into: args, generated: ["path": "notes.md"], fields: ["path"])
        XCTAssertEqual(merged["path"] as? String, "notes.md")
        XCTAssertEqual(merged["content"] as? String, "long content body")
    }

    func testMergeFieldsAcceptsArrayValue() {
        let merged = ArgRepair.mergeFields(
            into: ["path": "a.txt"],
            generated: ["edits": [["oldText": "x", "newText": "y"]]],
            fields: ["edits"])
        let edits = merged["edits"] as? [[String: Any]]
        XCTAssertEqual(edits?.count, 1)
    }

    func testArgsJSONStringRoundtrip() {
        let json = ArgRepair.argsJSONString(["path": "notes.md", "content": "hi"])
        let back = ArgRepair.parseArgs(json)
        XCTAssertEqual(back["path"] as? String, "notes.md")
        // python-style spacing contract (jsonDumps parity)
        XCTAssertTrue(json.contains("\"path\": \"notes.md\""), json)
    }

    func testMissingRequiredFields() {
        // DSH bash-like schema requires command + description.
        let bashLines = [
            #"{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command","description"]}}}"#,
        ]
        // description absent -> flagged.
        XCTAssertEqual(
            ArgRepair.missingRequiredFields(
                toolLines: bashLines, tool: "bash",
                args: ["command": "ls"]),
            ["description"])
        // Empty string counts as missing (the harness rejects empty required
        // strings the same way for our purposes).
        XCTAssertEqual(
            ArgRepair.missingRequiredFields(
                toolLines: bashLines, tool: "bash",
                args: ["command": "ls", "description": ""]),
            ["description"])
        // All present -> empty.
        XCTAssertTrue(
            ArgRepair.missingRequiredFields(
                toolLines: bashLines, tool: "bash",
                args: ["command": "ls", "description": "List files"]).isEmpty)
        // Unknown tool -> no schema, no opinion.
        XCTAssertTrue(
            ArgRepair.missingRequiredFields(
                toolLines: bashLines, tool: "nonexistent", args: [:]).isEmpty)
    }

    func testFirstJSONObjectToleratesProse() {
        let obj = ArgRepair.firstJSONObject(
            in: "Sure.\n\n{\"path\": \"a.md\"}\n\nDone.")
        XCTAssertEqual(obj?["path"] as? String, "a.md")
        XCTAssertNil(ArgRepair.firstJSONObject(in: "no object here"))
    }

    // MARK: extraction-first path fix (pathFromTask wiring)

    func testPathFromTask() {
        XCTAssertEqual(
            ArgRepair.pathFromTask(
                "Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py (this path is RELATIVE to the workspace root; always write paths in this relative form)."),
            "scratch/momij-subagent-test/count_verify_fires.py")
        XCTAssertEqual(
            ArgRepair.pathFromTask(
                "Create a new file named notes.md using the write tool. Put two short lines of text in it."),
            "notes.md")
        XCTAssertNil(ArgRepair.pathFromTask("Reply with exactly OK"))
    }

    func testPathFromTaskKeepsLeadingDot() {
        // Hidden-directory paths keep the dot: ".github/..." must not be
        // mangled into "github/..." (the live probe adopted exactly that
        // mangled path from the harness-injected context text).
        XCTAssertEqual(
            ArgRepair.pathFromTask(
                "One PR per workstream; closes the work per .github/PULL_REQUEST_TEMPLATE.md."),
            ".github/PULL_REQUEST_TEMPLATE.md")
        // "./debris" and leading-slash debris still normalize; trailing
        // sentence punctuation still strips.
        XCTAssertEqual(ArgRepair.pathFromTask("Save to ./notes.md."), "notes.md")
        XCTAssertEqual(ArgRepair.pathFromTask("Write /scratch/a.py now"), "scratch/a.py")
    }

    func testIsPathFix() {
        // The workspace scan's kind is always path-kind.
        XCTAssertTrue(ArgRepair.isPathFix(kind: "path_outside_workspace", fields: ["path"]))
        // Degenerate hits speaking of a path (aliased to file_path) too.
        XCTAssertTrue(ArgRepair.isPathFix(kind: "degenerate_tool_args", fields: ["file_path"]))
        // A required path property missing on the tool schema as well.
        XCTAssertTrue(ArgRepair.isPathFix(kind: "missing_required_property", fields: ["path"]))
        // Content/edits/description fixes are NOT answered by extraction.
        XCTAssertFalse(ArgRepair.isPathFix(kind: "degenerate_tool_args", fields: ["content"]))
        XCTAssertFalse(ArgRepair.isPathFix(kind: "missing_required_property", fields: ["description"]))
    }

    func testTaskPathMergeAnswersPathOutsideWorkspace() {
        // The live task shape: the task names the exact relative path the
        // model then hallucinated (/mijij-... instead of momij-...).
        let task = "Write a Python 3 script at scratch/momij-subagent-test/count_verify_fires.py (this path is RELATIVE to the workspace root; always write paths in this relative form)."
        let merged = ArgRepair.taskPathMerge(
            kind: "path_outside_workspace", fields: ["path"],
            args: ["path": "/mijij-subagent-test/count_verify_fires.py", "content": "print(1)"],
            task: task)
        XCTAssertEqual(
            merged?["path"] as? String, "scratch/momij-subagent-test/count_verify_fires.py")
        XCTAssertEqual(merged?["content"] as? String, "print(1)", "other fields survive")
    }

    func testTaskPathMergeAliasesFilePath() {
        let merged = ArgRepair.taskPathMerge(
            kind: "degenerate_tool_args", fields: ["file_path"],
            args: ["file_path": "", "content": "body"],
            task: "Create notes.md using the write tool. Put two short lines of text in it.")
        XCTAssertEqual(merged?["file_path"] as? String, "notes.md")
        XCTAssertEqual(merged?["content"] as? String, "body")
    }

    func testTaskPathMergeNilWhenNotPathKind() {
        XCTAssertNil(ArgRepair.taskPathMerge(
            kind: "degenerate_tool_args", fields: ["content"],
            args: ["path": "a.py", "content": "x"],
            task: "Write a.py with the content x"))
    }

    func testTaskPathMergeNilWhenTaskNamesNoPath() {
        // No path in the task -> the caller falls back to the model re-ask.
        XCTAssertNil(ArgRepair.taskPathMerge(
            kind: "path_outside_workspace", fields: ["path"],
            args: ["path": "/elsewhere/x.py"],
            task: "Reply with exactly OK"))
    }

    func testTaskPathMergeNilWhenNonPathFieldsPending() {
        // Extraction can only answer path fields; a mixed fix (path +
        // description) must fall back to the model re-ask.
        XCTAssertNil(ArgRepair.taskPathMerge(
            kind: "missing_required_property", fields: ["path", "description"],
            args: [:], task: "Write notes.md with the report"))
    }
}
