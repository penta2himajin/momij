import XCTest
import Foundation
@testable import MomijCore

/// Tests for workspace-relative path policy.
final class PathPolicyTests: XCTestCase {
    private let root = "/Users/penta2himajin/repos"

    func testNormalizeEscapesStripsDebris() {
        // The live escalation variants (backslash debris around separators).
        XCTAssertEqual(
            PathPolicy.normalizeEscapes(#"\\\/Users\/penta2himajin\/repos\/x.py"#),
            "/Users/penta2himajin/repos/x.py")
        // Quotes are debris in path values too.
        XCTAssertEqual(
            PathPolicy.normalizeEscapes(#""\\\\/repos/scratch/a.py""#),
            "/repos/scratch/a.py")
        // Plain path untouched.
        XCTAssertEqual(PathPolicy.normalizeEscapes("scratch/x.py"), "scratch/x.py")
    }

    func testResolveRelativeJoinsRoot() {
        XCTAssertEqual(
            PathPolicy.resolve("scratch/momij-subagent-test/count_verify_fires.py", in: root),
            root + "/scratch/momij-subagent-test/count_verify_fires.py")
        XCTAssertEqual(
            PathPolicy.resolve("notes.md", in: root),
            root + "/notes.md")
    }

    func testResolveAbsoluteInsideRootKept() {
        let abs = root + "/scratch/x.py"
        XCTAssertEqual(PathPolicy.resolve(abs, in: root), abs)
    }

    func testResolveAbsoluteOutsideRootKeptUnchanged() {
        let abs = "/Users/penta2himajin/other/repo/x.py"
        XCTAssertEqual(PathPolicy.resolve(abs, in: root), abs)
    }

    func testResolveNormalizesEscapeDebrisToAbsolute() {
        XCTAssertEqual(
            PathPolicy.resolve(#"\\\/Users\/penta2himajin\/repos\/scratch\/a.py"#, in: root),
            root + "/scratch/a.py")
    }

    func testRewrittenArgsResolvesPathFields() {
        let args = ToolMarkup.jsonDumps(["path": "scratch/a.py", "content": "body"])
        let out = PathPolicy.rewrittenArgs(args, root: root)
        XCTAssertNotNil(out)
        let parsed = ToolMarkup.jsonParseValue(out!) as? [String: Any]
        XCTAssertEqual(parsed?["path"] as? String, root + "/scratch/a.py")
        XCTAssertEqual(parsed?["content"] as? String, "body", "non-path fields untouched")
    }

    func testRewrittenArgsNilWhenNoChange() {
        let args = ToolMarkup.jsonDumps(["command": "ls"])
        XCTAssertNil(PathPolicy.rewrittenArgs(args, root: root))
        // Absolute inside root: after resolve it stays the same string -> nil.
        let absArgs = ToolMarkup.jsonDumps(["path": root + "/a.py"])
        XCTAssertNil(PathPolicy.rewrittenArgs(absArgs, root: root))
    }

    func testStripRootPrefix() {
        XCTAssertEqual(
            PathPolicy.stripRootPrefix(
                in: "wrote \(root)/scratch/a.py and \(root)/b.txt", root: root),
            "wrote scratch/a.py and b.txt")
    }

    func testSuffixLineContent() {
        let line = PathPolicy.suffixLine()
        XCTAssertTrue(line.contains("RELATIVE to the workspace root"), line)
        XCTAssertTrue(line.contains("Never write absolute paths"), line)
    }


    func testIsOutsideWorkspace() {
        XCTAssertTrue(PathPolicy.isOutsideWorkspace("/mijij-subagent-test/x.py", root: root))
        XCTAssertFalse(PathPolicy.isOutsideWorkspace(root + "/scratch/x.py", root: root))
        // Relative is always fine.
        XCTAssertFalse(PathPolicy.isOutsideWorkspace("scratch/x.py", root: root))
        // Escape debris normalized before the check.
        XCTAssertTrue(PathPolicy.isOutsideWorkspace(#"\\\/mnt\\\/x.py"#, root: root))
    }

    func testRepairFieldsAlias() {
        // The call used file_path; the detector speaks of path.
        let detail = ["reason": "path_not_filelike"]
        XCTAssertEqual(
            ArgRepair.repairFields(detail: detail, args: ["file_path": "x"]),
            ["file_path"])
        XCTAssertEqual(
            ArgRepair.repairFields(detail: detail, args: ["path": "x"]),
            ["path"])
    }

    func testWorkspaceRootFromEnv() {
        // Env unset in the test process -> nil (mode off).
        XCTAssertNil(PathPolicy.workspaceRoot())
    }
}