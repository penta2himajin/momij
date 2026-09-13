import XCTest
import Foundation
@testable import MomijCore

/// Roundtrip tests for the lightweight native trace store.
final class TraceStoreTests: XCTestCase {
    private func freshDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("momij-trace-test-" + UUID().uuidString)
        return url
    }

    func testWriteListGetRoundtrip() {
        let store = TraceStore(dir: freshDir())
        let id = TraceStore.newID()
        XCTAssertTrue(id.hasPrefix("tr-"), id)
        store.write(["ok": true, "locus": "none", "usage": ["prompt": 5]], id: id)
        let got = store.get(id)
        XCTAssertNotNil(got)
        XCTAssertEqual(got?["locus"] as? String, "none")
        let list = store.list(limit: 10)
        XCTAssertEqual(list.count, 1)
        XCTAssertEqual(list[0]["id"] as? String, nil)  // id is the filename, not duplicated
    }

    func testListNewestFirst() {
        let store = TraceStore(dir: freshDir())
        let ids = (0 ..< 5).map { _ in TraceStore.newID() }
        for (i, id) in ids.enumerated() {
            store.write(["ok": true, "n": i], id: id)
            Thread.sleep(forTimeInterval: 0.02)
        }
        let list = store.list(limit: 5)
        XCTAssertEqual(list.count, 5)
        XCTAssertEqual(list[0]["n"] as? Int, 4)
        let trimmed = store.list(limit: 2)
        XCTAssertEqual(trimmed.count, 2)
    }

    func testGetUnknownReturnsNil() {
        let store = TraceStore(dir: freshDir())
        XCTAssertNil(store.get("tr-nonexistent"))
    }
}
