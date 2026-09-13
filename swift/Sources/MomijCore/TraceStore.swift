import Foundation

/// Lightweight structured trace store: one JSON record per request,
/// persisted after the response (fire-and-forget). Ported from the
/// compositor TraceStore concept, kept minimal: write / list / get.
public final class TraceStore: @unchecked Sendable {
    private let dir: URL
    private let gate = NSLock()

    public init(dir: URL) {
        self.dir = dir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    public static func newID() -> String {
        let hex = (0 ..< 12).map { _ in
            String(format: "%02x", UInt8.random(in: 0 ... 255))
        }.joined()
        return "tr-" + hex
    }

    /// Atomically persist one trace record (fire-and-forget; failures are
    /// swallowed so tracing can never fail a request).
    public func write(_ record: [String: Any], id: String) {
        gate.lock()
        defer { gate.unlock() }
        let url = dir.appendingPathComponent(id + ".json")
        let payload: [String: Any] = (record as? [String: Any]) ?? [:]
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        else { return }
        let tmp = url.appendingPathExtension("tmp")
        try? data.write(to: tmp, options: .atomic)
        _ = try? FileManager.default.removeItem(at: url)
        _ = try? FileManager.default.moveItem(at: tmp, to: url)
    }

    /// Newest records by modification time, up to `limit`.
    public func list(limit: Int = 20) -> [[String: Any]] {
        gate.lock()
        defer { gate.unlock() }
        let fm = FileManager.default
        let files = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        let sorted = files.filter { $0.pathExtension == "json" }
            .sorted { (a, b) in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                return da > db
            }.prefix(max(1, min(limit, 200)))
        var records: [[String: Any]] = []
        for f in sorted {
            if let data = try? Data(contentsOf: f),
               let obj = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                records.append(obj)
            }
        }
        return records
    }

    public func get(_ id: String) -> [String: Any]? {
        gate.lock()
        defer { gate.unlock() }
        let url = dir.appendingPathComponent(id + ".json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
        else { return nil }
        return obj
    }
}
