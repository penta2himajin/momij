import Foundation
import Metal

extension SeedlessFlashHead {
    /// Deduped (token id, logit) pairs from probe gather. Optionally merges force-token dots.
    public func candidateLogits(
        inds: MTLBuffer? = nil,
        logits: MTLBuffer? = nil,
        hostH: UnsafePointer<Float16>? = nil
    ) -> (ids: [Int], logits: [Float]) {
        let ib = inds ?? indsBuf
        let lb = logits ?? logitsBuf
        let nLogits = nProbes * clusterSize
        let ip = ib.contents().bindMemory(to: Int32.self, capacity: nProbes)
        let lp = lb.contents().bindMemory(to: Float.self, capacity: nLogits)
        var best: [Int: Float] = [:]
        best.reserveCapacity(nLogits)
        for local in 0 ..< nLogits {
            let probe = local / clusterSize
            let row = local % clusterSize
            let cluster = Int(ip[probe])
            guard cluster >= 0, cluster < nClusters else { continue }
            let tid = Int(tokenMapHost[cluster * clusterSize + row])
            let v = lp[local]
            if let old = best[tid] {
                if v > old { best[tid] = v }
            } else {
                best[tid] = v
            }
        }
        if let hostH, !forceIds.isEmpty {
            for fi in 0 ..< forceIds.count {
                var dot: Float = 0
                let base = fi * H
                for k in 0 ..< H {
                    dot += Float(hostH[k]) * Float(forceRows[base + k])
                }
                let tid = forceIds[fi]
                if let old = best[tid] {
                    if dot > old { best[tid] = dot }
                } else {
                    best[tid] = dot
                }
            }
        }
        var ids: [Int] = []
        var scores: [Float] = []
        ids.reserveCapacity(best.count)
        scores.reserveCapacity(best.count)
        for (tid, v) in best {
            ids.append(tid)
            scores.append(v)
        }
        return (ids, scores)
    }

    public func sampleAfterFusedGather(
        hostH: UnsafePointer<Float16>,
        processor: LogitsProcessor,
        seen: Set<Int>,
        counts: [Int: Int],
        rng: inout some RandomNumberGenerator,
        inds: MTLBuffer? = nil,
        logits: MTLBuffer? = nil
    ) -> Int {
        let cands = candidateLogits(inds: inds, logits: logits, hostH: hostH)
        precondition(!cands.ids.isEmpty)
        return processor.sample(
            tokenIds: cands.ids, logits: cands.logits,
            seen: seen, counts: counts, rng: &rng)
    }

    public func sampleAfterCentroids(
        hBuf: MTLBuffer, hostH: UnsafePointer<Float16>,
        processor: LogitsProcessor,
        seen: Set<Int>,
        counts: [Int: Int],
        rng: inout some RandomNumberGenerator
    ) -> Int {
        _ = greedyAfterCentroids(hBuf: hBuf, hostH: hostH)
        return sampleAfterFusedGather(
            hostH: hostH, processor: processor,
            seen: seen, counts: counts, rng: &rng)
    }
}
