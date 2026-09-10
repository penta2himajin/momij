import Foundation
import Metal
import MLX

/// FlashHead: Metal centroid gemv fused into the layer CB → top-k → Metal 4-bit gather.
///
/// Default path: CPU top-k after layer wait, then a short gather CB (second wait).
/// Opt-in `MOMIJ_FLASH_FUSE=1`: hierarchical GPU top-k + gather in the layer CB
/// (same wait as layers) — avoids serial full-E×K GPU top-k (measured catastrophic).
public final class SeedlessFlashHead {
    let forceIds: [Int]
    let forceRows: [Float16]
    let tokenMapHost: [Int32]
    let nClusters: Int
    public let nProbes: Int
    public let clusterSize: Int
    let headGroupSize: Int
    let H: Int
    /// When true, encode top-k+gather into the layer CB (see `encodeFusedAfterCentroids`).
    public let fuseIntoLayerCB: Bool

    let centroidsBuf: MTLBuffer
    let scoresBuf: MTLBuffer
    let headWBuf: MTLBuffer
    let headSBuf: MTLBuffer
    let headBBuf: MTLBuffer
    let indsBuf: MTLBuffer
    let logitsBuf: MTLBuffer
    let candScoresBuf: MTLBuffer
    let candIndsBuf: MTLBuffer
    /// Device copy of token_map for GPU argmax → token id (greedy chain).
    public let tokenMapBuf: MTLBuffer
    public let forceIdsBuf: MTLBuffer?
    public let forceRowsBuf: MTLBuffer?
    public var forceCount: Int { forceIds.count }
    private var topScratch: [Int]

    public init?(store: WeightStore, device: MTLDevice) {
        let cfg = store.config
        guard let meta = cfg.flashHead,
              store.get("lm_head_flash.centroids.weight") != nil,
              store.get("lm_head_flash.token_map") != nil
        else { return nil }

        nClusters = meta.nClusters
        clusterSize = meta.clusterSize
        H = cfg.hiddenSize
        let envProbes = ProcessInfo.processInfo.environment["MOMIJ_FLASH_PROBES"].flatMap(Int.init)
        // Default 64: best measured stable band on M1 Max (96 similar quality, more gather tax).
        nProbes = min(envProbes ?? 64, nClusters)
        headGroupSize = meta.headGroupSize
        // Default on: hierarchical GPU top-k + gather in the layer CB kills the ~0.3ms
        // second-wait tax (measured ~208 tok/s vs ~180). Set MOMIJ_FLASH_FUSE=0 to disable.
        // Not the failed serial full-E×K top-k; chunk→merge only.
        fuseIntoLayerCB = ProcessInfo.processInfo.environment["MOMIJ_FLASH_FUSE"] != "0"

        let cw = store.req("lm_head_flash.centroids.weight")
        let cs = store.req("lm_head_flash.centroids.scales")
        let cb = store.req("lm_head_flash.centroids.biases")
        let tm = store.req("lm_head_flash.token_map").asType(.int32)
        let order = tm.reshaped([-1])
        let lw = store.req("lm_head.weight")
        let ls = store.req("lm_head.scales")
        let lb = store.req("lm_head.biases")
        let hw = MLX.take(lw, order, axis: 0).reshaped([nClusters, clusterSize, lw.dim(-1)])
        let hs = MLX.take(ls, order, axis: 0).reshaped([nClusters, clusterSize, ls.dim(-1)])
        let hb = MLX.take(lb, order, axis: 0).reshaped([nClusters, clusterSize, lb.dim(-1)])

        forceIds = meta.forceTokens
        var forceFlat: [Float16] = []
        if !forceIds.isEmpty {
            let fids = MLXArray(forceIds.map { Int32($0) })
            let fw = MLX.dequantized(
                MLX.take(lw, fids, axis: 0),
                scales: MLX.take(ls, fids, axis: 0),
                biases: MLX.take(lb, fids, axis: 0),
                groupSize: headGroupSize, bits: meta.headBits, mode: .affine
            ).asType(.float16)
            MLX.eval(fw)
            let n = forceIds.count * H
            forceFlat = [Float16](repeating: 0, count: n)
            fw.asData(access: .copy).data.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float16.self)
                for i in 0 ..< n { forceFlat[i] = src[i] }
            }
        }

        let centF = MLX.dequantized(
            cw, scales: cs, biases: cb,
            groupSize: meta.groupSize, bits: meta.bits, mode: .affine
        ).asType(.float16)
        MLX.eval(centF, tm, hw, hs, hb)

        var tmHost = [Int32](repeating: 0, count: nClusters * clusterSize)
        tm.asData(access: .copy).data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Int32.self)
            for i in 0 ..< tmHost.count { tmHost[i] = src[i] }
        }
        tokenMapHost = tmHost

        guard let cbuf = SeedlessMetal.mtlBuf(centF, device),
              let wbuf = SeedlessMetal.mtlBuf(hw, device),
              let sbuf = SeedlessMetal.mtlBuf(hs.asType(.float16), device),
              let bbuf = SeedlessMetal.mtlBuf(hb.asType(.float16), device)
        else { return nil }
        centroidsBuf = cbuf
        headWBuf = wbuf
        headSBuf = sbuf
        headBBuf = bbuf
        scoresBuf = device.makeBuffer(
            length: nClusters * MemoryLayout<Float>.size, options: .storageModeShared)!
        indsBuf = device.makeBuffer(
            length: nProbes * MemoryLayout<Int32>.size, options: .storageModeShared)!
        logitsBuf = device.makeBuffer(
            length: nProbes * clusterSize * MemoryLayout<Float>.size, options: .storageModeShared)!
        let nChunks = (nClusters + SeedlessMetal.flashTopKChunk - 1) / SeedlessMetal.flashTopKChunk
        let localTop = min(nProbes, SeedlessMetal.flashTopKChunk)
        let nCand = nChunks * localTop
        candScoresBuf = device.makeBuffer(
            length: nCand * MemoryLayout<Float>.size, options: .storageModeShared)!
        candIndsBuf = device.makeBuffer(
            length: nCand * MemoryLayout<Int32>.size, options: .storageModeShared)!
        let mapBuf = device.makeBuffer(
            length: tmHost.count * MemoryLayout<Int32>.size, options: .storageModeShared)!
        tmHost.withUnsafeBytes { mapBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        tokenMapBuf = mapBuf
        if !forceIds.isEmpty {
            let fib = device.makeBuffer(
                length: forceIds.count * MemoryLayout<Int32>.size, options: .storageModeShared)!
            let fip = fib.contents().bindMemory(to: Int32.self, capacity: forceIds.count)
            for i in 0 ..< forceIds.count { fip[i] = Int32(forceIds[i]) }
            forceIdsBuf = fib
            let frb = device.makeBuffer(
                length: forceFlat.count * MemoryLayout<Float16>.size, options: .storageModeShared)!
            forceFlat.withUnsafeBytes {
                frb.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            forceRowsBuf = frb
        } else {
            forceIdsBuf = nil
            forceRowsBuf = nil
        }
        topScratch = Array(repeating: 0, count: nClusters)
        forceRows = forceFlat
    }

    public func encodeCentroids(
        into enc: MTLComputeCommandEncoder, h: MTLBuffer, hByteOffset: Int = 0
    ) {
        SeedlessMetal.encodeBatchedGemv(
            into: enc, w: centroidsBuf, x: h, y: scoresBuf, E: nClusters, H: H,
            xByteOffset: hByteOffset)
    }

    /// Hierarchical GPU top-k + gather into the current encoder (no extra CB).
    public func encodeFusedAfterCentroids(into enc: MTLComputeCommandEncoder, h: MTLBuffer) {
        encodeFusedAfterCentroids(into: enc, h: h, inds: indsBuf, logits: logitsBuf)
    }

    public func encodeFusedAfterCentroids(
        into enc: MTLComputeCommandEncoder, h: MTLBuffer, inds: MTLBuffer, logits: MTLBuffer,
        hByteOffset: Int = 0
    ) {
        SeedlessMetal.encodeFlashTopK(
            into: enc, scores: scoresBuf, inds: inds,
            candScores: candScoresBuf, candInds: candIndsBuf,
            E: nClusters, K: nProbes)
        SeedlessMetal.encodeQmm4Gather(
            into: enc,
            w: headWBuf, scales: headSBuf, biases: headBBuf, x: h,
            inds: inds, y: logits,
            nProbes: nProbes, N: clusterSize, K: H, gs: headGroupSize,
            xByteOffset: hByteOffset)
    }

    /// After fused layer CB wait: read inds+logits, argmax, force tokens.
    public func greedyAfterFusedGather(hostH: UnsafePointer<Float16>) -> Int {
        greedyAfterFusedGather(hostH: hostH, inds: indsBuf, logits: logitsBuf)
    }

    public func greedyAfterFusedGather(
        hostH: UnsafePointer<Float16>, inds: MTLBuffer, logits: MTLBuffer
    ) -> Int {
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: nProbes)
        var top: [Int] = []
        top.reserveCapacity(nProbes)
        for i in 0 ..< nProbes { top.append(Int(ip[i])) }
        return argmaxWithForce(topClusters: top, hostH: hostH, logits: logits)
    }

    /// After layer CB wait (centroids ready): CPU top-k → Metal gather → argmax.
    /// Set `MOMIJ_FLASH_PROFILE=1` to print per-phase ms (first few calls).
    public func greedyAfterCentroids(hBuf: MTLBuffer, hostH: UnsafePointer<Float16>) -> Int {
        let profile = ProcessInfo.processInfo.environment["MOMIJ_FLASH_PROFILE"] == "1"
        var tTop = 0.0, tArgmax = 0.0, tForce = 0.0
        let t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        let scores = scoresBuf.contents().bindMemory(to: Float.self, capacity: nClusters)
        for i in 0 ..< nClusters { topScratch[i] = i }
        topScratch.select(nProbes, sortedBy: { scores[$0] < scores[$1] })
        let topClusters = Array(topScratch.suffix(nProbes))
        let ip = indsBuf.contents().bindMemory(to: Int32.self, capacity: nProbes)
        for (j, idx) in topClusters.enumerated() { ip[j] = Int32(idx) }
        if profile { tTop = (CFAbsoluteTimeGetCurrent() - t0) * 1000 }

        guard let q = SeedlessMetal.queue else { return 0 }
        let t1 = profile ? CFAbsoluteTimeGetCurrent() : 0
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeQmm4Gather(
            into: enc,
            w: headWBuf, scales: headSBuf, biases: headBBuf, x: hBuf,
            inds: indsBuf, y: logitsBuf,
            nProbes: nProbes, N: clusterSize, K: H, gs: headGroupSize)
        enc.endEncoding()
        var tEnc = 0.0, tCommit = 0.0
        if profile { tEnc = (CFAbsoluteTimeGetCurrent() - t1) * 1000 }
        let tC0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        cb.commit()
        if profile { tCommit = (CFAbsoluteTimeGetCurrent() - tC0) * 1000 }
        // Overlap force-token dots with gather GPU (CPU top-k stays host-side).
        var forceBestId = -1
        var forceBestScore = -Float.infinity
        let tF0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        if !forceIds.isEmpty {
            for fi in 0 ..< forceIds.count {
                var dot: Float = 0
                let base = fi * H
                for k in 0 ..< H {
                    dot += Float(hostH[k]) * Float(forceRows[base + k])
                }
                if dot > forceBestScore {
                    forceBestScore = dot
                    forceBestId = forceIds[fi]
                }
            }
        }
        if profile { tForce = (CFAbsoluteTimeGetCurrent() - tF0) * 1000 }
        let tW0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        cb.waitUntilCompleted()
        var tWait = 0.0, tGatherGpu = 0.0
        if profile {
            tWait = (CFAbsoluteTimeGetCurrent() - tW0) * 1000
            if cb.gpuEndTime > cb.gpuStartTime {
                tGatherGpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            }
        }

        let t2 = profile ? CFAbsoluteTimeGetCurrent() : 0
        let id = argmaxWithForce(
            topClusters: topClusters, hostH: hostH,
            forceBestId: forceBestId, forceBestScore: forceBestScore)
        if profile {
            tArgmax = (CFAbsoluteTimeGetCurrent() - t2) * 1000
            enum FlashProfGate {
                nonisolated(unsafe) static var n = 0
            }
            if FlashProfGate.n < 8 {
                FlashProfGate.n += 1
                print(String(format: "[flash] top=%.3f enc=%.3f commit=%.3f wait=%.3f gpu=%.3f force=%.3f argmax=%.3f",
                             tTop, tEnc, tCommit, tWait, tGatherGpu, tForce, tArgmax))
            }
        }
        return id
    }

    private func argmaxWithForce(
        topClusters: [Int], hostH: UnsafePointer<Float16>,
        forceBestId: Int = -1, forceBestScore: Float = -Float.infinity,
        logits: MTLBuffer? = nil
    ) -> Int {
        let logitsBuf = logits ?? self.logitsBuf
        let nLogits = nProbes * clusterSize
        let lp = logitsBuf.contents().bindMemory(to: Float.self, capacity: nLogits)
        var bestScore = -Float.infinity
        var bestLocal = 0
        for i in 0 ..< nLogits {
            let v = lp[i]
            if v > bestScore { bestScore = v; bestLocal = i }
        }
        let probe = bestLocal / clusterSize
        let row = bestLocal % clusterSize
        var bestId = Int(tokenMapHost[topClusters[probe] * clusterSize + row])

        var fId = forceBestId
        var fScore = forceBestScore
        if fId < 0, !forceIds.isEmpty {
            for fi in 0 ..< forceIds.count {
                var dot: Float = 0
                let base = fi * H
                for k in 0 ..< H {
                    dot += Float(hostH[k]) * Float(forceRows[base + k])
                }
                if dot > fScore {
                    fScore = dot
                    fId = forceIds[fi]
                }
            }
        }
        if fId >= 0, fScore > bestScore {
            bestId = fId
        }
        return bestId
    }
}

private extension Array where Element == Int {
    mutating func select(_ k: Int, sortedBy areInIncreasingOrder: (Int, Int) -> Bool) {
        guard k > 0, k < count else { return }
        var lo = 0, hi = count - 1
        let target = count - k
        while true {
            let pivot = self[hi]
            var i = lo
            for j in lo ..< hi {
                if areInIncreasingOrder(self[j], pivot) {
                    swapAt(i, j)
                    i += 1
                }
            }
            swapAt(i, hi)
            if i == target { return }
            if i < target { lo = i + 1 } else { hi = i - 1 }
        }
    }
}
