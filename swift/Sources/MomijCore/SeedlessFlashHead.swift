import Foundation
import Metal
import MLX

/// FlashHead: Metal centroid gemv fused into the layer CB → CPU top-k → Metal 4-bit
/// gather (separate short CB) → CPU argmax (+ force dots).
public final class SeedlessFlashHead {
    let forceIds: [Int]
    let forceRows: [Float16]
    let tokenMapHost: [Int32]
    let nClusters: Int
    public let nProbes: Int
    let clusterSize: Int
    let headGroupSize: Int
    let H: Int

    let centroidsBuf: MTLBuffer
    let scoresBuf: MTLBuffer
    let headWBuf: MTLBuffer
    let headSBuf: MTLBuffer
    let headBBuf: MTLBuffer
    let indsBuf: MTLBuffer
    let logitsBuf: MTLBuffer
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
        // Default 96: best measured band on M1 Max; set 512 for checkpoint-faithful FlashHead.
        nProbes = min(envProbes ?? 96, nClusters)
        headGroupSize = meta.headGroupSize

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
        topScratch = Array(repeating: 0, count: nClusters)
        forceRows = forceFlat
    }

    public func encodeCentroids(into enc: MTLComputeCommandEncoder, h: MTLBuffer) {
        SeedlessMetal.encodeBatchedGemv(
            into: enc, w: centroidsBuf, x: h, y: scoresBuf, E: nClusters, H: H)
    }

    /// After layer CB wait (centroids ready): CPU top-k → Metal gather → argmax.
    public func greedyAfterCentroids(hBuf: MTLBuffer, hostH: UnsafePointer<Float16>) -> Int {
        let scores = scoresBuf.contents().bindMemory(to: Float.self, capacity: nClusters)
        for i in 0 ..< nClusters { topScratch[i] = i }
        topScratch.select(nProbes, sortedBy: { scores[$0] < scores[$1] })
        let topClusters = Array(topScratch.suffix(nProbes))
        let ip = indsBuf.contents().bindMemory(to: Int32.self, capacity: nProbes)
        for (j, idx) in topClusters.enumerated() { ip[j] = Int32(idx) }

        guard let q = SeedlessMetal.queue else { return 0 }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeQmm4Gather(
            into: enc,
            w: headWBuf, scales: headSBuf, biases: headBBuf, x: hBuf,
            inds: indsBuf, y: logitsBuf,
            nProbes: nProbes, N: clusterSize, K: H, gs: headGroupSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

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

        if !forceIds.isEmpty {
            for fi in 0 ..< forceIds.count {
                var dot: Float = 0
                let base = fi * H
                for k in 0 ..< H {
                    dot += Float(hostH[k]) * Float(forceRows[base + k])
                }
                if dot > bestScore {
                    bestScore = dot
                    bestId = forceIds[fi]
                }
            }
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
