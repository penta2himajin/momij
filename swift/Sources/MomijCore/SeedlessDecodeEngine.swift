import Foundation
import Metal
import MLX
import MLXFast

/// End-to-end Seedless decode: MLX embed + Metal layers (commit→1wait) + MLX final norm/lm_head.
/// KV grows with `offset`; SWA layers ring-write after `slidingWindow`.
public final class SeedlessDecodeEngine: @unchecked Sendable {
    public let store: WeightStore
    public let config: MapleConfig
    public let stack: SeedlessLayerStack
    /// Capacity for full-attention KV (SWA layers use slidingWindow).
    public let fullMaxLen: Int
    /// Layers encoded per Metal command buffer (env `MOMIJ_LAYERS_PER_CB`, default 4).
    public var layersPerCB: Int
    private let embTable: MLXArray
    private let embBuf: MTLBuffer
    private let vocab: Int
    private let normW: MLXArray
    private let normWBuf: MTLBuffer
    private let lmHead: QuantProj
    private let flashHead: SeedlessFlashHead?
    public let useFlashHead: Bool
    /// When true, greedy argmax and SuffixSpec verify use exact `lm_head`
    /// (M-row 1-CB still packs). Default exact; `MOMIJ_EXACT_HEAD=0` opts into
    /// FlashHead probes (approximate, opt-in for speed probes).
    public let greedyExactHead: Bool
    /// FlashHead cluster probes (env `MOMIJ_FLASH_PROBES`, default 64). 0 if FlashHead off.
    public var flashProbes: Int { flashHead?.nProbes ?? 0 }
    /// True when `MOMIJ_FLASH_FUSE=1` (topk+gather in layer CB).
    public var flashFused: Bool { flashHead?.fuseIntoLayerCB ?? false }
    var decodeMaxM: Int { stack.maxM }
    private(set) var lastPrefillChunks: [Int] = []
    /// Raw-Metal exact lm_head argmax (env `MOMIJ_METAL_HEAD=0` disables).
    let useMetalHead: Bool
    private var metalHeadPending = false
    private var metalHeadBannedFor: [Int] = []
    private let finalNormBuf: MTLBuffer
    private var lmHeadW: MTLBuffer?
    private var lmHeadS: MTLBuffer?
    private var lmHeadB: MTLBuffer?
    private var lmHeadTgMax: MTLBuffer?
    private var lmHeadTgId: MTLBuffer?
    private var lmHeadArg: MTLBuffer?
    private var lmHeadBanned: MTLBuffer?
    private var lmHeadNumTG = 0
    private let H: Int
    /// Max chained verify feeds (draftK+1). Env `MOMIJ_SPEC_MAX_M`, default 9.
    private let specMaxM: Int
    /// True M-row chain verify. Default **on**; set `MOMIJ_MROW=0` to disable.
    /// Scratch is sized for `specMaxM` when on; actual packing only runs on hot batch verify.
    private let useMrow: Bool
    /// SuffixDecoding α for `MAX_SPEC = α·matchLen` (`MOMIJ_SPEC_ALPHA`, default 1.0).
    private let specAlpha: Double
    /// Cross-request suffix tree (previous outputs). Shared for the engine lifetime.
    private let globalSuffixIndex = SuffixDraftIndex(maxDepth: 64)
    /// FlashHead-approx Token Recycling adjacency (open-ended chat).
    private let recycleIndex = TokenRecycleIndex(topK: 16)
    private let useRecycle: Bool
    private var specFeedIds: MTLBuffer?
    private var specNormSlots: [MTLBuffer] = []
    private var specIndsSlots: [MTLBuffer] = []
    private var specLogitsSlots: [MTLBuffer] = []
    /// Packed final-norm rows for M-row FlashHead (`[maxM, H]`).
    private var specPackedNorm: MTLBuffer?
    /// Banned ids for the in-flight `generate` (e.g. `<think>` when thinking is off).
    private var decodeBanned: [Int] = []

    public struct PhaseMs: Sendable {
        public var embed = 0.0
        public var layers = 0.0
        public var head = 0.0
        public var sample = 0.0
        public var steps = 0
    }

    public private(set) var lastPhase = PhaseMs()

    public init(
        store: WeightStore, fullMaxLen: Int = 2048,
        enableFlashHead: Bool? = nil, enableExactHead: Bool? = nil
    ) throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        self.store = store
        self.config = store.config
        self.H = store.config.hiddenSize
        self.fullMaxLen = fullMaxLen
        let envM = ProcessInfo.processInfo.environment["MOMIJ_SPEC_MAX_M"].flatMap(Int.init)
        specMaxM = max(2, envM ?? 9)
        // Default on: hot SuffixSpec batch packs True M-row. Cold path never batches
        // (see useBatch below), so opt-out with MOMIJ_MROW=0 if scratch memory matters.
        useMrow = ProcessInfo.processInfo.environment["MOMIJ_MROW"] != "0"
        if let a = ProcessInfo.processInfo.environment["MOMIJ_SPEC_ALPHA"], let v = Double(a), v > 0 {
            specAlpha = v
        } else {
            specAlpha = SuffixSpec.defaultSpecAlpha
        }
        // Default on: TR adjacency helps open chat; set MOMIJ_SPEC_RECYCLE=0 to disable.
        useRecycle = ProcessInfo.processInfo.environment["MOMIJ_SPEC_RECYCLE"] != "0"
        let stackM = useMrow ? specMaxM : 1
        self.stack = try SeedlessLayerStack(
            store: store, device: device, fullMaxLen: fullMaxLen, maxM: stackM)
        if let s = ProcessInfo.processInfo.environment["MOMIJ_LAYERS_PER_CB"], let v = Int(s), v > 0 {
            layersPerCB = v
        } else {
            // Sweep sweet spot on M1 Max: g=4 under cooling.
            layersPerCB = 4
        }

        let embW = store.req("model.word_embeddings.weight")
        let embS = store.req("model.word_embeddings.scales")
        let embB = store.req("model.word_embeddings.biases")
        self.embTable = MLX.dequantized(
            embW, scales: embS, biases: embB,
            groupSize: 64, bits: 4, mode: .affine
        ).asType(.float16)
        self.normW = store.req("model.norm.weight").asType(.float16)
        self.lmHead = .quantized(
            weight: store.req("lm_head.weight"),
            scales: store.req("lm_head.scales"),
            biases: store.req("lm_head.biases"),
            bits: config.headBits,
            groupSize: config.headGroupSize)
        let flashEnv = ProcessInfo.processInfo.environment["MOMIJ_FLASH_HEAD"]
        let wantFlash = enableFlashHead ?? (flashEnv != "0")
        flashHead = wantFlash ? SeedlessFlashHead(store: store, device: device) : nil
        useFlashHead = flashHead != nil
        greedyExactHead = enableExactHead ?? SeedlessServeDefaults.exactHeadFromEnv
        if wantFlash && flashHead == nil {
            fputs("[momij] FlashHead weights missing; using exact lm_head\n", stderr)
        }
        MLX.eval(embTable, normW)
        SeedlessMetal.syncMLXStream()
        vocab = embTable.dim(0)
        guard let ebuf = SeedlessMetal.mtlBuf(embTable, device),
              let nbuf = SeedlessMetal.mtlBuf(normW, device)
        else { throw SeedlessError.notReady }
        embBuf = ebuf
        normWBuf = nbuf
        finalNormBuf = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        // Raw-Metal exact lm_head argmax: alias the quantized head weights and
        // stage small scratch buffers. Enabled only for the 4-bit head layout.
        useMetalHead = config.headBits == 4
            && ProcessInfo.processInfo.environment["MOMIJ_METAL_HEAD"] != "0"
        if useMetalHead {
            guard let wb = SeedlessMetal.mtlBuf(store.req("lm_head.weight"), device),
                  let sb = SeedlessMetal.mtlBuf(store.req("lm_head.scales"), device),
                  let bb = SeedlessMetal.mtlBuf(store.req("lm_head.biases"), device)
            else { throw SeedlessError.notReady }
            lmHeadW = wb
            lmHeadS = sb
            lmHeadB = bb
            let tgCount = (vocab + 255) / 256
            lmHeadNumTG = tgCount
            lmHeadTgMax = device.makeBuffer(length: specMaxM * tgCount * 4, options: .storageModeShared)!
            lmHeadTgId = device.makeBuffer(length: specMaxM * tgCount * 4, options: .storageModeShared)!
            lmHeadArg = device.makeBuffer(length: specMaxM * 4, options: .storageModeShared)!
            lmHeadBanned = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        } else {
            lmHeadW = nil
            lmHeadS = nil
            lmHeadB = nil
            lmHeadNumTG = 0
            lmHeadTgMax = nil
            lmHeadTgId = nil
            lmHeadArg = nil
            lmHeadBanned = nil
        }
    }

    /// True when this call should skip FlashHead argmax (exact `lm_head`).
    private var greedySkipFlash: Bool { greedyExactHead || !useFlashHead }

    private func ensureSpecSlots() {
        guard specFeedIds == nil, let device = SeedlessMetal.device else { return }
        specFeedIds = device.makeBuffer(length: (specMaxM + 1) * 4, options: .storageModeShared)
        for _ in 0 ..< specMaxM {
            specNormSlots.append(device.makeBuffer(length: H * 2, options: .storageModeShared)!)
            if let fh = flashHead {
                let logitBytes = fh.nProbes * fh.clusterSize * MemoryLayout<Float>.size
                specIndsSlots.append(device.makeBuffer(length: fh.nProbes * 4, options: .storageModeShared)!)
                specLogitsSlots.append(device.makeBuffer(length: logitBytes, options: .storageModeShared)!)
            }
        }
        if useMrow {
            specPackedNorm = device.makeBuffer(length: specMaxM * H * 2, options: .storageModeShared)
        }
    }

    /// C2: draft-driven chain verify in **one CB / one wait** (GPU embed; Qwisp chained style).
    /// `feeds` = [y] + draft (length D+1). Returns greedy evals[0..<feeds.count].
    /// FlashHead gather is the default head; `greedyExactHead` uses packed exact `lm_head`.
    public func stepChainFeeds(_ feeds: [Int]) throws -> [Int] {
        precondition(!feeds.isEmpty && feeds.count <= specMaxM)
        let wantFlash = !greedySkipFlash && flashFused && flashHead != nil
        let metalRows = (useMetalHead && !wantFlash) ? feeds.count : 0
        let packMrow = try encodeChainFeeds(feeds, withFlash: wantFlash, metalHeadRows: metalRows)
        let M = feeds.count
        if wantFlash, let fh = flashHead {
            var evals: [Int] = []
            evals.reserveCapacity(M)
            if packMrow, let packed = specPackedNorm {
                let base = packed.contents().bindMemory(to: Float16.self, capacity: M * H)
                for i in 0 ..< M {
                    evals.append(fh.greedyAfterFusedGather(
                        hostH: base.advanced(by: i * H),
                        inds: specIndsSlots[i], logits: specLogitsSlots[i]))
                }
            } else {
                for i in 0 ..< M {
                    let ptr = specNormSlots[i].contents().bindMemory(to: Float16.self, capacity: H)
                    evals.append(fh.greedyAfterFusedGather(
                        hostH: ptr, inds: specIndsSlots[i], logits: specLogitsSlots[i]))
                }
            }
            return evals
        }
        if useMetalHead, packMrow, metalHeadPending {
            var evals: [Int] = []
            evals.reserveCapacity(M)
            let ptr = lmHeadArg!.contents().bindMemory(to: Int32.self, capacity: M)
            for i in 0 ..< M { evals.append(Int(ptr[i])) }
            metalHeadPending = false
            return evals
        }
        return argmaxExactFromChain(M: M, packed: packMrow)
    }

    /// Same GPU chain as `stepChainFeeds`, but returns FlashHead candidate logits per row
    /// for rejection sampling (not argmax). Sequential fallback uses `stepCandidates`.
    public func stepChainCandidateRows(_ feeds: [Int]) throws -> [(ids: [Int], logits: [Float])] {
        precondition(!feeds.isEmpty && feeds.count <= specMaxM)
        guard useFlashHead, let fh = flashHead, fh.fuseIntoLayerCB else {
            var rows: [(ids: [Int], logits: [Float])] = []
            rows.reserveCapacity(feeds.count)
            for t in feeds { rows.append(try stepCandidates(t)) }
            return rows
        }
        let packMrow = try encodeChainFeeds(feeds, withFlash: true)
        let M = feeds.count
        var rows: [(ids: [Int], logits: [Float])] = []
        rows.reserveCapacity(M)
        if packMrow, let packed = specPackedNorm {
            let base = packed.contents().bindMemory(to: Float16.self, capacity: M * H)
            for i in 0 ..< M {
                rows.append(fh.candidateLogits(
                    inds: specIndsSlots[i], logits: specLogitsSlots[i],
                    hostH: base.advanced(by: i * H)))
            }
        } else {
            for i in 0 ..< M {
                let ptr = specNormSlots[i].contents().bindMemory(to: Float16.self, capacity: H)
                rows.append(fh.candidateLogits(
                    inds: specIndsSlots[i], logits: specLogitsSlots[i], hostH: ptr))
            }
        }
        return rows
    }

    /// One CB: embed `feeds` × layers ± packed M-row. FlashHead gather is optional.
    /// Returns whether True M-row packing ran.
    @discardableResult
    private func encodeChainFeeds(_ feeds: [Int], withFlash: Bool, metalHeadRows: Int = 0) throws -> Bool {
        ensureSpecSlots()
        guard let q = SeedlessMetal.queue, let feedBuf = specFeedIds else {
            throw SeedlessError.notReady
        }
        let fh = withFlash ? flashHead : nil
        let M = feeds.count
        let ip = feedBuf.contents().bindMemory(to: Int32.self, capacity: M)
        for (i, t) in feeds.enumerated() { ip[i] = Int32(t) }

        let packMrow = useMrow && M > 1 && stack.canEncodeMrow(M: M)
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        do {
            if packMrow, let packed = specPackedNorm {
                for i in 0 ..< M {
                    SeedlessMetal.encodeEmbedToken(
                        into: enc, table: embBuf, ids: feedBuf, out: stack.hBuf,
                        H: H, row: i, outRow: i)
                }
                for layer in stack.layers {
                    try layer.encodeStep(into: enc, M: M)
                }
                SeedlessMetal.encodeRms(
                    into: enc, h: stack.hBuf, w: normWBuf, out: packed,
                    H: H, eps: config.rmsNormEps, M: M)
                if metalHeadRows > 0 {
                    encodeMetalHead(enc, x: packed, rows: metalHeadRows)
                }
                if let fh {
                    for i in 0 ..< M {
                        let off = i * H * 2
                        fh.encodeCentroids(into: enc, h: packed, hByteOffset: off)
                        fh.encodeFusedAfterCentroids(
                            into: enc, h: packed, inds: specIndsSlots[i],
                            logits: specLogitsSlots[i], hByteOffset: off)
                    }
                }
            } else {
                for i in 0 ..< M {
                    SeedlessMetal.encodeEmbedToken(
                        into: enc, table: embBuf, ids: feedBuf, out: stack.hBuf, H: H, row: i)
                    for layer in stack.layers {
                        try layer.encodeStep(into: enc)
                    }
                    let norm = specNormSlots[i]
                    SeedlessMetal.encodeRms(
                        into: enc, h: stack.hBuf, w: normWBuf, out: norm, H: H, eps: config.rmsNormEps)
                    if let fh {
                        fh.encodeCentroids(into: enc, h: norm)
                        fh.encodeFusedAfterCentroids(
                            into: enc, h: norm, inds: specIndsSlots[i], logits: specLogitsSlots[i])
                    }
                }
            }
            enc.endEncoding()
        } catch {
            enc.endEncoding()
            throw error
        }
        cb.commit()
        cb.waitUntilCompleted()
        return packMrow
    }

    private func argmaxExactFromChain(M: Int, packed: Bool) -> [Int] {
        if packed, let packedBuf = specPackedNorm {
            let base = packedBuf.contents().bindMemory(to: Float16.self, capacity: M * H)
            return argmaxExactRows(base, M: M)
        }
        var out: [Int] = []
        out.reserveCapacity(M)
        for i in 0 ..< M {
            let ptr = specNormSlots[i].contents().bindMemory(to: Float16.self, capacity: H)
            out.append(contentsOf: argmaxExactRows(ptr, M: 1))
        }
        return out
    }

    private func argmaxExactRows(_ ptr: UnsafePointer<Float16>, M: Int) -> [Int] {
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: M * H)).reshaped([M, 1, H])
        let logits = lmHead.apply(h)
        var ids: [Int] = []
        ids.reserveCapacity(M)
        for i in 0 ..< M {
            var scores = ConstrainedPick.hostScores(logits[i])
            ConstrainedPick.applyBanned(&scores, banned: decodeBanned)
            ids.append(ConstrainedPick.argmaxAll(scores))
        }
        return ids
    }

    private func pickGreedyFromFinalNorm(skipFlash: Bool) -> Int {
        if let staged = takeMetalHeadArgmax() {
            return staged
        }
        if skipFlash || greedySkipFlash {
            return nextToken()
        }
        guard let fh = flashHead else { return nextToken() }
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        let id = fh.fuseIntoLayerCB
            ? fh.greedyAfterFusedGather(hostH: ptr)
            : fh.greedyAfterCentroids(hBuf: finalNormBuf, hostH: ptr)
        if !decodeBanned.isEmpty && decodeBanned.contains(id) {
            if let alt = fh.topTokenCandidates(k: 512).first(where: { !decodeBanned.contains($0) }) {
                return alt
            }
            return nextTokenBannedFallback()
        }
        return id
    }

    /// Greedy GPU token-feedback chain with **layersPerCB commits** + MTLSharedEvent
    /// between tokens (CPU encode overlaps GPU; no mega-CB stall). Force-tokens on GPU.
    public func stepGreedyChain(from start: Int, count K: Int) throws -> [Int] {
        precondition(K > 0 && K <= specMaxM)
        guard useFlashHead, let fh = flashHead, fh.fuseIntoLayerCB else {
            var out: [Int] = []
            var cur = start
            for _ in 0 ..< K {
                cur = try step(cur)
                out.append(cur)
            }
            return out
        }
        ensureSpecSlots()
        guard let q = SeedlessMetal.queue,
              let device = SeedlessMetal.device,
              let feedBuf = specFeedIds,
              let event = device.makeSharedEvent()
        else { throw SeedlessError.notReady }

        let ip = feedBuf.contents().bindMemory(to: Int32.self, capacity: K + 1)
        ip[0] = Int32(start)
        let g = max(1, layersPerCB)
        var lastCB: MTLCommandBuffer?

        for ti in 0 ..< K {
            var layerIdx = 0
            var isFirstCB = true
            while layerIdx < stack.layers.count {
                let cb = q.makeCommandBuffer()!
                if isFirstCB, ti > 0 {
                    // GPU-side wait: previous token's argmax must complete before embed.
                    cb.encodeWaitForEvent(event, value: UInt64(ti))
                }
                let enc = cb.makeComputeCommandEncoder()!
                do {
                    if isFirstCB {
                        SeedlessMetal.encodeEmbedToken(
                            into: enc, table: embBuf, ids: feedBuf, out: stack.hBuf, H: H, row: ti)
                    }
                    let end = min(layerIdx + g, stack.layers.count)
                    for j in layerIdx ..< end {
                        try stack.layers[j].encodeStep(into: enc)
                    }
                    let isLast = end == stack.layers.count
                    if isLast {
                        let norm = specNormSlots[ti]
                        let inds = specIndsSlots[ti]
                        let logits = specLogitsSlots[ti]
                        SeedlessMetal.encodeRms(
                            into: enc, h: stack.hBuf, w: normWBuf, out: norm,
                            H: H, eps: config.rmsNormEps)
                        fh.encodeCentroids(into: enc, h: norm)
                        fh.encodeFusedAfterCentroids(into: enc, h: norm, inds: inds, logits: logits)
                        SeedlessMetal.encodeFlashArgmaxToken(
                            into: enc, logits: logits, inds: inds, tokenMap: fh.tokenMapBuf,
                            ids: feedBuf, nProbes: fh.nProbes, clusterSize: fh.clusterSize,
                            outRow: ti + 1, h: norm, forceRows: fh.forceRowsBuf,
                            forceIds: fh.forceIdsBuf, nForce: fh.forceCount, H: H)
                    }
                    enc.endEncoding()
                    if isLast {
                        cb.encodeSignalEvent(event, value: UInt64(ti + 1))
                    }
                    cb.commit()
                    lastCB = cb
                    layerIdx = end
                    isFirstCB = false
                } catch {
                    enc.endEncoding()
                    throw error
                }
            }
        }
        lastCB?.waitUntilCompleted()

        var out: [Int] = []
        out.reserveCapacity(K)
        for i in 1 ... K { out.append(Int(ip[i])) }
        return out
    }

    public func reset() {
        stack.resetCaches()
        // Keep global suffix + recycle across resets (warm multi-turn / trials).
        metalHeadPending = false
    }

    /// Stage the banned-id list for the raw-Metal lm_head argmax kernel.
    private func syncMetalHeadBanned() {
        guard let buf = lmHeadBanned else { return }
        let p = buf.contents().bindMemory(to: Int32.self, capacity: 16)
        for i in 0 ..< 16 { p[i] = 0 }
        for (i, id) in decodeBanned.prefix(16).enumerated() { p[i] = Int32(id) }
    }

    /// Read the raw-Metal argmax result staged by the last encoded head kernel.
    private func takeMetalHeadArgmax(_ row: Int = 0) -> Int? {
        guard metalHeadPending, let arg = lmHeadArg else { return nil }
        metalHeadPending = false
        return Int(arg.contents().bindMemory(to: Int32.self, capacity: row + 1)[row])
    }

    /// Encode the exact lm_head argmax kernel for `rows` rows of `xBuf`.
    private func encodeMetalHead(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, rows: Int) {
        guard useMetalHead, let wB = lmHeadW, let sB = lmHeadS, let bB = lmHeadB,
              let tm = lmHeadTgMax, let ti = lmHeadTgId, let arg = lmHeadArg,
              let banned = lmHeadBanned else {
            metalHeadPending = false
            return
        }
        // Rows beyond the vocab (last partial TG) never write their partial;
        // seed the TG maxima to -inf so the fold ignores them.
        let tmPtr = tm.contents().bindMemory(to: Float.self, capacity: rows * lmHeadNumTG)
        for i in 0 ..< (rows * lmHeadNumTG) { tmPtr[i] = -Float.infinity }
        SeedlessMetal.encodeLMHeadArgmax(
            into: enc, x: x, w: wB, scales: sB, biases: bB,
            banned: banned, tgMax: tm, tgId: ti, out: arg,
            K: H, gs: config.headGroupSize, V: vocab, M: rows,
            numTG: lmHeadNumTG, nBanned: min(decodeBanned.count, 16))
        metalHeadPending = rows > 0
    }

    private func embedToken(_ id: Int) {
        precondition(id >= 0 && id < vocab)
        let src = embBuf.contents().advanced(by: id * H * 2)
        stack.hBuf.contents().copyMemory(from: src, byteCount: H * 2)
    }

    private func embedTokens(_ ids: ArraySlice<Int>) {
        let dst = stack.hBuf.contents()
        for (m, id) in ids.enumerated() {
            precondition(id >= 0 && id < vocab)
            let src = embBuf.contents().advanced(by: id * H * 2)
            dst.advanced(by: m * H * 2).copyMemory(from: src, byteCount: H * 2)
        }
    }

    /// Causal M-row prefill when capacity allows; last row → `finalNormBuf` via exact RMS.
    private func prefillPrompt(_ prompt: [Int], skipFlash: Bool) throws {
        lastPrefillChunks = []
        var i = 0
        while i < prompt.count {
            let remain = prompt.count - i
            var M = 1
            if remain > 1 {
                var cand = min(remain, stack.maxM)
                while cand > 1, !stack.canEncodeMrow(M: cand) { cand -= 1 }
                M = cand
            }
            lastPrefillChunks.append(M)
            let lastChunk = i + M == prompt.count
            let tEmbed = CFAbsoluteTimeGetCurrent()
            embedTokens(prompt[i ..< (i + M)])
            let tEmbedDone = CFAbsoluteTimeGetCurrent()
            try stack.stepCommitWait(layersPerCB: layersPerCB, M: M)
            let tCommit = CFAbsoluteTimeGetCurrent()
            if ProcessInfo.processInfo.environment["MOMIJ_PROFILE_PREFILL"] == "1" {
                fputs(String(format: "[prefill] chunk M=%d pos=%d wall=%.2fms embed=%.2fms commitWait=%.2fms\n",
                             M, i, (CFAbsoluteTimeGetCurrent() - tEmbed) * 1000,
                             (tEmbedDone - tEmbed) * 1000,
                             (CFAbsoluteTimeGetCurrent() - tEmbedDone) * 1000), stderr)
            }
            if lastChunk {
                try encodeFinalNorm(fromLastRow: M - 1, skipFlash: skipFlash)
            }
            i += M
        }
    }

    private func encodeFinalNorm(fromLastRow row: Int, skipFlash: Bool) throws {
        if row > 0 {
            let src = stack.hBuf.contents().advanced(by: row * H * 2)
            stack.hBuf.contents().copyMemory(from: src, byteCount: H * 2)
        }
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        SeedlessMetal.encodeRms(
            into: enc, h: stack.hBuf, w: normWBuf, out: finalNormBuf,
            H: H, eps: config.rmsNormEps)
        if !skipFlash, let fh = flashHead {
            fh.encodeCentroids(into: enc, h: finalNormBuf)
            if fh.fuseIntoLayerCB {
                fh.encodeFusedAfterCentroids(into: enc, h: finalNormBuf)
            }
        }
        if useMetalHead {
            encodeMetalHead(enc, x: finalNormBuf, rows: 1)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Observe FlashHead top candidates into the recycle adjacency for `fromToken`.
    private func observeRecycle(fromToken: Int, inds: MTLBuffer? = nil, logits: MTLBuffer? = nil) {
        guard useRecycle, let fh = flashHead else { return }
        let cands = fh.topTokenCandidates(k: recycleIndex.topK, inds: inds, logits: logits)
        recycleIndex.observe(fromToken: fromToken, candidates: cands)
    }

    private func nextToken() -> Int {
        // Greedy serve/generate: exact lm_head. FlashHead cluster probes are an
        // approximation and pick the wrong id even on short prompts (measured:
        // exact=100 vs flash=72887 on a length-8 dummy). Spec/recycle still use
        // FlashHead on their own paths.
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let prof = ProcessInfo.processInfo.environment["MOMIJ_PROFILE_HEAD"] == "1"
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = lmHead.apply(h.reshaped([1, 1, H]))
        let y = MLX.argMax(logits.reshaped([-1]), axis: -1)
        MLX.eval(y)
        let t1 = CFAbsoluteTimeGetCurrent()
        let id = y.item(Int.self)
        if prof {
            fputs(String(format: "[head] eval=%.3fms item=%.3fms\n",
                         (t1 - t0) * 1000, (CFAbsoluteTimeGetCurrent() - t1) * 1000), stderr)
        }
        if !decodeBanned.isEmpty && decodeBanned.contains(id) {
            return nextTokenBannedFallback()
        }
        return id
    }

    /// Same prefill hidden: exact `lm_head` argmax vs FlashHead probe argmax.
    func firstTokenExactVsFlash(_ prompt: [Int]) throws -> (
        exact: Int, flash: Int, exactInFlashTop: Bool, flashTop: [Int]
    ) {
        guard let fh = flashHead else { throw SeedlessError.notReady }
        reset()
        try prefillPrompt(prompt, skipFlash: false)
        let exact = nextToken()
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        let flash = fh.greedyAfterFusedGather(hostH: ptr)
        let top = fh.topTokenCandidates(k: 512)
        return (exact, flash, top.contains(exact), top)
    }

    /// FlashHead-greedy continuation (not the serve path). For quality probes.
    func generateFlashHeadGreedy(prompt: [Int], maxTokens: Int) throws -> [Int] {
        guard let fh = flashHead, !prompt.isEmpty, maxTokens > 0 else { return [] }
        reset()
        try prefillPrompt(prompt, skipFlash: false)
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        var y = fh.greedyAfterFusedGather(hostH: ptr)
        var out: [Int] = []
        while out.count < maxTokens {
            out.append(y)
            embedToken(y)
            try stack.stepCommitWait(layersPerCB: layersPerCB)
            try encodeFinalNorm(fromLastRow: 0, skipFlash: false)
            y = fh.greedyAfterFusedGather(
                hostH: finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H))
        }
        return out
    }

    /// Prefill `prompt` then return exact `lm_head` scores (for tests / banned-argmax).
    func greedyScoresAfterPrefill(prompt: [Int], skipFlash: Bool = true) throws -> [Float] {
        reset()
        guard !prompt.isEmpty else { return [] }
        try prefillPrompt(prompt, skipFlash: skipFlash)
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let logits = lmHead.apply(h.reshaped([1, 1, H]))
        return ConstrainedPick.hostScores(logits)
    }

    /// Sequential prefix, then last token layer-by-layer. Compare to batched mlx residuals.
    func lastTokenHiddenAfterEachLayer(prompt: [Int]) throws -> (embed: [Float], layers: [[Float]]) {
        reset()
        guard let last = prompt.last, !prompt.isEmpty else { return ([], []) }
        for id in prompt.dropLast() {
            _ = try step(id, skipFlash: true)
        }
        embedToken(last)
        let embedRow = copyLastH()
        var layersOut: [[Float]] = []
        layersOut.reserveCapacity(stack.layers.count)
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        for layer in stack.layers {
            let cb = q.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            do {
                try layer.encodeStep(into: enc)
                enc.endEncoding()
            } catch {
                enc.endEncoding()
                throw error
            }
            cb.commit()
            cb.waitUntilCompleted()
            layersOut.append(copyLastH())
        }
        return (embedRow, layersOut)
    }

    func lastTokenAfterFirstAttn(prompt: [Int]) throws -> [Float] {
        try lastTokenAfterFirstAttn(prompt: prompt, prefixFullStack: true)
    }

    /// If `prefixFullStack` is false, only layer 0 runs on prefix tokens (isolates KV clobber).
    func lastTokenAfterFirstAttn(prompt: [Int], prefixFullStack: Bool) throws -> [Float] {
        reset()
        guard let last = prompt.last, !prompt.isEmpty else { return [] }
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        let l0 = stack.layers[0]
        for id in prompt.dropLast() {
            if prefixFullStack {
                _ = try step(id, skipFlash: true)
            } else {
                embedToken(id)
                let cb = q.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                do {
                    try l0.encodeStep(into: enc)
                    enc.endEncoding()
                } catch {
                    enc.endEncoding()
                    throw error
                }
                cb.commit()
                cb.waitUntilCompleted()
            }
        }
        embedToken(last)
        let pos = l0.offset
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        do {
            try l0.encodeAttnOnly(into: enc, at: pos)
            enc.endEncoding()
        } catch {
            enc.endEncoding()
            throw error
        }
        cb.commit()
        cb.waitUntilCompleted()
        return copyLastH()
    }

    /// After L0-only prefix + last `encodeAttnOnly`: pre-RoPE QKV and post-norm/RoPE Q.
    func lastTokenLayer0QK(prompt: [Int]) throws -> (qkv: [Float], q: [Float]) {
        _ = try lastTokenAfterFirstAttn(prompt: prompt, prefixFullStack: false)
        let l0 = stack.layers[0]
        let qDim = l0.numHeads * l0.headDim
        let kvDim = l0.numKV * l0.headDim
        let qkvN = qDim + 2 * kvDim
        let qp = l0.qkvOut.contents().bindMemory(to: Float16.self, capacity: qkvN)
        let qk = l0.qkOut.contents().bindMemory(to: Float16.self, capacity: qDim + kvDim)
        return (
            (0 ..< qkvN).map { Float(qp[$0]) },
            (0 ..< qDim).map { Float(qk[$0]) }
        )
    }

    private func copyLastH() -> [Float] {
        let p = stack.hBuf.contents().bindMemory(to: Float16.self, capacity: H)
        return (0 ..< H).map { Float(p[$0]) }
    }

    private func nextTokenBannedFallback() -> Int {
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let logits = lmHead.apply(h.reshaped([1, 1, H]))
        var scores = ConstrainedPick.hostScores(logits)
        ConstrainedPick.applyBanned(&scores, banned: decodeBanned)
        return ConstrainedPick.argmaxAll(scores)
    }

    private func nextTokenSampled(
        processor: LogitsProcessor,
        seen: Set<Int>,
        counts: [Int: Int],
        rng: inout SplitMix64
    ) -> Int {
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        if useFlashHead, let fh = flashHead {
            if fh.fuseIntoLayerCB {
                return fh.sampleAfterFusedGather(
                    hostH: ptr, processor: processor,
                    seen: seen, counts: counts, rng: &rng)
            }
            return fh.sampleAfterCentroids(
                hBuf: finalNormBuf, hostH: ptr, processor: processor,
                seen: seen, counts: counts, rng: &rng)
        }
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let logits = lmHead.apply(h.reshaped([1, 1, H])).reshaped([-1]).asType(.float32)
        MLX.eval(logits)
        let n = logits.shape[0]
        var scores = [Float](repeating: 0, count: n)
        logits.asData(access: .copy).data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float.self)
            let m = min(n, raw.count / MemoryLayout<Float>.size)
            for i in 0 ..< m { scores[i] = src[i] }
        }
        let ids = Array(0 ..< n)
        return processor.sample(
            tokenIds: ids, logits: scores, seen: seen, counts: counts, rng: &rng)
    }

    /// Embed `token`, run layers, return FlashHead (or full) candidate logits for the next id.
    public func stepCandidates(_ token: Int) throws -> (ids: [Int], logits: [Float]) {
        embedToken(token)
        try stack.stepCommitWait(layersPerCB: layersPerCB) { [self] enc in
            SeedlessMetal.encodeRms(
                into: enc, h: stack.hBuf, w: normWBuf,
                out: finalNormBuf, H: H, eps: config.rmsNormEps)
            if let fh = flashHead {
                fh.encodeCentroids(into: enc, h: finalNormBuf)
                if fh.fuseIntoLayerCB {
                    fh.encodeFusedAfterCentroids(into: enc, h: finalNormBuf)
                }
            }
        }
        observeRecycle(fromToken: token)
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        if useFlashHead, let fh = flashHead {
            if !fh.fuseIntoLayerCB {
                // Fill inds/logits via gather CB (discard greedy pick).
                _ = fh.greedyAfterCentroids(hBuf: finalNormBuf, hostH: ptr)
            }
            return fh.candidateLogits(hostH: ptr)
        }
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let logits = lmHead.apply(h.reshaped([1, 1, H])).reshaped([-1]).asType(.float32)
        MLX.eval(logits)
        let n = logits.shape[0]
        var scores = [Float](repeating: 0, count: n)
        logits.asData(access: .copy).data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float.self)
            let m = min(n, raw.count / MemoryLayout<Float>.size)
            for i in 0 ..< m { scores[i] = src[i] }
        }
        return (Array(0 ..< n), scores)
    }

    /// One sampled token (FlashHead candidates + LogitsProcessor). Updates recycle.
    public func stepSampled(
        _ token: Int,
        processor: LogitsProcessor,
        seen: Set<Int>,
        counts: [Int: Int],
        rng: inout SplitMix64
    ) throws -> Int {
        embedToken(token)
        try stack.stepCommitWait(layersPerCB: layersPerCB) { [self] enc in
            SeedlessMetal.encodeRms(
                into: enc, h: stack.hBuf, w: normWBuf,
                out: finalNormBuf, H: H, eps: config.rmsNormEps)
            if let fh = flashHead {
                fh.encodeCentroids(into: enc, h: finalNormBuf)
                if fh.fuseIntoLayerCB {
                    fh.encodeFusedAfterCentroids(into: enc, h: finalNormBuf)
                }
            }
        }
        let next = nextTokenSampled(
            processor: processor, seen: seen, counts: counts, rng: &rng)
        observeRecycle(fromToken: token)
        return next
    }

    /// Non-speculative sampled decode. Penalties use prompt + generated history.
    /// Full-attn layers need prompt+generation ≤ fullMaxLen. Every public
    /// generate entry point checks BEFORE any Metal encode — an unchecked
    /// prefill would write KV past the full-layer cache max (OOM-buffer
    /// overrun) and, on the serve path, hang the request for minutes.
    private func checkPromptCapacity(_ prompt: [Int]) throws {
        if prompt.count >= fullMaxLen {
            throw SeedlessError.unsupportedShape(N: prompt.count, K: fullMaxLen, gs: 0)
        }
    }

    public func generateSampled(
        prompt: [Int], maxTokens: Int,
        processor: LogitsProcessor,
        eos: Int? = 151_645,
        seed: UInt64? = nil
    ) throws -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        try checkPromptCapacity(prompt)
        reset()
        var rng = SplitMix64(seed: seed ?? UInt64.random(in: 1 ... .max))
        var seen = Set(prompt)
        var counts: [Int: Int] = [:]
        for t in prompt { counts[t, default: 0] += 1 }

        var last = prompt[0]
        for i in 0 ..< prompt.count {
            let id = prompt[i]
            if i + 1 < prompt.count {
                // Prefill stays greedy for KV; sampling starts at first gen token.
                _ = try step(id)
            } else {
                last = try stepSampled(
                    id, processor: processor, seen: seen, counts: counts, rng: &rng)
            }
        }
        var out: [Int] = []
        var y = last
        while out.count < maxTokens {
            out.append(y)
            seen.insert(y)
            counts[y, default: 0] += 1
            if let eos, y == eos { break }
            if out.count >= maxTokens { break }
            y = try stepSampled(
                y, processor: processor, seen: seen, counts: counts, rng: &rng)
        }
        return out
    }

    /// Sampled decode with model-free draft + rejection sampling.
    /// Cold 1-step is the 150 tok/s floor. Periodic probes (and a warm accept
    /// window) enable sequential Leviathan; M-row batch only when drafts land.
    public func generateSampledSpeculative(
        prompt: [Int], maxTokens: Int,
        processor: LogitsProcessor,
        draftK: Int = 8,
        eos: Int? = 151_645,
        seed: UInt64? = nil
    ) throws -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        try checkPromptCapacity(prompt)
        reset()
        var rng = SplitMix64(seed: seed ?? UInt64.random(in: 1 ... .max))
        var seen = Set(prompt)
        var counts: [Int: Int] = [:]
        for t in prompt { counts[t, default: 0] += 1 }

        var last = prompt[0]
        for i in 0 ..< prompt.count {
            let id = prompt[i]
            if i + 1 < prompt.count {
                _ = try step(id)
            } else {
                last = try stepSampled(
                    id, processor: processor, seen: seen, counts: counts, rng: &rng)
            }
        }
        let r = try generateSampledSpeculativeFromPrefill(
            first: last, promptIds: prompt, maxTokens: maxTokens,
            processor: processor, draftK: draftK, eos: eos,
            rng: &rng, seen: &seen, counts: &counts)
        return r.tokens
    }

    /// Prefill already done; `first` is the first generated token.
    func generateSampledSpeculativeFromPrefill(
        first: Int, promptIds: [Int], maxTokens: Int,
        processor: LogitsProcessor, draftK: Int, eos: Int?,
        rng: inout SplitMix64, seen: inout Set<Int>, counts: inout [Int: Int]
    ) throws -> (tokens: [Int], accepted: Int, attempts: Int, batched: Int) {
        let localIndex = SuffixDraftIndex(maxDepth: 64)
        let useTree = ProcessInfo.processInfo.environment["MOMIJ_SPEC_TREE"] != "0"
        let batchEnv = ProcessInfo.processInfo.environment["MOMIJ_SPEC_BATCH"]
        let batchForce = batchEnv == "1"
        let batchOff = batchEnv == "0"
        let gateWindow = 8
        var acceptWindow: [Int] = []
        var idsHist = promptIds
        idsHist.append(first)
        var acceptedTotal = 0
        var attempts = 0
        var batched = 0
        var batching = false

        var out: [Int] = []
        var y = first
        while out.count < maxTokens {
            out.append(y)
            seen.insert(y)
            counts[y, default: 0] += 1
            if let eos, y == eos { break }
            let remain = maxTokens - out.count
            if remain == 0 { break }

            let meanAccept = acceptWindow.isEmpty ? 0.0
                : Double(acceptWindow.reduce(0, +)) / Double(acceptWindow.count)
            let wantDraft = batchForce || SpeculativeSampling.SampledSpecPolicy.useDraft(
                generated: out.count, meanAccept: meanAccept,
                windowCount: acceptWindow.count, currentlyBatching: batching)
            if !wantDraft {
                batching = false
                y = try stepSampled(
                    y, processor: processor, seen: seen, counts: counts, rng: &rng)
                idsHist.append(y)
                continue
            }

            let effK = SpeculativeSampling.SampledSpecPolicy.draftK(
                meanAccept: meanAccept, draftK: min(draftK, remain, specMaxM - 1))
            localIndex.clear()
            let treeSlice = idsHist.count > localIndex.maxDepth * 2
                ? Array(idsHist.suffix(localIndex.maxDepth * 2))
                : idsHist
            localIndex.insert(treeSlice)
            let treeHit = useTree
                ? SuffixDraftIndex.bestDraft(
                    local: localIndex, global: globalSuffixIndex,
                    history: idsHist, maxK: effK, alpha: specAlpha)
                : (matchLen: 0, tokens: [Int]())
            let pld = SuffixSpec.suffixDraft(
                history: idsHist, k: effK, promptLen: promptIds.count)
            let recycled = useRecycle ? recycleIndex.draft(from: y, maxK: min(effK, 4)) : []
            let fromTree: Bool
            let draft: [Int]
            if treeHit.matchLen >= 1, !treeHit.tokens.isEmpty {
                draft = treeHit.tokens
                fromTree = true
            } else if !recycled.isEmpty {
                draft = recycled
                fromTree = false
            } else {
                draft = pld
                fromTree = false
            }
            let qDraft: [Float] = fromTree
                ? Self.mergedDraftQ(
                    local: localIndex, global: globalSuffixIndex,
                    history: idsHist, draft: draft)
                : [Float](repeating: 1, count: draft.count)

            if draft.isEmpty {
                y = try stepSampled(
                    y, processor: processor, seen: seen, counts: counts, rng: &rng)
                idsHist.append(y)
                continue
            }

            attempts += 1
            let useBatch = !batchOff && useFlashHead && flashFused
                && (batchForce || SpeculativeSampling.SampledSpecPolicy.useBatch(
                    meanAccept: meanAccept, windowCount: acceptWindow.count,
                    currentlyBatching: batching))
            batching = useBatch

            let accepted: Int
            if useBatch {
                batched += 1
                let r = try verifyDraftSampled(
                    y: y, draft: draft, processor: processor,
                    seen: seen, counts: counts, rng: &rng, draftQ: qDraft)
                accepted = r.accepted
                if accepted > 0 {
                    let chunk = Array(draft.prefix(accepted))
                    out.append(contentsOf: chunk)
                    idsHist.append(contentsOf: chunk)
                    for t in chunk {
                        seen.insert(t)
                        counts[t, default: 0] += 1
                    }
                }
                y = r.next
                idsHist.append(y)
            } else {
                var acc = 0
                var cur = y
                var stopped = false
                for (i, d) in draft.enumerated() {
                    let cands = try stepCandidates(cur)
                    let prepared = processor.prepare(
                        tokenIds: cands.ids, logits: cands.logits, seen: seen, counts: counts)
                    let decision = SpeculativeSampling.tryAccept(
                        draftToken: d,
                        ids: prepared.ids, logits: prepared.logits,
                        processor: processor, rng: &rng,
                        draftQ: i < qDraft.count ? qDraft[i] : 1)
                    if decision.accepted {
                        acc += 1
                        out.append(d)
                        seen.insert(d)
                        counts[d, default: 0] += 1
                        idsHist.append(d)
                        cur = d
                        if let eos, d == eos {
                            y = d
                            stopped = true
                            break
                        }
                        if out.count >= maxTokens {
                            y = d
                            stopped = true
                            break
                        }
                        if i + 1 == draft.count {
                            y = try stepSampled(
                                d, processor: processor, seen: seen, counts: counts, rng: &rng)
                            idsHist.append(y)
                            stopped = true
                        }
                    } else {
                        y = decision.token
                        idsHist.append(y)
                        stopped = true
                        break
                    }
                }
                if !stopped {
                    y = try stepSampled(
                        cur, processor: processor, seen: seen, counts: counts, rng: &rng)
                    idsHist.append(y)
                }
                accepted = acc
            }
            acceptedTotal += accepted
            acceptWindow.append(accepted)
            if acceptWindow.count > gateWindow { acceptWindow.removeFirst() }
        }
        if out.count > maxTokens {
            out = Array(out.prefix(maxTokens))
        }
        globalSuffixIndex.insert(out)
        return (out, acceptedTotal, attempts, batched)
    }

    /// One token: embed → Metal layers (+final RMS + FlashHead centroids) → gather/sample.
    /// Updates Token Recycling adjacency from FlashHead candidates when enabled.
    @discardableResult
    public func step(
        _ token: Int, profile: Bool = false, skipFlash: Bool = false, allowed: Set<Int>? = nil
    ) throws -> Int {
        var ph = PhaseMs()
        let t0 = CFAbsoluteTimeGetCurrent()
        embedToken(token)
        let t1 = CFAbsoluteTimeGetCurrent()
        try stack.stepCommitWait(layersPerCB: layersPerCB) { [self] enc in
            SeedlessMetal.encodeRms(
                into: enc, h: stack.hBuf, w: normWBuf,
                out: finalNormBuf, H: H, eps: config.rmsNormEps)
            if !skipFlash, let fh = flashHead {
                fh.encodeCentroids(into: enc, h: finalNormBuf)
                if fh.fuseIntoLayerCB {
                    fh.encodeFusedAfterCentroids(into: enc, h: finalNormBuf)
                }
            }
            if useMetalHead, allowed == nil {
                encodeMetalHead(enc, x: finalNormBuf, rows: 1)
            }
        }
        let t2 = CFAbsoluteTimeGetCurrent()
        let next: Int
        if let allowed {
            next = nextTokenConstrained(allowed)
        } else {
            next = pickGreedyFromFinalNorm(skipFlash: skipFlash)
            observeRecycle(fromToken: token)
        }
        let t3 = CFAbsoluteTimeGetCurrent()
        if profile {
            ph.embed = (t1 - t0) * 1000
            ph.layers = (t2 - t1) * 1000
            ph.head = (t3 - t2) * 1000
            ph.sample = 0
            ph.steps = 1
            lastPhase = ph
        }
        return next
    }

    private func nextTokenConstrained(_ allowed: Set<Int>) -> Int {
        var allow = allowed
        for b in decodeBanned { allow.remove(b) }
        if allow.isEmpty { return 0 }
        if allow.count == 1, let only = allow.first { return only }
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let logits = lmHead.apply(h.reshaped([1, 1, H]))
        let scores = ConstrainedPick.hostScores(logits)
        return ConstrainedPick.argmax(allowed: allow, scores: scores) ?? allow.first!
    }

    public func generate(
        prompt: [Int], maxTokens: Int, eos: Int? = 151_645,
        allowedNext: (([Int]) -> Set<Int>)? = nil,
        bannedTokenIds: [Int] = []
    ) throws -> [Int] {
        decodeBanned = bannedTokenIds
        syncMetalHeadBanned()
        defer { decodeBanned = []; metalHeadPending = false }
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        // Full-attn layers need prompt+gen ≤ fullMaxLen. Fail cleanly before
        // Metal encode.
        try checkPromptCapacity(prompt)
        let capped = min(maxTokens, fullMaxLen - prompt.count)
        guard capped > 0 else { return [] }
        reset()
        let constrain = allowedNext != nil
        let skipH = constrain || greedySkipFlash
        try prefillPrompt(prompt, skipFlash: skipH)
        var last: Int
        if constrain, let allowedNext {
            last = nextTokenConstrained(allowedNext([]))
        } else {
            last = pickGreedyFromFinalNorm(skipFlash: skipH)
        }
        let chainK = constrain ? 0 : Self.envChainK
        var out: [Int] = []
        var y = last
        while out.count < capped {
            out.append(y)
            if let eos, y == eos { break }
            let remain = capped - out.count
            if remain == 0 { break }
            if constrain, let allowedNext {
                let allow = allowedNext(out)
                if allow.isEmpty { break }
                y = try step(y, skipFlash: true, allowed: allow)
                continue
            }
            if chainK > 1, useFlashHead, flashFused, !greedySkipFlash {
                let n = min(chainK, remain, specMaxM)
                let toks = try stepGreedyChain(from: y, count: n)
                if let eos, let ei = toks.firstIndex(of: eos) {
                    out.append(contentsOf: toks.prefix(ei + 1))
                    break
                }
                if toks.count == 1 {
                    y = toks[0]
                } else {
                    out.append(contentsOf: toks.dropLast())
                    y = toks.last!
                }
            } else {
                y = try step(y, skipFlash: skipH)
            }
        }
        return out
    }

    /// Greedy chain length. Default **0** (sequential) — event-pipelined chain is
    /// lossless with force-tokens but slower than seq on p128 (~180 vs ~197).
    /// Opt-in: `MOMIJ_CHAIN_K=8`.
    public static var envChainK: Int {
        if let s = ProcessInfo.processInfo.environment["MOMIJ_CHAIN_K"], let v = Int(s) {
            return max(0, v)
        }
        return 0
    }

    /// SuffixSpec decode: free draft from history, early-exit greedy verify.
    /// Head matches `generate` (FlashHead default; exact when `greedyExactHead`).
    /// Hot drafts batch via M-row 1-CB (`MOMIJ_SPEC_BATCH=0` forces sequential).
    public func generateSuffixSpec(
        prompt: [Int], maxTokens: Int, draftK: Int = 8, eos: Int? = 151_645,
        bannedTokenIds: [Int] = []
    ) throws -> (tokens: [Int], accepted: Int, attempts: Int, gated: Int) {
        decodeBanned = bannedTokenIds
        syncMetalHeadBanned()
        defer { decodeBanned = []; metalHeadPending = false }
        guard !prompt.isEmpty, maxTokens > 0 else { return ([], 0, 0, 0) }
        try checkPromptCapacity(prompt)
        reset()
        try prefillPrompt(prompt, skipFlash: greedySkipFlash)
        let y = pickGreedyFromFinalNorm(skipFlash: greedySkipFlash)
        let r = try generateSuffixSpecFromPrefill(
            first: y, promptIds: prompt, maxTokens: maxTokens, draftK: draftK, eos: eos)
        return (r.tokens, r.accepted, r.attempts, r.gated)
    }

    /// Compare greedy vs SuffixSpec effective tok/s.
    /// Default prompt is a long repeated block (PLD-friendly). Pass `prompt` for real text ids.
    public func benchmarkSuffixSpec(
        promptTokens: Int, genTokens: Int, trials: Int, draftK: Int = 8,
        prompt: [Int]? = nil
    ) throws -> String {
        var promptIds: [Int]
        if let prompt, !prompt.isEmpty {
            promptIds = prompt
        } else {
            // Longer unique block so PLD can copy contiguous spans (not a tiny motif).
            let block = Array(100 ..< 132)  // 32 tokens
            promptIds = []
            while promptIds.count < promptTokens {
                promptIds.append(contentsOf: block)
            }
            promptIds = Array(promptIds.prefix(promptTokens))
        }

        _ = try generate(prompt: Array(promptIds.prefix(min(32, promptIds.count))), maxTokens: 4, eos: nil)

        var greedyTps: [Double] = []
        var specTps: [Double] = []
        var acc = 0, att = 0, gat = 0
        var equal = true
        var acceptedTok = 0
        var producedTok = 0
        var srcTree = 0, srcRecycle = 0, srcPld = 0
        for _ in 0 ..< trials {
            reset()
            var last = promptIds[0]
            for i in 0 ..< promptIds.count {
                if i + 1 < promptIds.count {
                    _ = try step(promptIds[i])
                } else {
                    last = try step(promptIds[i])
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            var y = last
            var gOut: [Int] = []
            for _ in 0 ..< genTokens {
                gOut.append(y)
                y = try step(y)
            }
            greedyTps.append(Double(gOut.count) / max(CFAbsoluteTimeGetCurrent() - t0, 1e-9))

            let r = try generateSuffixSpec(
                prompt: promptIds, maxTokens: genTokens, draftK: draftK, eos: nil)
            reset()
            last = promptIds[0]
            for i in 0 ..< promptIds.count {
                if i + 1 < promptIds.count {
                    _ = try step(promptIds[i])
                } else {
                    last = try step(promptIds[i])
                }
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            let r2 = try generateSuffixSpecFromPrefill(
                first: last, promptIds: promptIds, maxTokens: genTokens, draftK: draftK, eos: nil)
            specTps.append(Double(r2.tokens.count) / max(CFAbsoluteTimeGetCurrent() - t1, 1e-9))
            acc += r2.accepted; att += r2.attempts; gat += r2.gated
            acceptedTok += r2.accepted
            producedTok += r2.tokens.count
            srcTree += r2.srcTree; srcRecycle += r2.srcRecycle; srcPld += r2.srcPld
            if gOut != r.tokens { equal = false }
            _ = r
        }
        let g = greedyTps.reduce(0, +) / Double(trials)
        let s = specTps.reduce(0, +) / Double(trials)
        let aps = att > 0 ? Double(acc) / Double(att) : 0
        let frac = producedTok > 0 ? Double(acceptedTok) / Double(producedTok) : 0
        let base = String(
            format: "suffix-spec bench (draftK=%d, n=%d, prompt=%d): greedy=%.1f tok/s  spec=%.1f tok/s  accept/attempt=%.2f  accept/gen=%.2f  attempts=%d gated=%d  lossless=%@  mrow=%@  global_tok=%d  alpha=%.1f  recycle_n=%d  src(tree/rec/pld)=%d/%d/%d",
            draftK, trials, promptIds.count, g, s, aps, frac, att, gat,
            equal ? "true" : "false", useMrow ? "on" : "off",
            globalSuffixIndex.tokenCount, specAlpha,
            recycleIndex.entryCount, srcTree, srcRecycle, srcPld)
        let chain = (try? benchmarkChainVerify(steps: 8, trials: 5)) ?? ""
        return base + (chain.isEmpty ? "" : "\n" + chain)
    }

    /// Forced full-accept path: same feeds via sequential steps vs one-CB chain.
    public func benchmarkChainVerify(steps: Int = 8, trials: Int = 5) throws -> String {
        let K = min(steps, specMaxM)
        let prompt = Array(repeating: 100, count: 64)
        _ = try generate(prompt: prompt, maxTokens: 4, eos: nil)
        let skipH = greedySkipFlash

        var seqTps: [Double] = []
        var chainTps: [Double] = []
        var match = true
        for _ in 0 ..< trials {
            reset()
            var last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count { _ = try step(prompt[i], skipFlash: skipH) }
                else { last = try step(prompt[i], skipFlash: skipH) }
            }
            var feeds: [Int] = [last]
            var cur = last
            var expected: [Int] = []
            for i in 0 ..< K {
                let q = try step(cur, skipFlash: skipH)
                expected.append(q)
                cur = q
                if i + 1 < K { feeds.append(q) }
            }
            let snapFeeds = feeds
            let snapExpected = expected

            reset()
            last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count { _ = try step(prompt[i], skipFlash: skipH) }
                else { last = try step(prompt[i], skipFlash: skipH) }
            }
            let snap = stack.snapshotForChain(steps: K)

            let t0 = CFAbsoluteTimeGetCurrent()
            cur = last
            var seq2: [Int] = []
            for _ in 0 ..< K {
                let q = try step(cur, skipFlash: skipH)
                seq2.append(q)
                cur = q
            }
            seqTps.append(Double(K) / max(CFAbsoluteTimeGetCurrent() - t0, 1e-9))

            stack.restoreCaches(snap)
            let t1 = CFAbsoluteTimeGetCurrent()
            let chainOut = try stepChainFeeds(snapFeeds)
            chainTps.append(Double(K) / max(CFAbsoluteTimeGetCurrent() - t1, 1e-9))
            if chainOut != seq2 || chainOut != snapExpected { match = false }
        }
        let s = seqTps.reduce(0, +) / Double(trials)
        let c = chainTps.reduce(0, +) / Double(trials)
        let head = greedySkipFlash ? "exact" : "flash"
        return String(format: "chain-verify K=%d head=%@: sequential=%.1f tok/s  chain1cb=%.1f tok/s  match=%@  speedup=%.2fx",
                      K, head, s, c, match ? "true" : "false", c / max(s, 1e-9))
    }

    /// Prefill already done; `first` is the first generated token; `promptIds` is the prompt.
    private func generateSuffixSpecFromPrefill(
        first: Int, promptIds: [Int], maxTokens: Int, draftK: Int, eos: Int?
    ) throws -> (
        tokens: [Int], accepted: Int, attempts: Int, gated: Int,
        srcTree: Int, srcRecycle: Int, srcPld: Int
    ) {
        var ids = promptIds
        var y = first
        var out: [Int] = [y]
        ids.append(y)
        var acceptedTotal = 0
        var attempts = 0
        var gated = 0
        var acceptWindow: [Int] = []
        let gateOn = ProcessInfo.processInfo.environment["MOMIJ_SPEC_GATE"] != "0"
        // Chain verify costs ~D forwards; only use when drafts have been landing.
        let batchEnv = ProcessInfo.processInfo.environment["MOMIJ_SPEC_BATCH"]
        let batchForce = batchEnv == "1"
        let batchOff = batchEnv == "0"
        let gateWindow = 8
        // Per-request suffix tree (SuffixDecoding). Rebuilt from `ids` each draft
        // (O(n·depth) on CPU ≪ one Metal step). Global tree caches prior outputs.
        let localIndex = SuffixDraftIndex(maxDepth: 64)
        let useTree = ProcessInfo.processInfo.environment["MOMIJ_SPEC_TREE"] != "0"
        var srcTree = 0, srcRecycle = 0, srcPld = 0

        while out.count < maxTokens {
            if let eos, y == eos { break }
            let remain = maxTokens - out.count
            let meanAccept = acceptWindow.isEmpty ? 1.0
                : Double(acceptWindow.reduce(0, +)) / Double(acceptWindow.count)
            // Cold / low-accept → shorter drafts via accept window; tree also caps via α·p.
            let draftKEnv = ProcessInfo.processInfo.environment["MOMIJ_DRAFT_K"]
                .flatMap(Int.init)
            let effK = SuffixSpec.adaptiveDraftK(
                meanAccept: meanAccept,
                draftK: min(draftKEnv ?? draftK, remain, specMaxM - 1))
            localIndex.clear()
            localIndex.insert(ids)
            let treeHit = useTree
                ? SuffixDraftIndex.bestDraft(
                    local: localIndex, global: globalSuffixIndex,
                    history: ids, maxK: effK, alpha: specAlpha)
                : (matchLen: 0, tokens: [Int]())
            let pld = SuffixSpec.suffixDraft(
                history: ids, k: effK, promptLen: promptIds.count)
            let recycleK = min(effK, 4)
            let recycled = useRecycle ? recycleIndex.draft(from: y, maxK: recycleK) : []
            // Hybrid: any suffix-tree hit → tree; else Token Recycling; else PLD.
            // matchLen≥1 (not 2): short open-chat n-grams still beat cold recycle.
            let draft: [Int]
            if treeHit.matchLen >= 1, !treeHit.tokens.isEmpty {
                draft = treeHit.tokens
                srcTree += 1
            } else if !recycled.isEmpty {
                draft = recycled
                srcRecycle += 1
            } else if !pld.isEmpty {
                draft = pld
                srcPld += 1
            } else {
                draft = treeHit.tokens
                if !draft.isEmpty { srcTree += 1 }
            }
            // Soft gate: recycle-only / low-accept stays opportunistic (reprobe often).
            let gateFloor: Double = useTree ? (useRecycle ? 0.25 : (useMrow ? 0.5 : 1.0)) : 0.15
            let suspend = gateOn
                && acceptWindow.count >= gateWindow && meanAccept < gateFloor
            // Periodic re-probe after gated stretch (late self-similarity / recycle warm-up).
            let reprobe = suspend && (out.count % 8 == 0) && !draft.isEmpty

            if (draft.isEmpty || suspend) && !reprobe {
                if !draft.isEmpty { gated += 1 }
                y = try step(y, skipFlash: greedySkipFlash)
                out.append(y)
                ids.append(y)
                continue
            }

            attempts += 1
            let gate = ProcessInfo.processInfo.environment["MOMIJ_SPEC_BATCH_GATE"]
                .flatMap(Double.init) ?? 1.0
            let hotEnough = meanAccept >= gate
            let canBatch = useMrow || (useFlashHead && flashFused)
            let useBatch = !batchOff && canBatch && (batchForce || hotEnough)
            let accepted: Int
            if useBatch {
                let r = try verifyDraftChain(y: y, draft: draft)
                accepted = r.accepted
                if accepted > 0 {
                    // Clamp to the request budget: a chain can accept more than
                    // `maxTokens - out.count` in one round; emitting past the
                    // budget violates the OpenAI max_tokens contract.
                    let budget = maxTokens - out.count
                    if budget > 0 {
                        var chunk = Array(draft.prefix(min(accepted, budget)))
                        // eos terminates output: keep it as the chunk tail and
                        // drop any post-eos bonus so it cannot leak into text.
                        if let eos, let ei = chunk.firstIndex(of: eos) {
                            chunk = Array(chunk.prefix(ei + 1))
                        }
                        out.append(contentsOf: chunk)
                        ids.append(contentsOf: chunk)
                    }
                }
                y = r.next
                if out.count < maxTokens, out.last != eos {
                    out.append(y)
                    ids.append(y)
                }
            } else {
                var acc = 0
                var cur = y
                for (di, d) in draft.enumerated() {
                    let q = try step(cur, skipFlash: greedySkipFlash)
                    out.append(q)
                    ids.append(q)
                    y = q
                    if q == d {
                        acc += 1
                        cur = q
                        if let eos, q == eos { break }
                        if out.count >= maxTokens { break }
                    } else {
                        // Rejected draft tail may still appear later (Token Recycling insight).
                        if useRecycle {
                            let rest = Array(draft[di...])
                            recycleIndex.observe(fromToken: cur, candidates: [q] + rest)
                        }
                        break
                    }
                }
                accepted = acc
            }
            acceptedTotal += accepted
            acceptWindow.append(accepted)
            if acceptWindow.count > gateWindow { acceptWindow.removeFirst() }
            if let eos, out.contains(eos) { break }
        }
        // Feed completed generation into the global tree for later requests / trials.
        if out.count > 1 {
            globalSuffixIndex.insert(out)
        }
        return (out, acceptedTotal, attempts, gated, srcTree, srcRecycle, srcPld)
    }

    private static func mergedDraftQ(
        local: SuffixDraftIndex, global: SuffixDraftIndex,
        history: [Int], draft: [Int]
    ) -> [Float] {
        let a = local.qAlong(history: history, draft: draft)
        if a.contains(where: { $0 < 1 - 1e-5 }) { return a }
        return global.qAlong(history: history, draft: draft)
    }

    /// Snapshot → one-CB chain (M-row when packed) → host reject walk → restore+replay.
    private func verifyDraftSampled(
        y: Int, draft: [Int],
        processor: LogitsProcessor,
        seen: Set<Int>,
        counts: [Int: Int],
        rng: inout SplitMix64,
        draftQ: [Float]
    ) throws -> (accepted: Int, next: Int) {
        let feeds = [y] + draft
        let snap = stack.snapshotForChain(steps: feeds.count)
        let rows = try stepChainCandidateRows(feeds)
        if useRecycle {
            for i in 0 ..< min(feeds.count, specIndsSlots.count) {
                observeRecycle(fromToken: feeds[i], inds: specIndsSlots[i], logits: specLogitsSlots[i])
            }
        }
        let walk = SpeculativeSampling.walkDraft(
            draft: draft, rows: rows, processor: processor,
            seen: seen, counts: counts, rng: &rng, draftQ: draftQ)
        if walk.accepted == draft.count {
            return walk
        }
        stack.restoreCaches(snap)
        if walk.accepted == 0 {
            _ = try step(y)
            return walk
        }
        let replay = [y] + Array(draft.prefix(walk.accepted))
        _ = try stepChainFeeds(replay)
        return walk
    }

    /// Snapshot → one-CB chain verify → restore+replay on partial accept.
    private func verifyDraftChain(y: Int, draft: [Int]) throws -> (accepted: Int, next: Int) {
        let feeds = [y] + draft  // M = D+1; evals[i] vs draft[i] for i<D; evals[D] is bonus next
        let snap = stack.snapshotForChain(steps: feeds.count)
        let evals = try stepChainFeeds(feeds)
        // Refresh recycle adjacency from each FlashHead row (open-chat TR).
        if useRecycle, !greedySkipFlash, !specIndsSlots.isEmpty {
            for i in 0 ..< min(feeds.count, specIndsSlots.count) {
                observeRecycle(fromToken: feeds[i], inds: specIndsSlots[i], logits: specLogitsSlots[i])
            }
        }
        var p = 0
        while p < draft.count, p < evals.count, evals[p] == draft[p] { p += 1 }

        if p == draft.count {
            // Full accept: KV already at post-M; next = bonus token.
            return (draft.count, evals[draft.count])
        }

        // Partial / zero accept: roll back and replay accepted prefix only.
        stack.restoreCaches(snap)
        if p == 0 {
            // Single greedy step from y (must match evals[0]).
            let q = try step(y, skipFlash: greedySkipFlash)
            return (0, q)
        }
        let replay = [y] + Array(draft.prefix(p))
        let revals = try stepChainFeeds(replay)
        // After replay of p+1 feeds, next mismatch token is evals[p] / revals[p].
        return (p, revals[p])
    }

    public func benchmark(promptTokens: Int, genTokens: Int, trials: Int, profile: Bool = true)
        throws -> (promptTps: Double, genTps: Double, phase: PhaseMs)
    {
        let prompt = Array(repeating: 100, count: promptTokens)
        // Warmup
        _ = try generate(prompt: Array(prompt.prefix(min(32, promptTokens))), maxTokens: 4, eos: nil)

        var pTps: [Double] = [], gTps: [Double] = []
        var accum = PhaseMs()
        for _ in 0 ..< trials {
            reset()
            let t0 = CFAbsoluteTimeGetCurrent()
            var last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count {
                    _ = try step(prompt[i], profile: false, skipFlash: greedySkipFlash)
                } else {
                    last = try step(prompt[i], profile: false, skipFlash: greedySkipFlash)
                }
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            var y = last
            let chainK = Self.envChainK
            var produced = 0
            while produced < genTokens {
                if chainK > 1, useFlashHead, flashFused {
                    let n = min(chainK, genTokens - produced, specMaxM)
                    let toks = try stepGreedyChain(from: y, count: n)
                    produced += toks.count
                    y = toks.last!
                } else {
                    y = try step(y, profile: profile, skipFlash: greedySkipFlash)
                    produced += 1
                    if profile {
                        accum.embed += lastPhase.embed
                        accum.layers += lastPhase.layers
                        accum.head += lastPhase.head
                        accum.steps += 1
                    }
                }
            }
            let t2 = CFAbsoluteTimeGetCurrent()
            pTps.append(Double(promptTokens) / max(t1 - t0, 1e-9))
            gTps.append(Double(genTokens) / max(t2 - t1, 1e-9))
        }
        if accum.steps > 0 {
            let n = Double(accum.steps)
            accum.embed /= n; accum.layers /= n; accum.head /= n; accum.sample /= n
        }
        lastPhase = accum
        let p = pTps.reduce(0, +) / Double(pTps.count)
        let g = gTps.reduce(0, +) / Double(gTps.count)
        return (p, g, accum)
    }

    /// Sampled decode floor (1 token / forward) vs speculative Leviathan, both gen-only.
    public func benchmarkSampled(
        promptTokens: Int, genTokens: Int, trials: Int,
        temperature: Float = 0.7,
        prompt: [Int]? = nil
    ) throws -> (seqTps: Double, specTps: Double, acceptPerAttempt: Double, batched: Int) {
        let promptIds: [Int]
        if let given = prompt, !given.isEmpty {
            promptIds = given
        } else {
            promptIds = Array(repeating: 100, count: promptTokens)
        }
        let proc = LogitsProcessor(temperature: temperature, topP: 1)
        _ = try generateSampled(
            prompt: Array(promptIds.prefix(min(16, promptIds.count))), maxTokens: 4,
            processor: proc, eos: nil, seed: 1)

        var seq: [Double] = []
        for trial in 0 ..< trials {
            reset()
            var rng = SplitMix64(seed: UInt64(trial) &+ 1)
            var seen = Set(promptIds)
            var counts: [Int: Int] = [:]
            for t in promptIds { counts[t, default: 0] += 1 }
            var last = promptIds[0]
            for i in 0 ..< promptIds.count {
                if i + 1 < promptIds.count {
                    _ = try step(promptIds[i])
                } else {
                    last = try stepSampled(
                        promptIds[i], processor: proc, seen: seen, counts: counts, rng: &rng)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            var y = last
            for _ in 0 ..< genTokens {
                seen.insert(y)
                counts[y, default: 0] += 1
                y = try stepSampled(y, processor: proc, seen: seen, counts: counts, rng: &rng)
            }
            seq.append(Double(genTokens) / max(CFAbsoluteTimeGetCurrent() - t0, 1e-9))
        }

        var spec: [Double] = []
        var acc = 0
        var att = 0
        var bat = 0
        for trial in 0 ..< trials {
            reset()
            var rng = SplitMix64(seed: UInt64(trial) &+ 11)
            var seen = Set(promptIds)
            var counts: [Int: Int] = [:]
            for t in promptIds { counts[t, default: 0] += 1 }
            var last = promptIds[0]
            for i in 0 ..< promptIds.count {
                if i + 1 < promptIds.count {
                    _ = try step(promptIds[i])
                } else {
                    last = try stepSampled(
                        promptIds[i], processor: proc, seen: seen, counts: counts, rng: &rng)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            let r = try generateSampledSpeculativeFromPrefill(
                first: last, promptIds: promptIds, maxTokens: genTokens,
                processor: proc, draftK: 8, eos: nil,
                rng: &rng, seen: &seen, counts: &counts)
            spec.append(Double(r.tokens.count) / max(CFAbsoluteTimeGetCurrent() - t0, 1e-9))
            acc += r.accepted
            att += r.attempts
            bat += r.batched
        }
        let s = seq.reduce(0, +) / Double(seq.count)
        let p = spec.reduce(0, +) / Double(spec.count)
        let apa = att > 0 ? Double(acc) / Double(att) : 0
        return (s, p, apa, bat)
    }

    /// L0 hidden vs MapleEngine MLX (decode step after same prompt token). Rel L2.
    public static func parityL0(store: WeightStore) throws -> Double {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        let token = 100
        _ = MapleEngine(store: store, enableSeedlessMoE: false) // ensure weights sanitize path OK
        let block = try SeedlessLayerBlock(store: store, layer: 0, device: device)
        let embW = store.req("model.word_embeddings.weight")
        let embS = store.req("model.word_embeddings.scales")
        let embB = store.req("model.word_embeddings.biases")
        let emb = MLX.dequantized(embW, scales: embS, biases: embB, groupSize: 64, bits: 4, mode: .affine)
            .asType(.float16)
        let row = MLX.take(emb, MLXArray(Int32(token)), axis: 0)
        MLX.eval(row)
        let arr = row.asArray(Float16.self)
        arr.withUnsafeBufferPointer {
            block.hBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: block.H * 2)
        }
        let cb = SeedlessMetal.queue!.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try block.encodeStep(into: enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        var out = [Float16](repeating: 0, count: block.H)
        out.withUnsafeMutableBytes {
            $0.copyMemory(from: UnsafeRawBufferPointer(start: block.hBuf.contents(), count: block.H * 2))
        }
        // MLX one layer
        let cfg = store.config
        let p = "model.layers.0"
        func q(_ name: String) -> QuantProj {
            .quantized(weight: store.req("\(name).weight"), scales: store.req("\(name).scales"),
                       biases: store.req("\(name).biases"), bits: cfg.expertBits, groupSize: cfg.expertGroupSize)
        }
        let attn = MapleAttention(
            numHeads: cfg.numAttentionHeads, numKVHeads: cfg.numKeyValueHeads, headDim: cfg.headDim,
            ropeDim: cfg.ropeDim, ropeBase: cfg.ropeTheta, eps: cfg.rmsNormEps, useRope: cfg.isSliding(0),
            qkvProj: q("\(p).self_attn.qkv_proj"), oProj: q("\(p).self_attn.o_proj"),
            qNorm: store.req("\(p).self_attn.q_norm.weight"), kNorm: store.req("\(p).self_attn.k_norm.weight"))
        let moe = MapleMoE(
            topK: cfg.numExpertsPerTok, numExperts: cfg.numExperts,
            bits: cfg.expertBits, groupSize: cfg.expertGroupSize,
            gateW: store.req("\(p).mlp.gate.weight"),
            upGateW: store.req("\(p).mlp.switch_mlp.up_gate_proj.weight"),
            upGateS: store.req("\(p).mlp.switch_mlp.up_gate_proj.scales"),
            upGateB: store.req("\(p).mlp.switch_mlp.up_gate_proj.biases"),
            downW: store.req("\(p).mlp.switch_mlp.down_proj.weight"),
            downS: store.req("\(p).mlp.switch_mlp.down_proj.scales"),
            downB: store.req("\(p).mlp.switch_mlp.down_proj.biases"))
        let cache = KVCache(maxSize: cfg.isSliding(0) ? cfg.slidingWindow : nil)
        var h = row.reshaped([1, 1, cfg.hiddenSize])
        let inN = store.req("\(p).input_layernorm.weight")
        let postN = store.req("\(p).post_attention_layernorm.weight")
        let hn = MLXFast.rmsNorm(h.asType(.float32), weight: inN.asType(.float32), eps: cfg.rmsNormEps).asType(.float16)
        let r = attn(hn, cache: cache)
        h = h + r
        let hn2 = MLXFast.rmsNorm(h.asType(.float32), weight: postN.asType(.float32), eps: cfg.rmsNormEps).asType(.float16)
        h = h + moe(hn2)
        MLX.eval(h)
        let ref = h.reshaped([-1]).asArray(Float16.self)
        var num: Float = 0, den: Float = 0
        for i in 0 ..< cfg.hiddenSize {
            let a = Float(out[i]), b = Float(ref[i])
            num += (a - b) * (a - b)
            den += b * b
        }
        return sqrt(Double(num / max(den, 1e-12)))
    }

    /// Sweep `layersPerCB` candidates on a warm engine.
    public static func sweepLayersPerCB(store: WeightStore, prompt: Int = 64, gen: Int = 64) throws -> String {
        let eng = try SeedlessDecodeEngine(store: store, fullMaxLen: max(prompt + gen + 64, 2048))
        var lines: [String] = []
        var bestTps = 0.0
        var bestG = 1
        for g in [1, 2, 3, 4, 6, 8, 12, 24] {
            eng.layersPerCB = g
            _ = try eng.generate(prompt: Array(repeating: 100, count: 16), maxTokens: 4, eos: nil)
            let r = try eng.benchmark(promptTokens: prompt, genTokens: gen, trials: 2, profile: true)
            let ph = eng.lastPhase
            lines.append(String(format: "  g=%2d  gen=%.1f tok/s  layers=%.3f head=%.3f ms",
                                g, r.genTps, ph.layers, ph.head))
            if r.genTps > bestTps { bestTps = r.genTps; bestG = g }
        }
        return "seedless layers_per_cb sweep (p\(prompt)/g\(gen)):\n" + lines.joined(separator: "\n")
            + String(format: "\nbest g=%d @ %.1f tok/s", bestG, bestTps)
    }
}
