import Foundation
import Metal
import MLX
import MLXFast

/// End-to-end Seedless decode: MLX embed + Metal layers (commit→1wait) + MLX final norm/lm_head.
/// KV grows with `offset`; SWA layers rotate via `maple_shift_kv`.
public final class SeedlessDecodeEngine: @unchecked Sendable {
    public let store: WeightStore
    public let config: MapleConfig
    public let stack: SeedlessLayerStack
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
    /// FlashHead cluster probes (env `MOMIJ_FLASH_PROBES`, default 64). 0 if FlashHead off.
    public var flashProbes: Int { flashHead?.nProbes ?? 0 }
    /// True when `MOMIJ_FLASH_FUSE=1` (topk+gather in layer CB).
    public var flashFused: Bool { flashHead?.fuseIntoLayerCB ?? false }
    private let finalNormBuf: MTLBuffer
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

    public struct PhaseMs: Sendable {
        public var embed = 0.0
        public var layers = 0.0
        public var head = 0.0
        public var sample = 0.0
        public var steps = 0
    }

    public private(set) var lastPhase = PhaseMs()

    public init(store: WeightStore, fullMaxLen: Int = 2048) throws {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        self.store = store
        self.config = store.config
        self.H = store.config.hiddenSize
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
        let wantFlash = flashEnv != "0"
        flashHead = wantFlash ? SeedlessFlashHead(store: store, device: device) : nil
        useFlashHead = flashHead != nil
        if wantFlash && flashHead == nil {
            fputs("[momij] FlashHead weights missing; using exact lm_head\n", stderr)
        }
        MLX.eval(embTable, normW)
        vocab = embTable.dim(0)
        guard let ebuf = SeedlessMetal.mtlBuf(embTable, device),
              let nbuf = SeedlessMetal.mtlBuf(normW, device)
        else { throw SeedlessError.notReady }
        embBuf = ebuf
        normWBuf = nbuf
        finalNormBuf = device.makeBuffer(length: H * 2, options: .storageModeShared)!
    }

    private func ensureSpecSlots() {
        guard specFeedIds == nil, let device = SeedlessMetal.device, let fh = flashHead else { return }
        specFeedIds = device.makeBuffer(length: (specMaxM + 1) * 4, options: .storageModeShared)
        let logitBytes = fh.nProbes * fh.clusterSize * MemoryLayout<Float>.size
        for _ in 0 ..< specMaxM {
            specNormSlots.append(device.makeBuffer(length: H * 2, options: .storageModeShared)!)
            specIndsSlots.append(device.makeBuffer(length: fh.nProbes * 4, options: .storageModeShared)!)
            specLogitsSlots.append(device.makeBuffer(length: logitBytes, options: .storageModeShared)!)
        }
        if useMrow {
            specPackedNorm = device.makeBuffer(length: specMaxM * H * 2, options: .storageModeShared)
        }
    }

    /// C2: draft-driven chain verify in **one CB / one wait** (GPU embed; Qwisp chained style).
    /// `feeds` = [y] + draft (length D+1). Returns greedy evals[0..<feeds.count].
    /// With M-row enabled (default) and capacity, packs True M-row layers instead of M×M=1.
    public func stepChainFeeds(_ feeds: [Int]) throws -> [Int] {
        precondition(!feeds.isEmpty && feeds.count <= specMaxM)
        guard useFlashHead, let fh = flashHead, fh.fuseIntoLayerCB else {
            var out: [Int] = []
            for t in feeds { out.append(try step(t)) }
            return out
        }
        let packMrow = try encodeChainFeeds(feeds)
        let M = feeds.count
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
        let packMrow = try encodeChainFeeds(feeds)
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

    /// One CB: embed `feeds` × layers ± packed M-row → per-row FlashHead gather into spec slots.
    /// Returns whether True M-row packing ran.
    @discardableResult
    private func encodeChainFeeds(_ feeds: [Int]) throws -> Bool {
        guard let fh = flashHead else { throw SeedlessError.notReady }
        ensureSpecSlots()
        guard let q = SeedlessMetal.queue, let feedBuf = specFeedIds else {
            throw SeedlessError.notReady
        }
        let M = feeds.count
        let ip = feedBuf.contents().bindMemory(to: Int32.self, capacity: M)
        for (i, t) in feeds.enumerated() { ip[i] = Int32(t) }

        let packMrow = useMrow && M > 1 && stack.canEncodeMrow(M: M)
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
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
            for i in 0 ..< M {
                let off = i * H * 2
                fh.encodeCentroids(into: enc, h: packed, hByteOffset: off)
                fh.encodeFusedAfterCentroids(
                    into: enc, h: packed, inds: specIndsSlots[i],
                    logits: specLogitsSlots[i], hByteOffset: off)
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
                fh.encodeCentroids(into: enc, h: norm)
                fh.encodeFusedAfterCentroids(
                    into: enc, h: norm, inds: specIndsSlots[i], logits: specLogitsSlots[i])
            }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return packMrow
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
    }

    private func embedToken(_ id: Int) {
        precondition(id >= 0 && id < vocab)
        let src = embBuf.contents().advanced(by: id * H * 2)
        stack.hBuf.contents().copyMemory(from: src, byteCount: H * 2)
    }

    /// Observe FlashHead top candidates into the recycle adjacency for `fromToken`.
    private func observeRecycle(fromToken: Int, inds: MTLBuffer? = nil, logits: MTLBuffer? = nil) {
        guard useRecycle, let fh = flashHead else { return }
        let cands = fh.topTokenCandidates(k: recycleIndex.topK, inds: inds, logits: logits)
        recycleIndex.observe(fromToken: fromToken, candidates: cands)
    }

    private func nextToken() -> Int {
        let ptr = finalNormBuf.contents().bindMemory(to: Float16.self, capacity: H)
        if useFlashHead, let fh = flashHead {
            if fh.fuseIntoLayerCB {
                return fh.greedyAfterFusedGather(hostH: ptr)
            }
            return fh.greedyAfterCentroids(hBuf: finalNormBuf, hostH: ptr)
        }
        let h = MLXArray(UnsafeBufferPointer(start: ptr, count: H)).reshaped([H])
        let logits = lmHead.apply(h.reshaped([1, 1, H]))
        let y = MLX.argMax(logits.reshaped([-1]), axis: -1)
        MLX.eval(y)
        return y.item(Int.self)
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
    public func generateSampled(
        prompt: [Int], maxTokens: Int,
        processor: LogitsProcessor,
        eos: Int? = 151_645,
        seed: UInt64? = nil
    ) throws -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
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
    public func step(_ token: Int, profile: Bool = false) throws -> Int {
        var ph = PhaseMs()
        let t0 = CFAbsoluteTimeGetCurrent()
        embedToken(token)
        let t1 = CFAbsoluteTimeGetCurrent()
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
        let t2 = CFAbsoluteTimeGetCurrent()
        let next = nextToken()
        observeRecycle(fromToken: token)
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

    public func generate(prompt: [Int], maxTokens: Int, eos: Int? = 151_645) throws -> [Int] {
        guard !prompt.isEmpty, maxTokens > 0 else { return [] }
        reset()
        var last = prompt[0]
        for i in 0 ..< prompt.count {
            let id = prompt[i]
            if i + 1 < prompt.count {
                _ = try step(id)
            } else {
                last = try step(id)
            }
        }
        let chainK = Self.envChainK
        var out: [Int] = []
        var y = last
        while out.count < maxTokens {
            out.append(y)
            if let eos, y == eos { break }
            let remain = maxTokens - out.count
            if remain == 0 { break }
            if chainK > 1, useFlashHead, flashFused {
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
                y = try step(y)
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
    /// Lossless vs `generate`. Sequential verify ≈ greedy speed; early exit + gate
    /// avoid wasted steps on cold drafts. Batched verify (future) is required for
    /// peak ≫ greedy.
    public func generateSuffixSpec(
        prompt: [Int], maxTokens: Int, draftK: Int = 8, eos: Int? = 151_645
    ) throws -> (tokens: [Int], accepted: Int, attempts: Int, gated: Int) {
        guard !prompt.isEmpty, maxTokens > 0 else { return ([], 0, 0, 0) }
        reset()
        var y = prompt[0]
        for i in 0 ..< prompt.count {
            if i + 1 < prompt.count {
                _ = try step(prompt[i])
            } else {
                y = try step(prompt[i])
            }
        }
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
        guard useFlashHead, flashFused else {
            return "chain-verify: skipped (need FlashHead fuse)"
        }
        let K = min(steps, specMaxM)
        let prompt = Array(repeating: 100, count: 64)
        _ = try generate(prompt: prompt, maxTokens: 4, eos: nil)

        var seqTps: [Double] = []
        var chainTps: [Double] = []
        var match = true
        for _ in 0 ..< trials {
            reset()
            var last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count { _ = try step(prompt[i]) }
                else { last = try step(prompt[i]) }
            }
            // Build greedy chain feeds: embed last → o0; embed o0 → o1; ...
            var feeds: [Int] = [last]
            var cur = last
            var expected: [Int] = []
            for i in 0 ..< K {
                let q = try step(cur)
                expected.append(q)
                cur = q
                if i + 1 < K { feeds.append(q) }
            }
            let snapFeeds = feeds
            let snapExpected = expected

            // Rewind to post-prefill: re-prefill cleanly
            reset()
            last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count { _ = try step(prompt[i]) }
                else { last = try step(prompt[i]) }
            }
            let snap = stack.snapshotForChain(steps: K)

            let t0 = CFAbsoluteTimeGetCurrent()
            cur = last
            var seq2: [Int] = []
            for _ in 0 ..< K {
                let q = try step(cur)
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
        return String(format: "chain-verify K=%d: sequential=%.1f tok/s  chain1cb=%.1f tok/s  match=%@  speedup=%.2fx",
                      K, s, c, match ? "true" : "false", c / max(s, 1e-9))
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
            let effK = SuffixSpec.adaptiveDraftK(
                meanAccept: meanAccept, draftK: min(draftK, remain, specMaxM - 1))
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
                y = try step(y)
                out.append(y)
                ids.append(y)
                continue
            }

            attempts += 1
            // Hot only: batch (+ M-row pack when useMrow). Cold must stay sequential
            // early-exit / greedy — forced MOMIJ_SPEC_BATCH=1 still allowed for benches.
            let hotEnough = meanAccept >= 1.5
            let useBatch = !batchOff && useFlashHead && flashFused
                && (batchForce || hotEnough)
            let accepted: Int
            if useBatch {
                let r = try verifyDraftChain(y: y, draft: draft)
                accepted = r.accepted
                if accepted > 0 {
                    let chunk = Array(draft.prefix(accepted))
                    out.append(contentsOf: chunk)
                    ids.append(contentsOf: chunk)
                }
                y = r.next
                out.append(y)
                ids.append(y)
            } else {
                var acc = 0
                var cur = y
                for (di, d) in draft.enumerated() {
                    let q = try step(cur)
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
        if useRecycle {
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
            let q = try step(y)
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
                    _ = try step(prompt[i], profile: false)
                } else {
                    last = try step(prompt[i], profile: false)
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
                    y = try step(y, profile: profile)
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
