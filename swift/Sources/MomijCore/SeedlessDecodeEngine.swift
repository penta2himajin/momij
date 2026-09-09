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
    private var specFeedIds: MTLBuffer?
    private var specNormSlots: [MTLBuffer] = []
    private var specIndsSlots: [MTLBuffer] = []
    private var specLogitsSlots: [MTLBuffer] = []

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
        self.stack = try SeedlessLayerStack(store: store, device: device, fullMaxLen: fullMaxLen)
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
        let envM = ProcessInfo.processInfo.environment["MOMIJ_SPEC_MAX_M"].flatMap(Int.init)
        specMaxM = max(2, envM ?? 9)
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
    }

    /// C2: draft-driven chain verify in **one CB / one wait** (GPU embed; Qwisp chained style).
    /// `feeds` = [y] + draft (length D+1). Returns greedy evals[0..<feeds.count].
    public func stepChainFeeds(_ feeds: [Int]) throws -> [Int] {
        precondition(!feeds.isEmpty && feeds.count <= specMaxM)
        guard useFlashHead, let fh = flashHead, fh.fuseIntoLayerCB else {
            // Fallback: sequential steps.
            var out: [Int] = []
            for t in feeds { out.append(try step(t)) }
            return out
        }
        ensureSpecSlots()
        guard let q = SeedlessMetal.queue, let feedBuf = specFeedIds else {
            throw SeedlessError.notReady
        }
        let ip = feedBuf.contents().bindMemory(to: Int32.self, capacity: feeds.count)
        for (i, t) in feeds.enumerated() { ip[i] = Int32(t) }

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        for i in 0 ..< feeds.count {
            SeedlessMetal.encodeEmbedToken(
                into: enc, table: embBuf, ids: feedBuf, out: stack.hBuf, H: H, row: i)
            for layer in stack.layers {
                try layer.encodeStep(into: enc)
            }
            let norm = specNormSlots[i]
            let inds = specIndsSlots[i]
            let logits = specLogitsSlots[i]
            SeedlessMetal.encodeRms(
                into: enc, h: stack.hBuf, w: normWBuf, out: norm, H: H, eps: config.rmsNormEps)
            fh.encodeCentroids(into: enc, h: norm)
            fh.encodeFusedAfterCentroids(into: enc, h: norm, inds: inds, logits: logits)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var evals: [Int] = []
        evals.reserveCapacity(feeds.count)
        for i in 0 ..< feeds.count {
            let ptr = specNormSlots[i].contents().bindMemory(to: Float16.self, capacity: H)
            evals.append(fh.greedyAfterFusedGather(
                hostH: ptr, inds: specIndsSlots[i], logits: specLogitsSlots[i]))
        }
        return evals
    }

    /// Greedy GPU token-feedback chain: K embeds+layers+FlashHead+argmax in **one wait**.
    /// No force-token override on GPU (EOS force may diverge vs sequential near stop).
    /// Env driver: `MOMIJ_CHAIN_K` via `generate`.
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
        guard let q = SeedlessMetal.queue, let feedBuf = specFeedIds else {
            throw SeedlessError.notReady
        }
        let ip = feedBuf.contents().bindMemory(to: Int32.self, capacity: K + 1)
        ip[0] = Int32(start)

        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        for i in 0 ..< K {
            SeedlessMetal.encodeEmbedToken(
                into: enc, table: embBuf, ids: feedBuf, out: stack.hBuf, H: H, row: i)
            for layer in stack.layers {
                try layer.encodeStep(into: enc)
            }
            let norm = specNormSlots[i]
            let inds = specIndsSlots[i]
            let logits = specLogitsSlots[i]
            SeedlessMetal.encodeRms(
                into: enc, h: stack.hBuf, w: normWBuf, out: norm, H: H, eps: config.rmsNormEps)
            fh.encodeCentroids(into: enc, h: norm)
            fh.encodeFusedAfterCentroids(into: enc, h: norm, inds: inds, logits: logits)
            SeedlessMetal.encodeFlashArgmaxToken(
                into: enc, logits: logits, inds: inds, tokenMap: fh.tokenMapBuf, ids: feedBuf,
                nProbes: fh.nProbes, clusterSize: fh.clusterSize, outRow: i + 1)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var out: [Int] = []
        out.reserveCapacity(K)
        for i in 1 ... K { out.append(Int(ip[i])) }
        return out
    }

    public func reset() { stack.resetCaches() }

    private func embedToken(_ id: Int) {
        precondition(id >= 0 && id < vocab)
        let src = embBuf.contents().advanced(by: id * H * 2)
        stack.hBuf.contents().copyMemory(from: src, byteCount: H * 2)
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

    /// One token: embed → Metal layers (+final RMS + FlashHead centroids) → gather/sample.
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

    /// Greedy chain length. `MOMIJ_CHAIN_K` (default 0 = off / sequential).
    public static var envChainK: Int {
        guard let s = ProcessInfo.processInfo.environment["MOMIJ_CHAIN_K"], let v = Int(s), v > 0
        else { return 0 }
        return v
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
        return try generateSuffixSpecFromPrefill(
            first: y, promptIds: prompt, maxTokens: maxTokens, draftK: draftK, eos: eos)
    }

    /// Compare greedy vs SuffixSpec effective tok/s on a repeating prompt (high draft hit rate).
    public func benchmarkSuffixSpec(
        promptTokens: Int, genTokens: Int, trials: Int, draftK: Int = 8
    ) throws -> String {
        var prompt: [Int] = []
        let motif = [101, 102, 103, 104, 105, 106, 107, 108]
        while prompt.count < promptTokens {
            prompt.append(contentsOf: motif)
        }
        prompt = Array(prompt.prefix(promptTokens))

        _ = try generate(prompt: Array(prompt.prefix(min(32, promptTokens))), maxTokens: 4, eos: nil)

        var greedyTps: [Double] = []
        var specTps: [Double] = []
        var acc = 0, att = 0, gat = 0
        var equal = true
        for _ in 0 ..< trials {
            // Match `benchmark`: time gen only (prefill untimed).
            reset()
            var last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count {
                    _ = try step(prompt[i])
                } else {
                    last = try step(prompt[i])
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
                prompt: prompt, maxTokens: genTokens, draftK: draftK, eos: nil)
            // Retimed: regenerate with timed gen-only
            reset()
            last = prompt[0]
            for i in 0 ..< prompt.count {
                if i + 1 < prompt.count {
                    _ = try step(prompt[i])
                } else {
                    last = try step(prompt[i])
                }
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            let r2 = try generateSuffixSpecFromPrefill(
                first: last, promptIds: prompt, maxTokens: genTokens, draftK: draftK, eos: nil)
            specTps.append(Double(r2.tokens.count) / max(CFAbsoluteTimeGetCurrent() - t1, 1e-9))
            acc += r2.accepted; att += r2.attempts; gat += r2.gated
            if gOut != r.tokens { equal = false }
            _ = r
        }
        let g = greedyTps.reduce(0, +) / Double(trials)
        let s = specTps.reduce(0, +) / Double(trials)
        let aps = att > 0 ? Double(acc) / Double(att) : 0
        let base = String(
            format: "suffix-spec bench (repeat-motif, draftK=%d, n=%d): greedy=%.1f tok/s  spec=%.1f tok/s  accept/attempt=%.2f  attempts=%d gated=%d  lossless=%@",
            draftK, trials, g, s, aps, att, gat, equal ? "true" : "false")
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
            let snap = stack.snapshotCaches()

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
    ) throws -> (tokens: [Int], accepted: Int, attempts: Int, gated: Int) {
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

        while out.count < maxTokens {
            if let eos, y == eos { break }
            let remain = maxTokens - out.count
            let draft = SuffixSpec.suffixDraft(history: ids, k: min(draftK, remain, specMaxM - 1))
            let meanAccept = acceptWindow.isEmpty ? 1.0
                : Double(acceptWindow.reduce(0, +)) / Double(acceptWindow.count)
            let suspend = gateOn && acceptWindow.count >= gateWindow && meanAccept < 1.0

            if draft.isEmpty || suspend {
                if !draft.isEmpty { gated += 1 }
                y = try step(y)
                out.append(y)
                ids.append(y)
                continue
            }

            attempts += 1
            let useBatch = !batchOff && useFlashHead && flashFused
                && (batchForce || meanAccept >= 1.5)
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
                for d in draft {
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
        return (out, acceptedTotal, attempts, gated)
    }

    /// Snapshot → one-CB chain verify → restore+replay on partial accept.
    private func verifyDraftChain(y: Int, draft: [Int]) throws -> (accepted: Int, next: Int) {
        let feeds = [y] + draft  // M = D+1; evals[i] vs draft[i] for i<D; evals[D] is bonus next
        let snap = stack.snapshotCaches()
        let evals = try stepChainFeeds(feeds)
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
