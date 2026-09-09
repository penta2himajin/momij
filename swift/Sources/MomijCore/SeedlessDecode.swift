import Foundation
import Metal
import MLX

/// One Maple MoE block weights + private scratch (for hazard-free multi-layer 1-CB).
/// Residual stream `h` may be shared across layers (in-place resid).
public final class SeedlessMoEBlockLayer {
    let layer: Int
    let H: Int
    let I: Int
    let E: Int
    let Ktop: Int
    let gs: Int
    let eps: Float

    let normW: MTLBuffer
    let gateW: MTLBuffer
    let upGateW: MTLBuffer
    let upGateS: MTLBuffer
    let upGateB: MTLBuffer
    let downW: MTLBuffer
    let downS: MTLBuffer
    let downB: MTLBuffer

    /// Residual stream. Shared across the stack when built via `SeedlessMoEStack`.
    let hBuf: MTLBuffer
    let xNorm: MTLBuffer
    let logits: MTLBuffer
    let inds: MTLBuffer
    let scores: MTLBuffer
    let ugOut: MTLBuffer
    let act: MTLBuffer
    let downOut: MTLBuffer
    let moeOut: MTLBuffer

    public init(store: WeightStore, layer: Int, device: MTLDevice, sharedH: MTLBuffer? = nil) throws {
        self.layer = layer
        let cfg = store.config
        H = cfg.hiddenSize
        I = cfg.moeIntermediateSize
        E = cfg.numExperts
        Ktop = cfg.numExpertsPerTok
        gs = cfg.expertGroupSize
        eps = cfg.rmsNormEps

        let p = "model.layers.\(layer)"
        let moe = "\(p).mlp.switch_mlp"
        let ug = "\(moe).up_gate_proj"
        let dn = "\(moe).down_proj"

        let nW = store.req("\(p).post_attention_layernorm.weight").asType(.float16)
        let gW = store.req("\(p).mlp.gate.weight").asType(.float16)
        let ugW = store.req("\(ug).weight")
        let ugS = store.req("\(ug).scales").asType(.float16)
        let ugB = store.req("\(ug).biases").asType(.float16)
        let dW = store.req("\(dn).weight")
        let dS = store.req("\(dn).scales").asType(.float16)
        let dB = store.req("\(dn).biases").asType(.float16)
        MLX.eval([nW, gW, ugW, ugS, ugB, dW, dS, dB])

        guard let bn = SeedlessMetal.mtlBuf(nW, device),
              let bg = SeedlessMetal.mtlBuf(gW, device),
              let b0 = SeedlessMetal.mtlBuf(ugW, device),
              let b1 = SeedlessMetal.mtlBuf(ugS, device),
              let b2 = SeedlessMetal.mtlBuf(ugB, device),
              let b3 = SeedlessMetal.mtlBuf(dW, device),
              let b4 = SeedlessMetal.mtlBuf(dS, device),
              let b5 = SeedlessMetal.mtlBuf(dB, device)
        else { throw SeedlessError.notReady }
        normW = bn; gateW = bg
        upGateW = b0; upGateS = b1; upGateB = b2
        downW = b3; downS = b4; downB = b5

        hBuf = sharedH ?? device.makeBuffer(length: H * 2, options: .storageModeShared)!
        xNorm = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        logits = device.makeBuffer(length: E * 4, options: .storageModeShared)!
        inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        scores = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        ugOut = device.makeBuffer(length: Ktop * 2 * I * 2, options: .storageModeShared)!
        act = device.makeBuffer(length: Ktop * I * 2, options: .storageModeShared)!
        downOut = device.makeBuffer(length: Ktop * H * 2, options: .storageModeShared)!
        moeOut = device.makeBuffer(length: H * 2, options: .storageModeShared)!
    }

    public func encode(into enc: MTLComputeCommandEncoder) throws {
        try SeedlessMetal.encodeMoEBlock(
            into: enc, h: hBuf, normW: normW, gateW: gateW,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xNorm: xNorm, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs)
    }

    /// Run one MoE block in-place on residual `h` (half). Returns updated h.
    public func run(h: [Float16]) throws -> [Float16] {
        precondition(h.count == H)
        h.withUnsafeBytes { hBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: H * 2) }
        try SeedlessMetal.moeBlockOneCB(
            h: hBuf, normW: normW, gateW: gateW,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xNorm: xNorm, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs)
        var out = [Float16](repeating: 0, count: H)
        out.withUnsafeMutableBytes {
            $0.copyMemory(from: UnsafeRawBufferPointer(start: hBuf.contents(), count: H * 2))
        }
        return out
    }
}

/// Full stack of MoE blocks: shared residual, **per-layer** weights + scratch.
/// Encode all layers into one CB → one wait (Milestone C MoE-only).
public final class SeedlessMoEStack {
    public let layers: [SeedlessMoEBlockLayer]
    public let hBuf: MTLBuffer
    public let H: Int

    public init(store: WeightStore, device: MTLDevice) throws {
        let n = store.config.numHiddenLayers
        H = store.config.hiddenSize
        hBuf = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        var built: [SeedlessMoEBlockLayer] = []
        built.reserveCapacity(n)
        for i in 0 ..< n {
            built.append(try SeedlessMoEBlockLayer(store: store, layer: i, device: device, sharedH: hBuf))
        }
        layers = built
    }

    public func fillH(_ v: Float16) {
        let p = hBuf.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { p[i] = v }
    }

    public func encodeAll(into enc: MTLComputeCommandEncoder) throws {
        for layer in layers {
            try layer.encode(into: enc)
        }
    }

    /// One CB / one wait for the full MoE stack (no attn).
    public func runOneCB() throws {
        try SeedlessMetal.ensureCompiled()
        guard let q = SeedlessMetal.queue else { throw SeedlessError.notReady }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try encodeAll(into: enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}

extension SeedlessEngine {
    /// Benchmark Milestone A MoE block (1 CB / wait). Returns blocks/s.
    public static func benchMoEBlock(store: WeightStore, layer: Int = 0, iters: Int = 50) throws -> Double {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        let block = try SeedlessMoEBlockLayer(store: store, layer: layer, device: device)
        let H = store.config.hiddenSize
        var h = [Float16](repeating: 0.01, count: H)
        for _ in 0 ..< 3 { h = try block.run(h: h) }
        h = [Float16](repeating: 0.01, count: H)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters {
            h = [Float16](repeating: 0.01, count: H)
            h = try block.run(h: h)
        }
        return Double(iters) / (CFAbsoluteTimeGetCurrent() - t0)
    }

    /// Real 24-layer MoE stack, per-layer scratch, one CB. Returns wall/gpu ms and tok/s floors.
    public static func profileMoEStack24(store: WeightStore, iters: Int = 20) throws -> String {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        fputs("[momij] loading \(store.config.numHiddenLayers) MoE layers (real weights, per-layer scratch)…\n", stderr)
        let stack = try SeedlessMoEStack(store: store, device: device)
        let n = stack.layers.count
        let H = stack.H
        let E = store.config.numExperts
        let I = store.config.moeIntermediateSize
        let Ktop = store.config.numExpertsPerTok
        let gs = store.config.expertGroupSize

        stack.fillH(0.01)
        for _ in 0 ..< 3 { try stack.runOneCB() }

        func avg(_ count: Int, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> (wall: Double, gpu: Double) {
            var w = 0.0, g = 0.0
            for _ in 0 ..< count {
                stack.fillH(0.01)
                let t = try SeedlessMetal.timeCB(body)
                w += t.wallMs; g += t.gpuMs
            }
            return (w / Double(count), g / Double(count))
        }

        // A: true Milestone C — all real layers, private scratch each, 1 CB
        let real24 = try avg(iters) { enc in
            try stack.encodeAll(into: enc)
        }

        // B: same L0 weights × n in 1 CB (previous probe; shared scratch)
        let l0 = stack.layers[0]
        let repeatL0 = try avg(iters) { enc in
            for _ in 0 ..< n {
                try l0.encode(into: enc)
            }
        }

        // C: n separate CBs (one wait per layer) — old Milestone A floor
        var perLayerWall = 0.0, perLayerGpu = 0.0
        for _ in 0 ..< iters {
            stack.fillH(0.01)
            let t0 = CFAbsoluteTimeGetCurrent()
            var gpuSum = 0.0
            for layer in stack.layers {
                let t = try SeedlessMetal.timeCB { enc in
                    try layer.encode(into: enc)
                }
                gpuSum += t.gpuMs
            }
            perLayerWall += (CFAbsoluteTimeGetCurrent() - t0) * 1000
            perLayerGpu += gpuSum
        }
        let perLayer = (wall: perLayerWall / Double(iters), gpu: perLayerGpu / Double(iters))

        // D: single layer for reference
        let one = try avg(iters) { enc in
            try l0.encode(into: enc)
        }

        func packed(_ N: Int, _ K: Int) -> Double { Double(Ktop * N * K) * 2.0 / 8.0 }
        let bytesPerLayer = packed(2 * I, H) + packed(H, I) + Double(E * H * 2)
            + Double(Ktop) * (Double(2 * I * (H / gs) * 4) + Double(H * (I / gs) * 4))
        let bytesTok = bytesPerLayer * Double(n)
        let gbsReal = (bytesTok / 1e9) / (real24.gpu / 1000.0)

        func line(_ name: String, _ t: (wall: Double, gpu: Double)) -> String {
            let busy = t.gpu / max(t.wall, 1e-9) * 100
            let tpsW = 1000.0 / t.wall
            let tpsG = 1000.0 / t.gpu
            return String(format: "  %-14@ wall=%.3f ms  gpu=%.3f ms  busy=%.0f%%  → tok/s wall=%.1f gpu=%.1f",
                          name as NSString, t.wall, t.gpu, busy, tpsW, tpsG)
        }

        let hp = stack.hBuf.contents().bindMemory(to: Float16.self, capacity: H)
        let finite = (0 ..< min(8, H)).allSatisfy { hp[$0].isFinite }

        return """
        seedless MoE stack 24L re-measure (real weights, per-layer scratch, no host memcpy, \(iters) iters)
        \(line("1L (L0)", one))
        \(line("24×L0 1cb", repeatL0))
        \(line("24L real 1cb", real24))
        \(line("24L ×wait", perLayer))
        streamed ≈ \(String(format: "%.2f", bytesTok / 1e6)) MB/token  →  \(String(format: "%.1f", gbsReal)) GB/s on 24L-real GPU (M1 Max peak ~400)
        h finite after run: \(finite)
        note: tok/s is MoE-only (no attn / embed / lm_head). oracle e2e ~180 includes those.
        """
    }

    /// Wall vs GPU split + per-kernel GPU time. No host memcpy in the timed loops.
    public static func profileMoEBlock(store: WeightStore, layer: Int = 0, iters: Int = 40) throws -> String {
        try SeedlessMetal.ensureCompiled()
        guard SeedlessMetal.device != nil else { throw SeedlessError.notReady }
        let block = try SeedlessMoEBlockLayer(store: store, layer: layer, device: SeedlessMetal.device!)
        let cfg = store.config
        let H = cfg.hiddenSize, I = cfg.moeIntermediateSize, E = cfg.numExperts
        let Ktop = cfg.numExpertsPerTok, gs = cfg.expertGroupSize, eps = cfg.rmsNormEps
        let layers = cfg.numHiddenLayers

        let hp = block.hBuf.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { hp[i] = 0.01 }
        for _ in 0 ..< 5 {
            try SeedlessMetal.moeBlockOneCB(
                h: block.hBuf, normW: block.normW, gateW: block.gateW,
                upGateW: block.upGateW, upGateS: block.upGateS, upGateB: block.upGateB,
                downW: block.downW, downS: block.downS, downB: block.downB,
                xNorm: block.xNorm, logits: block.logits, inds: block.inds, scores: block.scores,
                ugOut: block.ugOut, act: block.act, downOut: block.downOut, moeOut: block.moeOut,
                H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs)
        }

        func avg(_ n: Int, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> (wall: Double, gpu: Double) {
            var w = 0.0, g = 0.0
            for _ in 0 ..< n {
                let t = try SeedlessMetal.timeCB(body)
                w += t.wallMs; g += t.gpuMs
            }
            return (w / Double(n), g / Double(n))
        }

        let emptyWait: (wall: Double, gpu: Double) = {
            var w = 0.0, g = 0.0
            for _ in 0 ..< iters {
                let t0 = CFAbsoluteTimeGetCurrent()
                let cb = SeedlessMetal.queue!.makeCommandBuffer()!
                cb.commit()
                cb.waitUntilCompleted()
                w += (CFAbsoluteTimeGetCurrent() - t0) * 1000
                let gpu = (cb.gpuEndTime > cb.gpuStartTime) ? (cb.gpuEndTime - cb.gpuStartTime) * 1000 : 0
                g += gpu
            }
            return (w / Double(iters), g / Double(iters))
        }()
        let full = try avg(iters) { enc in
            try block.encode(into: enc)
        }
        let rms = try avg(iters) { enc in
            SeedlessMetal.encodeRms(into: enc, h: block.hBuf, w: block.normW, out: block.xNorm, H: H, eps: eps)
        }
        let gate = try avg(iters) { enc in
            SeedlessMetal.encodeGate(into: enc, w: block.gateW, x: block.xNorm, y: block.logits, E: E, H: H)
        }
        let route = try avg(iters) { enc in
            SeedlessMetal.encodeRoute(into: enc, logits: block.logits, inds: block.inds, scores: block.scores,
                                      E: E, Ktop: Ktop)
        }
        let expert = try avg(iters) { enc in
            try SeedlessMetal.encodeFusedExpert(
                into: enc, x: block.xNorm,
                upGateW: block.upGateW, upGateS: block.upGateS, upGateB: block.upGateB,
                downW: block.downW, downS: block.downS, downB: block.downB,
                inds: block.inds, scores: block.scores,
                ugOut: block.ugOut, act: block.act, downOut: block.downOut, y: block.moeOut,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        let up = try avg(iters) { enc in
            try SeedlessMetal.gqmm2(
                x: block.xNorm, w: block.upGateW, scales: block.upGateS, biases: block.upGateB,
                inds: block.inds, out: block.ugOut,
                Ktop: Ktop, K: H, N: 2 * I, gs: gs, lhsPerExpert: false, into: enc)
        }
        let down = try avg(iters) { enc in
            try SeedlessMetal.gqmm2(
                x: block.act, w: block.downW, scales: block.downS, biases: block.downB,
                inds: block.inds, out: block.downOut,
                Ktop: Ktop, K: I, N: H, gs: gs, lhsPerExpert: true, into: enc)
        }

        let stacked = try avg(max(8, iters / 4)) { enc in
            for _ in 0 ..< layers {
                try block.encode(into: enc)
            }
        }

        func packed(_ N: Int, _ K: Int) -> Double { Double(Ktop * N * K) * 2.0 / 8.0 }
        let upB = packed(2 * I, H)
        let downB = packed(H, I)
        let gateB = Double(E * H * 2)
        let scaleB = Double(Ktop) * (
            Double(2 * I * (H / gs) * 2 * 2) + Double(H * (I / gs) * 2 * 2))
        let bytes = upB + downB + gateB + scaleB
        let gbs = (bytes / 1e9) / (full.gpu / 1000.0)
        let busy = full.gpu / max(full.wall, 1e-9) * 100
        let floorWait = 1000.0 / (full.wall * Double(layers))
        let floorGPU = 1000.0 / (full.gpu * Double(layers))
        let floor24 = 1000.0 / stacked.gpu

        func line(_ name: String, _ t: (wall: Double, gpu: Double)) -> String {
            let busy = t.gpu / max(t.wall, 1e-9) * 100
            return String(format: "  %-10@ wall=%.3f ms  gpu=%.3f ms  busy=%.0f%%",
                          name as NSString, t.wall, t.gpu, busy)
        }

        return """
        seedless moe-block profile (L\(layer), \(iters) iters, no host memcpy)
        \(line("empty-wait", emptyWait))
        \(line("full", full))
        \(line("rms", rms))
        \(line("gate", gate))
        \(line("route", route))
        \(line("expert", expert))
        \(line("gqmm2-up", up))
        \(line("gqmm2-dn", down))
        \(line("24L-1cb", stacked))
        streamed ≈ \(String(format: "%.2f", bytes / 1e6)) MB/layer  →  \(String(format: "%.1f", gbs)) GB/s on full GPU time (M1 Max peak ~400)
        tok/s floor @\(layers)L MoE-only: per-layer-wait \(String(format: "%.1f", floorWait))  |  GPU-only \(String(format: "%.1f", floorGPU))  |  24L-1CB \(String(format: "%.1f", floor24))
        GPU busy in full block \(String(format: "%.0f", busy))%  (wall-gpu = commit/wait tax)
        """
    }
}
