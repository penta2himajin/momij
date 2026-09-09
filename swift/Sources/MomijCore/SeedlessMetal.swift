import Foundation
import Metal
import MLX

/// Raw Metal kernels for Maple ternary (2-bit affine) decode hot path.
///
/// `gqmm2_rows` is ported from Qwisp Seedless (MLX gather_qmv_fast bits=2 layout)
/// with `group_size` 64|128. Fused expert = up_gate gather → clamped SwiGLU →
/// down gather → score reduce, encoded on one command buffer.
public enum SeedlessMetal {
    nonisolated(unsafe) static var device: MTLDevice?
    nonisolated(unsafe) static var queue: MTLCommandQueue?
    nonisolated(unsafe) static var gqmm2Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var swigluPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var scoreReducePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var rmsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var residPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var gateGemvPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var routeTop8Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var stopBuf: MTLBuffer?
    nonisolated(unsafe) public static var ready = false

    public static func ensureCompiled() throws {
        if ready { return }
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { throw SeedlessError.noMetal }
        device = dev
        queue = q
        stopBuf = dev.makeBuffer(length: 4, options: .storageModeShared)
        stopBuf?.contents().storeBytes(of: Int32(0), as: Int32.self)

        let opts = mlxMatchCompileOpts()
        let lib = try dev.makeLibrary(source: metalSource, options: opts)
        func pipe(_ name: String) throws -> MTLComputePipelineState {
            try dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        }
        gqmm2Pipeline = try pipe("gqmm2_rows")
        swigluPipeline = try pipe("maple_clamped_swiglu")
        scoreReducePipeline = try pipe("maple_score_reduce")
        rmsPipeline = try pipe("maple_rms_norm")
        residPipeline = try pipe("maple_resid_add")
        gateGemvPipeline = try pipe("maple_gate_gemv")
        routeTop8Pipeline = try pipe("maple_route_top8")
        ready = true
    }

    static func mlxMatchCompileOpts() -> MTLCompileOptions {
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) { opts.mathMode = .safe }
        else { opts.fastMathEnabled = false }
        return opts
    }

    static func mtlBuf(_ a: MLXArray, _ device: MTLDevice) -> MTLBuffer? {
        if let b = a.asMTLBuffer(device: device, noCopy: true) { return b }
        return a.asMTLBuffer(device: device, noCopy: false)
    }

    // MARK: - Public API

    /// 2-bit gather-qmv: x[1,K], w[E,N,K/16], scales/biases[E,N,K/gs], inds[Ktop] → y[Ktop,N]
    public static func gqmm2(
        x: MTLBuffer, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, inds: MTLBuffer,
        out: MTLBuffer,
        Ktop: Int, K: Int, N: Int, gs: Int = 128, lhsPerExpert: Bool = false,
        into encoder: MTLComputeCommandEncoder? = nil,
        commandQueue: MTLCommandQueue? = nil
    ) throws {
        try ensureCompiled()
        guard let pipe = gqmm2Pipeline, let stop = stopBuf else { throw SeedlessError.notReady }
        guard N % 8 == 0, K % 512 == 0, gs == 64 || gs == 128 else {
            throw SeedlessError.unsupportedShape(N: N, K: K, gs: gs)
        }

        let ownsCB = encoder == nil
        let q = commandQueue ?? queue!
        let cb = ownsCB ? q.makeCommandBuffer()! : nil
        let enc = encoder ?? cb!.makeComputeCommandEncoder()!

        enc.setComputePipelineState(pipe)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(inds, offset: 0, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 6)
        enc.setBytes(&nn, length: 4, index: 7)
        enc.setBytes(&kt, length: 4, index: 8)
        enc.setBuffer(stop, offset: 0, index: 9)
        var lp = UInt32(lhsPerExpert ? 1 : 0)
        enc.setBytes(&lp, length: 4, index: 10)
        var gsv = Int32(gs)
        enc.setBytes(&gsv, length: 4, index: 11)
        // grid: width=1, height=N/8, depth=M*Ktop (M=1)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: N / 8, depth: Ktop),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        if ownsCB {
            enc.endEncoding()
            cb!.commit()
            cb!.waitUntilCompleted()
        }
    }

    /// Encode fused expert block into an existing encoder (no commit/wait).
    public static func encodeFusedExpert(
        into enc: MTLComputeCommandEncoder,
        x: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, y: MTLBuffer,
        H: Int, I: Int, Ktop: Int, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let swiglu = swigluPipeline, let reduce = scoreReducePipeline, let stop = stopBuf
        else { throw SeedlessError.notReady }

        try gqmm2(x: x, w: upGateW, scales: upGateS, biases: upGateB, inds: inds, out: ugOut,
                  Ktop: Ktop, K: H, N: 2 * I, gs: gs, lhsPerExpert: false, into: enc)

        enc.setComputePipelineState(swiglu)
        enc.setBuffer(ugOut, offset: 0, index: 0)
        enc.setBuffer(act, offset: 0, index: 1)
        var i32 = Int32(I), kt32 = Int32(Ktop)
        enc.setBytes(&i32, length: 4, index: 2)
        enc.setBytes(&kt32, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: I, height: Ktop, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, I), height: 1, depth: 1))

        try gqmm2(x: act, w: downW, scales: downS, biases: downB, inds: inds, out: downOut,
                  Ktop: Ktop, K: I, N: H, gs: gs, lhsPerExpert: true, into: enc)

        enc.setComputePipelineState(reduce)
        enc.setBuffer(downOut, offset: 0, index: 0)
        enc.setBuffer(scores, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var h32 = Int32(H)
        enc.setBytes(&h32, length: 4, index: 3)
        enc.setBytes(&kt32, length: 4, index: 4)
        enc.setBuffer(stop, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    /// One-token fused MoE expert block on a single command buffer (wait once).
    public static func fusedExpertStep(
        x: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, y: MTLBuffer,
        H: Int, I: Int, Ktop: Int, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let q = queue else { throw SeedlessError.notReady }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try encodeFusedExpert(
            into: enc, x: x,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, y: y,
            H: H, I: I, Ktop: Ktop, gs: gs)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Encode one MoE block into an existing encoder (no commit/wait).
    public static func encodeMoEBlock(
        into enc: MTLComputeCommandEncoder,
        h: MTLBuffer, normW: MTLBuffer, gateW: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        xNorm: MTLBuffer, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, moeOut: MTLBuffer,
        H: Int, I: Int, E: Int, Ktop: Int, eps: Float, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let rms = rmsPipeline, let resid = residPipeline,
              let gemv = gateGemvPipeline, let route = routeTop8Pipeline,
              let stop = stopBuf
        else { throw SeedlessError.notReady }

        enc.setComputePipelineState(rms)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(normW, offset: 0, index: 1)
        enc.setBuffer(xNorm, offset: 0, index: 2)
        var epsV = eps, h32 = Int32(H)
        enc.setBytes(&epsV, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(gemv)
        enc.setBuffer(gateW, offset: 0, index: 0)
        enc.setBuffer(xNorm, offset: 0, index: 1)
        enc.setBuffer(logits, offset: 0, index: 2)
        var e32 = Int32(E)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: E, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(route)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(inds, offset: 0, index: 1)
        enc.setBuffer(scores, offset: 0, index: 2)
        enc.setBytes(&e32, length: 4, index: 3)
        var k32 = Int32(Ktop)
        enc.setBytes(&k32, length: 4, index: 4)
        enc.setBuffer(stop, offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        try encodeFusedExpert(
            into: enc, x: xNorm,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, y: moeOut,
            H: H, I: I, Ktop: Ktop, gs: gs)

        enc.setComputePipelineState(resid)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(moeOut, offset: 0, index: 1)
        enc.setBytes(&h32, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, H), height: 1, depth: 1))
    }

    public static func moeBlockOneCB(
        h: MTLBuffer, normW: MTLBuffer, gateW: MTLBuffer,
        upGateW: MTLBuffer, upGateS: MTLBuffer, upGateB: MTLBuffer,
        downW: MTLBuffer, downS: MTLBuffer, downB: MTLBuffer,
        xNorm: MTLBuffer, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        ugOut: MTLBuffer, act: MTLBuffer, downOut: MTLBuffer, moeOut: MTLBuffer,
        H: Int, I: Int, E: Int, Ktop: Int, eps: Float, gs: Int = 128
    ) throws {
        try ensureCompiled()
        guard let q = queue else { throw SeedlessError.notReady }
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try encodeMoEBlock(
            into: enc, h: h, normW: normW, gateW: gateW,
            upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            xNorm: xNorm, logits: logits, inds: inds, scores: scores,
            ugOut: ugOut, act: act, downOut: downOut, moeOut: moeOut,
            H: H, I: I, E: E, Ktop: Ktop, eps: eps, gs: gs)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Wall + GPU time for one encoder body. GPU time is `gpuEndTime - gpuStartTime`.
    static func timeCB(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> (wallMs: Double, gpuMs: Double) {
        try ensureCompiled()
        guard let q = queue else { throw SeedlessError.notReady }
        let t0 = CFAbsoluteTimeGetCurrent()
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let wallMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        return (wallMs, gpuMs)
    }

    static func encodeRms(
        into enc: MTLComputeCommandEncoder, h: MTLBuffer, w: MTLBuffer, out: MTLBuffer,
        H: Int, eps: Float
    ) {
        enc.setComputePipelineState(rmsPipeline!)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(w, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var epsV = eps, h32 = Int32(H)
        enc.setBytes(&epsV, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    static func encodeGate(
        into enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
        E: Int, H: Int
    ) {
        enc.setComputePipelineState(gateGemvPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var e32 = Int32(E), h32 = Int32(H)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: E, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    static func encodeRoute(
        into enc: MTLComputeCommandEncoder, logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
        E: Int, Ktop: Int
    ) {
        enc.setComputePipelineState(routeTop8Pipeline!)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(inds, offset: 0, index: 1)
        enc.setBuffer(scores, offset: 0, index: 2)
        var e32 = Int32(E), k32 = Int32(Ktop)
        enc.setBytes(&e32, length: 4, index: 3)
        enc.setBytes(&k32, length: 4, index: 4)
        enc.setBuffer(stopBuf, offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Benches

    public static func benchQmv2(H: Int = 2048, N: Int = 512, iters: Int = 200) throws -> Double {
        try ensureCompiled()
        guard let device, let queue else { throw SeedlessError.notReady }
        let gs = 128
        let packedK = H * 2 / 32
        let nGroups = H / gs
        let E = 8, Ktop = 1

        let x = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        let w = device.makeBuffer(length: E * N * packedK * 4, options: .storageModeShared)!
        let s = device.makeBuffer(length: E * N * nGroups * 2, options: .storageModeShared)!
        let b = device.makeBuffer(length: E * N * nGroups * 2, options: .storageModeShared)!
        let inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        inds.contents().storeBytes(of: Int32(0), as: Int32.self)
        let y = device.makeBuffer(length: Ktop * N * 2, options: .storageModeShared)!

        for _ in 0 ..< 5 {
            try gqmm2(x: x, w: w, scales: s, biases: b, inds: inds, out: y,
                      Ktop: Ktop, K: H, N: N, gs: gs)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters {
            try gqmm2(x: x, w: w, scales: s, biases: b, inds: inds, out: y,
                      Ktop: Ktop, K: H, N: N, gs: gs)
        }
        _ = queue  // silence
        return Double(iters) / (CFAbsoluteTimeGetCurrent() - t0)
    }

    public static func benchFusedExpert(
        H: Int = 2048, I: Int = 512, E: Int = 256, Ktop: Int = 8, iters: Int = 100
    ) throws -> Double {
        try ensureCompiled()
        guard let device else { throw SeedlessError.notReady }
        let gs = 128
        let packedH = H * 2 / 32
        let packedI = I * 2 / 32
        let nGroupsH = H / gs
        let nGroupsI = I / gs

        let x = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        let ugW = device.makeBuffer(length: E * 2 * I * packedH * 4, options: .storageModeShared)!
        let ugS = device.makeBuffer(length: E * 2 * I * nGroupsH * 2, options: .storageModeShared)!
        let ugB = device.makeBuffer(length: E * 2 * I * nGroupsH * 2, options: .storageModeShared)!
        let dW = device.makeBuffer(length: E * H * packedI * 4, options: .storageModeShared)!
        let dS = device.makeBuffer(length: E * H * nGroupsI * 2, options: .storageModeShared)!
        let dB = device.makeBuffer(length: E * H * nGroupsI * 2, options: .storageModeShared)!
        let inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let scores = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        let sp = scores.contents().bindMemory(to: Float.self, capacity: Ktop)
        for i in 0 ..< Ktop { ip[i] = Int32(i % E); sp[i] = 1.0 / Float(Ktop) }

        let ugOut = device.makeBuffer(length: Ktop * 2 * I * 2, options: .storageModeShared)!
        let act = device.makeBuffer(length: Ktop * I * 2, options: .storageModeShared)!
        let downOut = device.makeBuffer(length: Ktop * H * 2, options: .storageModeShared)!
        let y = device.makeBuffer(length: H * 2, options: .storageModeShared)!

        for _ in 0 ..< 3 {
            try fusedExpertStep(
                x: x, upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters {
            try fusedExpertStep(
                x: x, upGateW: ugW, upGateS: ugS, upGateB: ugB,
                downW: dW, downS: dS, downB: dB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        return Double(iters) / (CFAbsoluteTimeGetCurrent() - t0)
    }

    // MARK: - Metal source

    private static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;
    #define SIMD_SIZE 32

    inline float ld16_b2(const device half* x, thread float* xt) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            xt[i]   = x[i];
            xt[i+1] = x[i+1] / 4.0f;
            xt[i+2] = x[i+2] / 16.0f;
            xt[i+3] = x[i+3] / 64.0f;
        }
        return sum;
    }
    inline float qd2(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
        float accum = 0.0f;
        for (int i = 0; i < 4; i++) {
            accum += (float)(w[i] & 0x03) * xt[4*i]
                   + (float)(w[i] & 0x0c) * xt[4*i+1]
                   + (float)(w[i] & 0x30) * xt[4*i+2]
                   + (float)(w[i] & 0xc0) * xt[4*i+3];
        }
        return scale * accum + sum * bias;
    }

    // Qwisp gqmm2_rows (bits=2 affine gather-qmv). gs=64|128.
    kernel void gqmm2_rows(
        device const uint32_t* w      [[buffer(0)]],
        device const half*     scales [[buffer(1)]],
        device const half*     biases [[buffer(2)]],
        device const half*     x      [[buffer(3)]],
        device const int*      inds   [[buffer(4)]],
        device half*           y      [[buffer(5)]],
        constant int& in_vec_size  [[buffer(6)]],
        constant int& out_vec_size [[buffer(7)]],
        constant int& ktop         [[buffer(8)]],
        device const int* stopFlag [[buffer(9)]],
        constant uint& lhsPer      [[buffer(10)]],
        constant int&  gsz         [[buffer(11)]],
        uint3 tid      [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]])
    {
        if (stopFlag[0] != 0) return;
        constexpr int packs_per_thread = 1, num_simdgroups = 2, results_per_simdgroup = 4;
        constexpr int pack_factor = 16, bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512;
        const int scale_step_per_thread = gsz / values_per_thread;
        const device uint8_t* ws = (const device uint8_t*)w;
        thread float x_thread[16];
        thread float result[4] = {0};
        const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
        const int in_vec_size_g = in_vec_size / gsz;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        biases += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)mk * out_vec_size + out_row;
        for (int k = 0; k < in_vec_size; k += block_size) {
            float sum = ld16_b2(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device half* sl = scales + row * in_vec_size_g;
                const device half* bl = biases + row * in_vec_size_g;
                result[row] += qd2(wl, x_thread, sl[0], bl[0], sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / gsz; biases += block_size / gsz; x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }

    kernel void maple_clamped_swiglu(
        device const half* ug [[buffer(0)]],  // [Ktop, 2I] = up || gate
        device half* act       [[buffer(1)]],  // [Ktop, I]
        constant int& I        [[buffer(2)]],
        constant int& Ktop     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint j = gid.x, ki = gid.y;
        if (j >= (uint)I || ki >= (uint)Ktop) return;
        float up = ug[ki * (2 * I) + j];
        float gate = ug[ki * (2 * I) + I + j];
        gate = metal::min(gate, 7.0f);
        up = metal::clamp(up, -7.0f, 7.0f);
        float silu = gate / (1.0f + metal::exp(-gate));
        act[ki * I + j] = half(silu * up);
    }

    kernel void maple_score_reduce(
        device const half* down [[buffer(0)]],   // [Ktop, H]
        device const float* scores [[buffer(1)]], // [Ktop]
        device half* y [[buffer(2)]],             // [H]
        constant int& H [[buffer(3)]],
        constant int& Ktop [[buffer(4)]],
        device const int* stopFlag [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if (stopFlag[0] != 0) return;
        if (gid >= (uint)H) return;
        float acc = 0.0f;
        for (int ki = 0; ki < Ktop; ++ki) {
            acc += float(down[ki * H + gid]) * scores[ki];
        }
        y[gid] = half(acc);
    }

    // Milestone A helpers: RMSNorm / residual / dense gate / top-8 route.
    kernel void maple_rms_norm(
        device const half* x [[buffer(0)]],
        device const half* w [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant float& eps [[buffer(3)]],
        constant int& H [[buffer(4)]],
        uint lid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        threadgroup float red[256];
        float acc = 0.0f;
        for (uint i = lid; i < (uint)H; i += tgs) {
            float xi = float(x[i]);
            acc += xi * xi;
        }
        red[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (lid < s) red[lid] += red[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = precise::rsqrt(red[0] / float(H) + eps);
        for (uint i = lid; i < (uint)H; i += tgs) {
            out[i] = half(float(x[i]) * inv * float(w[i]));
        }
    }

    kernel void maple_resid_add(
        device half* h [[buffer(0)]],
        device const half* delta [[buffer(1)]],
        constant int& H [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= (uint)H) return;
        h[gid] = half(float(h[gid]) + float(delta[gid]));
    }

    // Dense gate: y[e] = sum_k W[e,k] * x[k]  (1 TG / expert, parallel reduce)
    kernel void maple_gate_gemv(
        device const half* W [[buffer(0)]],  // [E, H]
        device const half* x [[buffer(1)]],  // [H]
        device float* y [[buffer(2)]],       // [E]
        constant int& E [[buffer(3)]],
        constant int& H [[buffer(4)]],
        uint e [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        if (e >= (uint)E) return;
        threadgroup float red[256];
        const device half* row = W + (size_t)e * (size_t)H;
        float acc = 0.0f;
        for (uint k = tid; k < (uint)H; k += tgs) {
            acc += float(row[k]) * float(x[k]);
        }
        red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) y[e] = red[0];
    }

    // Softmax + top-K with renorm. E<=256, K<=8. scores are float (for score_reduce).
    kernel void maple_route_top8(
        device const float* logits [[buffer(0)]],
        device int* inds [[buffer(1)]],
        device float* scores [[buffer(2)]],
        constant int& E [[buffer(3)]],
        constant int& K [[buffer(4)]],
        device const int* stopFlag [[buffer(5)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        if (stopFlag[0] != 0) return;
        threadgroup float red[256];
        threadgroup int redi[256];
        threadgroup float gates[256];
        threadgroup float work[256];
        threadgroup float bcast[1];
        float lg = (tid < (uint)E) ? logits[tid] : -INFINITY;
        red[tid] = lg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = max(red[tid], red[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) bcast[0] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m = bcast[0];
        float e = (tid < (uint)E) ? precise::exp(lg - m) : 0.0f;
        red[tid] = e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) bcast[0] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float Z = bcast[0];
        if (tid < (uint)E) { gates[tid] = e / Z; work[tid] = lg; }
        else { work[tid] = -INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < K; k++) {
            red[tid] = work[tid];
            redi[tid] = (int)tid;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (red[tid + s] > red[tid]) {
                        red[tid] = red[tid + s];
                        redi[tid] = redi[tid + s];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                int bi = redi[0];
                inds[k] = bi;
                scores[k] = gates[bi];
                work[bi] = -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            float ss = 0.0f;
            for (int k = 0; k < K; k++) ss += scores[k];
            for (int k = 0; k < K; k++) scores[k] = scores[k] / ss;
        }
    }
    """
}

public enum SeedlessError: Error, CustomStringConvertible {
    case noMetal, notReady
    case unsupportedShape(N: Int, K: Int, gs: Int)
    public var description: String {
        switch self {
        case .noMetal: return "no Metal device"
        case .notReady: return "SeedlessMetal not compiled"
        case let .unsupportedShape(N, K, gs): return "unsupported shape N=\(N) K=\(K) gs=\(gs)"
        }
    }
}

/// Bind Maple layer-0 expert weights into Seedless fused path and microbench.
public enum SeedlessEngine {
    public static func benchRealExpert(store: WeightStore, iters: Int = 50) throws -> Double {
        try SeedlessMetal.ensureCompiled()
        guard let device = SeedlessMetal.device else { throw SeedlessError.notReady }
        let cfg = store.config
        let H = cfg.hiddenSize, I = cfg.moeIntermediateSize, Ktop = cfg.numExpertsPerTok
        let gs = cfg.expertGroupSize

        let ugW = store.req("model.layers.0.mlp.switch_mlp.up_gate_proj.weight")
        let ugS = store.req("model.layers.0.mlp.switch_mlp.up_gate_proj.scales")
        let ugB = store.req("model.layers.0.mlp.switch_mlp.up_gate_proj.biases")
        let dW = store.req("model.layers.0.mlp.switch_mlp.down_proj.weight")
        let dS = store.req("model.layers.0.mlp.switch_mlp.down_proj.scales")
        let dB = store.req("model.layers.0.mlp.switch_mlp.down_proj.biases")
        MLX.eval([ugW, ugS, ugB, dW, dS, dB])

        guard let bugW = SeedlessMetal.mtlBuf(ugW, device),
              let bugS = SeedlessMetal.mtlBuf(ugS.asType(.float16), device),
              let bugB = SeedlessMetal.mtlBuf(ugB.asType(.float16), device),
              let bdW = SeedlessMetal.mtlBuf(dW, device),
              let bdS = SeedlessMetal.mtlBuf(dS.asType(.float16), device),
              let bdB = SeedlessMetal.mtlBuf(dB.asType(.float16), device)
        else { throw SeedlessError.notReady }

        let x = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        // fill x with ones
        let xp = x.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { xp[i] = 0.01 }

        let inds = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let scores = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        let ip = inds.contents().bindMemory(to: Int32.self, capacity: Ktop)
        let sp = scores.contents().bindMemory(to: Float.self, capacity: Ktop)
        for i in 0 ..< Ktop { ip[i] = Int32(i); sp[i] = 1.0 / Float(Ktop) }

        let ugOut = device.makeBuffer(length: Ktop * 2 * I * 2, options: .storageModeShared)!
        let act = device.makeBuffer(length: Ktop * I * 2, options: .storageModeShared)!
        let downOut = device.makeBuffer(length: Ktop * H * 2, options: .storageModeShared)!
        let y = device.makeBuffer(length: H * 2, options: .storageModeShared)!

        for _ in 0 ..< 3 {
            try SeedlessMetal.fusedExpertStep(
                x: x, upGateW: bugW, upGateS: bugS, upGateB: bugB,
                downW: bdW, downS: bdS, downB: bdB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters {
            try SeedlessMetal.fusedExpertStep(
                x: x, upGateW: bugW, upGateS: bugS, upGateB: bugB,
                downW: bdW, downS: bdS, downB: bdB, inds: inds, scores: scores,
                ugOut: ugOut, act: act, downOut: downOut, y: y,
                H: H, I: I, Ktop: Ktop, gs: gs)
        }
        return Double(iters) / (CFAbsoluteTimeGetCurrent() - t0)
    }
}
