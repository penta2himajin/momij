import Dispatch
import Foundation
import Metal
import MLX

/// Per-layer persistent Metal buffers for Maple MoE experts (Seedless path).
public final class SeedlessMoELayer {
    let upGateW: MTLBuffer
    let upGateS: MTLBuffer
    let upGateB: MTLBuffer
    let downW: MTLBuffer
    let downS: MTLBuffer
    let downB: MTLBuffer
    let H: Int
    let I: Int
    let Ktop: Int
    let gs: Int

    let xBuf: MTLBuffer
    let indsBuf: MTLBuffer
    let scoresBuf: MTLBuffer
    let ugOut: MTLBuffer
    let act: MTLBuffer
    let downOut: MTLBuffer
    let yBuf: MTLBuffer

    public init(store: WeightStore, layer: Int, device: MTLDevice) throws {
        let cfg = store.config
        self.H = cfg.hiddenSize
        self.I = cfg.moeIntermediateSize
        self.Ktop = cfg.numExpertsPerTok
        self.gs = cfg.expertGroupSize
        let p = "model.layers.\(layer).mlp.switch_mlp"
        let ugW = store.req("\(p).up_gate_proj.weight")
        let ugS = store.req("\(p).up_gate_proj.scales").asType(.float16)
        let ugB = store.req("\(p).up_gate_proj.biases").asType(.float16)
        let dW = store.req("\(p).down_proj.weight")
        let dS = store.req("\(p).down_proj.scales").asType(.float16)
        let dB = store.req("\(p).down_proj.biases").asType(.float16)
        MLX.eval([ugW, ugS, ugB, dW, dS, dB])

        guard let b0 = SeedlessMetal.mtlBuf(ugW, device),
              let b1 = SeedlessMetal.mtlBuf(ugS, device),
              let b2 = SeedlessMetal.mtlBuf(ugB, device),
              let b3 = SeedlessMetal.mtlBuf(dW, device),
              let b4 = SeedlessMetal.mtlBuf(dS, device),
              let b5 = SeedlessMetal.mtlBuf(dB, device)
        else { throw SeedlessError.notReady }
        upGateW = b0; upGateS = b1; upGateB = b2
        downW = b3; downS = b4; downB = b5

        xBuf = device.makeBuffer(length: H * 2, options: .storageModeShared)!
        indsBuf = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        scoresBuf = device.makeBuffer(length: Ktop * 4, options: .storageModeShared)!
        ugOut = device.makeBuffer(length: Ktop * 2 * I * 2, options: .storageModeShared)!
        act = device.makeBuffer(length: Ktop * I * 2, options: .storageModeShared)!
        downOut = device.makeBuffer(length: Ktop * H * 2, options: .storageModeShared)!
        yBuf = device.makeBuffer(length: H * 2, options: .storageModeShared)!
    }

    public func run(x: [Float16], inds: [Int32], scores: [Float]) throws -> [Float16] {
        precondition(x.count == H && inds.count == Ktop && scores.count == Ktop)
        x.withUnsafeBytes { xBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: H * 2) }
        inds.withUnsafeBytes { indsBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: Ktop * 4) }
        scores.withUnsafeBytes { scoresBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: Ktop * 4) }
        try SeedlessMetal.fusedExpertStep(
            x: xBuf, upGateW: upGateW, upGateS: upGateS, upGateB: upGateB,
            downW: downW, downS: downS, downB: downB,
            inds: indsBuf, scores: scoresBuf,
            ugOut: ugOut, act: act, downOut: downOut, y: yBuf,
            H: H, I: I, Ktop: Ktop, gs: gs)
        var y = [Float16](repeating: 0, count: H)
        y.withUnsafeMutableBytes {
            $0.copyMemory(from: UnsafeRawBufferPointer(start: yBuf.contents(), count: H * 2))
        }
        return y
    }
}

/// Phase timers for hybrid MoE (env `MOMIJ_PROFILE_MOE=1`). Printed once via `dumpIfEnabled()`.
public enum MoEProfile {
    nonisolated(unsafe) public static var enabled =
        ProcessInfo.processInfo.environment["MOMIJ_PROFILE_MOE"] == "1"
    nonisolated(unsafe) public static var calls = 0
    nonisolated(unsafe) public static var routerNs: UInt64 = 0
    nonisolated(unsafe) public static var hostInNs: UInt64 = 0
    nonisolated(unsafe) public static var metalNs: UInt64 = 0
    nonisolated(unsafe) public static var hostOutNs: UInt64 = 0
    nonisolated(unsafe) public static var mlxMoENs: UInt64 = 0
    nonisolated(unsafe) public static var mlxMoECalls = 0

    public static func reset() {
        calls = 0; routerNs = 0; hostInNs = 0; metalNs = 0; hostOutNs = 0
        mlxMoENs = 0; mlxMoECalls = 0
    }

    public static func dumpIfEnabled() {
        guard enabled, calls + mlxMoECalls > 0 else { return }
        let n = max(calls, 1)
        let m = max(mlxMoECalls, 1)
        func ms(_ ns: UInt64, _ c: Int) -> Double { Double(ns) / Double(c) / 1e6 }
        fputs(String(format:
            "[moe-profile] hybrid_calls=%d router_ms=%.3f host_in_ms=%.3f metal_ms=%.3f host_out_ms=%.3f | mlx_calls=%d mlx_moe_ms=%.3f\n",
            calls, ms(routerNs, n), ms(hostInNs, n), ms(metalNs, n), ms(hostOutNs, n),
            mlxMoECalls, ms(mlxMoENs, m)), stderr)
    }
}

/// Maple MoE with optional Seedless Metal expert path (router stays MLX).
/// Enable with `MOMIJ_SEEDLESS_MOE=1` for B=L=1 decode tokens.
public struct MapleMoEHybrid {
    let mlx: MapleMoE
    let metal: SeedlessMoELayer?
    let topK: Int
    let numExperts: Int
    let gateW: MLXArray
    public static var enabled: Bool {
        ProcessInfo.processInfo.environment["MOMIJ_SEEDLESS_MOE"] == "1"
    }

    public init(mlx: MapleMoE, metal: SeedlessMoELayer?, gateW: MLXArray, topK: Int, numExperts: Int) {
        self.mlx = mlx
        self.metal = metal
        self.gateW = gateW
        self.topK = topK
        self.numExperts = numExperts
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1), H = x.dim(2)
        guard let metal, B == 1, L == 1, Self.enabled else {
            if MoEProfile.enabled {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let y = mlx(x)
                eval(y)
                MoEProfile.mlxMoENs += DispatchTime.now().uptimeNanoseconds - t0
                MoEProfile.mlxMoECalls += 1
                return y
            }
            return mlx(x)
        }

        let t0 = DispatchTime.now().uptimeNanoseconds
        let flat = x.reshaped([H]).asType(.float16)
        let logits = MLX.matmul(flat.asType(.float32), gateW.transposed(1, 0).asType(.float32))
        let gates = MLX.softmax(logits, axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        let indsArr = order[(numExperts - topK)...].asType(.int32)
        var scoresArr = MLX.takeAlong(gates, indsArr, axis: -1)
        scoresArr = scoresArr / scoresArr.sum()
        MLX.eval([flat, indsArr, scoresArr])
        let t1 = DispatchTime.now().uptimeNanoseconds

        // Bulk host copy — per-element .item() is ~1000× slower here.
        let xHost = flat.asArray(Float16.self)
        let indHost = indsArr.asArray(Int32.self)
        let scoreHost = scoresArr.asArray(Float.self)
        let t2 = DispatchTime.now().uptimeNanoseconds

        do {
            let yHost = try metal.run(x: xHost, inds: indHost, scores: scoreHost)
            let t3 = DispatchTime.now().uptimeNanoseconds
            let out = MLXArray(yHost).reshaped([1, 1, H])
            let t4 = DispatchTime.now().uptimeNanoseconds
            if MoEProfile.enabled {
                MoEProfile.calls += 1
                MoEProfile.routerNs += t1 - t0
                MoEProfile.hostInNs += t2 - t1
                MoEProfile.metalNs += t3 - t2
                MoEProfile.hostOutNs += t4 - t3
            }
            return out
        } catch {
            fputs("[momij] seedless moe fallback: \(error)\n", stderr)
            return mlx(x)
        }
    }
}
