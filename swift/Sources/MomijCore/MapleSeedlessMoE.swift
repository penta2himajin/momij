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
            return mlx(x)
        }

        let flat = x.reshaped([H]).asType(.float16)
        let logits = MLX.matmul(flat.asType(.float32), gateW.transposed(1, 0).asType(.float32))
        let gates = MLX.softmax(logits, axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        let indsArr = order[(numExperts - topK)...].asType(.int32)
        var scoresArr = MLX.takeAlong(gates, indsArr, axis: -1)
        scoresArr = scoresArr / scoresArr.sum()
        MLX.eval([flat, indsArr, scoresArr])

        var xHost = [Float16](repeating: 0, count: H)
        var indHost = [Int32](repeating: 0, count: topK)
        var scoreHost = [Float](repeating: 0, count: topK)
        for i in 0 ..< H { xHost[i] = flat[i].item(Float16.self) }
        for i in 0 ..< topK {
            indHost[i] = indsArr[i].item(Int32.self)
            scoreHost[i] = scoresArr[i].item(Float.self)
        }

        do {
            let yHost = try metal.run(x: xHost, inds: indHost, scores: scoreHost)
            return MLXArray(yHost).reshaped([1, 1, H])
        } catch {
            fputs("[momij] seedless moe fallback: \(error)\n", stderr)
            return mlx(x)
        }
    }
}
