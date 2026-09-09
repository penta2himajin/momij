import Foundation
import MLX
import MLXFast

/// Maple decode helpers: residual add + RMSNorm in one Metal dispatch (oracle parity).
enum MapleFused {
    nonisolated(unsafe) private static var addRmsKernels: [String: MLXFast.MLXFastKernel] = [:]

    /// Returns `(h_out, hn_out)` where `h = round(x+r)` and `hn = rmsnorm(h)*w`.
    static func addRmsNorm(_ x: MLXArray, _ r: MLXArray, weight w: MLXArray, eps: Float) -> (MLXArray, MLXArray) {
        let key = String(format: "%.6e", eps)
        let kernel: MLXFast.MLXFastKernel
        if let k = addRmsKernels[key] {
            kernel = k
        } else {
            let epsLit = String(format: "%.10ef", eps)
            let source = """
                uint tid = thread_position_in_threadgroup.x;
                constexpr uint N = DIM;
                constexpr uint PT = N / 256u;
                float hb[PT];
                float ss = 0.0f;
                for (uint i = 0; i < PT; ++i) {
                    uint j = tid * PT + i;
                    float v = (float)x[j] + (float)r[j];
                    T_ vb = (T_)v;
                    h_out[j] = vb;
                    hb[i] = (float)vb;
                    ss += hb[i] * hb[i];
                }
                ss = simd_sum(ss);
                threadgroup float sums[8];
                uint sg = tid / 32u;
                uint lane = tid % 32u;
                if (lane == 0u) sums[sg] = ss;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot = 0.0f;
                for (uint i = 0; i < 8u; ++i) tot += sums[i];
                float scale = metal::rsqrt(tot / (float)N + \(epsLit));
                for (uint i = 0; i < PT; ++i) {
                    uint j = tid * PT + i;
                    hn_out[j] = (T_)(hb[i] * scale * (float)w[j]);
                }
            """
            let safe = key
                .replacingOccurrences(of: ".", with: "_")
                .replacingOccurrences(of: "-", with: "m")
                .replacingOccurrences(of: "+", with: "p")
            let k = MLXFast.metalKernel(
                name: "maple_add_rms_norm_\(safe)",
                inputNames: ["x", "r", "w"],
                outputNames: ["h_out", "hn_out"],
                source: source)
            addRmsKernels[key] = k
            kernel = k
        }

        let flatX = x.reshaped([-1])
        let flatR = r.reshaped([-1])
        let dim = flatX.dim(0)
        precondition(dim % 256 == 0, "addRmsNorm dim must be multiple of 256, got \(dim)")
        let outs = kernel(
            [flatX, flatR, w.asType(x.dtype)],
            template: [
                ("T_", x.dtype),
                ("DIM", dim),
            ],
            grid: (256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [x.shape, x.shape],
            outputDTypes: [x.dtype, x.dtype])
        return (outs[0], outs[1])
    }

    nonisolated(unsafe) private static var routerKernel: MLXFast.MLXFastKernel?

    /// Decode fused router: gemv + softmax + top-8 + renorm in one dispatch.
    /// `ctr` must be a persistent `[8] uint32` buffer owned by the gate (reset by kernel).
    static func fusedRouter(
        x: MLXArray, weight w: MLXArray, ctr: MLXArray, numExperts: Int, topK: Int
    ) -> (inds: MLXArray, scores: MLXArray) {
        precondition(topK == 8 && numExperts % 32 == 0)
        let kernel: MLXFast.MLXFastKernel
        if let k = routerKernel {
            kernel = k
        } else {
            // Source mirrored from mlx-lm-deepgrove maple.py `_make_fused_router_kernel`.
            let source = """
                constexpr uint NE = NEXP;
                constexpr uint D = DIM;
                constexpr uint NTG = NE / 32u;
                constexpr uint TM = 4u;
                constexpr uint TN = 4u;
                constexpr uint BLOCKN = 32u * TN;
                constexpr uint NITER = D / BLOCKN;

                uint tid = thread_position_in_threadgroup.x;
                uint tgid = threadgroup_position_in_grid.x;
                uint n_threads = 256u;
                uint sg_id = tid / 32u;
                uint lane = tid % 32u;
                uint n_sg = n_threads / 32u;

                uint row0 = tgid * (n_sg * TM) + sg_id * TM;
                float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
                uint bn = lane * TN;
                for (uint i = 0u; i < NITER; ++i) {
                    float v[TN];
                    for (uint tn = 0u; tn < TN; ++tn) v[tn] = float(x[bn + tn]);
                    for (uint tm = 0u; tm < TM; ++tm) {
                        const device T_* wrow = w + (ulong)(row0 + tm) * D;
                        T_ inter[TN];
                        for (uint tn = 0u; tn < TN; ++tn) inter[tn] = wrow[bn + tn];
                        for (uint tn = 0u; tn < TN; ++tn) result[tm] += inter[tn] * v[tn];
                    }
                    bn += BLOCKN;
                }
                for (uint tm = 0u; tm < TM; ++tm) {
                    for (ushort sn = 16; sn >= 1; sn >>= 1) {
                        result[tm] += simd_shuffle_down(result[tm], sn);
                    }
                }
                device atomic_float* ls = (device atomic_float*)logits_scratch;
                if (lane == 0u) {
                    for (uint tm = 0u; tm < TM; ++tm) {
                        atomic_store_explicit(&ls[row0 + tm], result[tm],
                                              memory_order_relaxed);
                    }
                }

                threadgroup_barrier(mem_flags::mem_device);
                threadgroup uint last_flag;
                if (tid == 0u) {
                    device atomic_uint* ctra = (device atomic_uint*)ctr_in;
                    uint prev = atomic_fetch_add_explicit(ctra, 1u, memory_order_relaxed);
                    uint last = (prev == NTG - 1u) ? 1u : 0u;
                    if (last == 1u) atomic_store_explicit(ctra, 0u, memory_order_relaxed);
                    last_flag = last;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (last_flag == 0u) return;
                threadgroup_barrier(mem_flags::mem_device);

                float my_max = -1e30f;
                for (uint e = tid; e < NE; e += n_threads) {
                    float v = atomic_load_explicit(&ls[e], memory_order_relaxed);
                    if (v > my_max) my_max = v;
                }
                for (int off = 16; off > 0; off >>= 1) {
                    float other = simd_shuffle_down(my_max, off);
                    if (other > my_max) my_max = other;
                }
                threadgroup float sg_red[16];
                if (lane == 0u) sg_red[sg_id] = my_max;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float m = sg_red[0];
                    for (uint s = 1u; s < n_sg; s++) if (sg_red[s] > m) m = sg_red[s];
                    sg_red[0] = m;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float lmax = sg_red[0];

                threadgroup float scores[NE];
                float my_sum = 0.0f;
                for (uint e = tid; e < NE; e += n_threads) {
                    float lv = atomic_load_explicit(&ls[e], memory_order_relaxed);
                    float v = metal::exp(lv - lmax);
                    scores[e] = v;
                    my_sum += v;
                }
                for (int off = 16; off > 0; off >>= 1) {
                    my_sum += simd_shuffle_down(my_sum, off);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0u) sg_red[sg_id] = my_sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float ssum = sg_red[0];
                    for (uint i = 1u; i < n_sg; i++) ssum += sg_red[i];
                    sg_red[0] = ssum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float inv_total = 1.0f / (sg_red[0] + 1e-20f);
                for (uint e = tid; e < NE; e += n_threads) {
                    scores[e] = scores[e] * inv_total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup int topk_idx[8];
                threadgroup float topk_val[8];
                threadgroup uint8_t used[NE];
                for (uint e = tid; e < NE; e += n_threads) used[e] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int k = 0; k < 8; k++) {
                    float my_best = -1e30f;
                    int my_idx = 0;
                    for (int e = int(tid); e < int(NE); e += int(n_threads)) {
                        if (!used[e] && scores[e] > my_best) {
                            my_best = scores[e];
                            my_idx = e;
                        }
                    }
                    for (int off = 16; off > 0; off >>= 1) {
                        float other_v = simd_shuffle_down(my_best, off);
                        int other_i = simd_shuffle_down(my_idx, off);
                        if (other_v > my_best) { my_best = other_v; my_idx = other_i; }
                    }
                    threadgroup float sg_vals[16];
                    threadgroup int sg_idxs[16];
                    if (lane == 0u) { sg_vals[sg_id] = my_best; sg_idxs[sg_id] = my_idx; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid == 0u) {
                        float bv = sg_vals[0]; int bi = sg_idxs[0];
                        for (uint s = 1u; s < n_sg; s++) {
                            if (sg_vals[s] > bv) { bv = sg_vals[s]; bi = sg_idxs[s]; }
                        }
                        topk_val[k] = bv; topk_idx[k] = bi;
                        used[bi] = 1;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (tid < 8u) {
                    float sel_sum = 0.0f;
                    for (int i = 0; i < 8; i++) sel_sum += topk_val[i];
                    out_indices[tid] = topk_idx[tid];
                    out_scores[tid] = float(topk_val[tid] / (sel_sum + 1e-20f));
                }
            """
            let k = MLXFast.metalKernel(
                name: "maple_fused_router",
                inputNames: ["x", "w", "ctr_in"],
                outputNames: ["out_indices", "out_scores", "logits_scratch"],
                source: source)
            routerKernel = k
            kernel = k
        }

        let flat = x.reshaped([-1])
        let dim = flat.dim(0)
        let outs = kernel(
            [flat, w, ctr],
            template: [
                ("T_", w.dtype),
                ("NEXP", numExperts),
                ("DIM", dim),
            ],
            grid: ((numExperts / 32) * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[topK], [topK], [numExperts]],
            outputDTypes: [.int32, .float32, .float32])
        return (outs[0], outs[1])
    }

    nonisolated(unsafe) private static var qkNormRopeKernel: MLXFast.MLXFastKernel?

    /// Decode: per-head RMSNorm + partial RoPE in one dispatch (oracle `maple_qk_norm_rope`).
    static func qkNormRope(
        qk: MLXArray, w: MLXArray, invFreq: MLXArray, offset: Int, eps: Float, ropeDim: Int
    ) -> MLXArray {
        let kernel: MLXFast.MLXFastKernel
        if let k = qkNormRopeKernel {
            kernel = k
        } else {
            let source = """
                uint head = thread_position_in_grid.y;
                uint lane = thread_position_in_grid.x;

                constexpr int per_lane = HEAD_DIM / 32;
                const device T_* xh = x + head * HEAD_DIM;
                const device T_* wh = w + head * HEAD_DIM;
                device T_* oh = out + head * HEAD_DIM;

                float ss = 0.0f;
                for (int i = 0; i < per_lane; ++i) {
                    float v = (float)xh[lane * per_lane + i];
                    ss += v * v;
                }
                ss = simd_sum(ss);
                float pos = pos_eps[0];
                float epsv = pos_eps[1];
                float scale = metal::rsqrt(ss / HEAD_DIM + epsv);

                for (int i = 0; i < per_lane; ++i) {
                    int j = lane * per_lane + i;
                    float v = (float)xh[j] * scale * (float)wh[j];
                    if (ROPE_DIM > 0 && j < ROPE_DIM) {
                        constexpr int rhalf = ROPE_DIM > 0 ? ROPE_DIM / 2 : 1;
                        int p = j < rhalf ? j : j - rhalf;
                        float theta = pos * inv_freq[p];
                        float c = metal::cos(theta);
                        float s = metal::sin(theta);
                        int j2 = j < rhalf ? j + rhalf : j - rhalf;
                        float u = (float)xh[j2] * scale * (float)wh[j2];
                        v = j < rhalf ? (v * c - u * s) : (v * c + u * s);
                    }
                    oh[j] = (T_)v;
                }
            """
            let k = MLXFast.metalKernel(
                name: "maple_qk_norm_rope",
                inputNames: ["x", "w", "inv_freq", "pos_eps"],
                outputNames: ["out"],
                source: source)
            qkNormRopeKernel = k
            kernel = k
        }
        let nHeads = qk.dim(0)
        let headDim = qk.dim(1)
        let posEps = MLXArray([Float(offset), eps])
        let outs = kernel(
            [qk, w, invFreq, posEps],
            template: [
                ("T_", qk.dtype),
                ("HEAD_DIM", headDim),
                ("ROPE_DIM", ropeDim),
            ],
            grid: (32, nHeads, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [qk.shape],
            outputDTypes: [qk.dtype])
        return outs[0]
    }
}
