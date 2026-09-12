import Foundation
import MLX

/// Attention masks aligned with mlx-lm `create_causal_mask` (additive for MLXFast SDPA).
public enum AttentionMasks {
    /// Sliding-window causal mask: query `i` (absolute pos `offset+i`) attends keys in
    /// `[offset+i-windowSize+1, offset+i]` clipped to available key indices `0..<offset+N`.
    ///
    /// Returns additive float mask shaped `[1, 1, N, offset+N]` with `0` allowed / `-inf` blocked.
    public static func slidingCausal(
        queryLen N: Int,
        offset: Int,
        windowSize: Int
    ) -> MLXArray {
        let kvLen = offset + N
        // Build on host for clarity (N is prefill length; typically ≤ few k).
        var data = [Float](repeating: -.infinity, count: N * kvLen)
        for qi in 0 ..< N {
            let absQ = offset + qi
            let kMin = max(0, absQ - windowSize + 1)
            let kMax = absQ  // inclusive
            for k in kMin ... min(kMax, kvLen - 1) {
                data[qi * kvLen + k] = 0
            }
        }
        return MLXArray(data).reshaped([1, 1, N, kvLen])
    }
}
