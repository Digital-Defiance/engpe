//
// keyswitch_mac.metal
// Galois key-switch multiply-accumulate in the NTT domain.
//
// For each coefficient index i:
//   acc[i] = Σ_d (digits[d*N + i] * evk[d*N + i]) mod q
//
// CONTRACT: modulus q must be < 2^31 so that (a*b) for a,b < q fits in 64 bits
// and `% q` is exact (matching the CPU `mul_mod` used by the RNS limb path).
//

#include "../common/fhe_common.metal"

/// Fused key-switch MAC over `n_digits` NTT-domain digit × EVK pairs.
///
/// @param digits  [n_digits * degree] ordinary residues (NTT domain)
/// @param evk     [n_digits * degree] ordinary residues (NTT domain, pre-transformed)
/// @param acc     [degree] output accumulator (overwritten)
/// @param modulus RNS limb prime q (< 2^31)
/// @param degree  polynomial degree N
/// @param n_digits gadget digit count
kernel void keyswitch_mac_batch(
    device const coeff_t* digits [[buffer(0)]],
    device const coeff_t* evk [[buffer(1)]],
    device coeff_t* acc [[buffer(2)]],
    constant coeff_t& modulus [[buffer(3)]],
    constant uint32_t& degree [[buffer(4)]],
    constant uint32_t& n_digits [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= degree) return;

    coeff_t sum = 0;
    for (uint32_t d = 0; d < n_digits; d++) {
        size_t idx = (size_t)d * (size_t)degree + (size_t)gid;
        // a,b < q < 2^31 ⇒ a*b fits in 64 bits; exact match to CPU mul_mod.
        coeff_t prod = (digits[idx] * evk[idx]) % modulus;
        sum = mod_add(sum, prod, modulus);
    }
    acc[gid] = sum;
}
