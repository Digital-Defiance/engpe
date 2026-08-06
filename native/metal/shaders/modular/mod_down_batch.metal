//
// mod_down_batch.metal
// Hybrid KS Stage D: scale residues by P^{-1} mod q (after host/GPU subtract of x_P).
//
// y[i] = (x[i] * p_inv) % q
// CONTRACT: q < 2^31 so (a*b)%q is exact for a,b < q.
//

#include "../common/fhe_common.metal"

kernel void mod_down_batch(
    device const coeff_t* x [[buffer(0)]],
    device coeff_t* y [[buffer(1)]],
    constant coeff_t& modulus [[buffer(2)]],
    constant coeff_t& p_inv [[buffer(3)]],
    constant uint32_t& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    y[gid] = (x[gid] * p_inv) % modulus;
}
