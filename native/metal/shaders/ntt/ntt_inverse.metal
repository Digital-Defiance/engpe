//
// ntt_inverse.metal
// Inverse Number Theoretic Transform (INTT) compute shader
//
// The inverse transform reuses the forward Cooley-Tukey butterfly network with
// inverse twiddle factors (powers of omega^{-1}), followed by a scaling pass
// that multiplies every coefficient by N^{-1} mod q.
//
// Arithmetic contract, twiddle layout and index derivation are identical to
// ntt_forward.metal — see that file's header for the derivation. In short:
//   stage_base  = (1 << stage) - 1
//   twiddle_idx = stage_base + (butterfly_idx % m)
// and all values are in Montgomery form with R = 2^32 while the stages run.
//
// The per-stage kernel (ntt_inverse_stage) is dispatched by the C++ Metal
// compute backend in batch_ntt_inverse, with the final scaling handled by
// ntt_inverse_scale, all inside a single command buffer.
//
// Design Reference: Section 4 - NTT Processor Implementation
// Requirements: 1.1, 1.4, 14.5
//

#include "../common/fhe_common.metal"

// ============================================================================
// Kernel: Inverse NTT Butterfly Stage
// ============================================================================

/// Perform one stage of the inverse NTT butterfly operations
///
/// Identical structure to ntt_forward_stage but reads inverse twiddle factors.
///
/// @param coeffs       Input/output coefficient buffer [batch_size][degree], Montgomery form
/// @param inv_twiddles Stage-major inverse twiddle table [degree - 1], Montgomery form
/// @param params       NTT parameters (degree, modulus, q_inv_neg, ...)
/// @param stage        Current NTT stage (0 to log2(degree)-1)
/// @param batch_size   Number of polynomials to process
kernel void ntt_inverse_stage(
    device coeff_t* coeffs [[buffer(0)]],
    device const coeff_t* inv_twiddles [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    constant uint32_t& stage [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.z;      // Which polynomial in the batch
    uint32_t butterfly_idx = gid.x;  // Which butterfly in this stage

    if (batch_idx >= batch_size) return;
    if (butterfly_idx >= params.degree / 2) return;

    // Stage geometry
    uint32_t m = 1u << stage;        // butterflies per group == half the group size
    uint32_t len = m << 1;           // group size

    uint32_t group = butterfly_idx / m;
    uint32_t offset = butterfly_idx % m;
    uint32_t j = len * group + offset;
    uint32_t k_idx = j + m;

    // Flat, stage-major twiddle index: exponent depends on the position within
    // the group (offset), not on which group this butterfly belongs to.
    uint32_t twiddle_idx = (m - 1u) + offset;

    coeff_t omega_inv = inv_twiddles[twiddle_idx];

    uint32_t poly_offset = batch_idx * params.degree;

    coeff_t a = coeffs[poly_offset + j];
    coeff_t b = coeffs[poly_offset + k_idx];

    //   out[j] = a + omega_inv * b
    //   out[k] = a - omega_inv * b
    coeff_t omega_b = mont_mul_32(omega_inv, b, params.modulus, params.q_inv_neg);
    coeff_t out_j = mod_add(a, omega_b, params.modulus);
    coeff_t out_k = mod_sub(a, omega_b, params.modulus);

    coeffs[poly_offset + j] = out_j;
    coeffs[poly_offset + k_idx] = out_k;
}

// ============================================================================
// Kernel: N^{-1} Scaling
// ============================================================================

/// Scale all coefficients by N^{-1} mod q after the inverse butterfly stages
///
/// params.inv_n_mont holds N^{-1} mod q already in Montgomery form, so the
/// Montgomery multiply below leaves the result in Montgomery form as well.
///
/// @param coeffs      Input/output coefficient buffer [batch_size][degree]
/// @param params      NTT parameters (degree, modulus, q_inv_neg, inv_n_mont)
/// @param batch_size  Number of polynomials to process
kernel void ntt_inverse_scale(
    device coeff_t* coeffs [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.z;
    uint32_t coeff_idx = gid.x;

    if (batch_idx >= batch_size || coeff_idx >= params.degree) return;

    uint32_t idx = batch_idx * params.degree + coeff_idx;
    coeffs[idx] = mont_mul_32(coeffs[idx], params.inv_n_mont,
                              params.modulus, params.q_inv_neg);
}

// ============================================================================
// Kernel: Batch Inverse NTT (threadgroup memory, degree <= 1024 only)
// ============================================================================

/// Perform a complete inverse NTT on a batch of polynomials in one dispatch.
///
/// LIMITATION: same fixed 1024-coefficient threadgroup scratch as
/// ntt_forward_batch. Returns without touching memory when
/// degree > NTT_TG_MAX_DEGREE; the host must not select it above that bound.
///
/// Requires a threadgroup of exactly `degree` threads and coefficients already
/// in Montgomery form.
///
/// @param input        Input coefficient buffer [batch_size][degree], Montgomery form
/// @param output       Output coefficient buffer [batch_size][degree], Montgomery form
/// @param inv_twiddles Stage-major inverse twiddle table [degree - 1], Montgomery form
/// @param params       NTT parameters
/// @param batch_size   Number of polynomials to process
kernel void ntt_inverse_batch(
    device const coeff_t* input [[buffer(0)]],
    device coeff_t* output [[buffer(1)]],
    device const coeff_t* inv_twiddles [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]]
) {
    uint32_t batch_idx = gid.z;

    if (batch_idx >= batch_size) return;
    // Hard guard: threadgroup scratch is a fixed 1024 entries.
    if (params.degree > NTT_TG_MAX_DEGREE) return;

    threadgroup coeff_t shared_coeffs[NTT_TG_MAX_DEGREE];

    uint32_t poly_offset = batch_idx * params.degree;
    uint32_t local_idx = tid.x;

    // Load coefficients into threadgroup memory with bit-reversal permutation
    if (local_idx < params.degree) {
        uint32_t reversed_idx = bit_reverse(local_idx, params.log_degree);
        shared_coeffs[local_idx] = input[poly_offset + reversed_idx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse butterfly stages
    for (uint32_t stage = 0; stage < params.log_degree; stage++) {
        uint32_t m = 1u << stage;
        uint32_t len = m << 1;
        uint32_t butterfly_idx = local_idx;

        if (butterfly_idx < params.degree / 2) {
            uint32_t group = butterfly_idx / m;
            uint32_t offset = butterfly_idx % m;
            uint32_t j = len * group + offset;
            uint32_t k = j + m;

            coeff_t omega_inv = inv_twiddles[(m - 1u) + offset];

            coeff_t a = shared_coeffs[j];
            coeff_t b = shared_coeffs[k];

            coeff_t omega_b = mont_mul_32(omega_inv, b, params.modulus, params.q_inv_neg);
            shared_coeffs[j] = mod_add(a, omega_b, params.modulus);
            shared_coeffs[k] = mod_sub(a, omega_b, params.modulus);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by N^{-1} mod q (Montgomery form) and write output
    if (local_idx < params.degree) {
        output[poly_offset + local_idx] = mont_mul_32(shared_coeffs[local_idx],
                                                      params.inv_n_mont,
                                                      params.modulus,
                                                      params.q_inv_neg);
    }
}
