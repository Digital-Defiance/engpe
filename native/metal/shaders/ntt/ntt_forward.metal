//
// ntt_forward.metal
// Forward Number Theoretic Transform (NTT) compute shader
//
// Implements the iterative Cooley-Tukey NTT (decimation-in-time: bit-reversed
// input, natural-order output) using GPU parallelism. Each thread performs one
// butterfly of one stage.
//
// ---------------------------------------------------------------------------
// Arithmetic contract
// ---------------------------------------------------------------------------
// All coefficients and twiddle factors are in Montgomery form with R = 2^32
// while the butterfly stages run. The host encodes ntt_to_montgomery before
// the stages and ntt_from_montgomery after them, so callers pass and receive
// ordinary residues in [0, q).
//
// ---------------------------------------------------------------------------
// Cyclic core, negacyclic transform
// ---------------------------------------------------------------------------
// The butterfly stages below are a CYCLIC length-N transform and require the
// twiddle table to hold powers of omega, a primitive N-th root of unity. For
// the negacyclic transform over Z_q[X]/(X^N + 1) that FHE needs, omega = psi^2
// where psi is a primitive 2N-th root, and the psi twist is applied by the
// ntt_twist kernel below (fused with the Montgomery domain crossing, so it
// costs no extra dispatch). The kernels here are unchanged by that: they never
// see psi.
//
// ---------------------------------------------------------------------------
// Twiddle table layout (must match the host, cpp/src/metal_compute.mm)
// ---------------------------------------------------------------------------
// A flat, stage-major table of N-1 entries. For stage s (0-based), with
//   m    = 1 << s          (half of the group size; butterflies per group)
//   len  = 2 * m           (group size)
// the stage owns `m` consecutive entries beginning at
//   stage_base = (1 << s) - 1 = m - 1
// and entry (stage_base + j) holds  omega^(j * N / len)  in Montgomery form,
// for j = 0 .. m-1. Total = 1 + 2 + 4 + ... + N/2 = N - 1 entries.
// (This mirrors the reference Rust table in fhe-evolve/native/src/ntt_accel.rs.)
//
// Index derivation for a butterfly thread:
//   butterfly_idx in [0, N/2)
//   group  = butterfly_idx / m      -> which group of `len` coefficients
//   offset = butterfly_idx % m      -> position *within* the group == j
//   top    = len * group + offset,  bottom = top + m
// The Cooley-Tukey twiddle exponent depends on the position within the group,
// i.e. on `offset`, never on `group`. Since the stage block already stores
// omega^(j * N / len) at position j, the flat index is exactly:
//   twiddle_idx = stage_base + offset
//
// Design Reference: Section 4 - NTT Processor Implementation
// Requirements: 1.1, 1.4, 14.5
//

#include "../common/fhe_common.metal"

// ============================================================================
// Kernel: Forward NTT Butterfly Stage
// ============================================================================

/// Perform one stage of the forward NTT butterfly operations
///
/// The NTT is computed in log2(N) stages, where each stage performs N/2 butterfly
/// operations. This kernel processes one stage for a batch of polynomials.
///
/// Thread organization:
/// - gid.x selects the butterfly within the polynomial (0 .. N/2-1)
/// - gid.z selects the polynomial within the batch
///
/// @param coeffs Input/output coefficient buffer [batch_size][degree], Montgomery form
/// @param twiddles Stage-major twiddle table [degree - 1], Montgomery form
/// @param params NTT parameters (degree, modulus, q_inv_neg, ...)
/// @param stage Current NTT stage (0 to log2(degree)-1)
/// @param batch_size Number of polynomials to process
kernel void ntt_forward_stage(
    device coeff_t* coeffs [[buffer(0)]],
    device const coeff_t* twiddles [[buffer(1)]],
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

    // Flat, stage-major twiddle index (see derivation in the file header).
    uint32_t stage_base = m - 1u;    // == (1 << stage) - 1
    uint32_t twiddle_idx = stage_base + offset;

    coeff_t omega = twiddles[twiddle_idx];

    uint32_t poly_offset = batch_idx * params.degree;

    coeff_t a = coeffs[poly_offset + j];
    coeff_t b = coeffs[poly_offset + k_idx];

    // Cooley-Tukey butterfly:
    //   out[j] = a + omega * b
    //   out[k] = a - omega * b
    coeff_t omega_b = mont_mul_32(omega, b, params.modulus, params.q_inv_neg);
    coeff_t out_j = mod_add(a, omega_b, params.modulus);
    coeff_t out_k = mod_sub(a, omega_b, params.modulus);

    coeffs[poly_offset + j] = out_j;
    coeffs[poly_offset + k_idx] = out_k;
}

// ============================================================================
// Kernel: Bit-Reversal Permutation
// ============================================================================

/// Perform bit-reversal permutation on polynomial coefficients
///
/// The decimation-in-time NTT requires input coefficients in bit-reversed order.
/// Each thread owns one index and swaps only when i < bit_reverse(i), so the
/// permutation is safe to run fully in parallel in place.
///
/// @param coeffs Input/output coefficient buffer [batch_size][degree]
/// @param params NTT parameters
/// @param batch_size Number of polynomials to process
kernel void ntt_bit_reverse(
    device coeff_t* coeffs [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.z;
    uint32_t i = gid.x;

    if (batch_idx >= batch_size || i >= params.degree) return;

    uint32_t j = bit_reverse(i, params.log_degree);

    if (i < j) {
        uint32_t poly_offset = batch_idx * params.degree;

        coeff_t temp = coeffs[poly_offset + i];
        coeffs[poly_offset + i] = coeffs[poly_offset + j];
        coeffs[poly_offset + j] = temp;
    }
}

// ============================================================================
// Kernels: Montgomery Domain Conversion
// ============================================================================

/// Convert ordinary residues to Montgomery form: x -> x * R mod q
///
/// Implemented as mont_mul_32(x, R^2 mod q) so the whole transform can stay on
/// the GPU inside a single command buffer.
///
/// @param coeffs Input/output coefficient buffer [batch_size][degree]
/// @param params NTT parameters (uses modulus, q_inv_neg, r2_mod_q)
/// @param batch_size Number of polynomials to process
kernel void ntt_to_montgomery(
    device coeff_t* coeffs [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.z;
    uint32_t i = gid.x;

    if (batch_idx >= batch_size || i >= params.degree) return;

    uint32_t idx = batch_idx * params.degree + i;
    coeffs[idx] = mont_mul_32(coeffs[idx], params.r2_mod_q,
                              params.modulus, params.q_inv_neg);
}

/// Fused negacyclic twist + Montgomery domain crossing.
///
/// coeffs[i] <- mont_mul_32(coeffs[i], twist[i])
///
/// One kernel serves both ends of the negacyclic transform; which end is
/// decided entirely by how the host scales the `twist` table, because
/// mont_mul_32(a, b) = a * b * R^(-1) mod q:
///
///   FORWARD PRE-PASS   twist[i] = psi^i * R^2 mod q
///       input  a_i (ordinary)  ->  output (a_i * psi^i) * R  (Montgomery form)
///       i.e. the psi twist and ntt_to_montgomery in a single pass.
///
///   INVERSE POST-PASS  twist[i] = psi^(-i) mod q   (ordinary form)
///       input  x_i * R (Montgomery) ->  output x_i * psi^(-i)  (ordinary)
///       i.e. the psi^(-1) untwist and ntt_from_montgomery in a single pass.
///
/// At i = 0 both tables degenerate to the plain conversions (R^2 and 1
/// respectively), so this kernel is a strict generalisation of
/// ntt_to_montgomery / ntt_from_montgomery. Fusing rather than adding a
/// separate elementwise dispatch keeps the dispatch count and the number of
/// command-buffer synchronisations unchanged.
///
/// @param coeffs Input/output coefficient buffer [batch_size][degree]
/// @param twist Per-coefficient multiplier table [degree], scaled as above
/// @param params NTT parameters (uses modulus, q_inv_neg)
/// @param batch_size Number of polynomials to process
kernel void ntt_twist(
    device coeff_t* coeffs [[buffer(0)]],
    device const coeff_t* twist [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    constant uint32_t& batch_size [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.z;
    uint32_t i = gid.x;

    if (batch_idx >= batch_size || i >= params.degree) return;

    uint32_t idx = batch_idx * params.degree + i;
    coeffs[idx] = mont_mul_32(coeffs[idx], twist[i],
                              params.modulus, params.q_inv_neg);
}

/// Convert Montgomery-form values back to ordinary residues: x*R -> x
///
/// This is a bare REDC (equivalently mont_mul_32(x, 1)).
///
/// @param coeffs Input/output coefficient buffer [batch_size][degree]
/// @param params NTT parameters (uses modulus, q_inv_neg)
/// @param batch_size Number of polynomials to process
kernel void ntt_from_montgomery(
    device coeff_t* coeffs [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.z;
    uint32_t i = gid.x;

    if (batch_idx >= batch_size || i >= params.degree) return;

    uint32_t idx = batch_idx * params.degree + i;
    coeffs[idx] = mont_redc_32(coeffs[idx], params.modulus, params.q_inv_neg);
}

// ============================================================================
// Kernel: Batch Forward NTT (threadgroup memory, degree <= 1024 only)
// ============================================================================

/// Perform a complete forward NTT on a batch of polynomials in one dispatch.
///
/// LIMITATION: the threadgroup scratch buffer is a fixed 1024 coefficients, and
/// one thread handles one coefficient, so this kernel only supports
/// degree <= NTT_TG_MAX_DEGREE. It returns without touching memory for larger
/// degrees; the host must not select it above that bound either.
///
/// Requires a threadgroup of exactly `degree` threads and coefficients already
/// in Montgomery form (the host runs ntt_to_montgomery / ntt_from_montgomery
/// around it, same as the per-stage path).
///
/// @param input Input coefficient buffer [batch_size][degree], Montgomery form
/// @param output Output NTT coefficient buffer [batch_size][degree], Montgomery form
/// @param twiddles Stage-major twiddle table [degree - 1], Montgomery form
/// @param params NTT parameters
/// @param batch_size Number of polynomials to process
kernel void ntt_forward_batch(
    device const coeff_t* input [[buffer(0)]],
    device coeff_t* output [[buffer(1)]],
    device const coeff_t* twiddles [[buffer(2)]],
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

    // Load coefficients into threadgroup memory with bit-reversal
    if (local_idx < params.degree) {
        uint32_t reversed_idx = bit_reverse(local_idx, params.log_degree);
        shared_coeffs[local_idx] = input[poly_offset + reversed_idx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform NTT stages
    for (uint32_t stage = 0; stage < params.log_degree; stage++) {
        uint32_t m = 1u << stage;
        uint32_t len = m << 1;
        uint32_t butterfly_idx = local_idx;

        if (butterfly_idx < params.degree / 2) {
            uint32_t group = butterfly_idx / m;
            uint32_t offset = butterfly_idx % m;
            uint32_t j = len * group + offset;
            uint32_t k = j + m;

            // Flat, stage-major twiddle index (see file header).
            coeff_t omega = twiddles[(m - 1u) + offset];

            coeff_t a = shared_coeffs[j];
            coeff_t b = shared_coeffs[k];

            coeff_t omega_b = mont_mul_32(omega, b, params.modulus, params.q_inv_neg);
            shared_coeffs[j] = mod_add(a, omega_b, params.modulus);
            shared_coeffs[k] = mod_sub(a, omega_b, params.modulus);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_idx < params.degree) {
        output[poly_offset + local_idx] = shared_coeffs[local_idx];
    }
}
