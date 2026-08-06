//
// fhe_common.metal
// Common utilities and types for FHE Metal shaders
//
// This file contains shared functions used across all FHE compute shaders.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Type Definitions
// ============================================================================

/// 64-bit unsigned integer for polynomial coefficients
typedef ulong coeff_t;

/// 32-bit unsigned integer for indices and sizes
typedef uint index_t;

/// Maximum polynomial degree supported by the threadgroup-memory NTT variants.
/// Their threadgroup scratch buffers are sized at compile time, so this is a
/// hard limit shared by ntt_forward_batch and ntt_inverse_batch. Kernels must
/// bail out above it, and the host must not select them above it either.
#define NTT_TG_MAX_DEGREE 1024u

/// Structure for passing modular arithmetic parameters
///
/// CONTRACT: `modulus` must be odd and < 2^32, and `inv_modulus` must hold
/// -q^(-1) mod 2^32 in its low 32 bits (the R = 2^32 Montgomery constant).
/// The Montgomery kernels in this project use R = 2^32 exclusively.
struct ModularParams {
    coeff_t modulus;           // Prime modulus q (< 2^32, odd)
    coeff_t inv_modulus;       // -q^(-1) mod 2^32 (low 32 bits are significant)
    coeff_t r_squared;         // R^2 mod q for Montgomery conversion
    uint32_t modulus_bits;     // Bit length of modulus
};

/// Structure for NTT parameters
///
/// This layout must stay byte-for-byte identical to the host-side struct in
/// cpp/src/metal_compute.mm (see NTTParamsHost + its static_asserts).
/// Metal aligns `ulong` to 8 bytes, hence the explicit padding word.
///
///   offset  0: degree       (uint32)
///   offset  4: log_degree   (uint32)
///   offset  8: q_inv_neg    (uint32)
///   offset 12: _pad         (uint32)
///   offset 16: modulus      (ulong)
///   offset 24: inv_n_mont   (ulong)
///   offset 32: r_mod_q      (ulong)
///   offset 40: r2_mod_q     (ulong)
///   sizeof   = 48
struct NTTParams {
    uint32_t degree;           // Polynomial degree N (power of 2)
    uint32_t log_degree;       // log2(N)
    uint32_t q_inv_neg;        // -q^(-1) mod 2^32
    uint32_t _pad;             // Explicit padding (keeps ulong 8-byte aligned)
    coeff_t modulus;           // Prime modulus q (odd, < 2^32)
    coeff_t inv_n_mont;        // N^(-1) mod q, in Montgomery form
    coeff_t r_mod_q;           // R mod q  (Montgomery form of 1)
    coeff_t r2_mod_q;          // R^2 mod q (for to-Montgomery conversion)
};

// ============================================================================
// Modular Arithmetic Functions
// ============================================================================

/// Add two coefficients modulo q
/// @param a First coefficient
/// @param b Second coefficient
/// @param q Modulus
/// @return (a + b) mod q
inline coeff_t mod_add(coeff_t a, coeff_t b, coeff_t q) {
    coeff_t sum = a + b;
    // Conditional subtraction to avoid branching
    return sum >= q ? sum - q : sum;
}

/// Subtract two coefficients modulo q
/// @param a First coefficient
/// @param b Second coefficient
/// @param q Modulus
/// @return (a - b) mod q
inline coeff_t mod_sub(coeff_t a, coeff_t b, coeff_t q) {
    // Add q to handle negative results
    return a >= b ? a - b : a + q - b;
}

// ----------------------------------------------------------------------------
// Montgomery arithmetic, R = 2^32
//
// Every modulus used by this project is odd and fits in 32 bits, so with
// R = 2^32 the product of two Montgomery-form values (each < q < 2^32) fits
// exactly in 64 bits. No 128-bit emulation is required.
//
// The previous R = 2^64 helpers (montgomery_reduce / montgomery_mul) were
// removed: they shifted a 64-bit value right by 64 (undefined, folds to 0)
// and therefore could never produce a correct reduction. Do not reintroduce
// them; use mont_mul_32 / mont_redc_32 below.
// ----------------------------------------------------------------------------

/// Montgomery reduction (REDC) for moduli < 2^32 with R = 2^32.
/// @param t Input value, must satisfy t < q * 2^32
/// @param q Modulus (odd, < 2^32)
/// @param q_inv_neg -q^(-1) mod 2^32
/// @return (t * R^(-1)) mod q, in [0, q)
inline coeff_t mont_redc_32(coeff_t t, coeff_t q, uint32_t q_inv_neg) {
    uint32_t m = (uint32_t)(t & 0xFFFFFFFF) * q_inv_neg;  // mod 2^32
    coeff_t r = (t + (coeff_t)m * q) >> 32;
    return r >= q ? r - q : r;
}

/// Montgomery multiplication for moduli < 2^32 with R = 2^32.
/// Both inputs must be in Montgomery form and < q. Result is in Montgomery form.
/// q_inv_neg = -q^{-1} mod 2^32.
inline coeff_t mont_mul_32(coeff_t a, coeff_t b, coeff_t q, uint32_t q_inv_neg) {
    coeff_t t = a * b;                                   // exact in 64 bits since a,b < 2^32
    uint32_t m = (uint32_t)(t & 0xFFFFFFFF) * q_inv_neg;  // mod 2^32
    coeff_t r = (t + (coeff_t)m * q) >> 32;
    return r >= q ? r - q : r;
}

/// Barrett reduction: compute a mod q using precomputed constants
/// @param a Input value
/// @param q Modulus
/// @param mu Precomputed floor(2^(2k) / q) where k = ceil(log2(q))
/// @param k Bit length parameter
/// @return a mod q
inline coeff_t barrett_reduce(coeff_t a, coeff_t q, coeff_t mu, uint32_t k) {
    // Barrett reduction algorithm
    // q_hat = floor((a * mu) / 2^(2k))
    // r = a - q_hat * q
    // if r >= q then r = r - q
    
    coeff_t q_hat = (a * mu) >> (2 * k);
    coeff_t r = a - q_hat * q;
    return r >= q ? r - q : r;
}

/// Modular negation: compute -a mod q
/// @param a Input coefficient
/// @param q Modulus
/// @return (-a) mod q
inline coeff_t mod_neg(coeff_t a, coeff_t q) {
    return a == 0 ? 0 : q - a;
}

// ============================================================================
// Bit Manipulation Functions
// ============================================================================

/// Reverse the bits of an index for NTT bit-reversal permutation
/// @param x Input index
/// @param bits Number of bits to reverse
/// @return Bit-reversed index
inline index_t bit_reverse(index_t x, uint32_t bits) {
    index_t result = 0;
    for (uint32_t i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/// Compute log2 of a power of 2
/// @param n Input value (must be power of 2)
/// @return log2(n)
inline uint32_t log2_pow2(uint32_t n) {
    uint32_t log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

// ============================================================================
// Memory Access Helpers
// ============================================================================

/// Load coefficient with bounds checking (debug builds)
/// @param buffer Coefficient buffer
/// @param index Index to load
/// @param size Buffer size
/// @return Coefficient value
inline coeff_t safe_load(device const coeff_t* buffer, index_t index, index_t size) {
#ifdef DEBUG
    return index < size ? buffer[index] : 0;
#else
    return buffer[index];
#endif
}

/// Store coefficient with bounds checking (debug builds)
/// @param buffer Coefficient buffer
/// @param index Index to store
/// @param value Value to store
/// @param size Buffer size
inline void safe_store(device coeff_t* buffer, index_t index, coeff_t value, index_t size) {
#ifdef DEBUG
    if (index < size) {
        buffer[index] = value;
    }
#else
    buffer[index] = value;
#endif
}

// ============================================================================
// Thread Group Utilities
// ============================================================================

/// Synchronize threads within a threadgroup
inline void threadgroup_barrier() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Get the number of threads in the threadgroup
inline uint32_t get_threadgroup_size(uint3 threads_per_threadgroup) {
    return threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z;
}

/// Get the linear thread index within the threadgroup
inline uint32_t get_threadgroup_linear_index(uint3 thread_position_in_threadgroup,
                                             uint3 threads_per_threadgroup) {
    return thread_position_in_threadgroup.z * threads_per_threadgroup.x * threads_per_threadgroup.y +
           thread_position_in_threadgroup.y * threads_per_threadgroup.x +
           thread_position_in_threadgroup.x;
}
