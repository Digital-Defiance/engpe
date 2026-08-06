//
// modmul_direct.metal
// Direct modular multiplication without Montgomery form
//
// Uses Barrett reduction for efficient modular multiplication.
// This avoids the complexity of Montgomery form conversion.
//

#include <metal_stdlib>
using namespace metal;

typedef ulong coeff_t;

/// Barrett reduction parameters
struct BarrettParams {
    coeff_t modulus;           // Prime modulus q
    coeff_t mu;                // Precomputed floor(2^(2k) / q)
    uint32_t k;                // Bit length of modulus
    uint32_t padding;          // Alignment padding
};

/// Barrett reduction for 128-bit product
/// Uses the approximation: q ≈ floor(x * mu / 2^(2k))
inline coeff_t barrett_reduce_128(coeff_t x_hi, coeff_t x_lo, 
                                   coeff_t modulus, coeff_t mu, uint32_t k) {
    // For 64-bit modulus, we need to handle 128-bit intermediate values
    // Approximation: q ≈ (x >> (k-1)) * mu >> (k+1)
    
    // Compute x >> (k-1)
    coeff_t x_shifted;
    if (k <= 64) {
        x_shifted = (x_hi << (65 - k)) | (x_lo >> (k - 1));
    } else {
        x_shifted = x_hi >> (k - 65);
    }
    
    // Compute q_approx = (x_shifted * mu) >> (k+1)
    // This is an approximation that may be off by at most 2
    coeff_t q_approx = (x_shifted * mu) >> (k + 1);
    
    // Compute r = x - q_approx * modulus
    // We only need the low 64 bits since result < 2*modulus
    coeff_t r = x_lo - q_approx * modulus;
    
    // At most 2 corrections needed
    if (r >= modulus) r -= modulus;
    if (r >= modulus) r -= modulus;
    
    return r;
}

/// Multiply two 64-bit values and return 128-bit result
inline void mul_64x64(coeff_t a, coeff_t b, thread coeff_t& hi, thread coeff_t& lo) {
    // Split into 32-bit parts
    uint32_t a_lo = a & 0xFFFFFFFF;
    uint32_t a_hi = a >> 32;
    uint32_t b_lo = b & 0xFFFFFFFF;
    uint32_t b_hi = b >> 32;
    
    // Compute partial products
    coeff_t p_ll = (coeff_t)a_lo * b_lo;
    coeff_t p_lh = (coeff_t)a_lo * b_hi;
    coeff_t p_hl = (coeff_t)a_hi * b_lo;
    coeff_t p_hh = (coeff_t)a_hi * b_hi;
    
    // Combine: result = p_hh * 2^64 + (p_lh + p_hl) * 2^32 + p_ll
    coeff_t mid = p_lh + p_hl;
    coeff_t carry1 = (mid < p_lh) ? 1UL : 0UL;
    
    lo = p_ll + (mid << 32);
    coeff_t carry2 = (lo < p_ll) ? 1UL : 0UL;
    
    hi = p_hh + (mid >> 32) + (carry1 << 32) + carry2;
}

/// Direct modular multiplication using Barrett reduction
/// result = (a * b) mod modulus
kernel void modmul_direct_batch(
    device const coeff_t* a [[buffer(0)]],
    device const coeff_t* b [[buffer(1)]],
    device coeff_t* result [[buffer(2)]],
    constant BarrettParams& params [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    coeff_t a_val = a[gid];
    coeff_t b_val = b[gid];
    
    // Compute 128-bit product
    coeff_t prod_hi, prod_lo;
    mul_64x64(a_val, b_val, prod_hi, prod_lo);
    
    // Barrett reduction
    result[gid] = barrett_reduce_128(prod_hi, prod_lo, 
                                      params.modulus, params.mu, params.k);
}

/// Batch modular addition: result[i] = (a[i] + b[i]) mod q
kernel void modadd_direct_batch(
    device const coeff_t* a [[buffer(0)]],
    device const coeff_t* b [[buffer(1)]],
    device coeff_t* result [[buffer(2)]],
    constant coeff_t& modulus [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    coeff_t sum = a[gid] + b[gid];
    result[gid] = sum >= modulus ? sum - modulus : sum;
}

/// Batch modular subtraction: result[i] = (a[i] - b[i]) mod q
kernel void modsub_direct_batch(
    device const coeff_t* a [[buffer(0)]],
    device const coeff_t* b [[buffer(1)]],
    device coeff_t* result [[buffer(2)]],
    constant coeff_t& modulus [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    coeff_t a_val = a[gid];
    coeff_t b_val = b[gid];
    result[gid] = a_val >= b_val ? a_val - b_val : a_val + modulus - b_val;
}
