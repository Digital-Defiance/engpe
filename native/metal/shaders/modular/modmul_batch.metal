//
// modmul_batch.metal
// Batch modular multiplication compute shader
//
// Performs parallel modular multiplication on arrays of coefficients.
// Uses Montgomery multiplication for efficiency.
//
// Design Reference: Section 2 - Modular Arithmetic Engine
// Requirements: 2.1, 2.6
//

#include "../common/fhe_common.metal"

// ============================================================================
// Kernel: Batch Modular Multiplication
// ============================================================================

/// Perform batch modular multiplication: result[i] = (a[i] * b[i]) mod q
///
/// This kernel multiplies corresponding elements from two arrays using
/// Montgomery multiplication. Each thread processes one multiplication.
///
/// Thread organization:
/// - Each thread processes one element
/// - Threads are organized in 1D grid
/// - Optimal threadgroup size: 256 threads
///
/// @param a First input array (Montgomery form)
/// @param b Second input array (Montgomery form)
/// @param result Output array (Montgomery form)
/// @param params Modular arithmetic parameters. Requires q < 2^32 and
///               params.inv_modulus == -q^(-1) mod 2^32 (see ModularParams).
/// @param size Number of elements to process
kernel void modmul_batch(
    device const coeff_t* a [[buffer(0)]],
    device const coeff_t* b [[buffer(1)]],
    device coeff_t* result [[buffer(2)]],
    constant ModularParams& params [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Bounds check
    if (gid >= size) return;
    
    // Load operands
    coeff_t a_val = a[gid];
    coeff_t b_val = b[gid];
    
    // Perform Montgomery multiplication (R = 2^32)
    coeff_t product = mont_mul_32(a_val, b_val, params.modulus,
                                  (uint32_t)params.inv_modulus);
    
    // Store result
    result[gid] = product;
}

// ============================================================================
// Kernel: Batch Modular Addition
// ============================================================================

/// Perform batch modular addition: result[i] = (a[i] + b[i]) mod q
///
/// @param a First input array
/// @param b Second input array
/// @param result Output array
/// @param modulus Modulus q
/// @param size Number of elements to process
kernel void modadd_batch(
    device const coeff_t* a [[buffer(0)]],
    device const coeff_t* b [[buffer(1)]],
    device coeff_t* result [[buffer(2)]],
    constant coeff_t& modulus [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    coeff_t sum = mod_add(a[gid], b[gid], modulus);
    result[gid] = sum;
}

// ============================================================================
// Kernel: Batch Modular Subtraction
// ============================================================================

/// Perform batch modular subtraction: result[i] = (a[i] - b[i]) mod q
///
/// @param a First input array
/// @param b Second input array
/// @param result Output array
/// @param modulus Modulus q
/// @param size Number of elements to process
kernel void modsub_batch(
    device const coeff_t* a [[buffer(0)]],
    device const coeff_t* b [[buffer(1)]],
    device coeff_t* result [[buffer(2)]],
    constant coeff_t& modulus [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    coeff_t diff = mod_sub(a[gid], b[gid], modulus);
    result[gid] = diff;
}

// ============================================================================
// Kernel: Scalar Modular Multiplication
// ============================================================================

/// Multiply array by scalar: result[i] = (a[i] * scalar) mod q
///
/// Useful for plaintext-ciphertext multiplication.
///
/// @param a Input array (Montgomery form)
/// @param scalar Scalar value (Montgomery form)
/// @param result Output array (Montgomery form)
/// @param params Modular arithmetic parameters
/// @param size Number of elements to process
kernel void modmul_scalar(
    device const coeff_t* a [[buffer(0)]],
    constant coeff_t& scalar [[buffer(1)]],
    device coeff_t* result [[buffer(2)]],
    constant ModularParams& params [[buffer(3)]],
    constant uint32_t& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    coeff_t product = mont_mul_32(a[gid], scalar, params.modulus,
                                  (uint32_t)params.inv_modulus);
    result[gid] = product;
}

// ============================================================================
// Kernel: Batch Modular Negation
// ============================================================================

/// Negate array elements: result[i] = (-a[i]) mod q
///
/// @param a Input array
/// @param result Output array
/// @param modulus Modulus q
/// @param size Number of elements to process
kernel void modneg_batch(
    device const coeff_t* a [[buffer(0)]],
    device coeff_t* result [[buffer(1)]],
    constant coeff_t& modulus [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    result[gid] = mod_neg(a[gid], modulus);
}
