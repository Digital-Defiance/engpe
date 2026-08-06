/**
 * C ABI for engpe Metal batch negacyclic NTT.
 * Wraps fhe_accelerate::metal::MetalComputeContext from node-fhe-accelerate.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Load metallib from path (or search defaults). Idempotent. */
bool engpe_metal_init(const char* metallib_path);

/** True when a Metal device is up and NTT pipelines are loaded. */
bool engpe_metal_available(void);

/**
 * In-place batch forward negacyclic NTT.
 * coeffs: [batch_size * degree] ordinary residues.
 * omega_powers / psi_powers: length `degree` flat tables (ordinary residues).
 * Returns false if refused (q >= 2^32, missing pipelines, bad args).
 */
bool engpe_metal_batch_ntt_forward(
    uint64_t* coeffs,
    size_t degree,
    size_t batch_size,
    uint64_t modulus,
    const uint64_t* omega_powers,
    const uint64_t* psi_powers);

/** In-place batch inverse negacyclic NTT. */
bool engpe_metal_batch_ntt_inverse(
    uint64_t* coeffs,
    size_t degree,
    size_t batch_size,
    uint64_t modulus,
    const uint64_t* omega_inv_powers,
    const uint64_t* psi_inv_powers);

/**
 * Pointwise modmul: out[i] = a[i]*b[i] mod q (CPU Barrett-style fallback ok).
 * Prefer GPU when available; always fills `out`.
 */
bool engpe_metal_batch_modmul(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    size_t count,
    uint64_t modulus);

/**
 * Key-switch MAC in NTT domain:
 *   acc[i] = Σ_d digits[d*N+i] * evk[d*N+i]  (mod q)
 * digits/evk are ordinary residues; q must be < 2^31.
 */
bool engpe_metal_keyswitch_mac(
    const uint64_t* digits_ntt,
    const uint64_t* evk_ntt,
    uint64_t* acc_out,
    size_t n_digits,
    size_t degree,
    uint64_t modulus);

/** Upload a static EVK NTT buffer (n_digits * degree) into a resident MTLBuffer. */
bool engpe_metal_upload_evk(
    uint64_t cache_key,
    const uint64_t* data,
    size_t n_digits,
    size_t degree);

/** Release every resident EVK buffer. Used to bound cache growth across re-keying. */
void engpe_metal_clear_evk_cache(void);

/** KS MAC using a previously uploaded EVK buffer. */
bool engpe_metal_keyswitch_mac_cached(
    uint64_t cache_key,
    const uint64_t* digits_ntt,
    uint64_t* acc_out,
    size_t n_digits,
    size_t degree,
    uint64_t modulus);

/**
 * Fused Digit-NTT → KS-MAC×2 → INTT×2 in one MTLCommandBuffer.
 * digits_coeff: [n_digits * degree] ordinary residues (not read back).
 * EVKs must be resident under evk_key_b / evk_key_a.
 */
bool engpe_metal_keyswitch_fused(
    const uint64_t* digits_coeff,
    uint64_t* acc_b_out,
    uint64_t* acc_a_out,
    size_t n_digits,
    size_t degree,
    uint64_t modulus,
    const uint64_t* omega_powers,
    const uint64_t* omega_inv_powers,
    const uint64_t* psi_powers,
    const uint64_t* psi_inv_powers,
    uint64_t evk_key_b,
    uint64_t evk_key_a);

/** Hybrid Stage D: y[i] = x[i] * p_inv mod q. */
bool engpe_metal_mod_down_batch(
    const uint64_t* x,
    uint64_t* y,
    size_t count,
    uint64_t modulus,
    uint64_t p_inv);

#ifdef __cplusplus
}
#endif
