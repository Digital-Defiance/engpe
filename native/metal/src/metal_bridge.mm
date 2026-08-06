/**
 * engpe Metal C ABI — thin wrappers around MetalComputeContext.
 * Keeps complexity out of Rust; each function is a straight passthrough.
 */

#include "metal_bridge.h"
#include "metal_compute.h"

#include <cstring>
#include <string>

using fhe_accelerate::metal::get_metal_context;
using fhe_accelerate::metal::MetalComputeContext;

static bool try_load_path(MetalComputeContext& ctx, const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
    return ctx.load_shaders(path);
}

bool engpe_metal_init(const char* metallib_path) {
    auto& ctx = get_metal_context();
    if (!ctx.is_available()) {
        return false;
    }
    if (ctx.has_pipeline("ntt_forward_gpu") && ctx.has_pipeline("ntt_inverse_gpu")) {
        return true;
    }
    if (try_load_path(ctx, metallib_path)) {
        return ctx.has_pipeline("ntt_forward_gpu");
    }
    // Search engpe-relative defaults.
    const char* fallbacks[] = {
        "native/fhe_shaders.metallib",
        "native/dist/shaders/fhe_shaders.metallib",
        "fhe_shaders.metallib",
        "dist/shaders/fhe_shaders.metallib",
        nullptr,
    };
    for (int i = 0; fallbacks[i] != nullptr; i++) {
        if (try_load_path(ctx, fallbacks[i]) && ctx.has_pipeline("ntt_forward_gpu")) {
            return true;
        }
    }
    return false;
}

bool engpe_metal_available(void) {
    auto& ctx = get_metal_context();
    return ctx.is_available()
        && ctx.has_pipeline("ntt_forward_gpu")
        && ctx.has_pipeline("ntt_inverse_gpu")
        && ctx.has_pipeline("ntt_twist");
}

bool engpe_metal_batch_ntt_forward(
    uint64_t* coeffs,
    size_t degree,
    size_t batch_size,
    uint64_t modulus,
    const uint64_t* omega_powers,
    const uint64_t* psi_powers) {
    if (modulus >= (1ULL << 32) || (modulus & 1ULL) == 0) {
        return false;
    }
    if (!engpe_metal_available()) {
        return false;
    }
    auto& ctx = get_metal_context();
    return ctx.batch_ntt_forward(
        coeffs, degree, batch_size, modulus, omega_powers, false, psi_powers);
}

bool engpe_metal_batch_ntt_inverse(
    uint64_t* coeffs,
    size_t degree,
    size_t batch_size,
    uint64_t modulus,
    const uint64_t* omega_inv_powers,
    const uint64_t* psi_inv_powers) {
    if (modulus >= (1ULL << 32) || (modulus & 1ULL) == 0) {
        return false;
    }
    if (!engpe_metal_available()) {
        return false;
    }
    auto& ctx = get_metal_context();
    return ctx.batch_ntt_inverse(
        coeffs, degree, batch_size, modulus, omega_inv_powers, false, psi_inv_powers);
}

bool engpe_metal_batch_modmul(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    size_t count,
    uint64_t modulus) {
    if (a == nullptr || b == nullptr || out == nullptr || modulus < 2) {
        return false;
    }
    auto& ctx = get_metal_context();
    if (ctx.is_available() && count >= 64 && ctx.has_pipeline("modmul_batch")) {
        ctx.batch_modmul(a, b, out, count, modulus);
        return true;
    }
    // CPU fallback for pointwise products.
    for (size_t i = 0; i < count; i++) {
        out[i] = (static_cast<__uint128_t>(a[i]) * b[i]) % modulus;
    }
    return true;
}

bool engpe_metal_keyswitch_mac(
    const uint64_t* digits_ntt,
    const uint64_t* evk_ntt,
    uint64_t* acc_out,
    size_t n_digits,
    size_t degree,
    uint64_t modulus) {
    if (digits_ntt == nullptr || evk_ntt == nullptr || acc_out == nullptr) {
        return false;
    }
    auto& ctx = get_metal_context();
    if (!ctx.is_available() || !ctx.has_pipeline("keyswitch_mac_batch")) {
        return false;
    }
    return ctx.keyswitch_mac_batch(
        digits_ntt, evk_ntt, acc_out, n_digits, degree, modulus);
}

bool engpe_metal_upload_evk(
    uint64_t cache_key,
    const uint64_t* data,
    size_t n_digits,
    size_t degree) {
    if (data == nullptr) {
        return false;
    }
    auto& ctx = get_metal_context();
    if (!ctx.is_available()) {
        return false;
    }
    return ctx.upload_evk(cache_key, data, n_digits, degree);
}

void engpe_metal_clear_evk_cache(void) {
    auto& ctx = get_metal_context();
    if (!ctx.is_available()) {
        return;
    }
    ctx.clear_evk_cache();
}

bool engpe_metal_keyswitch_mac_cached(
    uint64_t cache_key,
    const uint64_t* digits_ntt,
    uint64_t* acc_out,
    size_t n_digits,
    size_t degree,
    uint64_t modulus) {
    if (digits_ntt == nullptr || acc_out == nullptr) {
        return false;
    }
    auto& ctx = get_metal_context();
    if (!ctx.is_available() || !ctx.has_pipeline("keyswitch_mac_batch")) {
        return false;
    }
    return ctx.keyswitch_mac_cached(
        cache_key, digits_ntt, acc_out, n_digits, degree, modulus);
}

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
    uint64_t evk_key_a) {
    if (digits_coeff == nullptr || acc_b_out == nullptr || acc_a_out == nullptr) {
        return false;
    }
    auto& ctx = get_metal_context();
    if (!ctx.is_available() || !ctx.has_pipeline("keyswitch_fused")) {
        return false;
    }
    return ctx.keyswitch_fused(
        digits_coeff, acc_b_out, acc_a_out, n_digits, degree, modulus,
        omega_powers, omega_inv_powers, psi_powers, psi_inv_powers,
        evk_key_b, evk_key_a);
}

bool engpe_metal_mod_down_batch(
    const uint64_t* x,
    uint64_t* y,
    size_t count,
    uint64_t modulus,
    uint64_t p_inv) {
    if (x == nullptr || y == nullptr) {
        return false;
    }
    auto& ctx = get_metal_context();
    if (!ctx.is_available() || !ctx.has_pipeline("mod_down_batch")) {
        return false;
    }
    return ctx.mod_down_batch(x, y, count, modulus, p_inv);
}
