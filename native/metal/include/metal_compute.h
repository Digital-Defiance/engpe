/**
 * Metal GPU Compute Backend for FHE Operations
 * 
 * Provides GPU-accelerated batch operations for:
 * - Batch NTT (forward and inverse)
 * - Batch modular multiplication
 * - Batch polynomial operations
 * 
 * Optimized for M4 Max with 40 GPU cores.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace fhe_accelerate {
namespace metal {

/**
 * Metal compute context - manages device, command queue, and pipelines
 */
class MetalComputeContext {
public:
    MetalComputeContext();
    ~MetalComputeContext();
    
    // Prevent copying
    MetalComputeContext(const MetalComputeContext&) = delete;
    MetalComputeContext& operator=(const MetalComputeContext&) = delete;
    
    bool is_available() const { return device_ != nullptr; }
    
    // Device info
    std::string device_name() const;
    size_t max_buffer_size() const;
    size_t max_threadgroup_size() const;
    uint32_t gpu_cores() const { return gpu_cores_; }
    
    // Buffer management
    void* create_buffer(size_t size);
    void release_buffer(void* buffer);
    void copy_to_buffer(void* buffer, const void* data, size_t size);
    void copy_from_buffer(const void* buffer, void* data, size_t size);
    
    // Pipeline management
    bool load_shaders(const std::string& metallib_path);
    bool has_pipeline(const std::string& name) const;
    
    // Batch modular multiplication
    // result[i] = (a[i] * b[i]) mod modulus
    void batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus);
    
    // Batch modular addition
    void batch_modadd(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus);

    /// Key-switch MAC: acc[i] = Σ_d digits[d*N+i] * evk[d*N+i] mod q.
    /// Returns false if the KS pipeline is missing or args are invalid.
    bool keyswitch_mac_batch(const uint64_t* digits_ntt, const uint64_t* evk_ntt,
                             uint64_t* acc_out, size_t n_digits, size_t degree,
                             uint64_t modulus);

    /// Cache a static EVK NTT slab (n_digits * degree) under `cache_key`.
    bool upload_evk(uint64_t cache_key, const uint64_t* data, size_t n_digits,
                    size_t degree);

    /// Release every resident EVK slab.
    void clear_evk_cache();

    /// KS MAC against a previously uploaded EVK slab.
    bool keyswitch_mac_cached(uint64_t cache_key, const uint64_t* digits_ntt,
                              uint64_t* acc_out, size_t n_digits, size_t degree,
                              uint64_t modulus);

    /// Fused Digit-NTT → KS-MAC×2 → INTT×2 in a single MTLCommandBuffer.
    /// `digits_coeff` is ordinary-residue [n_digits * degree]; overwritten with
    /// NTT-domain digits on the GPU (not read back). Outputs `acc_b` / `acc_a`
    /// in coefficient domain. EVKs must already be uploaded under the given keys.
    bool keyswitch_fused(
        const uint64_t* digits_coeff, uint64_t* acc_b_out, uint64_t* acc_a_out,
        size_t n_digits, size_t degree, uint64_t modulus,
        const uint64_t* omega_powers, const uint64_t* omega_inv_powers,
        const uint64_t* psi_powers, const uint64_t* psi_inv_powers,
        uint64_t evk_key_b, uint64_t evk_key_a);

    /// Optional Stage D: y[i] = x[i] * p_inv mod q (hybrid mod-down scale).
    bool mod_down_batch(const uint64_t* x, uint64_t* y, size_t count,
                        uint64_t modulus, uint64_t p_inv);
    
    // ------------------------------------------------------------------------
    // Batch NTT
    //
    // CONTRACT
    //   coeffs   [batch_size][degree] ordinary residues in [0, q). Overwritten
    //            in place with ordinary residues; the Montgomery domain is
    //            entered and left on the GPU, inside one command buffer.
    //   degree   power of two, >= 2.
    //   modulus  must be ODD and < 2^32. The kernels use Montgomery arithmetic
    //            with R = 2^32; anything else is refused (returns false) rather
    //            than silently producing wrong output.
    //   twiddles a flat table of successive powers of OMEGA, the primitive
    //            N-th root of unity: twiddles[i] == omega^i mod q for i in
    //            [0, degree). With psi the primitive 2N-th root, omega == psi^2.
    //            This is exactly NTTProcessor::get_twiddles().forward (or
    //            .inverse, which holds omega^(-i), for the inverse transform).
    //            Powers of psi here would halve every butterfly exponent and
    //            silently produce a degenerate transform. Pass
    //            twiddles_in_montgomery_form = TwiddleFactors::in_montgomery_form
    //            so the host knows whether to convert; the host reshapes the
    //            table into the stage-major layout the kernels expect.
    //   psi_powers
    //            `degree` entries, ordinary residues. SELECTS WHICH TRANSFORM
    //            IS COMPUTED:
    //              nullptr  -> CYCLIC transform over Z_q[X]/(X^N - 1)
    //              non-null -> NEGACYCLIC transform over Z_q[X]/(X^N + 1)
    //            For the forward transform pass psi^i
    //            (NTTProcessor::get_twiddles().psi_powers); for the inverse
    //            pass psi^(-i) (.psi_inv_powers). The twist is applied ON THE
    //            GPU, fused into the Montgomery domain crossing that already
    //            runs, so it adds no dispatch and no extra synchronisation.
    //            Requires the "ntt_twist" pipeline.
    //
    //            DELIBERATELY NOT DEFAULTED. A default would let a call site
    //            silently pick the cyclic transform while the rest of the
    //            system computes the negacyclic one -- exactly the class of
    //            defect this contract exists to prevent. Pass nullptr
    //            explicitly if you really do want the cyclic transform.
    //
    // Returns true when the GPU produced a result, false when the request was
    // refused (unsupported modulus, missing pipelines, bad arguments) so the
    // caller can fall back to the CPU path.
    // ------------------------------------------------------------------------
    bool batch_ntt_forward(uint64_t* coeffs, size_t degree, size_t batch_size,
                           uint64_t modulus, const uint64_t* twiddles,
                           bool twiddles_in_montgomery_form,
                           const uint64_t* psi_powers);
    
    bool batch_ntt_inverse(uint64_t* coeffs, size_t degree, size_t batch_size,
                           uint64_t modulus, const uint64_t* inv_twiddles,
                           bool twiddles_in_montgomery_form,
                           const uint64_t* psi_inv_powers);
    
    // Batch polynomial multiplication (via NTT)
    // result[i] = poly_a[i] * poly_b[i] in NTT domain
    void batch_poly_mul(const uint64_t* poly_a, const uint64_t* poly_b,
                        uint64_t* result, size_t degree, size_t batch_size,
                        uint64_t modulus);
    
    // Synchronize - wait for all GPU operations to complete
    void synchronize();
    
private:
#ifdef __APPLE__
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;
    
    // Compute pipelines
    id<MTLComputePipelineState> modmul_pipeline_;
    id<MTLComputePipelineState> modadd_pipeline_;
    id<MTLComputePipelineState> ntt_stage_pipeline_;
    id<MTLComputePipelineState> ntt_bitrev_pipeline_;
    id<MTLComputePipelineState> ntt_batch_pipeline_;
    id<MTLComputePipelineState> ntt_to_mont_pipeline_;
    id<MTLComputePipelineState> ntt_from_mont_pipeline_;
    id<MTLComputePipelineState> ntt_twist_pipeline_;
    id<MTLComputePipelineState> ntt_inverse_stage_pipeline_;
    id<MTLComputePipelineState> ntt_inverse_scale_pipeline_;
    id<MTLComputePipelineState> keyswitch_mac_pipeline_;
    id<MTLComputePipelineState> mod_down_pipeline_;
    // Resident EVK NTT buffers keyed by host-provided cache_key.
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* evk_buffers_;
#else
    void* device_;
    void* command_queue_;
    void* library_;
    void* modmul_pipeline_;
    void* modadd_pipeline_;
    void* ntt_stage_pipeline_;
    void* ntt_bitrev_pipeline_;
    void* ntt_batch_pipeline_;
    void* ntt_to_mont_pipeline_;
    void* ntt_from_mont_pipeline_;
    void* ntt_twist_pipeline_;
    void* ntt_inverse_stage_pipeline_;
    void* ntt_inverse_scale_pipeline_;
    void* keyswitch_mac_pipeline_;
    void* mod_down_pipeline_;
    void* evk_buffers_;
#endif
    
    uint32_t gpu_cores_;
    size_t max_buffer_size_;
    size_t max_threadgroup_size_;
    
    bool create_pipelines();
    
    // Shared implementation of the forward/inverse batch transform. Encodes the
    // Montgomery conversion, bit-reversal, all log2(N) butterfly stages, the
    // inverse scaling pass and the conversion back into a SINGLE command buffer
    // with one commit/wait.
    bool ntt_execute(uint64_t* coeffs, size_t degree, size_t batch_size,
                     uint64_t modulus, const uint64_t* twiddles,
                     bool twiddles_in_montgomery_form, bool inverse,
                     const uint64_t* psi_powers);
};

/**
 * GPU-accelerated batch operations
 * 
 * These functions automatically choose between CPU and GPU based on workload size.
 * For small batches, CPU is faster due to GPU dispatch overhead.
 * For large batches (>4096 elements), GPU provides significant speedup.
 */

// Threshold for GPU dispatch (elements)
constexpr size_t GPU_DISPATCH_THRESHOLD = 4096;

// Get global Metal context (singleton)
MetalComputeContext& get_metal_context();

// Check if Metal is available
bool metal_available();

// Batch modular multiplication with automatic dispatch
void gpu_batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus);

// Batch NTT with automatic dispatch.
// `psi_powers` follows the same convention as MetalComputeContext::batch_ntt_*
// and is likewise not defaulted: nullptr selects the cyclic transform, a
// non-null table of `degree` entries (psi^i forward, psi^(-i) inverse) selects
// the negacyclic one.
void gpu_batch_ntt(uint64_t* coeffs, size_t degree, size_t batch_size,
                   uint64_t modulus, const uint64_t* twiddles, bool inverse,
                   const uint64_t* psi_powers);

} // namespace metal
} // namespace fhe_accelerate
