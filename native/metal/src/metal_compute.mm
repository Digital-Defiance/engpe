/**
 * Metal GPU Compute Backend Implementation
 * 
 * Provides GPU-accelerated batch operations for FHE.
 * Optimized for M4 Max with 40 GPU cores.
 */

#include "metal_compute.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstddef>
#include <vector>
#include <algorithm>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace fhe_accelerate {
namespace metal {

// ============================================================================
// MetalComputeContext Implementation
// ============================================================================

MetalComputeContext::MetalComputeContext()
    : device_(nil)
    , command_queue_(nil)
    , library_(nil)
    , modmul_pipeline_(nil)
    , modadd_pipeline_(nil)
    , ntt_stage_pipeline_(nil)
    , ntt_bitrev_pipeline_(nil)
    , ntt_batch_pipeline_(nil)
    , ntt_to_mont_pipeline_(nil)
    , ntt_from_mont_pipeline_(nil)
    , ntt_twist_pipeline_(nil)
    , ntt_inverse_stage_pipeline_(nil)
    , ntt_inverse_scale_pipeline_(nil)
    , keyswitch_mac_pipeline_(nil)
    , mod_down_pipeline_(nil)
    , evk_buffers_(nil)
    , gpu_cores_(0)
    , max_buffer_size_(0)
    , max_threadgroup_size_(0)
{
#ifdef __APPLE__
    // Get default Metal device
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
        std::cerr << "Metal: No GPU device found" << std::endl;
        return;
    }
    
    // Create command queue
    command_queue_ = [device_ newCommandQueue];
    if (command_queue_ == nil) {
        std::cerr << "Metal: Failed to create command queue" << std::endl;
        device_ = nil;
        return;
    }

    evk_buffers_ = [NSMutableDictionary new];
    
    // Get device info
    max_buffer_size_ = [device_ maxBufferLength];
    max_threadgroup_size_ = 1024;  // M4 Max supports up to 1024 threads per threadgroup
    
    // Detect GPU cores from device name
    NSString* name = [device_ name];
    if ([name containsString:@"M4 Max"]) {
        gpu_cores_ = 40;
    } else if ([name containsString:@"M4 Pro"]) {
        gpu_cores_ = 20;
    } else if ([name containsString:@"M4"]) {
        gpu_cores_ = 10;
    } else if ([name containsString:@"M3 Max"]) {
        gpu_cores_ = 40;
    } else if ([name containsString:@"M3 Pro"]) {
        gpu_cores_ = 18;
    } else {
        gpu_cores_ = 8;  // Default
    }
    
    std::cout << "Metal: Initialized with " << [name UTF8String] 
              << " (" << gpu_cores_ << " GPU cores)" << std::endl;
    std::cout << "Metal: Max buffer size: " << (max_buffer_size_ / 1024 / 1024) << " MB" << std::endl;
    
    // Try to load shaders
    if (!load_shaders("fhe_shaders.metallib")) {
        // Try alternate paths
        if (!load_shaders("dist/shaders/fhe_shaders.metallib")) {
            std::cerr << "Metal: Warning - shaders not loaded, GPU compute unavailable" << std::endl;
        }
    }
#endif
}

MetalComputeContext::~MetalComputeContext() {
#ifdef __APPLE__
    // ARC handles cleanup
    modmul_pipeline_ = nil;
    modadd_pipeline_ = nil;
    ntt_stage_pipeline_ = nil;
    ntt_bitrev_pipeline_ = nil;
    ntt_batch_pipeline_ = nil;
    ntt_to_mont_pipeline_ = nil;
    ntt_from_mont_pipeline_ = nil;
    ntt_twist_pipeline_ = nil;
    ntt_inverse_stage_pipeline_ = nil;
    ntt_inverse_scale_pipeline_ = nil;
    keyswitch_mac_pipeline_ = nil;
    mod_down_pipeline_ = nil;
    if (evk_buffers_ != nil) {
        [evk_buffers_ removeAllObjects];
        evk_buffers_ = nil;
    }
    library_ = nil;
    command_queue_ = nil;
    device_ = nil;
#endif
}

std::string MetalComputeContext::device_name() const {
#ifdef __APPLE__
    if (device_ == nil) return "No device";
    return [[device_ name] UTF8String];
#else
    return "Metal not available";
#endif
}

size_t MetalComputeContext::max_buffer_size() const {
    return max_buffer_size_;
}

size_t MetalComputeContext::max_threadgroup_size() const {
    return max_threadgroup_size_;
}

void* MetalComputeContext::create_buffer(size_t size) {
#ifdef __APPLE__
    if (device_ == nil) return nullptr;
    
    // Use shared memory for unified memory architecture (Apple Silicon)
    id<MTLBuffer> buffer = [device_ newBufferWithLength:size
                                                options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buffer;
#else
    return nullptr;
#endif
}

void MetalComputeContext::release_buffer(void* buffer) {
#ifdef __APPLE__
    if (buffer != nullptr) {
        id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)buffer;
        mtl_buffer = nil;
    }
#endif
}

void MetalComputeContext::copy_to_buffer(void* buffer, const void* data, size_t size) {
#ifdef __APPLE__
    if (buffer == nullptr) return;
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer;
    memcpy([mtl_buffer contents], data, size);
#endif
}

void MetalComputeContext::copy_from_buffer(const void* buffer, void* data, size_t size) {
#ifdef __APPLE__
    if (buffer == nullptr) return;
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer;
    memcpy(data, [mtl_buffer contents], size);
#endif
}

bool MetalComputeContext::load_shaders(const std::string& metallib_path) {
#ifdef __APPLE__
    if (device_ == nil) return false;
    
    NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
    NSError* error = nil;
    
    // Check if file exists
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        return false;
    }
    
    NSURL* url = [NSURL fileURLWithPath:path];
    library_ = [device_ newLibraryWithURL:url error:&error];
    
    if (library_ == nil) {
        if (error != nil) {
            std::cerr << "Metal: Failed to load library: " 
                      << [[error localizedDescription] UTF8String] << std::endl;
        }
        return false;
    }
    
    std::cout << "Metal: Loaded shaders from " << metallib_path << std::endl;
    
    // List available functions
    NSArray<NSString*>* functions = [library_ functionNames];
    std::cout << "Metal: Available kernels: ";
    for (NSString* name in functions) {
        std::cout << [name UTF8String] << " ";
    }
    std::cout << std::endl;
    
    return create_pipelines();
#else
    return false;
#endif
}

bool MetalComputeContext::create_pipelines() {
#ifdef __APPLE__
    if (library_ == nil) return false;
    
    NSError* error = nil;
    
    // Create modmul pipeline - prefer direct Barrett version
    id<MTLFunction> modmul_func = [library_ newFunctionWithName:@"modmul_direct_batch"];
    if (modmul_func == nil) {
        // Fallback to Montgomery version
        modmul_func = [library_ newFunctionWithName:@"modmul_batch"];
    }
    if (modmul_func != nil) {
        modmul_pipeline_ = [device_ newComputePipelineStateWithFunction:modmul_func error:&error];
        if (modmul_pipeline_ != nil) {
            std::cout << "Metal: Created modmul pipeline (max threads: " 
                      << [modmul_pipeline_ maxTotalThreadsPerThreadgroup] << ")" << std::endl;
        }
    }
    
    // Create modadd pipeline
    id<MTLFunction> modadd_func = [library_ newFunctionWithName:@"modadd_batch"];
    if (modadd_func != nil) {
        modadd_pipeline_ = [device_ newComputePipelineStateWithFunction:modadd_func error:&error];
    }
    
    // Create NTT stage pipeline
    id<MTLFunction> ntt_stage_func = [library_ newFunctionWithName:@"ntt_forward_stage"];
    if (ntt_stage_func != nil) {
        ntt_stage_pipeline_ = [device_ newComputePipelineStateWithFunction:ntt_stage_func error:&error];
        if (ntt_stage_pipeline_ != nil) {
            std::cout << "Metal: Created ntt_forward_stage pipeline" << std::endl;
        }
    }
    
    // Create NTT bit-reversal pipeline
    id<MTLFunction> ntt_bitrev_func = [library_ newFunctionWithName:@"ntt_bit_reverse"];
    if (ntt_bitrev_func != nil) {
        ntt_bitrev_pipeline_ = [device_ newComputePipelineStateWithFunction:ntt_bitrev_func error:&error];
    }
    
    // Create batch NTT pipeline (threadgroup-memory variant, degree <= 1024 only)
    id<MTLFunction> ntt_batch_func = [library_ newFunctionWithName:@"ntt_forward_batch"];
    if (ntt_batch_func != nil) {
        ntt_batch_pipeline_ = [device_ newComputePipelineStateWithFunction:ntt_batch_func error:&error];
    }
    
    // Create Montgomery domain conversion pipelines
    id<MTLFunction> to_mont_func = [library_ newFunctionWithName:@"ntt_to_montgomery"];
    if (to_mont_func != nil) {
        ntt_to_mont_pipeline_ = [device_ newComputePipelineStateWithFunction:to_mont_func error:&error];
    }
    id<MTLFunction> from_mont_func = [library_ newFunctionWithName:@"ntt_from_montgomery"];
    if (from_mont_func != nil) {
        ntt_from_mont_pipeline_ = [device_ newComputePipelineStateWithFunction:from_mont_func error:&error];
    }
    
    // Fused negacyclic twist + Montgomery domain crossing
    id<MTLFunction> twist_func = [library_ newFunctionWithName:@"ntt_twist"];
    if (twist_func != nil) {
        ntt_twist_pipeline_ = [device_ newComputePipelineStateWithFunction:twist_func error:&error];
    }
    
    // Create inverse NTT pipelines
    id<MTLFunction> inv_stage_func = [library_ newFunctionWithName:@"ntt_inverse_stage"];
    if (inv_stage_func != nil) {
        ntt_inverse_stage_pipeline_ = [device_ newComputePipelineStateWithFunction:inv_stage_func error:&error];
    }
    id<MTLFunction> inv_scale_func = [library_ newFunctionWithName:@"ntt_inverse_scale"];
    if (inv_scale_func != nil) {
        ntt_inverse_scale_pipeline_ = [device_ newComputePipelineStateWithFunction:inv_scale_func error:&error];
    }

    id<MTLFunction> ks_mac_func = [library_ newFunctionWithName:@"keyswitch_mac_batch"];
    if (ks_mac_func != nil) {
        keyswitch_mac_pipeline_ = [device_ newComputePipelineStateWithFunction:ks_mac_func error:&error];
        if (keyswitch_mac_pipeline_ != nil) {
            std::cout << "Metal: Created keyswitch_mac_batch pipeline" << std::endl;
        }
    }

    id<MTLFunction> mod_down_func = [library_ newFunctionWithName:@"mod_down_batch"];
    if (mod_down_func != nil) {
        mod_down_pipeline_ = [device_ newComputePipelineStateWithFunction:mod_down_func error:&error];
        if (mod_down_pipeline_ != nil) {
            std::cout << "Metal: Created mod_down_batch pipeline" << std::endl;
        }
    }
    
    if (ntt_stage_pipeline_ != nil) {
        std::cout << "Metal: NTT pipelines - to_mont:" << (ntt_to_mont_pipeline_ != nil)
                  << " bitrev:" << (ntt_bitrev_pipeline_ != nil)
                  << " fwd_stage:" << (ntt_stage_pipeline_ != nil)
                  << " inv_stage:" << (ntt_inverse_stage_pipeline_ != nil)
                  << " inv_scale:" << (ntt_inverse_scale_pipeline_ != nil)
                  << " from_mont:" << (ntt_from_mont_pipeline_ != nil)
                  << " twist:" << (ntt_twist_pipeline_ != nil)
                  << " ks_mac:" << (keyswitch_mac_pipeline_ != nil) << std::endl;
    }
    
    return modmul_pipeline_ != nil || ntt_stage_pipeline_ != nil;
#else
    return false;
#endif
}

bool MetalComputeContext::has_pipeline(const std::string& name) const {
#ifdef __APPLE__
    if (name == "modmul_batch") return modmul_pipeline_ != nil;
    if (name == "modadd_batch") return modadd_pipeline_ != nil;
    if (name == "ntt_forward_stage") return ntt_stage_pipeline_ != nil;
    if (name == "ntt_bit_reverse") return ntt_bitrev_pipeline_ != nil;
    if (name == "ntt_forward_batch") return ntt_batch_pipeline_ != nil;
    if (name == "ntt_to_montgomery") return ntt_to_mont_pipeline_ != nil;
    if (name == "ntt_from_montgomery") return ntt_from_mont_pipeline_ != nil;
    if (name == "ntt_twist") return ntt_twist_pipeline_ != nil;
    if (name == "ntt_inverse_stage") return ntt_inverse_stage_pipeline_ != nil;
    if (name == "ntt_inverse_scale") return ntt_inverse_scale_pipeline_ != nil;
    if (name == "keyswitch_mac_batch") return keyswitch_mac_pipeline_ != nil;
    if (name == "mod_down_batch") return mod_down_pipeline_ != nil;
    if (name == "keyswitch_fused") {
        return keyswitch_mac_pipeline_ != nil &&
               ntt_stage_pipeline_ != nil && ntt_bitrev_pipeline_ != nil &&
               ntt_to_mont_pipeline_ != nil && ntt_from_mont_pipeline_ != nil &&
               ntt_twist_pipeline_ != nil && ntt_inverse_stage_pipeline_ != nil &&
               ntt_inverse_scale_pipeline_ != nil;
    }
    // Complete GPU forward transform: every kernel the single-command-buffer
    // path needs. ntt_twist is required because the negacyclic transform is the
    // transform this backend is used for; without it only the cyclic core could
    // run, and a caller expecting negacyclic would get the wrong answer.
    if (name == "ntt_forward_gpu") {
        return ntt_stage_pipeline_ != nil && ntt_bitrev_pipeline_ != nil &&
               ntt_to_mont_pipeline_ != nil && ntt_from_mont_pipeline_ != nil &&
               ntt_twist_pipeline_ != nil;
    }
    if (name == "ntt_inverse_gpu") {
        return ntt_inverse_stage_pipeline_ != nil && ntt_bitrev_pipeline_ != nil &&
               ntt_inverse_scale_pipeline_ != nil && ntt_to_mont_pipeline_ != nil &&
               ntt_from_mont_pipeline_ != nil && ntt_twist_pipeline_ != nil;
    }
#endif
    return false;
}

void MetalComputeContext::synchronize() {
#ifdef __APPLE__
    // Create a command buffer and wait for completion
    if (command_queue_ == nil) return;
    
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    [cmd commit];
    [cmd waitUntilCompleted];
#endif
}

// ============================================================================
// Batch Operations
// ============================================================================

void MetalComputeContext::batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                                        size_t count, uint64_t modulus) {
#ifdef __APPLE__
    if (modmul_pipeline_ == nil || count == 0) return;
    
    size_t buffer_size = count * sizeof(uint64_t);
    
    // Create buffers with shared memory (zero-copy on Apple Silicon)
    id<MTLBuffer> buffer_a = [device_ newBufferWithBytes:a
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_b = [device_ newBufferWithBytes:b
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_result = [device_ newBufferWithLength:buffer_size
                                                       options:MTLResourceStorageModeShared];
    
    // Create Barrett params buffer
    struct {
        uint64_t modulus;
        uint64_t mu;
        uint32_t k;
        uint32_t padding;
    } params;
    
    params.modulus = modulus;
    params.k = 64 - __builtin_clzll(modulus);
    
    // Compute mu = floor(2^(2k) / modulus)
    if (params.k <= 32) {
        params.mu = (1ULL << (2 * params.k)) / modulus;
    } else {
        __uint128_t numerator = static_cast<__uint128_t>(1) << (2 * params.k);
        params.mu = static_cast<uint64_t>(numerator / modulus);
    }
    params.padding = 0;
    
    id<MTLBuffer> buffer_params = [device_ newBufferWithBytes:&params
                                                       length:sizeof(params)
                                                      options:MTLResourceStorageModeShared];
    
    uint32_t size = static_cast<uint32_t>(count);
    id<MTLBuffer> buffer_size_param = [device_ newBufferWithBytes:&size
                                                           length:sizeof(size)
                                                          options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    
    [encoder setComputePipelineState:modmul_pipeline_];
    [encoder setBuffer:buffer_a offset:0 atIndex:0];
    [encoder setBuffer:buffer_b offset:0 atIndex:1];
    [encoder setBuffer:buffer_result offset:0 atIndex:2];
    [encoder setBuffer:buffer_params offset:0 atIndex:3];
    [encoder setBuffer:buffer_size_param offset:0 atIndex:4];
    
    // Dispatch threads - use larger threadgroups for better GPU utilization
    NSUInteger threadgroup_size = std::min((NSUInteger)1024, [modmul_pipeline_ maxTotalThreadsPerThreadgroup]);
    NSUInteger num_threadgroups = (count + threadgroup_size - 1) / threadgroup_size;
    
    [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    // Copy result back
    memcpy(result, [buffer_result contents], buffer_size);
#endif
}

bool MetalComputeContext::keyswitch_mac_batch(const uint64_t* digits_ntt,
                                              const uint64_t* evk_ntt,
                                              uint64_t* acc_out, size_t n_digits,
                                              size_t degree, uint64_t modulus) {
#ifdef __APPLE__
    if (keyswitch_mac_pipeline_ == nil || digits_ntt == nullptr || evk_ntt == nullptr ||
        acc_out == nullptr || n_digits == 0 || degree == 0 || modulus < 2 ||
        modulus >= (1ULL << 31)) {
        return false;
    }

    size_t digit_bytes = n_digits * degree * sizeof(uint64_t);
    size_t acc_bytes = degree * sizeof(uint64_t);

    id<MTLBuffer> buffer_digits = [device_ newBufferWithBytes:digits_ntt
                                                       length:digit_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_evk = [device_ newBufferWithBytes:evk_ntt
                                                    length:digit_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_acc = [device_ newBufferWithLength:acc_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_mod = [device_ newBufferWithBytes:&modulus
                                                    length:sizeof(modulus)
                                                   options:MTLResourceStorageModeShared];
    uint32_t deg32 = static_cast<uint32_t>(degree);
    uint32_t nd32 = static_cast<uint32_t>(n_digits);
    id<MTLBuffer> buffer_deg = [device_ newBufferWithBytes:&deg32
                                                    length:sizeof(deg32)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_nd = [device_ newBufferWithBytes:&nd32
                                                   length:sizeof(nd32)
                                                  options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:keyswitch_mac_pipeline_];
    [encoder setBuffer:buffer_digits offset:0 atIndex:0];
    [encoder setBuffer:buffer_evk offset:0 atIndex:1];
    [encoder setBuffer:buffer_acc offset:0 atIndex:2];
    [encoder setBuffer:buffer_mod offset:0 atIndex:3];
    [encoder setBuffer:buffer_deg offset:0 atIndex:4];
    [encoder setBuffer:buffer_nd offset:0 atIndex:5];

    NSUInteger tg = std::min((NSUInteger)256,
                             [keyswitch_mac_pipeline_ maxTotalThreadsPerThreadgroup]);
    NSUInteger groups = (degree + tg - 1) / tg;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) {
        return false;
    }
    memcpy(acc_out, [buffer_acc contents], acc_bytes);
    return true;
#else
    (void)digits_ntt;
    (void)evk_ntt;
    (void)acc_out;
    (void)n_digits;
    (void)degree;
    (void)modulus;
    return false;
#endif
}

bool MetalComputeContext::upload_evk(uint64_t cache_key, const uint64_t* data,
                                     size_t n_digits, size_t degree) {
#ifdef __APPLE__
    if (device_ == nil || evk_buffers_ == nil || data == nullptr || n_digits == 0 ||
        degree == 0) {
        return false;
    }
    size_t bytes = n_digits * degree * sizeof(uint64_t);
    id<MTLBuffer> buf = [device_ newBufferWithBytes:data
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    if (buf == nil) {
        return false;
    }
    evk_buffers_[@(cache_key)] = buf;
    return true;
#else
    (void)cache_key;
    (void)data;
    (void)n_digits;
    (void)degree;
    return false;
#endif
}

void MetalComputeContext::clear_evk_cache() {
#ifdef __APPLE__
    if (evk_buffers_ != nil) {
        [evk_buffers_ removeAllObjects];
    }
#endif
}

bool MetalComputeContext::keyswitch_mac_cached(uint64_t cache_key,
                                               const uint64_t* digits_ntt,
                                               uint64_t* acc_out, size_t n_digits,
                                               size_t degree, uint64_t modulus) {
#ifdef __APPLE__
    if (keyswitch_mac_pipeline_ == nil || evk_buffers_ == nil || digits_ntt == nullptr ||
        acc_out == nullptr || n_digits == 0 || degree == 0 || modulus < 2 ||
        modulus >= (1ULL << 31)) {
        return false;
    }
    id<MTLBuffer> buffer_evk = evk_buffers_[@(cache_key)];
    if (buffer_evk == nil) {
        return false;
    }
    size_t expected = n_digits * degree * sizeof(uint64_t);
    if ([buffer_evk length] < expected) {
        return false;
    }

    size_t digit_bytes = expected;
    size_t acc_bytes = degree * sizeof(uint64_t);
    id<MTLBuffer> buffer_digits = [device_ newBufferWithBytes:digits_ntt
                                                       length:digit_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_acc = [device_ newBufferWithLength:acc_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_mod = [device_ newBufferWithBytes:&modulus
                                                    length:sizeof(modulus)
                                                   options:MTLResourceStorageModeShared];
    uint32_t deg32 = static_cast<uint32_t>(degree);
    uint32_t nd32 = static_cast<uint32_t>(n_digits);
    id<MTLBuffer> buffer_deg = [device_ newBufferWithBytes:&deg32
                                                    length:sizeof(deg32)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_nd = [device_ newBufferWithBytes:&nd32
                                                   length:sizeof(nd32)
                                                  options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:keyswitch_mac_pipeline_];
    [encoder setBuffer:buffer_digits offset:0 atIndex:0];
    [encoder setBuffer:buffer_evk offset:0 atIndex:1];
    [encoder setBuffer:buffer_acc offset:0 atIndex:2];
    [encoder setBuffer:buffer_mod offset:0 atIndex:3];
    [encoder setBuffer:buffer_deg offset:0 atIndex:4];
    [encoder setBuffer:buffer_nd offset:0 atIndex:5];

    NSUInteger tg = std::min((NSUInteger)256,
                             [keyswitch_mac_pipeline_ maxTotalThreadsPerThreadgroup]);
    NSUInteger groups = (degree + tg - 1) / tg;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) {
        return false;
    }
    memcpy(acc_out, [buffer_acc contents], acc_bytes);
    return true;
#else
    (void)cache_key;
    (void)digits_ntt;
    (void)acc_out;
    (void)n_digits;
    (void)degree;
    (void)modulus;
    return false;
#endif
}

void MetalComputeContext::batch_modadd(const uint64_t* a, const uint64_t* b, uint64_t* result,
                                        size_t count, uint64_t modulus) {
#ifdef __APPLE__
    if (modadd_pipeline_ == nil || count == 0) return;
    
    size_t buffer_size = count * sizeof(uint64_t);
    
    id<MTLBuffer> buffer_a = [device_ newBufferWithBytes:a
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_b = [device_ newBufferWithBytes:b
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_result = [device_ newBufferWithLength:buffer_size
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_modulus = [device_ newBufferWithBytes:&modulus
                                                        length:sizeof(modulus)
                                                       options:MTLResourceStorageModeShared];
    uint32_t size = static_cast<uint32_t>(count);
    id<MTLBuffer> buffer_size_param = [device_ newBufferWithBytes:&size
                                                           length:sizeof(size)
                                                          options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    
    [encoder setComputePipelineState:modadd_pipeline_];
    [encoder setBuffer:buffer_a offset:0 atIndex:0];
    [encoder setBuffer:buffer_b offset:0 atIndex:1];
    [encoder setBuffer:buffer_result offset:0 atIndex:2];
    [encoder setBuffer:buffer_modulus offset:0 atIndex:3];
    [encoder setBuffer:buffer_size_param offset:0 atIndex:4];
    
    NSUInteger threadgroup_size = std::min((NSUInteger)256, [modadd_pipeline_ maxTotalThreadsPerThreadgroup]);
    NSUInteger num_threadgroups = (count + threadgroup_size - 1) / threadgroup_size;
    
    [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    memcpy(result, [buffer_result contents], buffer_size);
#endif
}

// ============================================================================
// NTT: host-side Montgomery (R = 2^32) helpers
// ============================================================================

namespace {

/// Host mirror of `struct NTTParams` in cpp/shaders/common/fhe_common.metal.
/// The layout must stay byte-for-byte identical (Metal aligns `ulong` to 8).
struct NTTParamsHost {
    uint32_t degree;        // N
    uint32_t log_degree;    // log2(N)
    uint32_t q_inv_neg;     // -q^{-1} mod 2^32
    uint32_t _pad;
    uint64_t modulus;       // q
    uint64_t inv_n_mont;    // N^{-1} mod q, in Montgomery form
    uint64_t r_mod_q;       // R mod q  (Montgomery form of 1)
    uint64_t r2_mod_q;      // R^2 mod q (for to-Montgomery conversion)
};

static_assert(sizeof(NTTParamsHost) == 48, "NTTParams layout must match the Metal struct");
static_assert(offsetof(NTTParamsHost, degree) == 0, "NTTParams layout drift");
static_assert(offsetof(NTTParamsHost, log_degree) == 4, "NTTParams layout drift");
static_assert(offsetof(NTTParamsHost, q_inv_neg) == 8, "NTTParams layout drift");
static_assert(offsetof(NTTParamsHost, modulus) == 16, "NTTParams layout drift");
static_assert(offsetof(NTTParamsHost, inv_n_mont) == 24, "NTTParams layout drift");
static_assert(offsetof(NTTParamsHost, r_mod_q) == 32, "NTTParams layout drift");
static_assert(offsetof(NTTParamsHost, r2_mod_q) == 40, "NTTParams layout drift");

/// Montgomery context for R = 2^32. Mirrors MontCtx32 in
/// fhe-evolve/native/src/montgomery.rs.
struct MontCtx32 {
    uint64_t q = 0;
    uint32_t q_inv_neg = 0;   // -q^{-1} mod 2^32
    uint64_t r_mod_q = 0;     // 2^32 mod q
    uint64_t r2_mod_q = 0;    // (2^32)^2 mod q
    bool valid = false;
};

/// Build the Montgomery context for `q`.
/// Requires q odd and q < 2^32; returns an invalid context otherwise so the
/// caller can refuse the GPU path instead of producing wrong results.
MontCtx32 make_mont_ctx(uint64_t q) {
    MontCtx32 ctx;
    if (q < 3 || (q & 1ULL) == 0 || q >= (1ULL << 32)) {
        return ctx;  // invalid
    }

    const uint32_t q32 = static_cast<uint32_t>(q);

    // -q^{-1} mod 2^32 via Newton's method: x <- x * (2 - q*x), 5 iterations
    // starting from x = 1 gives full 32-bit precision (1->2->4->8->16->32 bits).
    uint32_t inv = 1;
    for (int i = 0; i < 5; i++) {
        inv = inv * (2u - q32 * inv);
    }
    const uint32_t q_inv_neg = 0u - inv;  // -q^{-1} mod 2^32

    // Verify q * q_inv_neg == -1 mod 2^32
    if (static_cast<uint32_t>(q32 * q_inv_neg) != 0xFFFFFFFFu) {
        return ctx;  // invalid
    }

    ctx.q = q;
    ctx.q_inv_neg = q_inv_neg;
    ctx.r_mod_q = (1ULL << 32) % q;
    ctx.r2_mod_q = static_cast<uint64_t>(
        (static_cast<__uint128_t>(ctx.r_mod_q) * ctx.r_mod_q) % q);
    ctx.valid = true;
    return ctx;
}

/// Convert an ordinary residue to Montgomery form: x -> x * 2^32 mod q.
inline uint64_t to_mont(uint64_t x, const MontCtx32& ctx) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(x % ctx.q) << 32) % ctx.q);
}

/// Modular inverse via the extended Euclidean algorithm.
/// Used for N^{-1} mod q; unlike Fermat it does not assume q is prime, only
/// that gcd(a, q) == 1 (guaranteed here: N is a power of two and q is odd).
/// Returns 0 when no inverse exists.
uint64_t mod_inverse_u64(uint64_t a, uint64_t q) {
    if (q <= 1) return 0;
    int64_t old_r = static_cast<int64_t>(a % q);
    int64_t r = static_cast<int64_t>(q);
    int64_t old_s = 1, s = 0;
    while (r != 0) {
        int64_t quot = old_r / r;
        int64_t tmp = old_r - quot * r;
        old_r = r; r = tmp;
        tmp = old_s - quot * s;
        old_s = s; s = tmp;
    }
    if (old_r != 1) return 0;  // not invertible
    if (old_s < 0) old_s += static_cast<int64_t>(q);
    return static_cast<uint64_t>(old_s);
}

/// Build the flat, stage-major twiddle table consumed by the NTT kernels.
///
/// Input `flat_powers` is the caller's table of successive powers of the root:
/// flat_powers[i] == omega^i mod q for i in [0, N).
///
/// Output has N-1 entries. Stage s (m = 1<<s, len = 2m) occupies the m entries
/// starting at (1<<s)-1, and entry j of that block is omega^(j*N/len), in
/// Montgomery form. This matches the layout documented in ntt_forward.metal and
/// the reference table in fhe-evolve/native/src/ntt_accel.rs.
std::vector<uint64_t> build_stage_twiddles(const uint64_t* flat_powers,
                                           size_t degree,
                                           const MontCtx32& ctx,
                                           bool already_montgomery) {
    std::vector<uint64_t> table;
    table.reserve(degree - 1);

    for (size_t len = 2; len <= degree; len <<= 1) {
        const size_t half = len / 2;      // == m, entries owned by this stage
        const size_t step = degree / len; // exponent stride within the stage
        for (size_t j = 0; j < half; j++) {
            const uint64_t w = flat_powers[j * step];
            table.push_back(already_montgomery ? w : to_mont(w, ctx));
        }
    }
    return table;
}

}  // namespace

// ============================================================================
// NTT: single-command-buffer GPU transform
// ============================================================================

bool MetalComputeContext::ntt_execute(uint64_t* coeffs, size_t degree, size_t batch_size,
                                      uint64_t modulus, const uint64_t* twiddles,
                                      bool twiddles_in_montgomery_form, bool inverse,
                                      const uint64_t* psi_powers) {
#ifdef __APPLE__
    if (device_ == nil || command_queue_ == nil) return false;
    if (coeffs == nullptr || twiddles == nullptr) return false;
    if (degree < 2 || batch_size == 0) return false;
    if ((degree & (degree - 1)) != 0) return false;  // must be a power of two

    id<MTLComputePipelineState> stage_pipeline =
        inverse ? ntt_inverse_stage_pipeline_ : ntt_stage_pipeline_;

    if (stage_pipeline == nil || ntt_bitrev_pipeline_ == nil ||
        ntt_to_mont_pipeline_ == nil || ntt_from_mont_pipeline_ == nil) {
        return false;
    }
    if (inverse && ntt_inverse_scale_pipeline_ == nil) return false;

    // A psi table means the caller wants the negacyclic transform, which needs
    // the fused twist kernel. Refuse rather than quietly computing the cyclic
    // transform instead.
    const bool negacyclic = (psi_powers != nullptr);
    if (negacyclic && ntt_twist_pipeline_ == nil) {
        std::cerr << "Metal: NTT refused - negacyclic transform requested but the "
                     "ntt_twist pipeline is unavailable" << std::endl;
        return false;
    }

    // Refuse rather than silently computing garbage: the kernels implement
    // Montgomery arithmetic with R = 2^32, which needs an odd modulus < 2^32.
    MontCtx32 ctx = make_mont_ctx(modulus);
    if (!ctx.valid) {
        std::cerr << "Metal: NTT refused - modulus " << modulus
                  << " must be odd and < 2^32 for the R=2^32 Montgomery kernels"
                  << std::endl;
        return false;
    }

    const size_t total_coeffs = degree * batch_size;
    const size_t coeff_buffer_size = total_coeffs * sizeof(uint64_t);

    // Twiddles: stage-major, Montgomery form, N-1 entries.
    std::vector<uint64_t> stage_twiddles =
        build_stage_twiddles(twiddles, degree, ctx, twiddles_in_montgomery_form);
    const size_t twiddle_buffer_size = stage_twiddles.size() * sizeof(uint64_t);

    // Negacyclic twist table for the fused ntt_twist pass, N entries.
    // The scaling differs by direction because mont_mul_32(a, b) = a*b*R^-1:
    //   forward: psi^i * R^2  -> ordinary in, Montgomery out (twist + to_mont)
    //   inverse: psi^(-i)     -> Montgomery in, ordinary out (untwist + from_mont)
    // psi_powers is expected in ordinary form.
    std::vector<uint64_t> twist_table;
    if (negacyclic) {
        twist_table.resize(degree);
        for (size_t i = 0; i < degree; i++) {
            const uint64_t p = psi_powers[i] % modulus;
            twist_table[i] = inverse ? p : to_mont(to_mont(p, ctx), ctx);
        }
    }

    NTTParamsHost params{};
    params.degree = static_cast<uint32_t>(degree);
    params.log_degree = 0;
    for (size_t t = degree; t > 1; t >>= 1) params.log_degree++;
    params.q_inv_neg = ctx.q_inv_neg;
    params._pad = 0;
    params.modulus = modulus;
    params.r_mod_q = ctx.r_mod_q;
    params.r2_mod_q = ctx.r2_mod_q;
    // N^{-1} mod q, then into Montgomery form for ntt_inverse_scale.
    const uint64_t inv_n = mod_inverse_u64(degree, modulus);
    if (inv_n == 0) {
        std::cerr << "Metal: NTT refused - N=" << degree
                  << " has no inverse modulo " << modulus << std::endl;
        return false;
    }
    params.inv_n_mont = to_mont(inv_n, ctx);

    id<MTLBuffer> buffer_coeffs = [device_ newBufferWithBytes:coeffs
                                                       length:coeff_buffer_size
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_twiddles = [device_ newBufferWithBytes:stage_twiddles.data()
                                                         length:twiddle_buffer_size
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_params = [device_ newBufferWithBytes:&params
                                                       length:sizeof(params)
                                                      options:MTLResourceStorageModeShared];
    uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
    id<MTLBuffer> buffer_batch = [device_ newBufferWithBytes:&batch_size_u32
                                                      length:sizeof(batch_size_u32)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_twist = nil;
    if (negacyclic) {
        buffer_twist = [device_ newBufferWithBytes:twist_table.data()
                                           length:twist_table.size() * sizeof(uint64_t)
                                          options:MTLResourceStorageModeShared];
        if (buffer_twist == nil) return false;
    }
    if (buffer_coeffs == nil || buffer_twiddles == nil ||
        buffer_params == nil || buffer_batch == nil) {
        return false;
    }

    // Stage index buffers, one per stage, so all stages can be encoded up front.
    std::vector<id<MTLBuffer>> buffer_stages(params.log_degree);
    for (uint32_t stage = 0; stage < params.log_degree; stage++) {
        buffer_stages[stage] = [device_ newBufferWithBytes:&stage
                                                   length:sizeof(stage)
                                                  options:MTLResourceStorageModeShared];
    }

    // ------------------------------------------------------------------------
    // Everything below is encoded into ONE command buffer with a single
    // commit/waitUntilCompleted pair. Compute encoders inside a command buffer
    // execute in submission order, so no explicit barriers are needed.
    // ------------------------------------------------------------------------
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];

    const NSUInteger coeff_tg = std::min((NSUInteger)256, max_threadgroup_size_);
    const NSUInteger coeff_groups = (degree + coeff_tg - 1) / coeff_tg;

    // Helper: encode a coefficient-wise kernel taking (coeffs, params, batch).
    auto encode_elementwise = [&](id<MTLComputePipelineState> pipeline) {
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_coeffs offset:0 atIndex:0];
        [encoder setBuffer:buffer_params offset:0 atIndex:1];
        [encoder setBuffer:buffer_batch offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, batch_size)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
        [encoder endEncoding];
    };

    // Helper: encode the fused twist kernel (coeffs, twist, params, batch).
    auto encode_twist = [&]() {
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:ntt_twist_pipeline_];
        [encoder setBuffer:buffer_coeffs offset:0 atIndex:0];
        [encoder setBuffer:buffer_twist offset:0 atIndex:1];
        [encoder setBuffer:buffer_params offset:0 atIndex:2];
        [encoder setBuffer:buffer_batch offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, batch_size)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
        [encoder endEncoding];
    };

    // 1. Enter the Montgomery domain. On the forward negacyclic transform the
    //    psi twist is FUSED into this pass (same dispatch count as the cyclic
    //    path, no additional synchronisation).
    if (negacyclic && !inverse) {
        encode_twist();
    } else {
        encode_elementwise(ntt_to_mont_pipeline_);
    }

    // 2. Bit-reversal permutation
    encode_elementwise(ntt_bitrev_pipeline_);

    // 3. log2(N) butterfly stages
    {
        const NSUInteger butterflies = degree / 2;
        const NSUInteger stage_tg = std::min((NSUInteger)256,
                                             [stage_pipeline maxTotalThreadsPerThreadgroup]);
        const NSUInteger stage_groups = (butterflies + stage_tg - 1) / stage_tg;

        for (uint32_t stage = 0; stage < params.log_degree; stage++) {
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            [encoder setComputePipelineState:stage_pipeline];
            [encoder setBuffer:buffer_coeffs offset:0 atIndex:0];
            [encoder setBuffer:buffer_twiddles offset:0 atIndex:1];
            [encoder setBuffer:buffer_params offset:0 atIndex:2];
            [encoder setBuffer:buffer_stages[stage] offset:0 atIndex:3];
            [encoder setBuffer:buffer_batch offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(stage_groups, 1, batch_size)
                    threadsPerThreadgroup:MTLSizeMake(stage_tg, 1, 1)];
            [encoder endEncoding];
        }
    }

    // 4. Inverse only: scale by N^{-1} mod q
    if (inverse) {
        encode_elementwise(ntt_inverse_scale_pipeline_);
    }

    // 5. Leave the Montgomery domain. On the inverse negacyclic transform the
    //    psi^(-1) untwist is FUSED into this pass.
    if (negacyclic && inverse) {
        encode_twist();
    } else {
        encode_elementwise(ntt_from_mont_pipeline_);
    }

    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
        std::cerr << "Metal: NTT command buffer failed (status "
                  << (long)[cmd status] << ")" << std::endl;
        return false;
    }

    memcpy(coeffs, [buffer_coeffs contents], coeff_buffer_size);
    return true;
#else
    (void)coeffs; (void)degree; (void)batch_size; (void)modulus;
    (void)twiddles; (void)twiddles_in_montgomery_form; (void)inverse;
    (void)psi_powers;
    return false;
#endif
}

bool MetalComputeContext::batch_ntt_forward(uint64_t* coeffs, size_t degree, size_t batch_size,
                                            uint64_t modulus, const uint64_t* twiddles,
                                            bool twiddles_in_montgomery_form,
                                            const uint64_t* psi_powers) {
    return ntt_execute(coeffs, degree, batch_size, modulus, twiddles,
                       twiddles_in_montgomery_form, /*inverse=*/false, psi_powers);
}

bool MetalComputeContext::batch_ntt_inverse(uint64_t* coeffs, size_t degree, size_t batch_size,
                                            uint64_t modulus, const uint64_t* inv_twiddles,
                                            bool twiddles_in_montgomery_form,
                                            const uint64_t* psi_inv_powers) {
    return ntt_execute(coeffs, degree, batch_size, modulus, inv_twiddles,
                       twiddles_in_montgomery_form, /*inverse=*/true, psi_inv_powers);
}

bool MetalComputeContext::mod_down_batch(const uint64_t* x, uint64_t* y, size_t count,
                                         uint64_t modulus, uint64_t p_inv) {
#ifdef __APPLE__
    if (mod_down_pipeline_ == nil || x == nullptr || y == nullptr || count == 0 ||
        modulus < 2 || modulus >= (1ULL << 31)) {
        return false;
    }
    size_t bytes = count * sizeof(uint64_t);
    id<MTLBuffer> bx = [device_ newBufferWithBytes:x length:bytes
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [device_ newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bm = [device_ newBufferWithBytes:&modulus length:sizeof(modulus)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> bp = [device_ newBufferWithBytes:&p_inv length:sizeof(p_inv)
                                           options:MTLResourceStorageModeShared];
    uint32_t cnt = static_cast<uint32_t>(count);
    id<MTLBuffer> bc = [device_ newBufferWithBytes:&cnt length:sizeof(cnt)
                                           options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:mod_down_pipeline_];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:by offset:0 atIndex:1];
    [enc setBuffer:bm offset:0 atIndex:2];
    [enc setBuffer:bp offset:0 atIndex:3];
    [enc setBuffer:bc offset:0 atIndex:4];
    NSUInteger tg = std::min((NSUInteger)256, [mod_down_pipeline_ maxTotalThreadsPerThreadgroup]);
    NSUInteger groups = (count + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) return false;
    memcpy(y, [by contents], bytes);
    return true;
#else
    (void)x; (void)y; (void)count; (void)modulus; (void)p_inv;
    return false;
#endif
}

bool MetalComputeContext::keyswitch_fused(
    const uint64_t* digits_coeff, uint64_t* acc_b_out, uint64_t* acc_a_out,
    size_t n_digits, size_t degree, uint64_t modulus,
    const uint64_t* omega_powers, const uint64_t* omega_inv_powers,
    const uint64_t* psi_powers, const uint64_t* psi_inv_powers,
    uint64_t evk_key_b, uint64_t evk_key_a) {
#ifdef __APPLE__
    if (!has_pipeline("keyswitch_fused")) return false;
    if (digits_coeff == nullptr || acc_b_out == nullptr || acc_a_out == nullptr) return false;
    if (n_digits == 0 || degree < 2 || (degree & (degree - 1)) != 0) return false;
    if (omega_powers == nullptr || omega_inv_powers == nullptr ||
        psi_powers == nullptr || psi_inv_powers == nullptr) return false;
    if (modulus >= (1ULL << 31) || (modulus & 1ULL) == 0) return false;

    id<MTLBuffer> evk_b = evk_buffers_[@(evk_key_b)];
    id<MTLBuffer> evk_a = evk_buffers_[@(evk_key_a)];
    if (evk_b == nil || evk_a == nil) return false;
    size_t digit_bytes = n_digits * degree * sizeof(uint64_t);
    size_t acc_bytes = degree * sizeof(uint64_t);
    if ([evk_b length] < digit_bytes || [evk_a length] < digit_bytes) return false;

    MontCtx32 ctx = make_mont_ctx(modulus);
    if (!ctx.valid) return false;

    auto build_params = [&](bool /*inv*/) -> NTTParamsHost {
        NTTParamsHost params{};
        params.degree = static_cast<uint32_t>(degree);
        params.log_degree = 0;
        for (size_t t = degree; t > 1; t >>= 1) params.log_degree++;
        params.q_inv_neg = ctx.q_inv_neg;
        params._pad = 0;
        params.modulus = modulus;
        params.r_mod_q = ctx.r_mod_q;
        params.r2_mod_q = ctx.r2_mod_q;
        const uint64_t inv_n = mod_inverse_u64(degree, modulus);
        params.inv_n_mont = to_mont(inv_n, ctx);
        return params;
    };

    NTTParamsHost fwd_params = build_params(false);
    NTTParamsHost inv_params = build_params(true);

    std::vector<uint64_t> fwd_tw =
        build_stage_twiddles(omega_powers, degree, ctx, false);
    std::vector<uint64_t> inv_tw =
        build_stage_twiddles(omega_inv_powers, degree, ctx, false);

    std::vector<uint64_t> twist_fwd(degree), twist_inv(degree);
    for (size_t i = 0; i < degree; i++) {
        twist_fwd[i] = to_mont(to_mont(psi_powers[i] % modulus, ctx), ctx);
        twist_inv[i] = psi_inv_powers[i] % modulus;
    }

    id<MTLBuffer> buf_digits = [device_ newBufferWithBytes:digits_coeff
                                                    length:digit_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_acc_b = [device_ newBufferWithLength:acc_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_acc_a = [device_ newBufferWithLength:acc_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_fwd_tw = [device_ newBufferWithBytes:fwd_tw.data()
                                                    length:fwd_tw.size() * sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_inv_tw = [device_ newBufferWithBytes:inv_tw.data()
                                                    length:inv_tw.size() * sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_fwd_params = [device_ newBufferWithBytes:&fwd_params
                                                        length:sizeof(fwd_params)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_inv_params = [device_ newBufferWithBytes:&inv_params
                                                        length:sizeof(inv_params)
                                                       options:MTLResourceStorageModeShared];
    uint32_t batch_digits = static_cast<uint32_t>(n_digits);
    uint32_t batch_one = 1;
    id<MTLBuffer> buf_batch_d = [device_ newBufferWithBytes:&batch_digits
                                                     length:sizeof(batch_digits)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_batch_1 = [device_ newBufferWithBytes:&batch_one
                                                     length:sizeof(batch_one)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_twist_fwd = [device_ newBufferWithBytes:twist_fwd.data()
                                                       length:twist_fwd.size() * sizeof(uint64_t)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_twist_inv = [device_ newBufferWithBytes:twist_inv.data()
                                                       length:twist_inv.size() * sizeof(uint64_t)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_mod = [device_ newBufferWithBytes:&modulus length:sizeof(modulus)
                                                options:MTLResourceStorageModeShared];
    uint32_t deg32 = static_cast<uint32_t>(degree);
    uint32_t nd32 = static_cast<uint32_t>(n_digits);
    id<MTLBuffer> buf_deg = [device_ newBufferWithBytes:&deg32 length:sizeof(deg32)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_nd = [device_ newBufferWithBytes:&nd32 length:sizeof(nd32)
                                               options:MTLResourceStorageModeShared];

    std::vector<id<MTLBuffer>> stage_bufs(fwd_params.log_degree);
    for (uint32_t s = 0; s < fwd_params.log_degree; s++) {
        stage_bufs[s] = [device_ newBufferWithBytes:&s length:sizeof(s)
                                            options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    const NSUInteger coeff_tg = std::min((NSUInteger)256, max_threadgroup_size_);
    const NSUInteger coeff_groups = (degree + coeff_tg - 1) / coeff_tg;

    auto encode_fwd_ntt = [&]() {
        // Twist (to mont + psi)
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_twist_pipeline_];
            [e setBuffer:buf_digits offset:0 atIndex:0];
            [e setBuffer:buf_twist_fwd offset:0 atIndex:1];
            [e setBuffer:buf_fwd_params offset:0 atIndex:2];
            [e setBuffer:buf_batch_d offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, n_digits)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
            [e endEncoding];
        }
        // Bitrev
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_bitrev_pipeline_];
            [e setBuffer:buf_digits offset:0 atIndex:0];
            [e setBuffer:buf_fwd_params offset:0 atIndex:1];
            [e setBuffer:buf_batch_d offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, n_digits)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
            [e endEncoding];
        }
        // Stages
        const NSUInteger butterflies = degree / 2;
        const NSUInteger stg = std::min((NSUInteger)256,
            [ntt_stage_pipeline_ maxTotalThreadsPerThreadgroup]);
        const NSUInteger sgroups = (butterflies + stg - 1) / stg;
        for (uint32_t stage = 0; stage < fwd_params.log_degree; stage++) {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_stage_pipeline_];
            [e setBuffer:buf_digits offset:0 atIndex:0];
            [e setBuffer:buf_fwd_tw offset:0 atIndex:1];
            [e setBuffer:buf_fwd_params offset:0 atIndex:2];
            [e setBuffer:stage_bufs[stage] offset:0 atIndex:3];
            [e setBuffer:buf_batch_d offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(sgroups, 1, n_digits)
                threadsPerThreadgroup:MTLSizeMake(stg, 1, 1)];
            [e endEncoding];
        }
        // From mont
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_from_mont_pipeline_];
            [e setBuffer:buf_digits offset:0 atIndex:0];
            [e setBuffer:buf_fwd_params offset:0 atIndex:1];
            [e setBuffer:buf_batch_d offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, n_digits)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
            [e endEncoding];
        }
    };

    auto encode_mac = [&](id<MTLBuffer> evk, id<MTLBuffer> acc) {
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:keyswitch_mac_pipeline_];
        [e setBuffer:buf_digits offset:0 atIndex:0];
        [e setBuffer:evk offset:0 atIndex:1];
        [e setBuffer:acc offset:0 atIndex:2];
        [e setBuffer:buf_mod offset:0 atIndex:3];
        [e setBuffer:buf_deg offset:0 atIndex:4];
        [e setBuffer:buf_nd offset:0 atIndex:5];
        NSUInteger tg = std::min((NSUInteger)256,
            [keyswitch_mac_pipeline_ maxTotalThreadsPerThreadgroup]);
        NSUInteger groups = (degree + tg - 1) / tg;
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [e endEncoding];
    };

    auto encode_inv_ntt = [&](id<MTLBuffer> acc) {
        // To mont
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_to_mont_pipeline_];
            [e setBuffer:acc offset:0 atIndex:0];
            [e setBuffer:buf_inv_params offset:0 atIndex:1];
            [e setBuffer:buf_batch_1 offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
            [e endEncoding];
        }
        // Bitrev
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_bitrev_pipeline_];
            [e setBuffer:acc offset:0 atIndex:0];
            [e setBuffer:buf_inv_params offset:0 atIndex:1];
            [e setBuffer:buf_batch_1 offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
            [e endEncoding];
        }
        const NSUInteger butterflies = degree / 2;
        const NSUInteger stg = std::min((NSUInteger)256,
            [ntt_inverse_stage_pipeline_ maxTotalThreadsPerThreadgroup]);
        const NSUInteger sgroups = (butterflies + stg - 1) / stg;
        for (uint32_t stage = 0; stage < inv_params.log_degree; stage++) {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_inverse_stage_pipeline_];
            [e setBuffer:acc offset:0 atIndex:0];
            [e setBuffer:buf_inv_tw offset:0 atIndex:1];
            [e setBuffer:buf_inv_params offset:0 atIndex:2];
            [e setBuffer:stage_bufs[stage] offset:0 atIndex:3];
            [e setBuffer:buf_batch_1 offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(sgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(stg, 1, 1)];
            [e endEncoding];
        }
        // Scale
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_inverse_scale_pipeline_];
            [e setBuffer:acc offset:0 atIndex:0];
            [e setBuffer:buf_inv_params offset:0 atIndex:1];
            [e setBuffer:buf_batch_1 offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
            [e endEncoding];
        }
        // Untwist (from mont + psi^{-1})
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:ntt_twist_pipeline_];
            [e setBuffer:acc offset:0 atIndex:0];
            [e setBuffer:buf_twist_inv offset:0 atIndex:1];
            [e setBuffer:buf_inv_params offset:0 atIndex:2];
            [e setBuffer:buf_batch_1 offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(coeff_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(coeff_tg, 1, 1)];
            [e endEncoding];
        }
    };

    // Single command buffer: Stage A → B → C (no intermediate host sync).
    encode_fwd_ntt();
    encode_mac(evk_b, buf_acc_b);
    encode_mac(evk_a, buf_acc_a);
    encode_inv_ntt(buf_acc_b);
    encode_inv_ntt(buf_acc_a);

    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) return false;

    memcpy(acc_b_out, [buf_acc_b contents], acc_bytes);
    memcpy(acc_a_out, [buf_acc_a contents], acc_bytes);
    return true;
#else
    (void)digits_coeff; (void)acc_b_out; (void)acc_a_out;
    (void)n_digits; (void)degree; (void)modulus;
    (void)omega_powers; (void)omega_inv_powers;
    (void)psi_powers; (void)psi_inv_powers;
    (void)evk_key_b; (void)evk_key_a;
    return false;
#endif
}

void MetalComputeContext::batch_poly_mul(const uint64_t* poly_a, const uint64_t* poly_b,
                                          uint64_t* result, size_t degree, size_t batch_size,
                                          uint64_t modulus) {
    // Pointwise multiplication in NTT domain
    size_t total = degree * batch_size;
    batch_modmul(poly_a, poly_b, result, total, modulus);
}

// ============================================================================
// Global Functions
// ============================================================================

static std::unique_ptr<MetalComputeContext> g_metal_context;

MetalComputeContext& get_metal_context() {
    if (!g_metal_context) {
        g_metal_context = std::make_unique<MetalComputeContext>();
    }
    return *g_metal_context;
}

bool metal_available() {
    return get_metal_context().is_available();
}

void gpu_batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus) {
    auto& ctx = get_metal_context();
    
    if (ctx.is_available() && count >= GPU_DISPATCH_THRESHOLD && ctx.has_pipeline("modmul_batch")) {
        ctx.batch_modmul(a, b, result, count, modulus);
    } else {
        // CPU fallback
        for (size_t i = 0; i < count; i++) {
            __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
            result[i] = product % modulus;
        }
    }
}

void gpu_batch_ntt(uint64_t* coeffs, size_t degree, size_t batch_size,
                   uint64_t modulus, const uint64_t* twiddles, bool inverse,
                   const uint64_t* psi_powers) {
    auto& ctx = get_metal_context();
    
    const char* required = inverse ? "ntt_inverse_gpu" : "ntt_forward_gpu";
    bool done = false;
    
    if (ctx.is_available() && batch_size >= 4 && ctx.has_pipeline(required)) {
        if (inverse) {
            done = ctx.batch_ntt_inverse(coeffs, degree, batch_size, modulus, twiddles,
                                         false, psi_powers);
        } else {
            done = ctx.batch_ntt_forward(coeffs, degree, batch_size, modulus, twiddles,
                                         false, psi_powers);
        }
    }
    
    if (!done) {
        // CPU fallback - process each polynomial
        // (Would call CPU NTT implementation here)
        std::cerr << "Metal: GPU NTT not available, falling back to CPU" << std::endl;
    }
}

} // namespace metal
} // namespace fhe_accelerate
