//! Metal-accelerated Galois key-switching (Phase 4).
//!
//! Host digit-decomposes `c1`, packs RNS limbs, batch-NTTs digits on the GPU,
//! MACs against resident (or cached) NTT-domain evaluation keys, then INTTs
//! and CRT-recombines. Falls back to the CPU RNS path when Metal is unavailable.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Counts limb key-switches served by the fused single-command-buffer path and
/// by each fallback. Audit gates read these to prove the fused pipeline is
/// actually exercised rather than silently degrading to the CPU route.
static FUSED_HITS: AtomicU64 = AtomicU64::new(0);
static FALLBACK_HITS: AtomicU64 = AtomicU64::new(0);

/// `(fused, fallback)` limb key-switch counts since process start.
pub fn ks_path_counters() -> (u64, u64) {
    (
        FUSED_HITS.load(Ordering::Relaxed),
        FALLBACK_HITS.load(Ordering::Relaxed),
    )
}

/// Nanoseconds accumulated in each key-switch phase, for profiling only.
static NS_DECOMPOSE: AtomicU64 = AtomicU64::new(0);
static NS_GPU: AtomicU64 = AtomicU64::new(0);
static NS_RECOMBINE: AtomicU64 = AtomicU64::new(0);
static NS_PLANS: AtomicU64 = AtomicU64::new(0);

/// `(digit decompose+pack, fused GPU, CRT recombine, NTT plan build)` nanoseconds.
pub fn ks_phase_nanos() -> (u64, u64, u64, u64) {
    (
        NS_DECOMPOSE.load(Ordering::Relaxed),
        NS_GPU.load(Ordering::Relaxed),
        NS_RECOMBINE.load(Ordering::Relaxed),
        NS_PLANS.load(Ordering::Relaxed),
    )
}

use crate::crypto::{
    digit_count, decompose_c1_digits, Ciphertext, CryptoResult, GaloisKey, KS_DIGIT_BASE,
};
use crate::metal_ntt::{metal_available, metal_forward_batch, metal_inverse_batch};
use crate::ntt::{forward_ntt_negacyclic, inverse_ntt_negacyclic, mul_mod, NegacyclicPlan};
use crate::rns::RnsBasis;
use crate::zq::{add_mod_zq, Zq};

/// Serialize Metal KS dispatches — the compute context is not thread-safe.
fn metal_gpu_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[link(name = "engpe_metal", kind = "static")]
#[cfg(engpe_metal)]
extern "C" {
    fn engpe_metal_keyswitch_mac(
        digits_ntt: *const u64,
        evk_ntt: *const u64,
        acc_out: *mut u64,
        n_digits: usize,
        degree: usize,
        modulus: u64,
    ) -> bool;
    fn engpe_metal_upload_evk(
        cache_key: u64,
        data: *const u64,
        n_digits: usize,
        degree: usize,
    ) -> bool;
    fn engpe_metal_clear_evk_cache();
    fn engpe_metal_keyswitch_mac_cached(
        cache_key: u64,
        digits_ntt: *const u64,
        acc_out: *mut u64,
        n_digits: usize,
        degree: usize,
        modulus: u64,
    ) -> bool;
    fn engpe_metal_keyswitch_fused(
        digits_coeff: *const u64,
        acc_b_out: *mut u64,
        acc_a_out: *mut u64,
        n_digits: usize,
        degree: usize,
        modulus: u64,
        omega_powers: *const u64,
        omega_inv_powers: *const u64,
        psi_powers: *const u64,
        psi_inv_powers: *const u64,
        evk_key_b: u64,
        evk_key_a: u64,
    ) -> bool;
}

/// Per-limb NTT plans, memoised in `crypto`'s basis registry. Rebuilding four
/// 16384-entry twiddle tables on every key-switch is pure overhead.
fn limb_plans(basis: &RnsBasis) -> CryptoResult<std::sync::Arc<Vec<NegacyclicPlan>>> {
    crate::crypto::cached_limb_plans(basis)
}

/// Pack digit polynomials into RNS limb-major flat buffers:
/// `out[limb] = concat(digit_0, digit_1, …)` each of length `N`.
///
/// Digits are small (`< B = 2^20`); decompose via direct `Zq % q_i`.
pub fn pack_digit_limbs(
    digit_polys: &[Vec<Zq>],
    basis: &RnsBasis,
) -> CryptoResult<Vec<Vec<u64>>> {
    let n = basis.n;
    let k = basis.limb_count();
    let n_digits = digit_polys.len();
    let mut out = vec![vec![0u64; n_digits * n]; k];
    for (d, dig) in digit_polys.iter().enumerate() {
        if dig.len() != n {
            return Err("digit poly degree mismatch".into());
        }
        let limbs = basis.decompose_poly_zq(dig)?;
        for limb in 0..k {
            out[limb][d * n..(d + 1) * n].copy_from_slice(&limbs[limb]);
        }
    }
    Ok(out)
}

/// Precompute NTT-domain EVK limbs for one Galois key: `[component][limb] = flat[digit][N]`.
/// component 0 = b, 1 = a.
fn ntt_evk_limbs(
    gk: &GaloisKey,
    basis: &RnsBasis,
    plans: &[NegacyclicPlan],
) -> CryptoResult<[Vec<Vec<u64>>; 2]> {
    let n = basis.n;
    let k = basis.limb_count();
    let n_digits = gk.digits.len();
    let mut b_out = vec![vec![0u64; n_digits * n]; k];
    let mut a_out = vec![vec![0u64; n_digits * n]; k];
    for (d, (b, a)) in gk.digits.iter().enumerate() {
        let b_limbs = basis.decompose_poly_zq(b)?;
        let a_limbs = basis.decompose_poly_zq(a)?;
        for limb in 0..k {
            let mut fb = b_limbs[limb].clone();
            let mut fa = a_limbs[limb].clone();
            forward_ntt_negacyclic(&mut fb, &plans[limb])
                .map_err(|_| "EVK forward NTT failed".to_string())?;
            forward_ntt_negacyclic(&mut fa, &plans[limb])
                .map_err(|_| "EVK forward NTT failed".to_string())?;
            b_out[limb][d * n..(d + 1) * n].copy_from_slice(&fb);
            a_out[limb][d * n..(d + 1) * n].copy_from_slice(&fa);
        }
    }
    Ok([b_out, a_out])
}

fn evk_cache() -> &'static Mutex<HashMap<EvkCacheKey, Vec<u64>>> {
    static CACHE: OnceLock<Mutex<HashMap<EvkCacheKey, Vec<u64>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `(degree, rotation step, key fingerprint, limb, component)`.
///
/// The fingerprint is essential: without it two distinct Galois keys sharing
/// `(N, k)` alias to the same entry and key-switching silently uses the wrong
/// evaluation material.
type EvkCacheKey = (usize, usize, u64, usize, u8);

/// Distinct Galois keys held in the NTT-domain cache before it is flushed.
/// Each entry is `n_digits * N * 8` bytes per limb on both host and GPU, so an
/// unbounded cache would grow without limit across re-keyed sessions.
const MAX_CACHED_KEYS: usize = 64;

fn cache_key(n: usize, k: usize, fingerprint: u64, limb: usize, component: u8) -> u64 {
    // Mix into a single stable MTL buffer key. The fingerprint dominates, so
    // distinct keys cannot collide on (n, k, limb, component) alone.
    let mut h = fingerprint ^ 0x9e37_79b9_7f4a_7c15;
    for v in [n as u64, k as u64, limb as u64, component as u64] {
        h ^= v.wrapping_add(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(h << 6)
            .wrapping_add(h >> 2);
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
    }
    h
}

/// Drop every cached EVK on host and GPU. Called when the cache is full.
fn flush_evk_cache(map: &mut std::sync::MutexGuard<'_, HashMap<EvkCacheKey, Vec<u64>>>) {
    map.clear();
    #[cfg(engpe_metal)]
    {
        if let Ok(_guard) = metal_gpu_lock().lock() {
            unsafe {
                engpe_metal_clear_evk_cache();
            }
        }
    }
}

fn ensure_evk_ntt_cached(
    gk: &GaloisKey,
    basis: &RnsBasis,
    plans: &[NegacyclicPlan],
) -> CryptoResult<()> {
    let mut map = evk_cache()
        .lock()
        .map_err(|_| "EVK cache poisoned".to_string())?;
    if map.contains_key(&(basis.n, gk.k, gk.fingerprint, 0, 0)) {
        return Ok(());
    }
    // Entries are inserted 2 per limb, so bound on distinct keys, not entries.
    if map.len() >= MAX_CACHED_KEYS * 2 * basis.limb_count() {
        flush_evk_cache(&mut map);
    }
    let [b_limbs, a_limbs] = ntt_evk_limbs(gk, basis, plans)?;
    for limb in 0..basis.limb_count() {
        map.insert((basis.n, gk.k, gk.fingerprint, limb, 0), b_limbs[limb].clone());
        map.insert((basis.n, gk.k, gk.fingerprint, limb, 1), a_limbs[limb].clone());
        #[cfg(engpe_metal)]
        {
            let _guard = metal_gpu_lock()
                .lock()
                .map_err(|_| "Metal GPU lock poisoned".to_string())?;
            let ck0 = cache_key(basis.n, gk.k, gk.fingerprint, limb, 0);
            let ck1 = cache_key(basis.n, gk.k, gk.fingerprint, limb, 1);
            let n_digits = gk.digits.len();
            let n = basis.n;
            unsafe {
                let _ = engpe_metal_upload_evk(
                    ck0,
                    b_limbs[limb].as_ptr(),
                    n_digits,
                    n,
                );
                let _ = engpe_metal_upload_evk(
                    ck1,
                    a_limbs[limb].as_ptr(),
                    n_digits,
                    n,
                );
            }
        }
    }
    Ok(())
}

fn cpu_ks_mac(digits_ntt: &[u64], evk_ntt: &[u64], n_digits: usize, n: usize, q: u64) -> Vec<u64> {
    let mut acc = vec![0u64; n];
    for d in 0..n_digits {
        let base = d * n;
        for i in 0..n {
            let prod = mul_mod(digits_ntt[base + i], evk_ntt[base + i], q);
            acc[i] = (acc[i] + prod) % q;
        }
    }
    acc
}

fn metal_ks_mac(
    digits_ntt: &[u64],
    evk_ntt: &[u64],
    n_digits: usize,
    n: usize,
    q: u64,
    cache_key_opt: Option<u64>,
) -> Option<Vec<u64>> {
    #[cfg(engpe_metal)]
    {
        let _guard = metal_gpu_lock().lock().ok()?;
        let mut acc = vec![0u64; n];
        let ok = unsafe {
            if let Some(ck) = cache_key_opt {
                engpe_metal_keyswitch_mac_cached(
                    ck,
                    digits_ntt.as_ptr(),
                    acc.as_mut_ptr(),
                    n_digits,
                    n,
                    q,
                )
            } else {
                engpe_metal_keyswitch_mac(
                    digits_ntt.as_ptr(),
                    evk_ntt.as_ptr(),
                    acc.as_mut_ptr(),
                    n_digits,
                    n,
                    q,
                )
            }
        };
        if ok {
            return Some(acc);
        }
    }
    #[cfg(not(engpe_metal))]
    {
        let _ = (digits_ntt, evk_ntt, n_digits, n, q, cache_key_opt);
    }
    None
}

fn ntt_batch_digits(
    digits: &mut [u64],
    n_digits: usize,
    plan: &NegacyclicPlan,
    prefer_metal: bool,
) -> CryptoResult<()> {
    if prefer_metal && metal_available() {
        let _guard = metal_gpu_lock()
            .lock()
            .map_err(|_| "Metal GPU lock poisoned".to_string())?;
        if metal_forward_batch(digits, n_digits, plan) {
            return Ok(());
        }
    }
    let n = plan.n;
    for d in 0..n_digits {
        forward_ntt_negacyclic(&mut digits[d * n..(d + 1) * n], plan)
            .map_err(|_| "digit forward NTT failed".to_string())?;
    }
    Ok(())
}

fn intt_acc(
    acc: &mut [u64],
    plan: &NegacyclicPlan,
    prefer_metal: bool,
) -> CryptoResult<()> {
    if prefer_metal && metal_available() {
        let _guard = metal_gpu_lock()
            .lock()
            .map_err(|_| "Metal GPU lock poisoned".to_string())?;
        if metal_inverse_batch(acc, 1, plan) {
            return Ok(());
        }
    }
    inverse_ntt_negacyclic(acc, plan).map_err(|_| "KS inverse NTT failed".to_string())
}

/// Key-switch using fused Metal Digit-NTT → KS-MAC → INTT (one command buffer
/// per limb) when available; otherwise host NTT + Metal/CPU MAC.
pub fn key_switch_accelerated(
    ct: &Ciphertext,
    gk: &GaloisKey,
    basis: &RnsBasis,
    prefer_metal: bool,
) -> CryptoResult<Ciphertext> {
    let n = ct.degree();
    let q = gk.q;
    let n_digits = gk.digits.len();
    let t_plans = std::time::Instant::now();
    let plans = limb_plans(basis)?;
    NS_PLANS.fetch_add(t_plans.elapsed().as_nanos() as u64, Ordering::Relaxed);
    ensure_evk_ntt_cached(gk, basis, &plans)?;

    let t_decompose = std::time::Instant::now();
    let digit_polys = decompose_c1_digits(&ct.c1, q, n_digits);
    let mut digit_limbs = pack_digit_limbs(&digit_polys, basis)?;
    NS_DECOMPOSE.fetch_add(t_decompose.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let k_limbs = basis.limb_count();
    let mut acc0_limbs = vec![vec![0u64; n]; k_limbs];
    let mut acc1_limbs = vec![vec![0u64; n]; k_limbs];

    let map = evk_cache()
        .lock()
        .map_err(|_| "EVK cache poisoned".to_string())?;

    for limb in 0..k_limbs {
        let plan = &plans[limb];
        let digits = &mut digit_limbs[limb];

        let evk_b = map
            .get(&(basis.n, gk.k, gk.fingerprint, limb, 0))
            .ok_or_else(|| "missing cached EVK b".to_string())?;
        let evk_a = map
            .get(&(basis.n, gk.k, gk.fingerprint, limb, 1))
            .ok_or_else(|| "missing cached EVK a".to_string())?;

        let use_metal = prefer_metal && metal_available();
        let t_gpu = std::time::Instant::now();
        let fused = if use_metal {
            metal_keyswitch_fused(
                digits,
                n_digits,
                plan,
                cache_key(basis.n, gk.k, gk.fingerprint, limb, 0),
                cache_key(basis.n, gk.k, gk.fingerprint, limb, 1),
            )
        } else {
            None
        };
        NS_GPU.fetch_add(t_gpu.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if let Some((acc_b, acc_a)) = fused {
            FUSED_HITS.fetch_add(1, Ordering::Relaxed);
            acc0_limbs[limb] = acc_b;
            acc1_limbs[limb] = acc_a;
            continue;
        }

        // Fallback: CPU digit NTT + Metal/CPU MAC + CPU INTT.
        FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
        ntt_batch_digits(digits, n_digits, plan, false)?;
        let mut acc_b = if use_metal {
            metal_ks_mac(
                digits,
                evk_b,
                n_digits,
                n,
                plan.q,
                Some(cache_key(basis.n, gk.k, gk.fingerprint, limb, 0)),
            )
            .unwrap_or_else(|| cpu_ks_mac(digits, evk_b, n_digits, n, plan.q))
        } else {
            cpu_ks_mac(digits, evk_b, n_digits, n, plan.q)
        };
        let mut acc_a = if use_metal {
            metal_ks_mac(
                digits,
                evk_a,
                n_digits,
                n,
                plan.q,
                Some(cache_key(basis.n, gk.k, gk.fingerprint, limb, 1)),
            )
            .unwrap_or_else(|| cpu_ks_mac(digits, evk_a, n_digits, n, plan.q))
        } else {
            cpu_ks_mac(digits, evk_a, n_digits, n, plan.q)
        };
        intt_acc(&mut acc_b, plan, false)?;
        intt_acc(&mut acc_a, plan, false)?;
        acc0_limbs[limb] = acc_b;
        acc1_limbs[limb] = acc_a;
        let _ = evk_b;
        let _ = evk_a;
    }
    drop(map);

    let t_recombine = std::time::Instant::now();
    let t0 = basis.recombine_poly_zq(&acc0_limbs)?;
    let t1 = basis.recombine_poly_zq(&acc1_limbs)?;
    NS_RECOMBINE.fetch_add(t_recombine.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let mut c0 = Vec::with_capacity(n);
    for i in 0..n {
        c0.push(add_mod_zq(ct.c0[i], t0[i], q));
    }
    Ok(Ciphertext { c0, c1: t1 })
}

fn metal_keyswitch_fused(
    digits_coeff: &[u64],
    n_digits: usize,
    plan: &NegacyclicPlan,
    evk_key_b: u64,
    evk_key_a: u64,
) -> Option<(Vec<u64>, Vec<u64>)> {
    #[cfg(engpe_metal)]
    {
        let _guard = metal_gpu_lock().lock().ok()?;
        let n = plan.n;
        let mut acc_b = vec![0u64; n];
        let mut acc_a = vec![0u64; n];
        let ok = unsafe {
            engpe_metal_keyswitch_fused(
                digits_coeff.as_ptr(),
                acc_b.as_mut_ptr(),
                acc_a.as_mut_ptr(),
                n_digits,
                n,
                plan.q,
                plan.omega_powers.as_ptr(),
                plan.omega_inv_powers.as_ptr(),
                plan.psi_powers.as_ptr(),
                plan.psi_inv_powers.as_ptr(),
                evk_key_b,
                evk_key_a,
            )
        };
        if ok {
            return Some((acc_b, acc_a));
        }
    }
    #[cfg(not(engpe_metal))]
    {
        let _ = (digits_coeff, n_digits, plan, evk_key_b, evk_key_a);
    }
    None
}

/// Public digit decomposition for host prep / tests.
pub fn decompose_and_pack_for_metal(
    c1: &[Zq],
    basis: &RnsBasis,
) -> CryptoResult<(usize, Vec<Vec<u64>>)> {
    let n_digits = digit_count(basis.modulus);
    let digits = decompose_c1_digits(c1, basis.modulus, n_digits);
    let packed = pack_digit_limbs(&digits, basis)?;
    Ok((n_digits, packed))
}

#[allow(dead_code)]
fn _ks_digit_base() -> u128 {
    KS_DIGIT_BASE
}
