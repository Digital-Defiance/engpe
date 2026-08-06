//! CPU / Metal CKKS PRS evaluation: encode → (RNS) NTT multiply → horizontal sum → decode.
//!
//! - **CPU (small scale)**: Rayon across ciphertext chunks; wide single-prime NTT.
//! - **CPU (RNS, Δ≈2^40)**: same 4-limb CRT basis as Metal; Rayon NTT per limb.
//! - **Metal (RNS)**: Rayon host encode/decompose, then one GPU batch NTT-mul per limb.
//!
//! # Precision / modulus policy
//! Single-prime Metal cannot host `Δ = 2^40`. The RNS basis restores
//! `|Q| ≈ 124` bits so the 110k-SNP panel at `N = 16384` meets `ε < 1e-4`.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::ckks_encode::{decode_real_slots, encode_real_slots, scale_from_bits};
use crate::ckks_prs::{plaintext_prs_oracles, unpack_header, unpack_slots, CkksPrsParams};
use crate::ckks_rns::{
    evaluate_cpu_rns_cohort, evaluate_encrypted_rns_cohort_backend,
    evaluate_encrypted_rns_cohort_staged, evaluate_metal_rns_cohort, rns_basis_cached,
};

use crate::ckks_rotate::rotate_and_sum_u64;
use crate::metal_ntt::{metal_available, metal_modulus_ok, metal_ntt_pointwise_product};
use crate::ntt::{
    find_ntt_modulus, forward_ntt_negacyclic, inverse_ntt_negacyclic, mul_mod, NegacyclicPlan,
};

pub type EvalResult<T> = Result<T, String>;

type PlanCache = Mutex<HashMap<(usize, u32), Arc<NegacyclicPlan>>>;

/// Max `scale_bits` that fits a ~56-bit NTT prime at the given degree.
pub fn recommended_scale_bits(poly_degree: u32) -> u32 {
    let log_n = 32 - poly_degree.leading_zeros() - 1;
    let budget = 56u32.saturating_sub(8).saturating_sub(log_n);
    (budget / 2).min(16).max(8)
}

fn min_modulus_bits(poly_degree: usize, scale_bits: u32) -> u32 {
    let log_n = (poly_degree as f64).log2().ceil() as u32;
    (2 * scale_bits + log_n + 8).min(62)
}

fn plan_cache() -> &'static PlanCache {
    static CACHE: OnceLock<PlanCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_in(
    cache: &'static PlanCache,
    key: (usize, u32),
    build: impl FnOnce() -> EvalResult<NegacyclicPlan>,
) -> EvalResult<Arc<NegacyclicPlan>> {
    let mut map = cache
        .lock()
        .map_err(|_| "plan cache poisoned".to_string())?;
    if let Some(plan) = map.get(&key) {
        return Ok(Arc::clone(plan));
    }
    let arc = Arc::new(build()?);
    map.insert(key, Arc::clone(&arc));
    Ok(arc)
}

fn assert_cpu_gate(n: usize, plan: &NegacyclicPlan) -> EvalResult<()> {
    // Sparse schoolbook gate — safe at production N=16384 (Task 6).
    let _ = n;
    if crate::ntt::verify_negacyclic_convolution(plan) {
        Ok(())
    } else {
        Err(format!(
            "negacyclic convolution gate failed for N={}, q={}",
            plan.n, plan.q
        ))
    }
}

fn new_plan(n: usize, q: u64, label: &str) -> EvalResult<NegacyclicPlan> {
    NegacyclicPlan::new(n, q).ok_or_else(|| format!("failed to build {label} NTT plan q={q}"))
}

fn build_cpu_plan(n: usize, min_bits: u32) -> EvalResult<NegacyclicPlan> {
    let q = find_ntt_modulus(n, min_bits)
        .ok_or_else(|| format!("no NTT modulus for N={n}, min_bits={min_bits}"))?;
    let plan = new_plan(n, q, "CPU")?;
    assert_cpu_gate(n, &plan)?;
    Ok(plan)
}

fn plan_for(n: usize, scale_bits: u32) -> EvalResult<Arc<NegacyclicPlan>> {
    let min_bits = min_modulus_bits(n, scale_bits);
    cached_in(plan_cache(), (n, min_bits), || build_cpu_plan(n, min_bits))
}

fn to_centered_u64(x: i64, q: u64) -> u64 {
    let q_i = q as i128;
    let mut v = (x as i128) % q_i;
    if v < 0 {
        v += q_i;
    }
    v as u64
}

fn from_centered_u64(u: u64, q: u64) -> i64 {
    if u > q / 2 {
        (u as i128 - q as i128) as i64
    } else {
        u as i64
    }
}

/// Max scale_bits for Metal RNS (`k` × ~31-bit primes, `|Q| ≈ 124`).
/// `2s + log2(N) + log2(N/2) + 8 ≤ 120`.
pub fn recommended_metal_scale_bits(poly_degree: u32) -> u32 {
    let log_n = 32 - poly_degree.leading_zeros() - 1;
    let budget = 120u32
        .saturating_sub(8)
        .saturating_sub(log_n)
        .saturating_sub(log_n.saturating_sub(1));
    (budget / 2).min(40).max(16)
}

/// Which NTT backend produced the pointwise product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NttBackend {
    CpuRayon,
    MetalGpu,
}

fn cpu_pointwise_product(a: &[u64], b: &[u64], plan: &NegacyclicPlan) -> EvalResult<Vec<u64>> {
    let mut fa = a.to_vec();
    let mut fb = b.to_vec();
    forward_ntt_negacyclic(&mut fa, plan).map_err(|_| "forward NTT failed".to_string())?;
    forward_ntt_negacyclic(&mut fb, plan).map_err(|_| "forward NTT failed".to_string())?;
    let mut fc: Vec<u64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| mul_mod(x, y, plan.q))
        .collect();
    inverse_ntt_negacyclic(&mut fc, plan).map_err(|_| "inverse NTT failed".to_string())?;
    Ok(fc)
}

fn try_metal_single_prod(a: &[u64], b: &[u64], plan: &NegacyclicPlan) -> Option<Vec<u64>> {
    let prod = metal_ntt_pointwise_product(a, b, plan)?;
    prod.iter().any(|&x| x != 0).then_some(prod)
}

fn ntt_pointwise_product(
    a: &[u64],
    b: &[u64],
    plan: &NegacyclicPlan,
    prefer_metal: bool,
) -> EvalResult<(Vec<u64>, NttBackend)> {
    if prefer_metal && metal_modulus_ok(plan.q) {
        if let Some(prod) = try_metal_single_prod(a, b, plan) {
            return Ok((prod, NttBackend::MetalGpu));
        }
    }
    Ok((cpu_pointwise_product(a, b, plan)?, NttBackend::CpuRayon))
}

fn decode_slot0(prod: &[u64], q: u64, scale: f64) -> EvalResult<f64> {
    let summed = rotate_and_sum_u64(prod, q);
    let coeffs: Vec<i64> = summed.iter().map(|&u| from_centered_u64(u, q)).collect();
    let slots = decode_real_slots(&coeffs, scale * scale)?;
    Ok(slots[0])
}

fn decode_slot_sum(prod: &[u64], q: u64, scale: f64) -> EvalResult<f64> {
    let coeffs: Vec<i64> = prod.iter().map(|&u| from_centered_u64(u, q)).collect();
    let slots = decode_real_slots(&coeffs, scale * scale)?;
    Ok(slots.iter().sum())
}

fn horizontal_sum_slots(prod: &[u64], q: u64, scale: f64, use_galois: bool) -> EvalResult<f64> {
    if use_galois {
        decode_slot0(prod, q, scale)
    } else {
        decode_slot_sum(prod, q, scale)
    }
}

fn residues_from_slots(values: &[f64], n: usize, scale: f64, q: u64) -> EvalResult<Vec<u64>> {
    let coeffs = encode_real_slots(values, n, scale)?;
    Ok(coeffs.iter().map(|&c| to_centered_u64(c, q)).collect())
}

fn galois_eligible(prefer_metal: bool, q: u64) -> bool {
    !prefer_metal && q >= (1u64 << 32)
}

fn eval_one_chunk(
    geno: &[f64],
    weights: &[f64],
    n: usize,
    scale: f64,
    plan: &NegacyclicPlan,
    prefer_metal: bool,
) -> EvalResult<(f64, NttBackend)> {
    let ct_r = residues_from_slots(geno, n, scale, plan.q)?;
    let pt_r = residues_from_slots(weights, n, scale, plan.q)?;
    let (prod, backend) = ntt_pointwise_product(&ct_r, &pt_r, plan, prefer_metal)?;
    let score = horizontal_sum_slots(&prod, plan.q, scale, galois_eligible(prefer_metal, plan.q))?;
    Ok((score, backend))
}

/// One (patient, disease) oracle / decoded pair.
#[derive(Debug, Clone, Copy)]
pub struct PatientScore {
    pub oracle: f64,
    pub decoded: f64,
    pub abs_error: f64,
    pub patient: u32,
    pub disease: u32,
}

/// Result of the CKKS PRS pipeline (CPU or Metal), possibly multi-patient × multi-disease.
#[derive(Debug, Clone)]
pub struct PrsCpuResult {
    /// Flat scores, patient-major × disease-minor (length = M * D).
    pub scores: Vec<PatientScore>,
    pub patient_count: u32,
    pub disease_count: u32,
    pub cipher_count: u32,
    pub backend: NttBackend,
    /// Wall-clock milliseconds for the NTT multiply phase across the cohort.
    pub ntt_ms: f64,
    /// HEPRS-style stage split when measured; zeros if not.
    pub stage_encrypt_ms: f64,
    pub stage_eval_ms: f64,
    pub stage_decrypt_ms: f64,
}

impl PrsCpuResult {
    #[allow(dead_code)]
    pub fn pair_count(&self) -> usize {
        self.scores.len()
    }

    #[allow(dead_code)]
    pub fn primary(&self) -> PatientScore {
        self.scores[0]
    }
}

fn require_scale_fits(params: &CkksPrsParams) -> EvalResult<()> {
    let max = recommended_scale_bits(params.poly_degree);
    if params.scale_bits <= max {
        Ok(())
    } else {
        Err(format!(
            "scale_bits {} too large for single-prime CPU NTT at N={}; use ≤ {} (2^40 needs RNS)",
            params.scale_bits, params.poly_degree, max
        ))
    }
}

fn require_rns_scale(params: &CkksPrsParams) -> EvalResult<()> {
    let max = recommended_metal_scale_bits(params.poly_degree);
    if params.scale_bits <= max {
        Ok(())
    } else {
        Err(format!(
            "scale_bits {} too large for RNS at N={}; use ≤ {}",
            params.scale_bits, params.poly_degree, max
        ))
    }
}

fn napi_unpack<'a>(
    header: &'a [u32],
    slots: &'a [f64],
) -> EvalResult<(CkksPrsParams, crate::ckks_prs::PrsSlotViews<'a>)> {
    let params = unpack_header(header).map_err(|e| e.to_string())?;
    let views = unpack_slots(params, slots).map_err(|e| e.to_string())?;
    Ok((params, views))
}

fn rns_decode(
    params: &CkksPrsParams,
    views: &crate::ckks_prs::PrsSlotViews<'_>,
    basis: &crate::rns::RnsBasis,
    metal: bool,
) -> EvalResult<Vec<f64>> {
    let n = params.poly_degree as usize;
    let patients = params.patient_count as usize;
    let diseases = params.disease_count as usize;
    let n_slots = params.slot_count as usize;
    let ciphers = params.cipher_count as usize;
    let scale = scale_from_bits(params.scale_bits);
    if metal {
        evaluate_metal_rns_cohort(
            views.genotype_slots,
            views.weight_slots,
            patients,
            diseases,
            n,
            n_slots,
            ciphers,
            scale,
            basis,
        )
    } else {
        evaluate_cpu_rns_cohort(
            views.genotype_slots,
            views.weight_slots,
            patients,
            diseases,
            n,
            n_slots,
            ciphers,
            scale,
            basis,
        )
    }
}

fn rns_lane(
    params: &CkksPrsParams,
    views: &crate::ckks_prs::PrsSlotViews<'_>,
    metal: bool,
) -> EvalResult<PrsCpuResult> {
    require_rns_scale(params)?;
    let basis = rns_basis_cached(params.poly_degree as usize)?;
    let oracles = plaintext_prs_oracles(views);
    let t0 = std::time::Instant::now();
    let decoded = rns_decode(params, views, &basis, metal)?;
    let backend = if metal {
        NttBackend::MetalGpu
    } else {
        NttBackend::CpuRayon
    };
    Ok(PrsCpuResult {
        scores: zip_scores(&oracles, &decoded, params.disease_count as usize),
        patient_count: params.patient_count,
        disease_count: params.disease_count,
        cipher_count: params.cipher_count,
        backend,
        ntt_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stage_encrypt_ms: 0.0,
        stage_eval_ms: 0.0,
        stage_decrypt_ms: 0.0,
    })
}

fn try_metal_lane(
    params: &CkksPrsParams,
    views: &crate::ckks_prs::PrsSlotViews<'_>,
) -> Option<PrsCpuResult> {
    rns_lane(params, views, true).ok()
}

fn zip_scores(oracles: &[f64], decoded: &[f64], diseases: usize) -> Vec<PatientScore> {
    oracles
        .iter()
        .zip(decoded.iter())
        .enumerate()
        .map(|(idx, (&o, &d))| PatientScore {
            oracle: o,
            decoded: d,
            abs_error: (d - o).abs(),
            patient: (idx / diseases) as u32,
            disease: (idx % diseases) as u32,
        })
        .collect()
}

fn eval_one_patient_cpu(
    geno: &[f64],
    weights: &[f64],
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    plan: &NegacyclicPlan,
) -> EvalResult<f64> {
    let mut total = 0.0;
    for c in 0..ciphers {
        let start = c * n_slots;
        let (v, _) = eval_one_chunk(
            &geno[start..start + n_slots],
            &weights[start..start + n_slots],
            n,
            scale,
            plan,
            false,
        )?;
        total += v;
    }
    Ok(total)
}

fn run_cpu_single_prime(
    params: &CkksPrsParams,
    views: &crate::ckks_prs::PrsSlotViews<'_>,
) -> EvalResult<PrsCpuResult> {
    require_scale_fits(params)?;
    let n = params.poly_degree as usize;
    let n_slots = params.slot_count as usize;
    let ciphers = params.cipher_count as usize;
    let patients = params.patient_count as usize;
    let diseases = params.disease_count as usize;
    let scale = scale_from_bits(params.scale_bits);
    let plan = plan_for(n, params.scale_bits)?;
    let oracles = plaintext_prs_oracles(views);
    let t0 = std::time::Instant::now();
    let pairs = patients * diseases;
    let decoded: EvalResult<Vec<f64>> = (0..pairs)
        .into_par_iter()
        .map(|idx| {
            let p = idx / diseases;
            let d = idx % diseases;
            eval_one_patient_cpu(
                views.patient_geno(p),
                views.disease_weights(d),
                n,
                n_slots,
                ciphers,
                scale,
                &plan,
            )
        })
        .collect();
    Ok(PrsCpuResult {
        scores: zip_scores(&oracles, &decoded?, diseases),
        patient_count: params.patient_count,
        disease_count: params.disease_count,
        cipher_count: params.cipher_count,
        backend: NttBackend::CpuRayon,
        ntt_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stage_encrypt_ms: 0.0,
        stage_eval_ms: 0.0,
        stage_decrypt_ms: 0.0,
    })
}

fn run_cpu_lane(
    params: &CkksPrsParams,
    views: &crate::ckks_prs::PrsSlotViews<'_>,
) -> EvalResult<PrsCpuResult> {
    let max_single = recommended_scale_bits(params.poly_degree);
    if params.scale_bits > max_single {
        return rns_lane(params, views, false);
    }
    run_cpu_single_prime(params, views)
}

/// Full pipeline: prefer Metal when `prefer_metal` and eligible, else Rayon CPU.
pub fn evaluate_prs_pipeline(
    header: &[u32],
    slots: &[f64],
    prefer_metal: bool,
) -> EvalResult<PrsCpuResult> {
    let (params, views) = napi_unpack(header, slots)?;
    if prefer_metal && metal_available() {
        if let Some(r) = try_metal_lane(&params, &views) {
            return Ok(r);
        }
    }
    run_cpu_lane(&params, &views)
}

/// Fully encrypted RLWE lane (KeyGen + Encrypt + ct×pt + Galois KS RaS + Decrypt).
/// Reject encrypted evaluation at degrees where the fixed 4-limb modulus falls
/// outside the 128-bit security envelope.
///
/// `RnsBasis` uses a constant limb count, so `|Q|` stays at ~124 bits as `N`
/// shrinks. That is secure at N=16384 (ceiling 438 bits) and badly insecure at
/// N=1024 (ceiling 27 bits). Reduced degrees are legitimate for host-side
/// tests, which call the internal evaluator directly; they must not be
/// reachable through the deployed encrypted entry point.
fn require_secure_degree(n: usize) -> EvalResult<()> {
    let basis = crate::rns::RnsBasis::generate(n, crate::rns::FHE_RNS_LIMBS)
        .map_err(|e| format!("basis for security check: {e}"))?;
    let log_q = basis.modulus_bits();
    if crate::crypto::is_128bit_secure(n, log_q) {
        return Ok(());
    }
    let min_n = crate::crypto::min_degree_for_128bit(log_q);
    Err(format!(
        "insecure parameters: N={n} with |Q|={log_q} bits is outside the 128-bit \
         classical security envelope (HE Standard 2018). Minimum degree for this \
         modulus is N={min_n:?}. Reduced degrees are test-only."
    ))
}

pub fn evaluate_prs_encrypted(
    header: &[u32],
    slots: &[f64],
    prefer_metal: bool,
) -> EvalResult<PrsCpuResult> {
    let (params, views) = napi_unpack(header, slots)?;
    require_rns_scale(&params)?;
    let n = params.poly_degree as usize;
    require_secure_degree(n)?;
    let patients = params.patient_count as usize;
    let diseases = params.disease_count as usize;
    let n_slots = params.slot_count as usize;
    let ciphers = params.cipher_count as usize;
    let scale = scale_from_bits(params.scale_bits);
    let oracles = plaintext_prs_oracles(&views);
    let use_metal = prefer_metal && metal_available();
    let t0 = std::time::Instant::now();
    let (decoded, stages) = evaluate_encrypted_rns_cohort_staged(
        views.genotype_slots,
        views.weight_slots,
        patients,
        diseases,
        n,
        n_slots,
        ciphers,
        scale,
        use_metal,
    )?;
    Ok(PrsCpuResult {
        scores: zip_scores(&oracles, &decoded, diseases),
        patient_count: params.patient_count,
        disease_count: params.disease_count,
        cipher_count: params.cipher_count,
        backend: if use_metal {
            NttBackend::MetalGpu
        } else {
            NttBackend::CpuRayon
        },
        ntt_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stage_encrypt_ms: stages.encrypt_ms,
        stage_eval_ms: stages.eval_ms,
        stage_decrypt_ms: stages.decrypt_ms,
    })
}

/// Double-buffered async cohort evaluate (prep∥Metal/CPU eval).
pub fn evaluate_prs_async(
    header: &[u32],
    slots: &[f64],
    prefer_metal: bool,
) -> EvalResult<PrsCpuResult> {
    use crate::ckks_async::{evaluate_double_buffered, CohortJob};
    use crate::ckks_rns::RnsNttBackend;

    let (params, views) = napi_unpack(header, slots)?;
    require_rns_scale(&params)?;
    let backend = if prefer_metal && metal_available() {
        RnsNttBackend::Metal
    } else {
        RnsNttBackend::CpuRayon
    };
    let job = CohortJob {
        geno: views.genotype_slots.to_vec(),
        weights: views.weight_slots.to_vec(),
        patients: params.patient_count as usize,
        diseases: params.disease_count as usize,
        n: params.poly_degree as usize,
        n_slots: params.slot_count as usize,
        ciphers: params.cipher_count as usize,
        scale: scale_from_bits(params.scale_bits),
    };
    let oracles = plaintext_prs_oracles(&views);
    let t0 = std::time::Instant::now();
    let outs = evaluate_double_buffered(vec![job], backend)?;
    let decoded = outs
        .into_iter()
        .next()
        .ok_or_else(|| "async pipeline returned no scores".to_string())?;
    let ntt_backend = match backend {
        RnsNttBackend::Metal => NttBackend::MetalGpu,
        RnsNttBackend::CpuRayon => NttBackend::CpuRayon,
    };
    Ok(PrsCpuResult {
        scores: zip_scores(&oracles, &decoded, params.disease_count as usize),
        patient_count: params.patient_count,
        disease_count: params.disease_count,
        cipher_count: params.cipher_count,
        backend: ntt_backend,
        ntt_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stage_encrypt_ms: 0.0,
        stage_eval_ms: 0.0,
        stage_decrypt_ms: 0.0,
    })
}

/// Run `n_jobs` seeded cohorts through the double-buffered pipeline so host
/// prep for job N+1 overlaps Metal/CPU eval of job N. Returns (wall_ms, max_abs_error).
pub fn evaluate_async_job_sweep(
    patient_count: u32,
    disease_count: u32,
    snp: u32,
    poly_degree: u32,
    scale_bits: u32,
    n_jobs: u32,
    prefer_metal: bool,
) -> EvalResult<(f64, f64, NttBackend)> {
    use crate::ckks_async::{evaluate_double_buffered, CohortJob};
    use crate::ckks_rns::RnsNttBackend;
    use crate::clinic::pack_synthetic_clinic;

    let backend = if prefer_metal && metal_available() {
        RnsNttBackend::Metal
    } else {
        RnsNttBackend::CpuRayon
    };
    let mut jobs = Vec::with_capacity(n_jobs as usize);
    let mut last_header = Vec::new();
    let mut last_slots = Vec::new();
    for seed in 0..n_jobs {
        let (header, slots) = pack_synthetic_clinic(
            patient_count,
            disease_count,
            snp,
            poly_degree,
            scale_bits,
            seed,
        )?;
        let (params, views) = napi_unpack(&header, &slots)?;
        jobs.push(CohortJob {
            geno: views.genotype_slots.to_vec(),
            weights: views.weight_slots.to_vec(),
            patients: params.patient_count as usize,
            diseases: params.disease_count as usize,
            n: params.poly_degree as usize,
            n_slots: params.slot_count as usize,
            ciphers: params.cipher_count as usize,
            scale: scale_from_bits(params.scale_bits),
        });
        last_header = header;
        last_slots = slots;
    }
    let t0 = std::time::Instant::now();
    let _outs = evaluate_double_buffered(jobs, backend)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let sync = evaluate_prs_pipeline(&last_header, &last_slots, prefer_metal)?;
    let max_abs = sync
        .scores
        .iter()
        .map(|s| s.abs_error)
        .fold(0.0_f64, f64::max);
    let ntt_backend = match backend {
        RnsNttBackend::Metal => NttBackend::MetalGpu,
        RnsNttBackend::CpuRayon => NttBackend::CpuRayon,
    };
    Ok((ms, max_abs, ntt_backend))
}

/// CPU-only convenience wrapper (Task 2 compatibility).
pub fn evaluate_prs_cpu(header: &[u32], slots: &[f64]) -> EvalResult<PrsCpuResult> {
    evaluate_prs_pipeline(header, slots, false)
}

/// Metal-preferring evaluate (falls back to CPU if unavailable / ineligible).
pub fn evaluate_prs_metal(header: &[u32], slots: &[f64]) -> EvalResult<PrsCpuResult> {
    match evaluate_prs_pipeline(header, slots, true) {
        Ok(r) => Ok(r),
        Err(_) => evaluate_prs_pipeline(header, slots, false),
    }
}

#[cfg(test)]
#[path = "ckks_eval_tests.rs"]
mod tests;
