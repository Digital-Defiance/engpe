//! RNS CKKS multi-disease cohort: encode genotypes once, broadcast across D
//! disease weight panels (Task 7 SIMD sweep).
//!
//! Host encode + CRT decompose run on Rayon. Per limb, patient cipher residues
//! are expanded across D diseases (no genotype re-encode) and Metal/CPU batch
//! NTT-muls `M × D × cipher_count` pairs in one dispatch.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::ckks_encode::{decode_real_slots_i128, encode_real_slots};
use crate::ckks_rotate::rotate_and_sum_zq;
use crate::crypto::{Ciphertext, EncryptedStageMs};
use crate::metal_ntt::metal_batch_mul_pairs;
use crate::ntt::{
    forward_ntt_negacyclic, inverse_ntt_negacyclic, mul_mod, NegacyclicPlan,
};
use crate::rns::{RnsBasis, DEFAULT_RNS_LIMBS, FHE_RNS_LIMBS};
use crate::zq::{center_i128_to_zq, zq_to_center_i128, Zq};

pub type RnsEvalResult<T> = Result<T, String>;

/// Which NTT engine multiplies each RNS limb batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnsNttBackend {
    Metal,
    CpuRayon,
}

/// `decomposed[poly_idx][limb][coeff]`
pub type Decomposed = Vec<Vec<Vec<u64>>>;

fn encode_one(
    values: &[f64],
    start: usize,
    n_slots: usize,
    n: usize,
    scale: f64,
) -> RnsEvalResult<Vec<i64>> {
    encode_real_slots(&values[start..start + n_slots], n, scale)
}

fn encode_panels(
    values: &[f64],
    panels: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
) -> RnsEvalResult<Vec<Vec<i64>>> {
    let per_side = ciphers * n_slots;
    let total = panels * ciphers;
    (0..total)
        .into_par_iter()
        .map(|idx| {
            let start = (idx / ciphers) * per_side + (idx % ciphers) * n_slots;
            encode_one(values, start, n_slots, n, scale)
        })
        .collect()
}

fn decompose_all(polys: &[Vec<i64>], basis: &RnsBasis) -> RnsEvalResult<Decomposed> {
    polys.par_iter().map(|p| basis.decompose_poly(p)).collect()
}

fn pack_limb(decomposed: &Decomposed, limb: usize, n: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(decomposed.len() * n);
    for limbs in decomposed {
        out.extend_from_slice(&limbs[limb]);
    }
    out
}

/// Expand patient limb polys across D diseases (copy residues, no re-encode).
fn broadcast_cts(ct_limb: &[u64], patients: usize, diseases: usize, ciphers: usize, n: usize) -> Vec<u64> {
    let panel = ciphers * n;
    let mut out = Vec::with_capacity(patients * diseases * panel);
    for p in 0..patients {
        let slice = &ct_limb[p * panel..(p + 1) * panel];
        for _ in 0..diseases {
            out.extend_from_slice(slice);
        }
    }
    out
}

/// Tile disease limb polys once per patient so pairs align with `broadcast_cts`.
fn broadcast_pts(pt_limb: &[u64], patients: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(patients * pt_limb.len());
    for _ in 0..patients {
        out.extend_from_slice(pt_limb);
    }
    out
}

fn limb_plans(basis: &RnsBasis) -> RnsEvalResult<Vec<NegacyclicPlan>> {
    let mut plans = Vec::with_capacity(basis.limb_count());
    for &q in &basis.primes {
        let plan =
            NegacyclicPlan::new(basis.n, q).ok_or_else(|| format!("failed limb plan q={q}"))?;
        plans.push(plan);
    }
    Ok(plans)
}

fn cpu_mul_one_pair(a: &[u64], b: &[u64], plan: &NegacyclicPlan) -> RnsEvalResult<Vec<u64>> {
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

fn cpu_batch_mul_pairs(
    cts: &mut [u64],
    pts: &[u64],
    batch: usize,
    plan: &NegacyclicPlan,
) -> RnsEvalResult<()> {
    let n = plan.n;
    let products: RnsEvalResult<Vec<Vec<u64>>> = (0..batch)
        .into_par_iter()
        .map(|i| cpu_mul_one_pair(&cts[i * n..(i + 1) * n], &pts[i * n..(i + 1) * n], plan))
        .collect();
    for (i, prod) in products?.into_iter().enumerate() {
        cts[i * n..(i + 1) * n].copy_from_slice(&prod);
    }
    Ok(())
}

fn dispatch_limb_mul(
    cts: &mut [u64],
    pts: &[u64],
    batch: usize,
    plan: &NegacyclicPlan,
    limb: usize,
    backend: RnsNttBackend,
) -> RnsEvalResult<()> {
    match backend {
        RnsNttBackend::Metal => metal_batch_mul_pairs(cts, pts, batch, plan)
            .ok_or_else(|| format!("Metal batch mul refused on limb {limb}")),
        RnsNttBackend::CpuRayon => cpu_batch_mul_pairs(cts, pts, batch, plan),
    }
}

fn mul_one_limb(
    ct_dec: &Decomposed,
    pt_dec: &Decomposed,
    patients: usize,
    diseases: usize,
    ciphers: usize,
    plan: &NegacyclicPlan,
    limb: usize,
    backend: RnsNttBackend,
) -> RnsEvalResult<Vec<u64>> {
    let n = plan.n;
    let batch = patients * diseases * ciphers;
    let mut cts = broadcast_cts(&pack_limb(ct_dec, limb, n), patients, diseases, ciphers, n);
    let pts = broadcast_pts(&pack_limb(pt_dec, limb, n), patients);
    dispatch_limb_mul(&mut cts, &pts, batch, plan, limb, backend)?;
    Ok(cts)
}

fn mul_all_limbs(
    ct_dec: &Decomposed,
    pt_dec: &Decomposed,
    patients: usize,
    diseases: usize,
    ciphers: usize,
    basis: &RnsBasis,
    plans: &[NegacyclicPlan],
    backend: RnsNttBackend,
) -> RnsEvalResult<Vec<Vec<u64>>> {
    let k = basis.limb_count();
    let mut prod_limbs = Vec::with_capacity(k);
    for limb in 0..k {
        prod_limbs.push(mul_one_limb(
            ct_dec, pt_dec, patients, diseases, ciphers, &plans[limb], limb, backend,
        )?);
    }
    Ok(prod_limbs)
}

fn cipher_limb_slice(prod_limbs: &[Vec<u64>], basis: &RnsBasis, index: usize) -> Vec<Vec<u64>> {
    let n = basis.n;
    let start = index * n;
    prod_limbs
        .iter()
        .map(|limb| limb[start..start + n].to_vec())
        .collect()
}

fn recombine_products(
    prod_limbs: &[Vec<u64>],
    basis: &RnsBasis,
    poly_count: usize,
) -> RnsEvalResult<Vec<Vec<i128>>> {
    (0..poly_count)
        .into_par_iter()
        .map(|i| {
            let limbs = cipher_limb_slice(prod_limbs, basis, i);
            basis.recombine_poly(&limbs)
        })
        .collect()
}

fn score_from_wide_product(prod: &[i128], q: Zq, scale: f64) -> RnsEvalResult<f64> {
    let residues: Vec<Zq> = prod.iter().map(|&c| center_i128_to_zq(c, q)).collect();
    let summed = rotate_and_sum_zq(&residues, q);
    let centered: Vec<i128> = summed
        .iter()
        .map(|&u| zq_to_center_i128(u, q))
        .collect::<Result<Vec<_>, _>>()?;
    let slots = decode_real_slots_i128(&centered, scale * scale)?;
    Ok(slots[0])
}

fn pair_score_from_products(
    products: &[Vec<i128>],
    patient: usize,
    disease: usize,
    diseases: usize,
    ciphers: usize,
    q: Zq,
    scale: f64,
) -> RnsEvalResult<f64> {
    let start = (patient * diseases + disease) * ciphers;
    let mut total = 0.0;
    for c in 0..ciphers {
        total += score_from_wide_product(&products[start + c], q, scale)?;
    }
    Ok(total)
}

fn score_matrix(
    products: &[Vec<i128>],
    patients: usize,
    diseases: usize,
    ciphers: usize,
    q: Zq,
    scale: f64,
) -> RnsEvalResult<Vec<f64>> {
    let total = patients * diseases;
    (0..total)
        .into_par_iter()
        .map(|idx| {
            let p = idx / diseases;
            let d = idx % diseases;
            pair_score_from_products(products, p, d, diseases, ciphers, q, scale)
        })
        .collect()
}

/// Host-prepared RNS polys (encode + CRT) ready for limb NTT.
pub struct PreparedRnsCohort {
    pub ct_dec: Decomposed,
    pub pt_dec: Decomposed,
    pub patients: usize,
    pub diseases: usize,
    pub ciphers: usize,
    pub scale: f64,
    pub basis: Arc<RnsBasis>,
}

/// Encode + CRT-decompose genotypes and disease weights (Rayon host prep).
pub fn prepare_rns_cohort(
    geno: &[f64],
    weights: &[f64],
    patients: usize,
    diseases: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    basis: &Arc<RnsBasis>,
) -> RnsEvalResult<PreparedRnsCohort> {
    let ct_polys = encode_panels(geno, patients, n, n_slots, ciphers, scale)?;
    let pt_polys = encode_panels(weights, diseases, n, n_slots, ciphers, scale)?;
    let ct_dec = decompose_all(&ct_polys, basis)?;
    let pt_dec = decompose_all(&pt_polys, basis)?;
    Ok(PreparedRnsCohort {
        ct_dec,
        pt_dec,
        patients,
        diseases,
        ciphers,
        scale,
        basis: Arc::clone(basis),
    })
}

/// Limb NTT + CRT recombine + rotate-and-sum for a prepared cohort.
pub fn evaluate_prepared_rns(
    prep: &PreparedRnsCohort,
    backend: RnsNttBackend,
) -> RnsEvalResult<Vec<f64>> {
    let plans = limb_plans(&prep.basis)?;
    let prod_limbs = mul_all_limbs(
        &prep.ct_dec,
        &prep.pt_dec,
        prep.patients,
        prep.diseases,
        prep.ciphers,
        &prep.basis,
        &plans,
        backend,
    )?;
    let products = recombine_products(
        &prod_limbs,
        &prep.basis,
        prep.patients * prep.diseases * prep.ciphers,
    )?;
    score_matrix(
        &products,
        prep.patients,
        prep.diseases,
        prep.ciphers,
        prep.basis.modulus,
        prep.scale,
    )
}

/// RNS PRS: M patients × D diseases. Genotypes encoded once; broadcast across D.
///
/// Returns flat scores length `M * D` (patient-major, disease-minor).
pub fn evaluate_rns_cohort(
    geno: &[f64],
    weights: &[f64],
    patients: usize,
    diseases: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    basis: &RnsBasis,
    backend: RnsNttBackend,
) -> RnsEvalResult<Vec<f64>> {
    let arc = Arc::new(basis.clone());
    let prep = prepare_rns_cohort(
        geno, weights, patients, diseases, n, n_slots, ciphers, scale, &arc,
    )?;
    evaluate_prepared_rns(&prep, backend)
}

/// Metal convenience wrapper.
pub fn evaluate_metal_rns_cohort(
    geno: &[f64],
    weights: &[f64],
    patients: usize,
    diseases: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    basis: &RnsBasis,
) -> RnsEvalResult<Vec<f64>> {
    evaluate_rns_cohort(
        geno,
        weights,
        patients,
        diseases,
        n,
        n_slots,
        ciphers,
        scale,
        basis,
        RnsNttBackend::Metal,
    )
}

/// CPU Rayon RNS at the same scale / limb basis as Metal.
pub fn evaluate_cpu_rns_cohort(
    geno: &[f64],
    weights: &[f64],
    patients: usize,
    diseases: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    basis: &RnsBasis,
) -> RnsEvalResult<Vec<f64>> {
    evaluate_rns_cohort(
        geno,
        weights,
        patients,
        diseases,
        n,
        n_slots,
        ciphers,
        scale,
        basis,
        RnsNttBackend::CpuRayon,
    )
}

/// Cached RNS basis for degree `n` (`k = DEFAULT_RNS_LIMBS`).
pub fn rns_basis_cached(n: usize) -> RnsEvalResult<Arc<RnsBasis>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<RnsBasis>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().map_err(|_| "RNS cache poisoned".to_string())?;
    if let Some(b) = map.get(&n) {
        return Ok(Arc::clone(b));
    }
    let basis = RnsBasis::generate(n, DEFAULT_RNS_LIMBS)?;
    let arc = Arc::new(basis);
    map.insert(n, Arc::clone(&arc));
    Ok(arc)
}

/// Cached evaluation keys for the encrypted path (`k = FHE_RNS_LIMBS`).
pub fn eval_keys_cached(n: usize) -> RnsEvalResult<Arc<crate::crypto::EvaluationKeys>> {
    use crate::crypto::{setup_evaluation_keys, DEFAULT_NOISE_ETA, EvaluationKeys};
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<EvaluationKeys>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().map_err(|_| "eval-key cache poisoned".to_string())?;
    if let Some(k) = map.get(&n) {
        return Ok(Arc::clone(k));
    }
    let basis = RnsBasis::generate(n, FHE_RNS_LIMBS)?;
    let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA)?;
    let arc = Arc::new(keys);
    map.insert(n, Arc::clone(&arc));
    Ok(arc)
}

fn score_encrypted_patient_disease_cached(
    geno_side: &[f64],
    weight_cts: &[Ciphertext],
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    keys: &crate::crypto::EvaluationKeys,
    prefer_metal: bool,
    stages: &mut EncryptedStageMs,
) -> RnsEvalResult<f64> {
    use crate::ckks_encode::decode_real_slots_i128;
    use crate::crypto::{
        add_ct, decrypt, encrypt_encoded, mul_ct_ct, Ciphertext, DEFAULT_NOISE_ETA,
    };
    let q = keys.basis.modulus;
    let mut acc: Option<Ciphertext> = None;
    for c in 0..ciphers {
        let g0 = c * n_slots;
        let t_enc = std::time::Instant::now();
        let ca = encode_real_slots(&geno_side[g0..g0 + n_slots], n, scale)?;
        let ct_g = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA)?;
        stages.encrypt_ms += t_enc.elapsed().as_secs_f64() * 1e3;

        let t_ev = std::time::Instant::now();
        let prod = mul_ct_ct(&ct_g, &weight_cts[c], &keys.relin, &keys.basis, prefer_metal)?;
        acc = Some(match acc {
            None => prod,
            Some(prev) => add_ct(&prev, &prod, q)?,
        });
        stages.eval_ms += t_ev.elapsed().as_secs_f64() * 1e3;
    }
    let acc = acc.ok_or_else(|| "no ciphertexts for patient/disease".to_string())?;
    let t_ev = std::time::Instant::now();
    let summed = crate::crypto::rotate_and_sum_encrypted_backend(
        &acc,
        &keys.galois,
        &keys.basis,
        prefer_metal,
    )?;
    stages.eval_ms += t_ev.elapsed().as_secs_f64() * 1e3;

    let t_dec = std::time::Instant::now();
    let phase = decrypt(&keys.sk, &summed, q)?;
    let slots = decode_real_slots_i128(&phase, scale * scale)?;
    stages.decrypt_ms += t_dec.elapsed().as_secs_f64() * 1e3;
    Ok(slots[0])
}

fn encrypt_weight_blocks(
    weight_side: &[f64],
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    keys: &crate::crypto::EvaluationKeys,
    stages: &mut EncryptedStageMs,
) -> RnsEvalResult<Vec<Ciphertext>> {
    use crate::crypto::{encrypt_encoded, DEFAULT_NOISE_ETA};
    let mut out = Vec::with_capacity(ciphers);
    for c in 0..ciphers {
        let w0 = c * n_slots;
        let t_enc = std::time::Instant::now();
        let cb = encode_real_slots(&weight_side[w0..w0 + n_slots], n, scale)?;
        let ct_w = encrypt_encoded(&keys.pk, &cb, DEFAULT_NOISE_ETA)?;
        stages.encrypt_ms += t_enc.elapsed().as_secs_f64() * 1e3;
        out.push(ct_w);
    }
    Ok(out)
}

/// Score one (patient, disease) by accumulating every ciphertext×ciphertext
/// product (with relinearization) first, then folding with a single rotate-and-sum.
///
/// This matches HEPRS Algorithm 1 (`MulRelinNew` + `InnerSumLog` once outside
/// the SNP loop): \(K\) ciphertexts cost \(K\) relinearizations plus
/// \(\log_2(N/2)\) rotation key-switches, not \(K \cdot \log_2(N/2)\) folds.
/// Partial sums are linear, so the result is identical to folding each
/// ciphertext and summing the scalars.
fn score_encrypted_patient_disease(
    geno_side: &[f64],
    weight_side: &[f64],
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    keys: &crate::crypto::EvaluationKeys,
    prefer_metal: bool,
) -> RnsEvalResult<f64> {
    let mut stages = EncryptedStageMs::default();
    let weight_cts = encrypt_weight_blocks(
        weight_side, n, n_slots, ciphers, scale, keys, &mut stages,
    )?;
    score_encrypted_patient_disease_cached(
        geno_side, &weight_cts, n, n_slots, ciphers, scale, keys, prefer_metal, &mut stages,
    )
}

/// Fully encrypted RLWE PRS: encrypt genotypes and weights, ct×ct multiply
/// with relinearization, Galois key-switched rotate-and-sum, decrypt.
/// Returns flat M×D scores.
pub fn evaluate_encrypted_rns_cohort(
    geno: &[f64],
    weights: &[f64],
    patients: usize,
    diseases: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
) -> RnsEvalResult<Vec<f64>> {
    evaluate_encrypted_rns_cohort_backend(
        geno, weights, patients, diseases, n, n_slots, ciphers, scale, true,
    )
}

/// Encrypted RLWE PRS with explicit Metal KS preference.
pub fn evaluate_encrypted_rns_cohort_backend(
    geno: &[f64],
    weights: &[f64],
    patients: usize,
    diseases: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    prefer_metal: bool,
) -> RnsEvalResult<Vec<f64>> {
    Ok(evaluate_encrypted_rns_cohort_staged(
        geno, weights, patients, diseases, n, n_slots, ciphers, scale, prefer_metal,
    )?
    .0)
}

/// Like [`evaluate_encrypted_rns_cohort_backend`], plus HEPRS-style stage timings.
///
/// Weight ciphertexts are encrypted **once per disease** and reused across every
/// patient — the dominant win on multi-patient panels.
pub fn evaluate_encrypted_rns_cohort_staged(
    geno: &[f64],
    weights: &[f64],
    patients: usize,
    diseases: usize,
    n: usize,
    n_slots: usize,
    ciphers: usize,
    scale: f64,
    prefer_metal: bool,
) -> RnsEvalResult<(Vec<f64>, EncryptedStageMs)> {
    let keys = eval_keys_cached(n)?;
    let per_side = ciphers * n_slots;
    let mut stages = EncryptedStageMs::default();

    // Encrypt each disease's weight blocks once (wall-clock).
    let t_w = std::time::Instant::now();
    let mut weight_cts: Vec<Vec<Ciphertext>> = Vec::with_capacity(diseases);
    for d in 0..diseases {
        let w0 = d * per_side;
        let mut local = EncryptedStageMs::default();
        weight_cts.push(encrypt_weight_blocks(
            &weights[w0..w0 + per_side],
            n,
            n_slots,
            ciphers,
            scale,
            &keys,
            &mut local,
        )?);
    }
    stages.encrypt_ms = t_w.elapsed().as_secs_f64() * 1e3;

    // Parallel over patients; stage counters inside are thread-local sums — we
    // report wall-clock of this section as eval (includes per-patient encrypt /
    // fold / decrypt), matching how HEPRS `-print` reports a single "Run model"
    // span rather than a sum of threads.
    let t_p = std::time::Instant::now();
    let patient_results: RnsEvalResult<Vec<Vec<f64>>> = (0..patients)
        .into_par_iter()
        .map(|p| {
            let g0 = p * per_side;
            let mut local = EncryptedStageMs::default();
            let mut row = Vec::with_capacity(diseases);
            for d in 0..diseases {
                let score = score_encrypted_patient_disease_cached(
                    &geno[g0..g0 + per_side],
                    &weight_cts[d],
                    n,
                    n_slots,
                    ciphers,
                    scale,
                    &keys,
                    prefer_metal,
                    &mut local,
                )?;
                row.push(score);
            }
            Ok(row)
        })
        .collect();
    let patient_results = patient_results?;
    stages.eval_ms = t_p.elapsed().as_secs_f64() * 1e3;
    // Decrypt is inside the per-patient path; not separated under Rayon.
    stages.decrypt_ms = 0.0;

    let mut scores = Vec::with_capacity(patients * diseases);
    for row in patient_results {
        scores.extend(row);
    }
    Ok((scores, stages))
}
