//! ENGPE native addon — CKKS PRS FFI entry points (CPU + Metal).

use napi::bindgen_prelude::*;
use napi_derive::napi;

mod ckks_encode;
mod ckks_eval;
mod ckks_prs;
mod ckks_rns;
mod ckks_async;
mod ckks_rotate;
mod clinic;
mod crypto;
mod metal_ks;
mod metal_ntt;
mod ntt;
mod rns;
mod zq;

#[cfg(test)]
mod audit_gates;
mod packing;
#[cfg(test)]
mod validation_gates;

pub use clinic::{evaluate_clinic_sweep_inner, ClinicScoreCell, ClinicSweepResult};

use ckks_eval::{
    evaluate_async_job_sweep, evaluate_prs_async, evaluate_prs_cpu, evaluate_prs_encrypted,
    evaluate_prs_metal, NttBackend,
};
use ckks_prs::{unpack_header, unpack_slots, PrsSlotViews};
use metal_ntt::metal_available;

fn to_napi(err: String) -> Error {
    Error::new(Status::GenericFailure, err)
}

fn backend_code(b: NttBackend) -> f64 {
    match b {
        NttBackend::CpuRayon => 0.0,
        NttBackend::MetalGpu => 1.0,
    }
}

/// Pack cohort result:
/// `[patientCount, diseaseCount, backend, nttMs, cipherCount, accepted,
///   o0, d0, e0, o1, d1, e1, …]`  (patient-major × disease-minor)
fn pack_result(result: ckks_eval::PrsCpuResult) -> Float64Array {
    let pairs = result.scores.len();
    let mut out = Vec::with_capacity(6 + 3 * pairs);
    out.push(result.patient_count as f64);
    out.push(result.disease_count as f64);
    out.push(backend_code(result.backend));
    out.push(result.ntt_ms);
    out.push(result.cipher_count as f64);
    out.push(1.0);
    for s in &result.scores {
        out.push(s.oracle);
        out.push(s.decoded);
        out.push(s.abs_error);
    }
    Float64Array::new(out)
}

/// Extended pack with HEPRS-style stages after the standard header:
/// `[…accepted, encryptMs, evalMs, decryptMs, o0, d0, e0, …]`
fn pack_result_staged(result: ckks_eval::PrsCpuResult) -> Float64Array {
    let pairs = result.scores.len();
    let mut out = Vec::with_capacity(9 + 3 * pairs);
    out.push(result.patient_count as f64);
    out.push(result.disease_count as f64);
    out.push(backend_code(result.backend));
    out.push(result.ntt_ms);
    out.push(result.cipher_count as f64);
    out.push(1.0);
    out.push(result.stage_encrypt_ms);
    out.push(result.stage_eval_ms);
    out.push(result.stage_decrypt_ms);
    for s in &result.scores {
        out.push(s.oracle);
        out.push(s.decoded);
        out.push(s.abs_error);
    }
    Float64Array::new(out)
}

/// Validate header + slots; returns `cipher_count` on success.
#[napi]
pub fn validate_prs_batch(header: Uint32Array, slots: Float64Array) -> Result<u32> {
    let params = unpack_header(header.as_ref())?;
    let _views = unpack_slots(params, slots.as_ref())?;
    Ok(params.cipher_count)
}

/// Echo the first `count` genotype slots after unpack (roundtrip test aid).
#[napi]
pub fn echo_prs_genotype_slots(
    header: Uint32Array,
    slots: Float64Array,
    count: u32,
) -> Result<Float64Array> {
    let params = unpack_header(header.as_ref())?;
    let views: PrsSlotViews<'_> = unpack_slots(params, slots.as_ref())?;
    let n = (count as usize).min(views.params.snp_count as usize);
    let geno0 = views.patient_geno(0);
    let out: Vec<f64> = geno0[..n].to_vec();
    Ok(Float64Array::new(out))
}

/// Whether Metal NTT pipelines are available in this process.
#[napi]
pub fn is_metal_available() -> bool {
    metal_available()
}

/// CPU CKKS PRS evaluate (cohort-capable).
#[napi]
pub fn evaluate_prs_stub(header: Uint32Array, slots: Float64Array) -> Result<Float64Array> {
    let result = evaluate_prs_cpu(header.as_ref(), slots.as_ref()).map_err(to_napi)?;
    Ok(pack_result(result))
}

/// Prefer Metal GPU path; fall back to CPU.
#[napi]
pub fn evaluate_prs(header: Uint32Array, slots: Float64Array) -> Result<Float64Array> {
    let result = evaluate_prs_metal(header.as_ref(), slots.as_ref()).map_err(to_napi)?;
    Ok(pack_result(result))
}

/// Number of leading summary values in `evaluate_clinic_sweep` output.
const CLINIC_HEADER_LEN: usize = 6;
/// Values per (patient, disease) cell: `[patient, disease, oracle, decoded, absError]`.
const CLINIC_CELL_LEN: usize = 5;

/// Offline clinic sweep — same engine entry point as the Tauri
/// `evaluate_clinic_sweep` command, exposed for the air-gap test gate.
///
/// Returns `[patientCount, diseaseCount, backend, nttMs, maxAbsError, airgapped,
///           p, d, oracle, decoded, absError, …]`
#[napi]
pub fn evaluate_clinic_sweep(
    patient_count: u32,
    disease_count: u32,
    seed: u32,
) -> Result<Float64Array> {
    let sweep =
        evaluate_clinic_sweep_inner(patient_count, disease_count, seed).map_err(to_napi)?;
    let mut out = Vec::with_capacity(CLINIC_HEADER_LEN + CLINIC_CELL_LEN * sweep.scores.len());
    out.push(sweep.patient_count as f64);
    out.push(sweep.disease_count as f64);
    out.push(if sweep.backend == "metal" { 1.0 } else { 0.0 });
    out.push(sweep.ntt_ms);
    out.push(sweep.max_abs_error);
    out.push(if sweep.airgapped { 1.0 } else { 0.0 });
    for cell in &sweep.scores {
        out.push(cell.patient as f64);
        out.push(cell.disease as f64);
        out.push(cell.plaintext);
        out.push(cell.decoded);
        out.push(cell.abs_error);
    }
    Ok(Float64Array::new(out))
}

/// Explicit backend select: `prefer_metal` tries GPU when true.
#[napi]
pub fn evaluate_prs_pipeline_napi(
    header: Uint32Array,
    slots: Float64Array,
    prefer_metal: bool,
) -> Result<Float64Array> {
    let result = if prefer_metal {
        evaluate_prs_metal(header.as_ref(), slots.as_ref()).map_err(to_napi)?
    } else {
        evaluate_prs_cpu(header.as_ref(), slots.as_ref()).map_err(to_napi)?
    };
    Ok(pack_result(result))
}

/// Fully encrypted RLWE evaluate (KeyGen/Encrypt/ct×ct+relin/Galois-KS/Decrypt).
/// `prefer_metal` routes Galois key-switching through the Metal KS pipeline.
#[napi]
pub fn evaluate_prs_encrypted_napi(
    header: Uint32Array,
    slots: Float64Array,
    prefer_metal: bool,
) -> Result<Float64Array> {
    let result =
        evaluate_prs_encrypted(header.as_ref(), slots.as_ref(), prefer_metal).map_err(to_napi)?;
    Ok(pack_result(result))
}

/// Same as `evaluate_prs_encrypted_napi` but prepends encrypt/eval/decrypt stage ms
/// after `accepted` (indices 6–8) before score triples.
#[napi]
pub fn evaluate_prs_encrypted_staged_napi(
    header: Uint32Array,
    slots: Float64Array,
    prefer_metal: bool,
) -> Result<Float64Array> {
    let result =
        evaluate_prs_encrypted(header.as_ref(), slots.as_ref(), prefer_metal).map_err(to_napi)?;
    Ok(pack_result_staged(result))
}

/// Streamed patient-packed ct×ct PRS (zero Galois KS; SNPs streamed).
///
/// Input: patient-major genotypes (`patients * snp_count`), weight vector,
/// secure `poly_degree`, `scale_bits`. Output layout matches
/// `evaluate_prs_encrypted_staged_napi`.
#[napi]
pub fn evaluate_prs_patient_packed_napi(
    geno: Float64Array,
    weights: Float64Array,
    patients: u32,
    snp_count: u32,
    poly_degree: u32,
    scale_bits: u32,
    prefer_metal: bool,
) -> Result<Float64Array> {
    use ckks_eval::{PatientScore, PrsCpuResult, NttBackend};
    use ckks_rns::eval_keys_cached;
    use crypto::is_128bit_secure;
    use packing::evaluate_patient_packed_ctct;
    use rns::RnsBasis;

    let n = poly_degree as usize;
    let patients = patients as usize;
    let snp_count = snp_count as usize;
    let scale = (scale_bits as f64).exp2();
    let basis = RnsBasis::generate(n, rns::FHE_RNS_LIMBS).map_err(to_napi)?;
    let log_q = basis.modulus_bits();
    if !is_128bit_secure(n, log_q) {
        return Err(to_napi(format!(
            "insecure parameters: N={n} with |Q|={log_q} bits"
        )));
    }
    let keys = eval_keys_cached(n).map_err(to_napi)?;
    let use_metal = prefer_metal && metal_available();
    let t0 = std::time::Instant::now();
    let packed = evaluate_patient_packed_ctct(
        geno.as_ref(),
        weights.as_ref(),
        patients,
        snp_count,
        &keys,
        scale,
        use_metal,
    )
    .map_err(to_napi)?;

    // Plaintext oracles for abs-error.
    let mut scores = Vec::with_capacity(patients);
    for p in 0..patients {
        let mut oracle = 0.0f64;
        for s in 0..snp_count {
            oracle += geno[p * snp_count + s] * weights[s];
        }
        let decoded = packed.scores[p];
        scores.push(PatientScore {
            oracle,
            decoded,
            abs_error: (decoded - oracle).abs(),
            patient: p as u32,
            disease: 0,
        });
    }
    let result = PrsCpuResult {
        scores,
        patient_count: patients as u32,
        disease_count: 1,
        cipher_count: packed.ciphertexts as u32,
        backend: if use_metal {
            NttBackend::MetalGpu
        } else {
            NttBackend::CpuRayon
        },
        ntt_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stage_encrypt_ms: packed.stages.encrypt_ms,
        stage_eval_ms: packed.stages.eval_ms,
        stage_decrypt_ms: packed.stages.decrypt_ms,
    };
    Ok(pack_result_staged(result))
}

/// Double-buffered async evaluate overlapping host prep with Metal/CPU NTT.
#[napi]
pub fn evaluate_prs_async_napi(
    header: Uint32Array,
    slots: Float64Array,
    prefer_metal: bool,
) -> Result<Float64Array> {
    let result =
        evaluate_prs_async(header.as_ref(), slots.as_ref(), prefer_metal).map_err(to_napi)?;
    Ok(pack_result(result))
}

/// Multi-job async sweep: `[wallMs, maxAbsError, backendCode, nJobs]`.
#[napi]
pub fn evaluate_async_job_sweep_napi(
    patient_count: u32,
    disease_count: u32,
    snp: u32,
    poly_degree: u32,
    scale_bits: u32,
    n_jobs: u32,
    prefer_metal: bool,
) -> Result<Float64Array> {
    let (ms, max_err, backend) = evaluate_async_job_sweep(
        patient_count,
        disease_count,
        snp,
        poly_degree,
        scale_bits,
        n_jobs,
        prefer_metal,
    )
    .map_err(to_napi)?;
    Ok(Float64Array::new(vec![
        ms,
        max_err,
        backend_code(backend),
        n_jobs as f64,
    ]))
}

/// Report CRT recombination latency (ms) for a synthetic degree-`n` poly.
#[napi]
pub fn benchmark_crt_recombine_ms(poly_degree: u32) -> Result<f64> {
    use rns::RnsBasis;
    use std::time::Instant;
    let n = poly_degree as usize;
    let basis = RnsBasis::generate(n, 4).map_err(to_napi)?;
    let coeffs: Vec<i64> = (0..n as i64)
        .map(|i| ((i * 1_000_000) % 1_000_000_007) - 500_000_000)
        .collect();
    let limbs = basis.decompose_poly(&coeffs).map_err(to_napi)?;
    let _ = basis.recombine_poly(&limbs).map_err(to_napi)?; // warmup
    let t0 = Instant::now();
    let _ = basis.recombine_poly(&limbs).map_err(to_napi)?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0)
}

/// Synthetic streamed patient-packed bench (no JS `patients×snps` matrix).
/// Returns the staged pack layout of `evaluate_prs_patient_packed_napi`.
#[napi]
pub fn bench_patient_packed_synthetic_napi(
    patients: u32,
    snp_count: u32,
    poly_degree: u32,
    scale_bits: u32,
    prefer_metal: bool,
) -> Result<Float64Array> {
    use ckks_eval::{PatientScore, PrsCpuResult, NttBackend};
    use ckks_rns::eval_keys_cached;
    use crypto::is_128bit_secure;
    use packing::evaluate_patient_packed_synthetic_stream;
    use rns::RnsBasis;

    let n = poly_degree as usize;
    let patients_usz = patients as usize;
    let snp_usz = snp_count as usize;
    let scale = (scale_bits as f64).exp2();
    let basis = RnsBasis::generate(n, rns::FHE_RNS_LIMBS).map_err(to_napi)?;
    let log_q = basis.modulus_bits();
    if !is_128bit_secure(n, log_q) {
        return Err(to_napi(format!(
            "insecure parameters: N={n} with |Q|={log_q} bits"
        )));
    }
    let keys = eval_keys_cached(n).map_err(to_napi)?;
    let use_metal = prefer_metal && metal_available();
    let t0 = std::time::Instant::now();
    let packed = evaluate_patient_packed_synthetic_stream(
        patients_usz,
        snp_usz,
        &keys,
        scale,
        use_metal,
    )
    .map_err(to_napi)?;

    let mut scores = Vec::with_capacity(patients_usz);
    for p in 0..patients_usz {
        let mut oracle = 0.0f64;
        for s in 0..snp_usz {
            let dosage = ((s * 7 + p * 3) % 3) as f64;
            let w = 1e-4 * (((s % 11) as f64) - 5.0);
            oracle += dosage * w;
        }
        let decoded = packed.scores[p];
        scores.push(PatientScore {
            oracle,
            decoded,
            abs_error: (decoded - oracle).abs(),
            patient: p as u32,
            disease: 0,
        });
    }
    let result = PrsCpuResult {
        scores,
        patient_count: patients,
        disease_count: 1,
        cipher_count: packed.ciphertexts as u32,
        backend: if use_metal {
            NttBackend::MetalGpu
        } else {
            NttBackend::CpuRayon
        },
        ntt_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stage_encrypt_ms: packed.stages.encrypt_ms,
        stage_eval_ms: packed.stages.eval_ms,
        stage_decrypt_ms: packed.stages.decrypt_ms,
    };
    Ok(pack_result_staged(result))
}
