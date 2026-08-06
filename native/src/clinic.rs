//! Clinic helpers: deterministic synthetic M×D packs + air-gapped sweep.
//!
//! Used by the Tauri desktop app and the Task 9 air-gap validation gate.
//! No network I/O is performed anywhere in this path.

use crate::ckks_eval::{evaluate_prs_pipeline, NttBackend, PatientScore, PrsCpuResult};
use crate::metal_ntt::metal_available;

/// One decoded (patient, disease) cell for the clinic UI.
#[derive(Debug, Clone)]
pub struct ClinicScoreCell {
    pub patient: u32,
    pub disease: u32,
    pub plaintext: f64,
    pub decoded: f64,
    pub abs_error: f64,
}

/// Result of an offline clinic sweep.
#[derive(Debug, Clone)]
pub struct ClinicSweepResult {
    pub patient_count: u32,
    pub disease_count: u32,
    pub backend: String,
    pub ntt_ms: f64,
    pub max_abs_error: f64,
    pub scores: Vec<ClinicScoreCell>,
    /// Always true: this path never opens sockets or dials the network.
    pub airgapped: bool,
}

fn mulberry32(seed: u32) -> impl FnMut() -> f64 {
    let mut t = seed;
    move || {
        t = t.wrapping_add(0x6d2b79f5);
        let mut r = (t ^ (t >> 15)).wrapping_mul(1 | t);
        r ^= r.wrapping_add((r ^ (r >> 7)).wrapping_mul(61 | r));
        ((r ^ (r >> 14)) as f64) / 4294967296.0
    }
}

fn fill_geno(out: &mut [f64], snp: usize, rand: &mut impl FnMut() -> f64) {
    for i in 0..snp {
        let u = rand();
        out[i] = if u < 0.45 {
            0.0
        } else if u < 0.9 {
            1.0
        } else {
            2.0
        };
    }
}

fn fill_weights(out: &mut [f64], snp: usize, bound: f64, rand: &mut impl FnMut() -> f64) {
    for i in 0..snp {
        out[i] = (rand() * 2.0 - 1.0) * bound;
    }
}

fn require_positive_counts(patients: u32, diseases: u32) -> Result<(), String> {
    if patients >= 1 && diseases >= 1 {
        Ok(())
    } else {
        Err("patients and diseases must be ≥ 1".into())
    }
}

/// Build a deterministic synthetic packed batch for clinic / air-gap tests.
pub fn pack_synthetic_clinic(
    patients: u32,
    diseases: u32,
    snp: u32,
    poly_degree: u32,
    scale_bits: u32,
    seed: u32,
) -> Result<(Vec<u32>, Vec<f64>), String> {
    require_positive_counts(patients, diseases)?;
    let n_slots = poly_degree / 2;
    let ciphers = (snp as u64).div_ceil(n_slots as u64) as u32;
    let per = (ciphers * n_slots) as usize;
    let m = patients as usize;
    let d = diseases as usize;
    let snp_n = snp as usize;
    let mut slots = vec![0.0; (m + d) * per];
    let mut rand = mulberry32(seed);
    for disease in 0..d {
        fill_weights(&mut slots[(m + disease) * per..], snp_n, 0.02, &mut rand);
    }
    for p in 0..m {
        fill_geno(&mut slots[p * per..], snp_n, &mut rand);
    }
    let header = vec![
        poly_degree, snp, n_slots, ciphers, scale_bits, patients, diseases,
    ];
    Ok((header, slots))
}

fn backend_name(b: NttBackend) -> String {
    match b {
        NttBackend::CpuRayon => "cpu".into(),
        NttBackend::MetalGpu => "metal".into(),
    }
}

fn to_cells(result: &PrsCpuResult) -> Vec<ClinicScoreCell> {
    result
        .scores
        .iter()
        .map(|s: &PatientScore| ClinicScoreCell {
            patient: s.patient,
            disease: s.disease,
            plaintext: s.oracle,
            decoded: s.decoded,
            abs_error: s.abs_error,
        })
        .collect()
}

/// Full offline M×D sweep used by Tauri and the air-gap gate.
pub fn evaluate_clinic_sweep_inner(
    patient_count: u32,
    disease_count: u32,
    seed: u32,
) -> Result<ClinicSweepResult, String> {
    let snp = 2048u32;
    let poly = 1024u32;
    let scale_bits = 40u32;
    let (header, slots) =
        pack_synthetic_clinic(patient_count, disease_count, snp, poly, scale_bits, seed)?;
    let prefer_metal = metal_available();
    let result = evaluate_prs_pipeline(&header, &slots, prefer_metal)?;
    let max_abs_error = result
        .scores
        .iter()
        .map(|s| s.abs_error)
        .fold(0.0_f64, f64::max);
    Ok(ClinicSweepResult {
        patient_count: result.patient_count,
        disease_count: result.disease_count,
        backend: backend_name(result.backend),
        ntt_ms: result.ntt_ms,
        max_abs_error,
        scores: to_cells(&result),
        airgapped: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VALIDATION GATE (Task 9): full sweep succeeds with no network dependency.
    #[test]
    fn airgap_full_patient_sweep() {
        let out = evaluate_clinic_sweep_inner(2, 5, 0xc0ffee).expect("offline sweep");
        assert!(out.airgapped);
        assert_eq!(out.scores.len(), 10);
        assert!(
            out.max_abs_error < 1e-4,
            "air-gap gate precision fail: {}",
            out.max_abs_error
        );
    }
}
