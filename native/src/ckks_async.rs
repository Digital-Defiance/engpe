//! Heterogeneous double-buffered RNS pipeline (Task 8).
//!
//! Thread A (prep): Rayon CKKS encode + CRT decompose for cohort N+1.
//! Thread B (eval): Metal/CPU limb NTT for cohort N, waiting on the command buffer.
//! A `sync_channel(1)` keeps one prepared cohort in flight so the GPU never
//! idles waiting on host prep (and prep never races ahead unboundedly).

use std::sync::mpsc;
use std::thread;

use crate::ckks_rns::{
    evaluate_prepared_rns, prepare_rns_cohort, rns_basis_cached, PreparedRnsCohort, RnsEvalResult,
    RnsNttBackend,
};

/// One cohort job: flat genotype + disease-weight slot buffers.
#[derive(Clone)]
pub struct CohortJob {
    pub geno: Vec<f64>,
    pub weights: Vec<f64>,
    pub patients: usize,
    pub diseases: usize,
    pub n: usize,
    pub n_slots: usize,
    pub ciphers: usize,
    pub scale: f64,
}

fn prepare_job(job: &CohortJob) -> RnsEvalResult<PreparedRnsCohort> {
    let basis = rns_basis_cached(job.n)?;
    prepare_rns_cohort(
        &job.geno,
        &job.weights,
        job.patients,
        job.diseases,
        job.n,
        job.n_slots,
        job.ciphers,
        job.scale,
        &basis,
    )
}

fn eval_prepared_batch(
    rx: mpsc::Receiver<RnsEvalResult<PreparedRnsCohort>>,
    backend: RnsNttBackend,
) -> RnsEvalResult<Vec<Vec<f64>>> {
    let mut results = Vec::new();
    while let Ok(prep_res) = rx.recv() {
        let prep = prep_res?;
        results.push(evaluate_prepared_rns(&prep, backend)?);
    }
    Ok(results)
}

fn spawn_preparer(
    jobs: Vec<CohortJob>,
) -> (
    thread::JoinHandle<()>,
    mpsc::Receiver<RnsEvalResult<PreparedRnsCohort>>,
) {
    let (tx, rx) = mpsc::sync_channel(1);
    let handle = thread::spawn(move || {
        for job in jobs {
            let prep = prepare_job(&job);
            if tx.send(prep).is_err() {
                break;
            }
        }
    });
    (handle, rx)
}

fn evaluate_single_job(job: &CohortJob, backend: RnsNttBackend) -> RnsEvalResult<Vec<f64>> {
    let prep = prepare_job(job)?;
    evaluate_prepared_rns(&prep, backend)
}

/// Evaluate a sequence of cohorts with host/GPU overlap.
pub fn evaluate_double_buffered(
    jobs: Vec<CohortJob>,
    backend: RnsNttBackend,
) -> RnsEvalResult<Vec<Vec<f64>>> {
    match jobs.len() {
        0 => Ok(Vec::new()),
        1 => Ok(vec![evaluate_single_job(&jobs[0], backend)?]),
        _ => evaluate_piped(jobs, backend),
    }
}

fn evaluate_piped(
    jobs: Vec<CohortJob>,
    backend: RnsNttBackend,
) -> RnsEvalResult<Vec<Vec<f64>>> {
    let (preparer, rx) = spawn_preparer(jobs);
    let results = eval_prepared_batch(rx, backend)?;
    preparer
        .join()
        .map_err(|_| "prep thread panicked".to_string())?;
    Ok(results)
}

/// Bit-identical compare of two f64 score vectors (Task 8 validation gate).
pub fn scores_bit_identical(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| x.to_bits() == y.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckks_rns::{evaluate_rns_cohort, rns_basis_cached, RnsNttBackend};
    use crate::metal_ntt::metal_available;

    fn seeded_job(seed: u64, patients: usize, diseases: usize) -> CohortJob {
        let n = 64usize;
        let n_slots = n / 2;
        let snp = 16usize;
        let ciphers = snp.div_ceil(n_slots);
        let per = ciphers * n_slots;
        let mut geno = vec![0.0; patients * per];
        let mut weights = vec![0.0; diseases * per];
        for p in 0..patients {
            for i in 0..snp {
                geno[p * per + i] = ((i as u64 + seed + p as u64) % 3) as f64;
            }
        }
        for d in 0..diseases {
            for i in 0..snp {
                weights[d * per + i] =
                    0.01 * ((i as f64) - 5.0) * (1.0 + 0.05 * d as f64 + seed as f64 * 1e-6);
            }
        }
        CohortJob {
            geno,
            weights,
            patients,
            diseases,
            n,
            n_slots,
            ciphers,
            scale: (40u32 as f64).exp2(),
        }
    }

    /// VALIDATION GATE (Task 8): async double-buffer ≡ sync, bit-for-bit.
    #[test]
    fn async_pipeline_bit_identical_to_sync() {
        let backend = if metal_available() {
            RnsNttBackend::Metal
        } else {
            RnsNttBackend::CpuRayon
        };
        let jobs: Vec<CohortJob> = (0..3).map(|s| seeded_job(s, 2, 3)).collect();
        let basis = rns_basis_cached(jobs[0].n).unwrap();

        let mut sync_scores = Vec::new();
        for job in &jobs {
            sync_scores.push(
                evaluate_rns_cohort(
                    &job.geno,
                    &job.weights,
                    job.patients,
                    job.diseases,
                    job.n,
                    job.n_slots,
                    job.ciphers,
                    job.scale,
                    &basis,
                    backend,
                )
                .unwrap(),
            );
        }

        let async_scores = evaluate_double_buffered(jobs, backend).unwrap();
        assert_eq!(async_scores.len(), sync_scores.len());
        for (i, (a, b)) in async_scores.iter().zip(sync_scores.iter()).enumerate() {
            assert!(
                scores_bit_identical(a, b),
                "GATE FAIL: cohort {i} async ≠ sync (len a={} b={})",
                a.len(),
                b.len()
            );
        }
    }
}
