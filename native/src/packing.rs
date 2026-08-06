//! Patient-packed (SNP-major) encrypted PRS — streamed evaluator.
//!
//! Production SNP packing puts SNPs in slots and rotates once per patient.
//! This module transposes: ciphertext `i` holds SNP `i`'s dosage across up to
//! `N/2` patients, and the score is
//!
//! ```text
//!   acc = Σ_i Enc(g_{·,i}) · Enc(w_i)   // broadcast weight, ct×ct + relin
//! ```
//!
//! Slot `p` of `acc` is patient `p`'s PRS. There is **no horizontal sum**, so
//! Galois key-switches are zero. The accumulator is a single ciphertext;
//! SNPs are streamed, so peak encrypted memory does not grow with `n_snp` or
//! with cohort size beyond one slot-batch (`≤ N/2` patients).

use crate::ckks_encode::{decode_real_slots_i128, encode_real_slots};
use crate::crypto::{
    add_ct, decrypt, encrypt_encoded, mul_ct_ct, Ciphertext, CryptoResult, EncryptedStageMs,
    EvaluationKeys, DEFAULT_NOISE_ETA,
};

/// Outcome of one patient-packed evaluation.
#[derive(Debug, Clone)]
pub struct PackedEval {
    /// PRS per patient (length = requested patient count; padded slots dropped).
    pub scores: Vec<f64>,
    /// Ciphertexts consumed (one per SNP per slot-batch).
    pub ciphertexts: usize,
    /// Galois key-switches performed. Zero by construction.
    pub key_switches: usize,
    /// Stage split (encrypt / eval / decrypt).
    pub stages: EncryptedStageMs,
}

/// Streamed patient-packed ct×ct PRS.
///
/// `geno_patient_major` is length `patients * snp_count` (patient-major).
/// `weights` is length `snp_count`. Patients are processed in batches of at
/// most `N/2` slots; only one accumulator ciphertext is live at a time.
pub fn evaluate_patient_packed_ctct(
    geno_patient_major: &[f64],
    weights: &[f64],
    patients: usize,
    snp_count: usize,
    keys: &EvaluationKeys,
    scale: f64,
    prefer_metal: bool,
) -> CryptoResult<PackedEval> {
    let n = keys.basis.n;
    let slots = n / 2;
    let q = keys.basis.modulus;
    if geno_patient_major.len() != patients * snp_count {
        return Err(format!(
            "geno length {} ≠ patients ({patients}) × snps ({snp_count})",
            geno_patient_major.len()
        ));
    }
    if weights.len() != snp_count {
        return Err("weight length mismatch".into());
    }
    if patients == 0 || snp_count == 0 {
        return Err("empty cohort".into());
    }

    let mut scores = vec![0.0f64; patients];
    let mut stages = EncryptedStageMs::default();
    let mut ciphertexts = 0usize;
    let mut batch_start = 0usize;

    while batch_start < patients {
        let batch_len = (patients - batch_start).min(slots);
        let mut acc: Option<Ciphertext> = None;

        // Stream SNPs: encrypt geno + weight, mul-relin into acc, drop both.
        // Peak encrypted working set ≈ one accumulator + two fresh ciphertexts,
        // independent of snp_count (and of patients beyond one slot-batch).
        for s in 0..snp_count {
            let mut geno_slots = vec![0.0f64; slots];
            for p in 0..batch_len {
                geno_slots[p] = geno_patient_major[(batch_start + p) * snp_count + s];
            }
            let w_slots = vec![weights[s]; slots];

            let t_enc = std::time::Instant::now();
            let ca = encode_real_slots(&geno_slots, n, scale)?;
            let cb = encode_real_slots(&w_slots, n, scale)?;
            let ct_g = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA)?;
            let ct_w = encrypt_encoded(&keys.pk, &cb, DEFAULT_NOISE_ETA)?;
            stages.encrypt_ms += t_enc.elapsed().as_secs_f64() * 1e3;
            ciphertexts += 1;

            let t_ev = std::time::Instant::now();
            let term = mul_ct_ct(&ct_g, &ct_w, &keys.relin, &keys.basis, prefer_metal)?;
            // Drop ct_g / ct_w as they go out of scope each iteration.
            acc = Some(match acc {
                None => term,
                Some(prev) => add_ct(&prev, &term, q)?,
            });
            stages.eval_ms += t_ev.elapsed().as_secs_f64() * 1e3;
        }

        let acc = acc.ok_or_else(|| "no SNPs supplied".to_string())?;
        let t_dec = std::time::Instant::now();
        let phase = decrypt(&keys.sk, &acc, q)?;
        let decoded = decode_real_slots_i128(&phase, scale * scale)?;
        stages.decrypt_ms += t_dec.elapsed().as_secs_f64() * 1e3;

        for p in 0..batch_len {
            scores[batch_start + p] = decoded[p];
        }
        batch_start += batch_len;
    }

    Ok(PackedEval {
        scores,
        ciphertexts,
        key_switches: 0,
        stages,
    })
}

/// Fully synthetic stream: dosages from a closed-form PRNG, never materialising
/// a `patients × snps` matrix. Isolates evaluator RSS for the memory claim.
pub fn evaluate_patient_packed_synthetic_stream(
    patients: usize,
    snp_count: usize,
    keys: &EvaluationKeys,
    scale: f64,
    prefer_metal: bool,
) -> CryptoResult<PackedEval> {
    let n = keys.basis.n;
    let slots = n / 2;
    let q = keys.basis.modulus;
    if patients == 0 || snp_count == 0 {
        return Err("empty cohort".into());
    }

    let mut scores = vec![0.0f64; patients];
    let mut stages = EncryptedStageMs::default();
    let mut ciphertexts = 0usize;
    let mut batch_start = 0usize;

    while batch_start < patients {
        let batch_len = (patients - batch_start).min(slots);
        let mut acc: Option<Ciphertext> = None;
        let mut oracles = vec![0.0f64; batch_len];

        for s in 0..snp_count {
            let w = 1e-4 * (((s % 11) as f64) - 5.0);
            let mut geno_slots = vec![0.0f64; slots];
            for p in 0..batch_len {
                let dosage = ((s * 7 + (batch_start + p) * 3) % 3) as f64;
                geno_slots[p] = dosage;
                oracles[p] += dosage * w;
            }
            let w_slots = vec![w; slots];

            let t_enc = std::time::Instant::now();
            let ca = encode_real_slots(&geno_slots, n, scale)?;
            let cb = encode_real_slots(&w_slots, n, scale)?;
            let ct_g = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA)?;
            let ct_w = encrypt_encoded(&keys.pk, &cb, DEFAULT_NOISE_ETA)?;
            stages.encrypt_ms += t_enc.elapsed().as_secs_f64() * 1e3;
            ciphertexts += 1;

            let t_ev = std::time::Instant::now();
            let term = mul_ct_ct(&ct_g, &ct_w, &keys.relin, &keys.basis, prefer_metal)?;
            acc = Some(match acc {
                None => term,
                Some(prev) => add_ct(&prev, &term, q)?,
            });
            stages.eval_ms += t_ev.elapsed().as_secs_f64() * 1e3;
        }

        let acc = acc.ok_or_else(|| "no SNPs supplied".to_string())?;
        let t_dec = std::time::Instant::now();
        let phase = decrypt(&keys.sk, &acc, q)?;
        let decoded = decode_real_slots_i128(&phase, scale * scale)?;
        stages.decrypt_ms += t_dec.elapsed().as_secs_f64() * 1e3;

        for p in 0..batch_len {
            scores[batch_start + p] = decoded[p];
            let _ = oracles[p]; // oracle available for gates; NAPI recomputes
        }
        batch_start += batch_len;
    }

    Ok(PackedEval {
        scores,
        ciphertexts,
        key_switches: 0,
        stages,
    })
}

/// Per-patient key-switch count for a block-packed layout (per-ciphertext fold model).
///
/// With the *hoisted* fold used in production, `patients_per_ct = 1` costs
/// `log2(N/2)` rotations total (not `ciphers × log2`), plus one relin per
/// ciphertext. This helper documents the older per-ciphertext-fold frontier.
pub fn block_packed_key_switches_per_patient(
    n: usize,
    n_snp: usize,
    patients_per_ct: usize,
) -> Option<usize> {
    let slots = n / 2;
    if patients_per_ct == 0 || patients_per_ct > slots || !patients_per_ct.is_power_of_two() {
        return None;
    }
    let snps_per_ct = slots / patients_per_ct;
    let ciphertexts = n_snp.div_ceil(snps_per_ct);
    let rotations = snps_per_ct.ilog2() as usize;
    Some(ciphertexts * rotations / patients_per_ct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{setup_evaluation_keys, DEFAULT_NOISE_ETA};
    use crate::rns::RnsBasis;

    #[test]
    fn patient_packed_ctct_matches_oracle_zero_ks() {
        let n = 256usize;
        let slots = n / 2;
        let n_snp = 40usize;
        let patients = 17usize; // not a full slot vector — pad internally
        let basis = RnsBasis::generate(n, 4).expect("basis");
        let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA).expect("keys");
        let scale = (40u32 as f64).exp2();

        let mut geno = vec![0.0f64; patients * n_snp];
        let weights: Vec<f64> = (0..n_snp)
            .map(|i| 0.001 * ((i % 11) as f64 - 5.0))
            .collect();
        for p in 0..patients {
            for s in 0..n_snp {
                geno[p * n_snp + s] = ((s * 7 + p * 3) % 3) as f64;
            }
        }

        let out = evaluate_patient_packed_ctct(
            &geno, &weights, patients, n_snp, &keys, scale, false,
        )
        .expect("packed");
        assert_eq!(out.key_switches, 0);
        assert_eq!(out.scores.len(), patients);
        assert!(out.stages.encrypt_ms > 0.0);
        assert!(out.stages.eval_ms > 0.0);

        for p in 0..patients {
            let want: f64 = (0..n_snp).map(|s| geno[p * n_snp + s] * weights[s]).sum();
            assert!(
                (out.scores[p] - want).abs() < 1e-3,
                "patient {p}: got {} want {want}",
                out.scores[p]
            );
        }
        let _ = slots; // silence if unused in assert path
    }

    #[test]
    fn block_packing_interpolates_between_layouts() {
        let n = 16_384usize;
        let n_snp = 110_000usize;
        let slots = n / 2;

        let snp_packed = block_packed_key_switches_per_patient(n, n_snp, 1).unwrap();
        assert_eq!(
            snp_packed,
            n_snp.div_ceil(slots) * slots.ilog2() as usize,
            "P=1 must reproduce the per-ciphertext-fold 14x13 cost"
        );

        let patient_packed = block_packed_key_switches_per_patient(n, n_snp, slots).unwrap();
        assert_eq!(patient_packed, 0);

        let mut prev = usize::MAX;
        for p in [1usize, 8, 64, 512, 4096, slots] {
            let ks = block_packed_key_switches_per_patient(n, n_snp, p).unwrap();
            assert!(ks <= prev);
            prev = ks;
        }
    }
}
