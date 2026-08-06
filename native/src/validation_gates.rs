//! Task 10 validation gates — deterministic, seeded, mathematically strict.
//!
//! Every gate here builds its cohort from one seeded generator
//! (`clinic::pack_synthetic_clinic`) so the same seed always yields the same
//! M patients and D disease weight panels. `plaintext_prs_oracles` is the sole
//! ground truth; no gate compares one FHE path against another FHE path except
//! where bit-equality is the property under test.

use crate::ckks_async::{evaluate_double_buffered, scores_bit_identical, CohortJob};
use crate::ckks_prs::{plaintext_prs_oracles, unpack_header, unpack_slots, CkksPrsParams};
use crate::ckks_rns::{
    evaluate_cpu_rns_cohort, evaluate_metal_rns_cohort, evaluate_rns_cohort, rns_basis_cached,
    RnsNttBackend,
};
use crate::metal_ntt::metal_available;

/// Per-coordinate tolerance mandated by the directive.
const EPSILON: f64 = 1e-4;

const SNP: u32 = 512;
const POLY_DEGREE: u32 = 256;
const SCALE_BITS: u32 = 40;

/// A seeded cohort plus everything the RNS entry points need.
struct SeededCohort {
    params: CkksPrsParams,
    geno: Vec<f64>,
    weights: Vec<f64>,
    oracle: Vec<f64>,
}

impl SeededCohort {
    fn patients(&self) -> usize {
        self.params.patient_count as usize
    }

    fn diseases(&self) -> usize {
        self.params.disease_count as usize
    }

    fn scale(&self) -> f64 {
        (self.params.scale_bits as f64).exp2()
    }

    fn job(&self) -> CohortJob {
        CohortJob {
            geno: self.geno.clone(),
            weights: self.weights.clone(),
            patients: self.patients(),
            diseases: self.diseases(),
            n: self.params.poly_degree as usize,
            n_slots: self.params.slot_count as usize,
            ciphers: self.params.cipher_count as usize,
            scale: self.scale(),
        }
    }
}

/// Build the same cohort every time for a given `(patients, diseases, seed)`.
fn seeded_cohort(patients: u32, diseases: u32, seed: u32) -> SeededCohort {
    let (header, slots) =
        crate::clinic::pack_synthetic_clinic(patients, diseases, SNP, POLY_DEGREE, SCALE_BITS, seed)
            .expect("seeded pack");
    let params = unpack_header(&header).expect("header");
    let views = unpack_slots(params, &slots).expect("slots");
    SeededCohort {
        params,
        geno: views.genotype_slots.to_vec(),
        weights: views.weight_slots.to_vec(),
        oracle: plaintext_prs_oracles(&views),
    }
}

/// Assert every (p, d) coordinate is within `EPSILON` of the plaintext oracle.
fn assert_matrix_within_epsilon(cohort: &SeededCohort, decoded: &[f64], label: &str) {
    assert_eq!(
        decoded.len(),
        cohort.oracle.len(),
        "GATE FAIL [{label}]: matrix has {} cells, oracle has {}",
        decoded.len(),
        cohort.oracle.len()
    );
    let diseases = cohort.diseases();
    for (i, (&got, &want)) in decoded.iter().zip(cohort.oracle.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err < EPSILON,
            "GATE FAIL [{label}]: cell (p={}, d={}) err {err:.3e} ≥ {EPSILON:.0e} \
             (decoded {got}, oracle {want})",
            i / diseases,
            i % diseases
        );
    }
}

fn run_rns(cohort: &SeededCohort, backend: RnsNttBackend) -> Vec<f64> {
    let basis = rns_basis_cached(cohort.params.poly_degree as usize).expect("basis");
    let job = cohort.job();
    let run = match backend {
        RnsNttBackend::Metal => evaluate_metal_rns_cohort,
        RnsNttBackend::CpuRayon => evaluate_cpu_rns_cohort,
    };
    run(
        &job.geno,
        &job.weights,
        job.patients,
        job.diseases,
        job.n,
        job.n_slots,
        job.ciphers,
        job.scale,
        basis.as_ref(),
    )
    .expect("rns cohort")
}

/// The seeded generator must be reproducible, or every other gate is vacuous.
#[test]
fn seeded_generator_is_reproducible() {
    let a = seeded_cohort(4, 5, 0x5eed);
    let b = seeded_cohort(4, 5, 0x5eed);
    assert!(scores_bit_identical(&a.geno, &b.geno), "genotypes drifted");
    assert!(scores_bit_identical(&a.weights, &b.weights), "weights drifted");
    assert!(scores_bit_identical(&a.oracle, &b.oracle), "oracle drifted");

    let c = seeded_cohort(4, 5, 0x5eed + 1);
    assert!(
        !scores_bit_identical(&a.oracle, &c.oracle),
        "distinct seeds produced identical cohorts — generator ignores its seed"
    );
}

/// GATE 1 (Task 10): 4×5 Metal SIMD matrix vs. plaintext oracle, ε < 1e-4.
#[test]
fn gate1_multi_disease_simd_precision() {
    let cohort = seeded_cohort(4, 5, 0xc0ffee);
    assert_eq!(cohort.oracle.len(), 20);

    assert_matrix_within_epsilon(&cohort, &run_rns(&cohort, RnsNttBackend::CpuRayon), "cpu");
    if metal_available() {
        assert_matrix_within_epsilon(&cohort, &run_rns(&cohort, RnsNttBackend::Metal), "metal");
    }
}

/// GATE 2 (Task 10): async double-buffered queue ≡ sync pipeline, bit-for-bit.
#[test]
fn gate2_async_double_buffer_no_race() {
    let backend = if metal_available() {
        RnsNttBackend::Metal
    } else {
        RnsNttBackend::CpuRayon
    };
    let cohorts: Vec<SeededCohort> = (0..4).map(|s| seeded_cohort(4, 5, 0xa5a5 + s)).collect();
    let basis = rns_basis_cached(POLY_DEGREE as usize).expect("basis");

    let sync: Vec<Vec<f64>> = cohorts
        .iter()
        .map(|c| {
            let job = c.job();
            evaluate_rns_cohort(
                &job.geno,
                &job.weights,
                job.patients,
                job.diseases,
                job.n,
                job.n_slots,
                job.ciphers,
                job.scale,
                basis.as_ref(),
                backend,
            )
            .expect("sync cohort")
        })
        .collect();

    let jobs: Vec<CohortJob> = cohorts.iter().map(SeededCohort::job).collect();
    let piped = evaluate_double_buffered(jobs, backend).expect("double-buffered");

    assert_eq!(piped.len(), sync.len(), "double-buffer dropped a cohort");
    for (i, (a, b)) in piped.iter().zip(sync.iter()).enumerate() {
        assert!(
            scores_bit_identical(a, b),
            "GATE FAIL: cohort {i} async ≠ sync bitwise — race or dropped polynomial"
        );
    }

    // The sync reference itself must still be numerically correct.
    for (cohort, scores) in cohorts.iter().zip(sync.iter()) {
        assert_matrix_within_epsilon(cohort, scores, "sync-reference");
    }
}

/// GATE (Phase 3): fully encrypted RLWE path matches plaintext oracle within ε.
#[test]
fn gate_encrypted_rlwe_precision() {
    use crate::ckks_rns::evaluate_encrypted_rns_cohort;
    // Smaller panel keeps Galois key-switch cost tractable under test.
    let snp = 64u32;
    let poly = 128u32;
    let scale_bits = 40u32;
    let (header, slots) =
        crate::clinic::pack_synthetic_clinic(2, 3, snp, poly, scale_bits, 0xebc0)
            .expect("pack");
    let params = unpack_header(&header).expect("header");
    let views = unpack_slots(params, &slots).expect("slots");
    let oracle = plaintext_prs_oracles(&views);
    let decoded = evaluate_encrypted_rns_cohort(
        views.genotype_slots,
        views.weight_slots,
        params.patient_count as usize,
        params.disease_count as usize,
        params.poly_degree as usize,
        params.slot_count as usize,
        params.cipher_count as usize,
        (scale_bits as f64).exp2(),
    )
    .expect("encrypted cohort");
    assert_eq!(decoded.len(), oracle.len());
    for (i, (&got, &want)) in decoded.iter().zip(oracle.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err < EPSILON,
            "ENCRYPTED GATE FAIL: cell {i} err {err:.3e} (decoded {got}, oracle {want})"
        );
    }
}

/// GATE (Phase 5): fused hybrid Metal KS path matches plaintext oracle within ε.
#[test]
fn assert_fused_hybrid_keyswitch_matches_oracle() {
    use crate::ckks_rns::evaluate_encrypted_rns_cohort_backend;
    use crate::crypto::digit_count;
    use crate::rns::{AuxiliaryModulus, RnsBasis, DEFAULT_AUX_LIMBS};

    // Hybrid digit width must reduce the gadget vs the Phase-3 10-bit base.
    let basis = RnsBasis::generate(128, 4).expect("basis");
    let aux = AuxiliaryModulus::generate(128, DEFAULT_AUX_LIMBS, &basis.primes).expect("aux");
    assert_eq!(aux.primes.len(), 1);
    assert!(aux.modulus > 1);
    let digits = digit_count(basis.modulus);
    assert!(
        digits <= 8,
        "hybrid KS digit count {digits} should be ≤ 8 at |Q|≈124 with 20-bit digits"
    );

    // Mod-down round-trip: x = a*P + r → floor(x/P) recovers a on Q limbs.
    let p = aux.modulus as i128;
    let a = 12345i128;
    let r = 7i128;
    let x = a * p + r;
    let q_limbs = basis.decompose_coeff_i128(x);
    let p_limb = (x.rem_euclid(p) as u64) % aux.primes[0];
    let y = aux
        .mod_down_coeff(&q_limbs, &[p_limb], &basis.primes)
        .expect("mod_down");
    let back = basis.recombine_coeff(&y).expect("recombine");
    assert_eq!(back, a, "mod_down failed: got {back} want {a}");

    if !metal_available() {
        eprintln!("skip fused hybrid oracle: Metal unavailable");
        return;
    }

    let snp = 64u32;
    let poly = 128u32;
    let scale_bits = 40u32;
    let (header, slots) =
        crate::clinic::pack_synthetic_clinic(2, 3, snp, poly, scale_bits, 0xf05e)
            .expect("pack");
    let params = unpack_header(&header).expect("header");
    let views = unpack_slots(params, &slots).expect("slots");
    let oracle = plaintext_prs_oracles(&views);
    let decoded = evaluate_encrypted_rns_cohort_backend(
        views.genotype_slots,
        views.weight_slots,
        params.patient_count as usize,
        params.disease_count as usize,
        params.poly_degree as usize,
        params.slot_count as usize,
        params.cipher_count as usize,
        (scale_bits as f64).exp2(),
        true,
    )
    .expect("fused hybrid encrypted cohort");
    assert_eq!(decoded.len(), oracle.len());
    for (i, (&got, &want)) in decoded.iter().zip(oracle.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err < EPSILON,
            "FUSED HYBRID GATE FAIL: cell {i} err {err:.3e} (decoded {got}, oracle {want})"
        );
    }
}

/// GATE (Phase 4): Metal KS must be bit-identical to CPU KS on c0/c1 limbs.
#[test]
fn assert_metal_keyswitch_matches_cpu() {
    use crate::crypto::{
        automorphism_ct, encrypt_encoded, gen_galois_key, key_switch_with_backend, keygen,
        DEFAULT_NOISE_ETA,
    };
    use crate::ckks_encode::encode_real_slots;
    use crate::rns::RnsBasis;

    if !metal_available() {
        eprintln!("skip assert_metal_keyswitch_matches_cpu: Metal unavailable");
        return;
    }

    let n = 128usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let k = 5usize; // directive oracle automorphism
    let gk = gen_galois_key(&sk, k, &basis, DEFAULT_NOISE_ETA).expect("galois");

    let scale = (20u32 as f64).exp2();
    // Two meaningful slots; remaining slots pad with zeros via encode.
    let slots = vec![0.37f64, -0.11];
    let mut padded = vec![0.0f64; n / 2];
    padded[0] = slots[0];
    padded[1] = slots[1];
    let coeffs = encode_real_slots(&padded, n, scale).expect("encode");
    let ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt");
    let rotated = automorphism_ct(&ct, k, basis.modulus);

    let cpu = key_switch_with_backend(&rotated, &gk, &basis, false).expect("cpu ks");
    let metal = key_switch_with_backend(&rotated, &gk, &basis, true).expect("metal ks");

    assert_eq!(
        cpu.c0.len(),
        metal.c0.len(),
        "GATE FAIL [metal-ks]: c0 length mismatch"
    );
    assert_eq!(
        cpu.c1.len(),
        metal.c1.len(),
        "GATE FAIL [metal-ks]: c1 length mismatch"
    );
    for i in 0..cpu.c0.len() {
        assert_eq!(
            cpu.c0[i], metal.c0[i],
            "GATE FAIL [metal-ks]: c0[{i}] cpu={} metal={}",
            cpu.c0[i], metal.c0[i]
        );
        assert_eq!(
            cpu.c1[i], metal.c1[i],
            "GATE FAIL [metal-ks]: c1[{i}] cpu={} metal={}",
            cpu.c1[i], metal.c1[i]
        );
    }
}
