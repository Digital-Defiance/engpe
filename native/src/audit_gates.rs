//! Pre-publication audit gates.
//!
//! `validation_gates` proves the pipeline is *accurate*. This module proves the
//! pipeline is *what the paper says it is*: that the ciphertexts are genuine
//! RLWE rather than encoded plaintext, that Galois key-switching performs a
//! real slot rotation, that the fused Metal command buffer is actually taken
//! (and not silently degrading to the CPU route), and that the hybrid gadget
//! shrank the digit count as claimed.
//!
//! Every assertion here is a claim made in `paper.md`.

use crate::ckks_encode::{decode_real_slots_i128, encode_real_slots};
use crate::crypto::{
    automorphism_ct, decrypt, digit_count, encrypt_encoded, gen_galois_key, is_128bit_secure,
    key_switch_with_backend, keygen, min_degree_for_128bit, mul_ct_pt,
    rotate_and_sum_encrypted_backend, setup_evaluation_keys, DEFAULT_NOISE_ETA, KS_DIGIT_BITS,
};
use crate::metal_ks::ks_path_counters;
use crate::metal_ntt::metal_available;
use crate::rns::{AuxiliaryModulus, RnsBasis, DEFAULT_AUX_LIMBS};
use crate::zq::{center_i128_to_zq, zq_to_center_i128, Zq};

const EPSILON: f64 = 1e-4;

fn center(u: Zq, q: Zq) -> i128 {
    zq_to_center_i128(u, q).expect("centered residue fits i128")
}

fn to_residue(c: i128, q: Zq) -> Zq {
    center_i128_to_zq(c, q)
}

/// AUDIT 1 — the ciphertext must be a real RLWE ciphertext, not an encoding.
///
/// Checks the three properties that separate encryption from encoding:
///   (a) the ciphertext does not reveal the message (c0 ≠ Δm),
///   (b) encryption is randomised (two encryptions of one message differ),
///   (c) a wrong secret key does not recover the message.
#[test]
fn audit_ciphertexts_are_real_rlwe_not_encoded_plaintext() {
    let n = 256usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let scale = (40u32 as f64).exp2();

    let slots: Vec<f64> = (0..n / 2).map(|i| 0.01 * (i as f64 % 17.0 - 8.0)).collect();
    let coeffs = encode_real_slots(&slots, n, scale).expect("encode");
    let ct1 = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt 1");
    let ct2 = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt 2");

    // (a) The ciphertext must not equal the encoded plaintext.
    let encoded_residues: Vec<Zq> = coeffs.iter().map(|&c| to_residue(c as i128, q)).collect();
    assert_ne!(
        ct1.c0, encoded_residues,
        "AUDIT FAIL: c0 equals the encoded plaintext — this is encoding, not encryption"
    );
    let matching = ct1
        .c0
        .iter()
        .zip(encoded_residues.iter())
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        matching < n / 8,
        "AUDIT FAIL: {matching}/{n} c0 coefficients equal the plaintext — masking is not applied"
    );

    // c1 must be non-trivial: an all-zero c1 would make c0 a plain encoding.
    assert!(
        ct1.c1.iter().any(|&x| x != Zq::ZERO),
        "AUDIT FAIL: c1 is identically zero — no RLWE mask present"
    );

    // (b) Encryption must be randomised (IND-CPA requires fresh randomness).
    assert_ne!(
        ct1.c0, ct2.c0,
        "AUDIT FAIL: two encryptions of the same message are identical — deterministic encryption"
    );
    assert_ne!(
        ct1.c1, ct2.c1,
        "AUDIT FAIL: c1 is reused across encryptions — randomness is not fresh"
    );

    // (c) The correct key decrypts; a wrong key does not.
    let phase = decrypt(&sk, &ct1, q).expect("decrypt");
    let back = decode_real_slots_i128(&phase, scale).expect("decode");
    let max_err = slots
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_err < EPSILON,
        "AUDIT FAIL: correct key did not decrypt (max err {max_err:.3e})"
    );

    let (wrong_sk, _) = keygen(&basis, DEFAULT_NOISE_ETA).expect("second keygen");
    let bad_phase = decrypt(&wrong_sk, &ct1, q).expect("decrypt wrong");
    let bad = decode_real_slots_i128(&bad_phase, scale).expect("decode wrong");
    let bad_err = slots
        .iter()
        .zip(bad.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        bad_err > 1.0,
        "AUDIT FAIL: a WRONG secret key recovered the message (err {bad_err:.3e}) — \
         the ciphertext is not bound to the key"
    );
}

/// AUDIT 2 — fresh ciphertexts must actually carry RLWE noise, and that noise
/// must be small relative to Q (otherwise "noise" is cosmetic or fatal).
#[test]
fn audit_rlwe_noise_is_present_and_bounded() {
    let n = 256usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");

    // Encrypt the zero message: the decrypted phase is then exactly the noise.
    let zero = vec![0i64; n];
    let ct = encrypt_encoded(&pk, &zero, DEFAULT_NOISE_ETA).expect("encrypt zero");
    let phase = decrypt(&sk, &ct, q).expect("decrypt");

    let max_noise = phase.iter().map(|&c| c.abs()).max().unwrap_or(0);
    assert!(
        max_noise > 0,
        "AUDIT FAIL: zero-message ciphertext decrypts to exactly zero — no noise, not RLWE"
    );

    // Noise must be tiny versus Q or the scheme has no security/precision margin.
    let q_bits = crate::zq::zq_ilog2(q) + 1;
    let noise_bits = if max_noise <= 1 {
        1
    } else {
        (max_noise as u128).ilog2() + 1
    };
    assert!(
        noise_bits + 40 < q_bits,
        "AUDIT FAIL: noise is {noise_bits} bits against a {q_bits}-bit Q — no usable budget"
    );
    eprintln!("[audit] fresh noise ≈ 2^{noise_bits}, Q ≈ 2^{q_bits}");
}

/// AUDIT 3 — the Galois key-switch must perform a genuine slot rotation.
///
/// A key-switch that returned its input unchanged, or that silently dropped the
/// automorphism, would still pass a sum-only test on symmetric data. This uses
/// an asymmetric slot vector and checks the rotated slot values directly.
#[test]
fn audit_galois_keyswitch_performs_real_rotation() {
    let n = 128usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let slots_n = n / 2;
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let k = 5usize; // φ_5 rotates CKKS slots by one position
    let gk = gen_galois_key(&sk, k, &basis, DEFAULT_NOISE_ETA).expect("galois key");

    let scale = (40u32 as f64).exp2();
    // Asymmetric, distinct values so any rotation is observable.
    let slots: Vec<f64> = (0..slots_n).map(|i| 0.001 * (i as f64 + 1.0)).collect();
    let coeffs = encode_real_slots(&slots, n, scale).expect("encode");
    let ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt");

    let rotated = automorphism_ct(&ct, k, q);
    let switched = key_switch_with_backend(&rotated, &gk, &basis, false).expect("cpu ks");
    let phase = decrypt(&sk, &switched, q).expect("decrypt");
    let got = decode_real_slots_i128(&phase, scale).expect("decode");

    // This encoding indexes slots by the NATURAL embedding order ζ_j =
    // exp(iπ(2j+1)/N), not the canonical power-of-5 order. Under that
    // labelling φ_5 is the permutation j ↦ ((5(2j+1) mod 2N) − 1)/2 folded
    // back through conjugate symmetry — NOT a one-position cyclic shift.
    let sigma = |j: usize| -> usize {
        let t = (5 * (2 * j + 1)) % (2 * n);
        let idx = (t - 1) / 2;
        if idx < slots_n {
            idx
        } else {
            n - 1 - idx
        }
    };
    let expected: Vec<f64> = (0..slots_n).map(|j| slots[sigma(j)]).collect();
    let err = expected
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    // Tolerance is set by key-switch noise, which at the Δ (not Δ²) scale used
    // here is ≈ n·B·σ·√d / Δ ≈ 2^28.6 / 2^40 ≈ 4e-4 for 20-bit digits. The PRS
    // pipeline key-switches at Δ² where this term is ~1e-12 (see AUDIT 6).
    const ROTATION_EPS: f64 = 5e-3;
    assert!(
        err < ROTATION_EPS,
        "AUDIT FAIL: key-switched automorphism does not match the φ_5 slot \
         permutation (max err {err:.3e})"
    );
    eprintln!("[audit] φ_5 permutation matches, single-rotation KS noise = {err:.3e} at Δ=2^40");

    // It must genuinely differ from the unrotated plaintext, or "rotation" is a no-op.
    let identity_err = slots
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        identity_err > EPSILON,
        "AUDIT FAIL: rotation returned the original slots — automorphism was a no-op"
    );
}

/// AUDIT 4 — the fused single-command-buffer Metal path must actually run.
///
/// `key_switch_accelerated` falls back to the CPU route whenever the fused
/// call declines. Without this counter check, the bit-identity gate could pass
/// while the GPU pipeline never executes.
#[test]
fn audit_fused_metal_pipeline_is_actually_executed() {
    if !metal_available() {
        eprintln!("[audit] skip fused-path check: Metal unavailable");
        return;
    }

    let n = 128usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let k = 5usize;
    let gk = gen_galois_key(&sk, k, &basis, DEFAULT_NOISE_ETA).expect("galois key");
    let scale = (40u32 as f64).exp2();
    let slots: Vec<f64> = (0..n / 2).map(|i| 0.002 * (i as f64 - 3.0)).collect();
    let coeffs = encode_real_slots(&slots, n, scale).expect("encode");
    let ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt");
    let rotated = automorphism_ct(&ct, k, basis.modulus);

    let (fused_before, fallback_before) = ks_path_counters();
    let metal = key_switch_with_backend(&rotated, &gk, &basis, true).expect("metal ks");
    let (fused_after, fallback_after) = ks_path_counters();

    let fused_delta = fused_after - fused_before;
    let fallback_delta = fallback_after - fallback_before;
    eprintln!("[audit] fused limbs={fused_delta} fallback limbs={fallback_delta}");

    assert!(
        fused_delta > 0,
        "AUDIT FAIL: the fused Metal command buffer never executed \
         (fused={fused_delta}, fallback={fallback_delta}). The paper's Section 3.2 \
         claim is not backed by the running code."
    );
    assert_eq!(
        fallback_delta, 0,
        "AUDIT FAIL: {fallback_delta} limbs silently fell back off the fused path"
    );

    // And the fused result must still be bit-identical to the CPU oracle.
    let cpu = key_switch_with_backend(&rotated, &gk, &basis, false).expect("cpu ks");
    assert_eq!(cpu.c0, metal.c0, "AUDIT FAIL: fused c0 differs from CPU oracle");
    assert_eq!(cpu.c1, metal.c1, "AUDIT FAIL: fused c1 differs from CPU oracle");
}

/// AUDIT 5 — the hybrid gadget must actually reduce the digit count, and the
/// auxiliary modulus P must be well formed and co-prime to Q.
#[test]
fn audit_hybrid_gadget_and_auxiliary_modulus() {
    assert_eq!(
        KS_DIGIT_BITS, 20,
        "AUDIT FAIL: paper claims 20-bit hybrid digits"
    );

    let n = 16384usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let digits = digit_count(basis.modulus);
    let q_bits = basis.modulus_bits();
    eprintln!("[audit] |Q| = {q_bits} bits, digit count = {digits}");

    // Phase 3 used 10-bit digits => 13 digits. The paper claims ~7.
    assert_eq!(
        digits, 7,
        "AUDIT FAIL: expected 7 hybrid digits at |Q|={q_bits}, got {digits}"
    );

    let aux = AuxiliaryModulus::generate(n, DEFAULT_AUX_LIMBS, &basis.primes).expect("aux P");
    for &p in &aux.primes {
        assert!(p < (1u64 << 31), "AUDIT FAIL: aux prime {p} exceeds 2^31");
        assert!(
            (p - 1) % (2 * n as u64) == 0,
            "AUDIT FAIL: aux prime {p} is not NTT-eligible for N={n}"
        );
        assert!(
            !basis.primes.contains(&p),
            "AUDIT FAIL: aux prime {p} collides with a Q limb — P and Q must be co-prime"
        );
    }

    // P^{-1} mod q_i must be a genuine inverse.
    for (i, &q) in basis.primes.iter().enumerate() {
        let p_mod_q = (aux.modulus % q as u128) as u128;
        let inv = aux.p_inv_mod_q[i] as u128;
        assert_eq!(
            (p_mod_q * inv) % q as u128,
            1,
            "AUDIT FAIL: P^-1 mod q_{i} is not an inverse"
        );
    }
}

/// AUDIT 6 — end-to-end three-way agreement on one encrypted PRS.
///
/// Encrypted RLWE result, CPU key-switch result, and the plaintext oracle must
/// all agree. This is the claim a reader of the paper cares about most.
#[test]
fn audit_encrypted_prs_matches_plaintext_oracle_three_ways() {
    let n = 256usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA).expect("eval keys");
    let scale = (40u32 as f64).exp2();
    let slots_n = n / 2;

    let geno: Vec<f64> = (0..slots_n).map(|i| ((i % 3) as f64) / 2.0).collect();
    let weights: Vec<f64> = (0..slots_n)
        .map(|i| 0.0005 * ((i % 11) as f64 - 5.0))
        .collect();

    // Plaintext oracle: the dot product we are trying to compute privately.
    let oracle: f64 = geno.iter().zip(weights.iter()).map(|(g, w)| g * w).sum();

    let ca = encode_real_slots(&geno, n, scale).expect("encode geno");
    let cb = encode_real_slots(&weights, n, scale).expect("encode weights");
    let ct = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA).expect("encrypt");
    let pt: Vec<Zq> = cb
        .iter()
        .map(|&c| to_residue(c as i128, basis.modulus))
        .collect();
    let prod = mul_ct_pt(&ct, &pt, &basis).expect("ct x pt");

    let cpu_sum =
        rotate_and_sum_encrypted_backend(&prod, &keys.galois, &basis, false).expect("cpu ras");
    let cpu_phase = decrypt(&keys.sk, &cpu_sum, basis.modulus).expect("decrypt cpu");
    let cpu_score = decode_real_slots_i128(&cpu_phase, scale * scale).expect("decode cpu")[0];

    let cpu_err = (cpu_score - oracle).abs();
    assert!(
        cpu_err < EPSILON,
        "AUDIT FAIL: encrypted CPU PRS {cpu_score} vs oracle {oracle} (err {cpu_err:.3e})"
    );

    if metal_available() {
        let gpu_sum =
            rotate_and_sum_encrypted_backend(&prod, &keys.galois, &basis, true).expect("gpu ras");
        assert_eq!(
            cpu_sum.c0, gpu_sum.c0,
            "AUDIT FAIL: Metal rotate-and-sum c0 differs from CPU bitwise"
        );
        assert_eq!(
            cpu_sum.c1, gpu_sum.c1,
            "AUDIT FAIL: Metal rotate-and-sum c1 differs from CPU bitwise"
        );
    }

    eprintln!("[audit] encrypted PRS {cpu_score:.12} vs oracle {oracle:.12} (err {cpu_err:.3e})");
}

/// AUDIT 14 — cost of the patient-packed layout at production degree.
///
/// Section 5 argues the throughput gap against cohort-amortized HE-PRS is a
/// packing choice rather than a kernel deficit. This measures the per-SNP
/// primitives at N=16384 and projects both layouts onto a common per-patient
/// basis, so the claim rests on measurement instead of assertion.
/// Ignored by default: `cargo test --release -- --ignored packed_cost`.
#[test]
#[ignore]
fn audit_patient_packed_cost_projection() {
    use crate::crypto::mul_ct_pt;
    use crate::packing::block_packed_key_switches_per_patient;

    let n = 16_384usize;
    let slots = n / 2;
    let n_snp = 110_000usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let (_sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let scale = (40u32 as f64).exp2();

    let geno: Vec<f64> = (0..slots).map(|p| (p % 3) as f64).collect();
    let coeffs = encode_real_slots(&geno, n, scale).expect("encode");
    let w = vec![0.0031f64; slots];
    let w_coeffs = encode_real_slots(&w, n, scale).expect("encode w");
    let pt: Vec<Zq> = w_coeffs
        .iter()
        .map(|&c| to_residue(c as i128, q))
        .collect();

    const REPS: usize = 8;
    let t = std::time::Instant::now();
    let mut ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt");
    for _ in 1..REPS {
        ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt");
    }
    let encrypt_ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

    let t = std::time::Instant::now();
    for _ in 0..REPS {
        let _ = mul_ct_pt(&ct, &pt, &basis).expect("ct x pt");
    }
    let mul_ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

    // Both layouts move the same number of ciphertexts per patient:
    // n_snp / (N/2) rounded up. They differ only in rotation depth.
    let ct_per_patient = n_snp as f64 / slots as f64;
    let packed_ms = ct_per_patient * (encrypt_ms + mul_ms);
    let ks_snp_packed = block_packed_key_switches_per_patient(n, n_snp, 1).unwrap();

    eprintln!(
        "[audit] N={n} per-ciphertext: encrypt {encrypt_ms:.1} ms, ct x pt {mul_ms:.1} ms"
    );
    eprintln!(
        "[audit] patient-packed projection: {ct_per_patient:.1} ciphertexts/patient, \
         0 key-switches, ~{packed_ms:.0} ms/patient (evaluator working set = 1 ciphertext, \
         independent of SNP count)"
    );
    eprintln!(
        "[audit] deployed SNP-packed: {ks_snp_packed} key-switches/patient, measured 3274 ms"
    );

    assert!(
        packed_ms < 3274.0,
        "AUDIT FAIL: patient-packed projection {packed_ms:.0} ms is not faster than the \
         measured SNP-packed 3274 ms — Section 5's argument would not hold"
    );
}

/// AUDIT 13 — the parameter set must sit inside a published security envelope.
///
/// A privacy paper has to state a security level. Rather than quote a
/// hand-rolled estimator, this pins the parameters against the Homomorphic
/// Encryption Security Standard (2018) tables for uniform ternary secrets at
/// classical 128-bit security, and fails if anyone raises |Q| or lowers N out
/// of that envelope. It also pins σ, because those tables assume σ ≈ 3.2 and
/// the envelope is not applicable at materially smaller noise.
#[test]
fn audit_security_parameters_are_inside_he_standard() {
    let sigma = (DEFAULT_NOISE_ETA as f64 / 2.0).sqrt();
    assert!(
        sigma >= 3.0,
        "AUDIT FAIL: sigma {sigma:.2} is below the ~3.2 assumed by the HE Standard \
         tables, so the (N, log q) envelope quoted in the paper does not apply"
    );

    // The production parameter set must be inside the envelope.
    let prod_n = 16_384usize;
    let basis = RnsBasis::generate(prod_n, 4).expect("basis");
    let log_q = basis.modulus_bits();
    assert!(
        is_128bit_secure(prod_n, log_q),
        "AUDIT FAIL: production N={prod_n}, |Q|={log_q} is outside the 128-bit envelope"
    );
    eprintln!(
        "[audit] production params: N={prod_n}, |Q|={log_q} bits (ceiling 438 at 128-bit \
         classical), ternary secret, CBD(eta={DEFAULT_NOISE_ETA}) sigma={sigma:.2}, \
         margin {} bits of log q",
        438 - log_q
    );

    // The limb count is fixed, so |Q| does not shrink with N. Reduced-degree
    // configurations reuse the same ~124-bit modulus and are therefore NOT
    // secure — they exist for test speed only. This must be detected, not
    // silently accepted, or a reduced-degree lane could be mistaken for a
    // deployable configuration.
    for small_n in [1024usize, 2048, 4096] {
        let b = RnsBasis::generate(small_n, 4).expect("basis");
        let lq = b.modulus_bits();
        assert!(
            !is_128bit_secure(small_n, lq),
            "AUDIT FAIL: N={small_n} with |Q|={lq} was reported secure; the HE Standard \
             ceiling at that degree is far below 124 bits"
        );
    }
    eprintln!(
        "[audit] reduced-degree lanes (N<=4096, |Q|=124) correctly flagged insecure; \
         minimum secure degree for |Q|=124 is N={:?}",
        min_degree_for_128bit(log_q)
    );
}

/// AUDIT 12 — locate the real cost centre of the encrypted path.
///
/// Profiling the full panel attributes only ~4% of warm runtime to the fused
/// GPU key-switch and ~91% to unattributed host work. `poly_mul_as` — the
/// ring multiply used by encrypt, decrypt, and Galois key generation — is
/// schoolbook O(N²) over `u128`, which at N=16384 is ~1.8e8 modular operations
/// per call. This times the primitives that call it so the claim is measured.
/// Ignored by default (~1 min): `cargo test --release -- --ignored primitive_cost`.
#[test]
#[ignore]
fn audit_primitive_cost_at_production_degree() {
    let n = 16_384usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let scale = (40u32 as f64).exp2();

    let t = std::time::Instant::now();
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let keygen_ms = t.elapsed().as_secs_f64() * 1e3;

    let slots: Vec<f64> = (0..n / 2).map(|i| 0.001 * ((i % 7) as f64)).collect();
    let coeffs = encode_real_slots(&slots, n, scale).expect("encode");

    let t = std::time::Instant::now();
    let ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt");
    let encrypt_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    let _ = decrypt(&sk, &ct, q).expect("decrypt");
    let decrypt_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    let _ = gen_galois_key(&sk, 5, &basis, DEFAULT_NOISE_ETA).expect("galois key");
    let galois_ms = t.elapsed().as_secs_f64() * 1e3;

    let rotations = (n / 2).ilog2() as usize;
    let ciphers = 110_000usize.div_ceil(n / 2);
    eprintln!(
        "[audit] N={n} primitive cost: keygen {keygen_ms:.0} ms, encrypt {encrypt_ms:.0} ms, \
         decrypt {decrypt_ms:.0} ms, one Galois key {galois_ms:.0} ms"
    );
    eprintln!(
        "[audit] implied per patient: {ciphers} encrypt+decrypt = {:.0} ms; \
         implied setup: {rotations} Galois keys = {:.0} ms",
        ciphers as f64 * (encrypt_ms + decrypt_ms),
        rotations as f64 * galois_ms
    );

    // These are O(N^2) today. The assertion is deliberately loose: it exists to
    // fail loudly if someone "optimises" by making them asymptotically worse.
    assert!(
        encrypt_ms < 5000.0 && galois_ms < 20_000.0,
        "AUDIT FAIL: primitive cost regressed badly"
    );
}

/// AUDIT 11 — count the key-switches a full 110k-SNP patient actually costs.
///
/// Section 5 attributes the throughput gap against cohort-amortized HE-PRS to
/// this number, so it is measured rather than derived: 14 ciphertexts covering
/// 110,000 SNPs at 8,192 slots, 13 automorphisms each, 4 RNS limbs per
/// key-switch. Ignored by default because it runs the full encrypted panel
/// (~20 s); run with `cargo test --release -- --ignored keyswitch_count`.
#[test]
#[ignore]
fn audit_full_panel_keyswitch_count_matches_paper() {
    use crate::ckks_rns::evaluate_encrypted_rns_cohort_backend;

    let snp = 110_000u32;
    let poly = 16_384u32;
    let scale_bits = 40u32;
    let (header, slots) = crate::clinic::pack_synthetic_clinic(1, 1, snp, poly, scale_bits, 0x5eed)
        .expect("pack");
    let params = crate::ckks_prs::unpack_header(&header).expect("header");
    let views = crate::ckks_prs::unpack_slots(params, &slots).expect("slots");

    let slot_count = (poly / 2) as usize;
    let ciphers = (snp as usize).div_ceil(slot_count);
    let rotations = slot_count.ilog2() as usize;
    let limbs = 4usize;
    // Hoisted rotate-and-sum (log slots) + one relinearization per ciphertext.
    let expected_limb_switches = (rotations + ciphers) * limbs;

    let mut run = || {
        let t = std::time::Instant::now();
        evaluate_encrypted_rns_cohort_backend(
            views.genotype_slots,
            views.weight_slots,
            params.patient_count as usize,
            params.disease_count as usize,
            params.poly_degree as usize,
            params.slot_count as usize,
            params.cipher_count as usize,
            (scale_bits as f64).exp2(),
            metal_available(),
        )
        .expect("encrypted full panel");
        t.elapsed().as_secs_f64() * 1e3
    };

    // First call pays KeyGen + EVK NTT warmup; profile the second (steady state).
    let cold_ms = run();
    let (fused_before, fallback_before) = ks_path_counters();
    let (d0, g0, r0, p0) = crate::metal_ks::ks_phase_nanos();
    let elapsed_ms = run();
    eprintln!("[audit] cold {cold_ms:.0} ms, warm {elapsed_ms:.0} ms");
    let (fused_after, fallback_after) = ks_path_counters();
    let (d1, g1, r1, p1) = crate::metal_ks::ks_phase_nanos();

    let ms = |ns: u64| ns as f64 / 1e6;
    let (dec, gpu, rec, plans) = (ms(d1 - d0), ms(g1 - g0), ms(r1 - r0), ms(p1 - p0));
    let pct = |x: f64| 100.0 * x / elapsed_ms;
    eprintln!(
        "[audit] full-panel {elapsed_ms:.0} ms breakdown: NTT plan build {plans:.0} ms \
         ({:.1}%), digit decompose+pack {dec:.0} ms ({:.1}%), fused GPU {gpu:.0} ms \
         ({:.1}%), CRT recombine {rec:.0} ms ({:.1}%), other {:.0} ms ({:.1}%)",
        pct(plans),
        pct(dec),
        pct(gpu),
        pct(rec),
        elapsed_ms - dec - gpu - rec - plans,
        pct(elapsed_ms - dec - gpu - rec - plans)
    );

    let total = (fused_after - fused_before) + (fallback_after - fallback_before);
    eprintln!(
        "[audit] full panel: {ciphers} ciphertexts (relin each) + {rotations} rotations \
         (hoisted) x {limbs} limbs = {} key-switches/patient ({} limb ops measured, \
         {} fused / {} fallback)",
        rotations + ciphers,
        total,
        fused_after - fused_before,
        fallback_after - fallback_before
    );
    assert_eq!(
        total, expected_limb_switches as u64,
        "AUDIT FAIL: measured {total} limb key-switches, paper implies {expected_limb_switches}"
    );
}

/// AUDIT 10 — quantify the noise cost of the 20-bit hybrid gadget.
///
/// Widening B from 2^10 to 2^20 bought roughly 2x throughput at the cost of
/// ~2^10 more key-switch noise. This measures the actual per-rotation error at
/// the Δ scale and at the Δ² scale the PRS pipeline actually uses, so the
/// trade-off in the paper is backed by numbers rather than asserted.
#[test]
fn audit_keyswitch_noise_budget_is_measured() {
    let n = 1024usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let gk = gen_galois_key(&sk, 5, &basis, DEFAULT_NOISE_ETA).expect("galois key");

    // Encrypting zero makes the decrypted phase equal to the accumulated noise,
    // so the key-switch contribution is measured directly rather than inferred.
    let zero = vec![0i64; n];
    let ct = encrypt_encoded(&pk, &zero, DEFAULT_NOISE_ETA).expect("encrypt zero");
    let fresh = decrypt(&sk, &ct, q).expect("decrypt fresh");
    let fresh_max = fresh.iter().map(|&c| c.abs()).max().unwrap_or(1).max(1);

    let rotated = automorphism_ct(&ct, 5, q);
    let switched = key_switch_with_backend(&rotated, &gk, &basis, false).expect("ks");
    let after = decrypt(&sk, &switched, q).expect("decrypt switched");
    let ks_max = after.iter().map(|&c| c.abs()).max().unwrap_or(1).max(1);

    let fresh_bits = (fresh_max as u128).ilog2() + 1;
    let ks_bits = (ks_max as u128).ilog2() + 1;
    let q_bits = crate::zq::zq_ilog2(q) + 1;
    let digits = digit_count(q);

    eprintln!(
        "[audit] N={n} |Q|={q_bits} B=2^{KS_DIGIT_BITS} digits={digits}: \
         fresh noise 2^{fresh_bits}, post-key-switch noise 2^{ks_bits}"
    );
    eprintln!(
        "[audit] implied relative error: {:.2e} at Δ=2^40, {:.2e} at Δ²=2^80",
        (ks_max as f64) / 2f64.powi(40),
        (ks_max as f64) / 2f64.powi(80)
    );

    // Key-switching must add noise (it is a real operation) but must leave a
    // wide margin under Q, and must be negligible at the Δ² working scale.
    assert!(
        ks_bits > fresh_bits,
        "AUDIT FAIL: key-switching added no noise — suspicious, it should"
    );
    assert!(
        ks_bits + 40 < q_bits,
        "AUDIT FAIL: post-key-switch noise 2^{ks_bits} leaves <40 bits under 2^{q_bits}"
    );
    assert!(
        (ks_max as f64) / 2f64.powi(80) < 1e-6,
        "AUDIT FAIL: key-switch noise is not negligible at the Δ² PRS scale"
    );
}

/// AUDIT 8 — REGRESSION: two distinct Galois keys sharing `(N, k)` must not
/// alias in the NTT-domain EVK cache.
///
/// The cache was originally keyed on `(N, k, limb, component)` only. A second
/// key-switch under a freshly generated key — a re-keyed clinic session — then
/// reused the first session's evaluation material and produced garbage with no
/// error raised. Both backends are exercised because the host map and the
/// resident `MTLBuffer` table are keyed independently.
#[test]
fn audit_distinct_galois_keys_do_not_alias_in_evk_cache() {
    let n = 128usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let k = 5usize;
    // Run at Δ=2^46 so single-rotation key-switch noise (≈2^31.5 at N=128,
    // B=2^20, σ≈3.16) sits ~4e-5 below the signal. A tight tolerance is what
    // makes this gate a discriminator: stale evaluation keys produce errors of
    // order the slot magnitude, thousands of times larger.
    let scale = (46u32 as f64).exp2();
    let slots: Vec<f64> = (0..n / 2).map(|i| 0.001 * (i as f64 + 1.0)).collect();
    let coeffs = encode_real_slots(&slots, n, scale).expect("encode");

    let sigma = |j: usize| -> usize {
        let idx = ((5 * (2 * j + 1)) % (2 * n) - 1) / 2;
        if idx < n / 2 {
            idx
        } else {
            n - 1 - idx
        }
    };
    let expected: Vec<f64> = (0..n / 2).map(|j| slots[sigma(j)]).collect();

    // Three independent key sets, all with the same (N, k). Under the aliasing
    // bug, sessions 2 and 3 silently key-switch with session 1's material.
    for session in 0..3 {
        let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
        let gk = gen_galois_key(&sk, k, &basis, DEFAULT_NOISE_ETA).expect("galois key");
        let ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).expect("encrypt");
        let rotated = automorphism_ct(&ct, k, q);

        for &prefer_metal in &[false, true] {
            if prefer_metal && !metal_available() {
                continue;
            }
            let switched =
                key_switch_with_backend(&rotated, &gk, &basis, prefer_metal).expect("ks");
            let phase = decrypt(&sk, &switched, q).expect("decrypt");
            let got = decode_real_slots_i128(&phase, scale).expect("decode");
            let err = expected
                .iter()
                .zip(got.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                err < 1e-3,
                "AUDIT FAIL: session {session} (prefer_metal={prefer_metal}) key-switched \
                 with stale evaluation keys — EVK cache aliased on (N, k). max err {err:.3e}"
            );
        }
    }
}

/// AUDIT 9 — distinct Galois keys must produce distinct fingerprints, and the
/// fingerprint must be stable for a given key.
#[test]
fn audit_galois_key_fingerprints_are_distinct_and_stable() {
    let n = 128usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let k = 5usize;

    let mut seen = Vec::new();
    for _ in 0..8 {
        let (sk, _) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
        let gk = gen_galois_key(&sk, k, &basis, DEFAULT_NOISE_ETA).expect("galois key");
        assert_eq!(
            gk.fingerprint,
            gk.clone().fingerprint,
            "AUDIT FAIL: fingerprint is not stable under clone"
        );
        assert!(
            !seen.contains(&gk.fingerprint),
            "AUDIT FAIL: two independently generated Galois keys share a fingerprint"
        );
        seen.push(gk.fingerprint);
    }

    // Different rotation steps under one secret key must also differ.
    let (sk, _) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let g1 = gen_galois_key(&sk, 5, &basis, DEFAULT_NOISE_ETA).expect("gk 5");
    let g2 = gen_galois_key(&sk, 25, &basis, DEFAULT_NOISE_ETA).expect("gk 25");
    assert_ne!(
        g1.fingerprint, g2.fingerprint,
        "AUDIT FAIL: Galois keys for different rotation steps share a fingerprint"
    );
}

/// AUDIT 7 — the ct×pt product must depend on the ciphertext, i.e. the
/// "encrypted" pipeline cannot be quietly computing on plaintext.
#[test]
fn audit_encrypted_pipeline_is_not_secretly_plaintext() {
    let n = 128usize;
    let basis = RnsBasis::generate(n, 4).expect("basis");
    let q = basis.modulus;
    let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).expect("keygen");
    let scale = (40u32 as f64).exp2();

    let a: Vec<f64> = (0..n / 2).map(|i| 0.01 * (i as f64 % 5.0)).collect();
    let b: Vec<f64> = (0..n / 2).map(|i| 0.02 * ((i % 7) as f64 - 3.0)).collect();
    let ca = encode_real_slots(&a, n, scale).expect("encode a");
    let cb = encode_real_slots(&b, n, scale).expect("encode b");
    let ct = encrypt_encoded(&pk, &ca, DEFAULT_NOISE_ETA).expect("encrypt");
    let pt: Vec<Zq> = cb.iter().map(|&c| to_residue(c as i128, q)).collect();
    let prod = mul_ct_pt(&ct, &pt, &basis).expect("ct x pt");

    // The product ciphertext must still require the key: c1 must be non-zero,
    // and the raw c0 must not decode to the answer without the secret key.
    assert!(
        prod.c1.iter().any(|&x| x != Zq::ZERO),
        "AUDIT FAIL: product c1 is zero — result is unmasked plaintext"
    );

    let no_key: Vec<i128> = prod.c0.iter().map(|&u| center(u, q)).collect();
    let leaked = decode_real_slots_i128(&no_key, scale * scale).expect("decode c0");
    let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    let leak_err = expected
        .iter()
        .zip(leaked.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    assert!(
        leak_err > EPSILON,
        "AUDIT FAIL: c0 alone decodes to the product without the secret key — \
         the pipeline is not actually encrypting"
    );

    // With the key it must decrypt correctly.
    let phase = decrypt(&sk, &prod, q).expect("decrypt");
    let got = decode_real_slots_i128(&phase, scale * scale).expect("decode");
    let err = expected
        .iter()
        .zip(got.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    assert!(
        err < EPSILON,
        "AUDIT FAIL: ct x pt product does not decrypt to the plaintext product (err {err:.3e})"
    );
}
