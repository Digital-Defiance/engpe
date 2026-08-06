use super::*;

fn sample_header(snp: u32, n: u32, scale_bits: u32) -> Vec<u32> {
    sample_header_md(snp, n, scale_bits, 1, 1)
}

fn sample_header_m(snp: u32, n: u32, scale_bits: u32, patients: u32) -> Vec<u32> {
    sample_header_md(snp, n, scale_bits, patients, 1)
}

fn sample_header_md(
    snp: u32,
    n: u32,
    scale_bits: u32,
    patients: u32,
    diseases: u32,
) -> Vec<u32> {
    let slots = n / 2;
    let ciphers = snp.div_ceil(slots);
    vec![n, snp, slots, ciphers, scale_bits, patients, diseases]
}

#[test]
fn cpu_prs_matches_oracle_small() {
    let n = 64u32;
    let snp = 20u32;
    let scale_bits = 12u32;
    let h = sample_header(snp, n, scale_bits);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    for i in 0..snp as usize {
        slots[i] = (i % 3) as f64;
        slots[params.per_side() + i] = 0.05 * ((i as f64) - 10.0);
    }
    let result = evaluate_prs_cpu(&h, &slots).unwrap();
    let p = result.primary();
    assert!(
        p.abs_error < 1e-2,
        "oracle={} decoded={} err={}",
        p.oracle,
        p.decoded,
        p.abs_error
    );
}

#[test]
fn cpu_cohort_matches_oracles() {
    let n = 64u32;
    let snp = 20u32;
    let m = 4u32;
    let h = sample_header_m(snp, n, 12, m);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    let per = params.per_side();
    for p in 0..m as usize {
        for i in 0..snp as usize {
            slots[p * per + i] = ((i + p) % 3) as f64;
        }
    }
    for i in 0..snp as usize {
        slots[m as usize * per + i] = 0.05 * ((i as f64) - 10.0);
    }
    let result = evaluate_prs_cpu(&h, &slots).unwrap();
    assert_eq!(result.patient_count, m);
    assert_eq!(result.disease_count, 1);
    for s in &result.scores {
        assert!(
            s.abs_error < 1e-2,
            "oracle={} decoded={} err={}",
            s.oracle,
            s.decoded,
            s.abs_error
        );
    }
}

#[test]
fn cpu_multi_disease_matrix_matches_oracles() {
    let n = 64u32;
    let snp = 20u32;
    let m = 2u32;
    let d = 3u32;
    let h = sample_header_md(snp, n, 12, m, d);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    let per = params.per_side();
    for p in 0..m as usize {
        for i in 0..snp as usize {
            slots[p * per + i] = ((i + p) % 3) as f64;
        }
    }
    for disease in 0..d as usize {
        for i in 0..snp as usize {
            slots[(m as usize + disease) * per + i] =
                0.02 * ((i as f64) - 5.0) * (1.0 + 0.1 * disease as f64);
        }
    }
    let result = evaluate_prs_cpu(&h, &slots).unwrap();
    assert_eq!(result.patient_count, m);
    assert_eq!(result.disease_count, d);
    assert_eq!(result.scores.len(), (m * d) as usize);
    for s in &result.scores {
        assert!(
            s.abs_error < 1e-2,
            "p={} d={} oracle={} decoded={} err={}",
            s.patient,
            s.disease,
            s.oracle,
            s.decoded,
            s.abs_error
        );
    }
}

#[test]
fn sum_of_decoded_products_matches_oracle() {
    let n = 64usize;
    let scale = (12u32 as f64).exp2();
    let snp = 20usize;
    let mut a = vec![0.0; n / 2];
    let mut b = vec![0.0; n / 2];
    let mut oracle = 0.0;
    for i in 0..snp {
        a[i] = (i % 3) as f64;
        b[i] = 0.05 * ((i as f64) - 10.0);
        oracle += a[i] * b[i];
    }
    let plan = plan_for(n, 12).unwrap();
    let ca = encode_real_slots(&a, n, scale).unwrap();
    let cb = encode_real_slots(&b, n, scale).unwrap();
    let ra: Vec<u64> = ca.iter().map(|&c| to_centered_u64(c, plan.q)).collect();
    let rb: Vec<u64> = cb.iter().map(|&c| to_centered_u64(c, plan.q)).collect();
    let prod = ntt_pointwise_product(&ra, &rb, &plan, false).unwrap().0;
    let coeffs: Vec<i64> = prod.iter().map(|&u| from_centered_u64(u, plan.q)).collect();
    let decoded = decode_real_slots(&coeffs, scale * scale).unwrap();
    let summed: f64 = decoded.iter().sum();
    assert!(
        (summed - oracle).abs() < 1e-2,
        "oracle={oracle} slot_sum={summed}"
    );
}

#[test]
fn ntt_mul_is_approximately_slotwise() {
    let n = 64usize;
    let scale = (14u32 as f64).exp2();
    let a: Vec<f64> = (0..n / 2).map(|i| 0.05 * (i as f64 - 10.0)).collect();
    let b: Vec<f64> = (0..n / 2).map(|i| 0.02 * ((i % 7) as f64)).collect();
    let plan = plan_for(n, 14).unwrap();
    let ca = encode_real_slots(&a, n, scale).unwrap();
    let cb = encode_real_slots(&b, n, scale).unwrap();
    let ra: Vec<u64> = ca.iter().map(|&c| to_centered_u64(c, plan.q)).collect();
    let rb: Vec<u64> = cb.iter().map(|&c| to_centered_u64(c, plan.q)).collect();
    let prod = ntt_pointwise_product(&ra, &rb, &plan, false).unwrap().0;
    let coeffs: Vec<i64> = prod.iter().map(|&u| from_centered_u64(u, plan.q)).collect();
    let decoded = decode_real_slots(&coeffs, scale * scale).unwrap();
    for i in 0..n / 2 {
        let expected = a[i] * b[i];
        assert!(
            (decoded[i] - expected).abs() < 5e-2,
            "slot {i}: got {} want {expected}",
            decoded[i]
        );
    }
}

#[test]
fn cpu_rns_matches_oracle_at_scale_40() {
    let n = 64u32;
    let snp = 16u32;
    let h = sample_header(snp, n, 40);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    for i in 0..snp as usize {
        slots[i] = (i % 3) as f64;
        slots[params.per_side() + i] = 0.01 * ((i as f64) - 5.0);
    }
    let result = evaluate_prs_cpu(&h, &slots).unwrap();
    assert_eq!(result.backend, NttBackend::CpuRayon);
    let p = result.primary();
    assert!(
        p.abs_error < 1e-4,
        "oracle={} decoded={} err={}",
        p.oracle,
        p.decoded,
        p.abs_error
    );
}

#[test]
fn metal_prs_matches_oracle_when_available() {
    if !metal_available() {
        return;
    }
    let n = 64u32;
    let snp = 16u32;
    let scale_bits = 40u32;
    let h = sample_header(snp, n, scale_bits);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    for i in 0..snp as usize {
        slots[i] = (i % 3) as f64;
        slots[params.per_side() + i] = 0.01 * ((i as f64) - 5.0);
    }
    let result = evaluate_prs_pipeline(&h, &slots, true).unwrap();
    assert_eq!(result.backend, NttBackend::MetalGpu);
    let p = result.primary();
    assert!(
        p.abs_error < 1e-4,
        "oracle={} decoded={} err={}",
        p.oracle,
        p.decoded,
        p.abs_error
    );
}

#[test]
fn metal_rns_prs_n16384_epsilon() {
    if !metal_available() {
        return;
    }
    let n = 16384u32;
    let snp = 2048u32;
    let scale_bits = 40u32;
    let h = sample_header(snp, n, scale_bits);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    for i in 0..snp as usize {
        slots[i] = (i % 3) as f64;
        slots[params.per_side() + i] = 0.01 * ((i as f64) * 0.1 - 1.0);
    }
    let result = evaluate_prs_pipeline(&h, &slots, true).unwrap();
    assert_eq!(result.backend, NttBackend::MetalGpu);
    let p = result.primary();
    assert!(
        p.abs_error < 1e-4,
        "oracle={} decoded={} err={}",
        p.oracle,
        p.decoded,
        p.abs_error
    );
}

fn fill_snp_panel(
    slots: &mut [f64],
    per: usize,
    patients: usize,
    diseases: usize,
    snp: usize,
    weight_scale: f64,
) {
    for p in 0..patients {
        for i in 0..snp {
            slots[p * per + i] = ((i + p) % 3) as f64;
        }
    }
    for d in 0..diseases {
        for i in 0..snp {
            slots[(patients + d) * per + i] =
                weight_scale * ((i as f64) - 5.0) * (1.0 + 0.05 * d as f64);
        }
    }
}

#[test]
fn metal_cohort_batches_patients() {
    if !metal_available() {
        return;
    }
    let n = 64u32;
    let snp = 16usize;
    let m = 4usize;
    let h = sample_header_m(snp as u32, n, 40, m as u32);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    fill_snp_panel(&mut slots, params.per_side(), m, 1, snp, 0.01);
    let result = evaluate_prs_pipeline(&h, &slots, true).unwrap();
    assert_eq!(result.backend, NttBackend::MetalGpu);
    assert_eq!(result.patient_count, m as u32);
    for s in &result.scores {
        assert!(s.abs_error < 1e-4, "err={}", s.abs_error);
    }
}

/// VALIDATION GATE (Task 7): Metal M×D matrix must match oracle within ε < 1e-4.
#[test]
fn metal_multi_disease_matrix_gate() {
    if !metal_available() {
        return;
    }
    let n = 64u32;
    let snp = 16usize;
    let m = 2usize;
    let d = 5usize;
    let h = sample_header_md(snp as u32, n, 40, m as u32, d as u32);
    let params = unpack_header(&h).unwrap();
    let mut slots = vec![0.0; params.expected_slots_len()];
    fill_snp_panel(&mut slots, params.per_side(), m, d, snp, 0.01);
    let result = evaluate_prs_pipeline(&h, &slots, true).unwrap();
    assert_eq!(result.backend, NttBackend::MetalGpu);
    assert_eq!(result.patient_count, m as u32);
    assert_eq!(result.disease_count, d as u32);
    assert_eq!(result.scores.len(), m * d);
    for s in &result.scores {
        assert!(
            s.abs_error < 1e-4,
            "GATE FAIL p={} d={} oracle={} decoded={} err={}",
            s.patient,
            s.disease,
            s.oracle,
            s.decoded,
            s.abs_error
        );
    }
}
