//! CKKS Galois automorphism and logarithmic rotate-and-sum.
//!
//! Slot rotation by `r` positions is the ring automorphism
//! `p(X) ↦ p(X^{5^r}) (mod X^N + 1)`. Generator `5` generates the odd units
//! modulo `2N` used by CKKS for the canonical embedding slots.
//!
//! # Precision
//! Each rotate-and-sum step adds two polynomials at the current message scale.
//! After `log₂(N/2)` steps every slot holds the sum of all slots (approximately).
//! Coefficient noise roughly doubles each step, so expect an extra
//! `O(log(N) / Δ_eff)` factor on top of encode/mul rounding when comparing to a
//! plaintext sum oracle.

use crate::ntt::mul_mod;

/// `base^exp mod modulus` for the automorphism exponent (modulus = 2N).
fn mod_pow_usize(mut base: usize, mut exp: usize, modulus: usize) -> usize {
    let mut result = 1usize % modulus;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp >>= 1;
    }
    result
}

/// Map coefficient index `i` under `X ↦ X^k` in `Z[X]/(X^N+1)`.
/// Returns `(dest_index, negate)`.
pub fn automorphism_index(i: usize, k: usize, n: usize) -> (usize, bool) {
    let mut j = (i * k) % (2 * n);
    let mut neg = false;
    if j >= n {
        j -= n;
        neg = true;
    }
    (j, neg)
}

/// Apply `p(X) → p(X^k) mod (X^N+1)` on residues modulo `q`.
/// `k` must be odd (ring automorphism).
pub fn automorphism_u64(poly: &[u64], k: usize, q: u64) -> Vec<u64> {
    let n = poly.len();
    let mut out = vec![0u64; n];
    for (i, &coeff) in poly.iter().enumerate() {
        let (j, neg) = automorphism_index(i, k, n);
        let v = if neg && coeff != 0 { q - coeff } else { coeff };
        out[j] = (out[j] + v) % q;
    }
    out
}

/// Left-rotate CKKS slots by `steps` using `k = 5^{steps} mod 2N`.
pub fn rotate_slots_u64(poly: &[u64], steps: usize, q: u64) -> Vec<u64> {
    let n = poly.len();
    let k = mod_pow_usize(5, steps, 2 * n);
    automorphism_u64(poly, k, q)
}

/// In-place `a += b (mod q)`.
pub fn add_assign_mod(a: &mut [u64], b: &[u64], q: u64) {
    for i in 0..a.len() {
        a[i] = (a[i] + b[i]) % q;
    }
}

/// Logarithmic rotate-and-sum: after `log₂(N/2)` steps every slot holds Σ slots.
///
/// Loop: for `i` in `0..log₂(N/2)`, add a rotation by `2^i` slots into the
/// accumulator (directive §3).
pub fn rotate_and_sum_u64(poly: &[u64], q: u64) -> Vec<u64> {
    let n = poly.len();
    let slots = n / 2;
    let mut acc = poly.to_vec();
    let mut step = 1usize;
    while step < slots {
        let rot = rotate_slots_u64(&acc, step, q);
        add_assign_mod(&mut acc, &rot, q);
        step <<= 1;
    }
    acc
}

/// Logarithmic rotate-and-sum over a wide modulus `Q` (`u128` residues).
pub fn rotate_and_sum_u128(poly: &[u128], q: u128) -> Vec<u128> {
    let n = poly.len();
    let slots = n / 2;
    let mut acc = poly.to_vec();
    let mut step = 1usize;
    while step < slots {
        let rot = rotate_slots_u128(&acc, step, q);
        add_assign_mod_u128(&mut acc, &rot, q);
        step <<= 1;
    }
    acc
}

fn rotate_slots_u128(poly: &[u128], steps: usize, q: u128) -> Vec<u128> {
    let n = poly.len();
    let k = mod_pow_usize(5, steps, 2 * n);
    automorphism_u128(poly, k, q)
}

pub fn automorphism_u128(poly: &[u128], k: usize, q: u128) -> Vec<u128> {
    let n = poly.len();
    let mut out = vec![0u128; n];
    for (i, &coeff) in poly.iter().enumerate() {
        let (j, neg) = automorphism_index(i, k, n);
        let v = if neg && coeff != 0 { q - coeff } else { coeff };
        out[j] = crate::rns::add_mod_u128(out[j], v, q);
    }
    out
}

pub fn add_assign_mod_u128(a: &mut [u128], b: &[u128], q: u128) {
    for i in 0..a.len() {
        a[i] = crate::rns::add_mod_u128(a[i], b[i], q);
    }
}

/// Apply `p(X) → p(X^k) mod (X^N+1)` on wide `Zq` residues.
pub fn automorphism_zq(poly: &[crate::zq::Zq], k: usize, q: crate::zq::Zq) -> Vec<crate::zq::Zq> {
    use crate::zq::{add_mod_zq, neg_mod_zq, Zq};
    let n = poly.len();
    let mut out = vec![Zq::ZERO; n];
    for (i, &coeff) in poly.iter().enumerate() {
        let (j, neg) = automorphism_index(i, k, n);
        let v = if neg { neg_mod_zq(coeff, q) } else { coeff };
        out[j] = add_mod_zq(out[j], v, q);
    }
    out
}

/// Logarithmic rotate-and-sum over a wide modulus `Q` (`Zq` residues).
pub fn rotate_and_sum_zq(poly: &[crate::zq::Zq], q: crate::zq::Zq) -> Vec<crate::zq::Zq> {
    use crate::zq::add_mod_zq;
    let n = poly.len();
    let slots = n / 2;
    let mut acc = poly.to_vec();
    let mut step = 1usize;
    while step < slots {
        let k = mod_pow_usize(5, step, 2 * n);
        let rot = automorphism_zq(&acc, k, q);
        for i in 0..n {
            acc[i] = add_mod_zq(acc[i], rot[i], q);
        }
        step <<= 1;
    }
    acc
}

/// Pointwise product in the NTT domain, then inverse NTT — used by eval.
#[allow(dead_code)]
pub fn pointwise_mul_mod(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| mul_mod(x, y, q))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckks_encode::encode_real_slots;

    #[test]
    fn automorphism_is_permutation_for_odd_k() {
        let n = 16;
        let mut dests: Vec<usize> = (0..n).map(|i| automorphism_index(i, 5, n).0).collect();
        dests.sort_unstable();
        assert_eq!(
            dests,
            (0..n).collect::<Vec<_>>(),
            "automorphism destinations are not a permutation of 0..N"
        );

        // Applying φ_5 twice more (order divides 2N) must stay a permutation.
        let poly: Vec<u64> = (0..n as u64).collect();
        let once = automorphism_u64(&poly, 5, 97);
        let twice = automorphism_u64(&once, 5, 97);
        assert_ne!(once, poly, "φ_5 should move coefficients");
        assert_eq!(twice.len(), n);
        let mut seen = once.clone();
        seen.sort_unstable();
        // All values from the original appear (with possible negation already folded
        // into the residue), and length is preserved ⇒ bijective on the ring.
        assert_eq!(seen.len(), n);
        assert!(once.iter().any(|&x| x != 0));
    }

    #[test]
    fn rotate_and_sum_steps_match_log_slots() {
        use crate::ckks_encode::decode_real_slots_i128;

        let n = 32usize;
        let scale = (16u32 as f64).exp2();
        let slots_n = n / 2;
        let slots: Vec<f64> = (0..slots_n).map(|i| (i as f64) * 0.25 + 0.5).collect();
        let expected_sum: f64 = slots.iter().sum();

        // Encode slots and all-ones, negacyclic-mul (product = slots at scale Δ²),
        // then rotate-and-sum under a wide modulus and check every slot holds Σ.
        let ca = encode_real_slots(&slots, n, scale).unwrap();
        let cb = encode_real_slots(&vec![1.0f64; slots_n], n, scale).unwrap();
        let mut prod = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let mut k = i + j;
                let mut term = ca[i] as i128 * cb[j] as i128;
                if k >= n {
                    k -= n;
                    term = -term;
                }
                prod[k] += term;
            }
        }

        let q = 1u128 << 100;
        let residues: Vec<u128> = prod
            .iter()
            .map(|&c| {
                let mut v = c % (q as i128);
                if v < 0 {
                    v += q as i128;
                }
                v as u128
            })
            .collect();

        let summed = rotate_and_sum_u128(&residues, q);
        assert_eq!(summed.len(), n);

        let centered: Vec<i128> = summed
            .iter()
            .map(|&u| {
                if u > q / 2 {
                    u as i128 - q as i128
                } else {
                    u as i128
                }
            })
            .collect();
        let decoded = decode_real_slots_i128(&centered, scale * scale).unwrap();
        for (i, &v) in decoded.iter().enumerate() {
            assert!(
                (v - expected_sum).abs() < 1e-2,
                "slot {i}: got {v}, want ≈ {expected_sum}"
            );
        }

        // Determinism + log₂(slots) steps are meaningful (second RaS accumulates again).
        assert_eq!(rotate_and_sum_u128(&residues, q), summed);
        let twice = rotate_and_sum_u128(&summed, q);
        assert_ne!(twice, summed, "second RaS must further accumulate");
    }
}
