//! Residue Number System (RNS) basis and CRT limb helpers for CKKS.
//!
//! Decomposes a wide modulus `Q = ∏ q_i` into Montgomery-friendly primes
//! `q_i < 2^31` with `q_i ≡ 1 (mod 2N)`, so each limb can ride the existing
//! Metal NTT kernels unchanged. Pairwise CRT reconstitutes `Z_Q` coefficients
//! after limb-wise ring multiplication.
//!
//! Ring / plaintext paths use four limbs (`|Q| ≈ 124` bits, fits centered
//! `i128`). The encrypted HEPRS-matched path uses seven limbs
//! (`|Q| ≈ 200`–`210` bits) with coefficients stored as [`crate::zq::Zq`].

use rayon::prelude::*;

use crate::ntt::{find_ntt_modulus_below, mod_inv, NegacyclicPlan};
use crate::zq::{
    add_mod_zq, mul_mod_zq, product_zq, zq_ilog2, zq_to_center_i128, Zq,
};

pub type RnsResult<T> = Result<T, String>;

/// Ring / plaintext RNS limb count: four ~31-bit primes → `|Q| ≈ 124` bits.
pub const RING_RNS_LIMBS: usize = 4;

/// HEPRS-matched encrypted path: seven Metal-eligible limbs → `|Q| ≈ 202` bits.
pub const FHE_RNS_LIMBS: usize = 7;

/// Default limb count for the ring / plaintext path.
pub const DEFAULT_RNS_LIMBS: usize = RING_RNS_LIMBS;

/// RNS basis for degree-`N` CKKS with CRT recombination tables.
#[derive(Debug, Clone)]
pub struct RnsBasis {
    pub n: usize,
    pub primes: Vec<u64>,
    /// `Q = ∏ q_i` as a wide residue.
    pub modulus: Zq,
    /// `Q_i = Q / q_i`.
    pub q_hat: Vec<Zq>,
    /// `ŷ_i = Q_i^{-1} mod q_i`.
    pub q_hat_inv: Vec<u64>,
    /// Precomputed `w_i = (Q_i · ŷ_i) mod Q` — one mul per limb at recombine.
    pub crt_weight: Vec<Zq>,
}

/// `(a * b) % m` via widening multiply + reduction (no bit-serial loop).
/// Retained for AuxiliaryModulus / u128 helpers.
#[inline(always)]
pub fn mul_mod_u128(a: u128, b: u128, m: u128) -> u128 {
    if m == 0 {
        return 0;
    }
    let a = a % m;
    let b = b % m;
    if a == 0 || b == 0 {
        return 0;
    }
    let (hi, lo) = widening_mul_u128(a, b);
    rem_u256(hi, lo, m, two128_mod(m))
}

#[inline(always)]
fn two128_mod(m: u128) -> u128 {
    // 2^128 ≡ (u128::MAX % m + 1) (mod m)
    let t = u128::MAX % m;
    if t == m - 1 {
        0
    } else {
        t + 1
    }
}

/// 128×128 → 256-bit product as `(hi, lo)`.
#[inline(always)]
pub fn widening_mul_u128(a: u128, b: u128) -> (u128, u128) {
    let a0 = a as u64 as u128;
    let a1 = a >> 64;
    let b0 = b as u64 as u128;
    let b1 = b >> 64;

    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    // mid = p01 + p10 may exceed 2^128; capture the carry bit.
    let (mid, carry1) = p01.overflowing_add(p10);
    let carry1 = if carry1 { 1u128 } else { 0 };

    let lo_hi = (p00 >> 64) + (mid & 0xffff_ffff_ffff_ffff);
    let lo = (p00 & 0xffff_ffff_ffff_ffff) | ((lo_hi & 0xffff_ffff_ffff_ffff) << 64);
    let hi = p11 + (mid >> 64) + (lo_hi >> 64) + (carry1 << 64);
    (hi, lo)
}

/// `(hi · 2^128 + lo) mod m`.
#[inline(always)]
fn rem_u256(hi: u128, lo: u128, m: u128, two128: u128) -> u128 {
    if m == 0 {
        return 0;
    }
    if hi == 0 {
        return lo % m;
    }
    let hi_term = if two128 == 0 {
        0
    } else {
        // hi < 2^128, two128 < m → product may need one more reduction step
        let (h2, l2) = widening_mul_u128(hi % m, two128);
        if h2 == 0 {
            l2 % m
        } else {
            // hi*two128 < m*m ≤ 2^248; recurse once (h2 is tiny)
            rem_u256(h2, l2, m, two128)
        }
    };
    add_mod_u128(hi_term, lo % m, m)
}

#[inline(always)]
pub fn add_mod_u128(a: u128, b: u128, m: u128) -> u128 {
    let a = a % m;
    let b = b % m;
    // Callers keep m ≤ 2^124 so a+b < 2m fits in u128.
    let sum = a + b;
    if sum >= m {
        sum - m
    } else {
        sum
    }
}

fn to_residue(c: i128, q: u64) -> u64 {
    let q_i = q as i128;
    let mut v = c % q_i;
    if v < 0 {
        v += q_i;
    }
    v as u64
}

fn mul_mod_u64(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

fn zq_to_u64(z: Zq) -> u64 {
    u64::try_from(z).unwrap_or(0)
}

impl RnsBasis {
    /// Build a `k`-limb RNS basis of Metal-eligible NTT primes for degree `n`.
    pub fn generate(n: usize, k: usize) -> RnsResult<Self> {
        if !(1..=7).contains(&k) {
            return Err(format!("RNS limb count {k} out of range 1..=7"));
        }
        let primes = find_rns_primes(n, k)
            .ok_or_else(|| format!("could not find {k} RNS primes for N={n}"))?;
        Self::from_primes(n, primes)
    }

    pub fn from_primes(n: usize, primes: Vec<u64>) -> RnsResult<Self> {
        validate_primes(n, &primes)?;
        let modulus = product_zq(&primes)?;
        let mut q_hat = Vec::with_capacity(primes.len());
        let mut q_hat_inv = Vec::with_capacity(primes.len());
        let mut crt_weight = Vec::with_capacity(primes.len());
        for &q in &primes {
            let hat = modulus / Zq::from(q);
            let hat_mod_q = zq_to_u64(hat % Zq::from(q));
            let inv = mod_inv(hat_mod_q, q);
            if mul_mod_u64(hat_mod_q, inv, q) != 1 {
                return Err(format!("CRT inverse failed for q={q}"));
            }
            // w = (Q_i * ŷ_i) mod Q — residues are tiny so this is one fast mul.
            let w = mul_mod_zq(hat % modulus, Zq::from(inv), modulus);
            q_hat.push(hat);
            q_hat_inv.push(inv);
            crt_weight.push(w);
        }
        Ok(Self {
            n,
            primes,
            modulus,
            q_hat,
            q_hat_inv,
            crt_weight,
        })
    }

    pub fn limb_count(&self) -> usize {
        self.primes.len()
    }

    pub fn modulus_bits(&self) -> u32 {
        if self.modulus == Zq::ZERO {
            0
        } else {
            zq_ilog2(self.modulus) + 1
        }
    }

    /// `c ↦ (c mod q_i)_i` with centered `i64` input.
    pub fn decompose_coeff(&self, c: i64) -> Vec<u64> {
        self.primes
            .iter()
            .map(|&q| to_residue(c as i128, q))
            .collect()
    }

    /// Decompose a centered `i128` coefficient (for RLWE polys mod Q, small Q).
    pub fn decompose_coeff_i128(&self, c: i128) -> Vec<u64> {
        self.primes.iter().map(|&q| to_residue(c, q)).collect()
    }

    /// Direct residue: `c mod q_i` with no i128 centering (safe for `|Q| > 128`).
    pub fn decompose_coeff_zq(&self, c: Zq) -> Vec<u64> {
        self.primes
            .iter()
            .map(|&q| zq_to_u64(c % Zq::from(q)))
            .collect()
    }

    /// Decompose a length-`N` polynomial into `k` residue limbs (`limb-major`).
    pub fn decompose_poly(&self, coeffs: &[i64]) -> RnsResult<Vec<Vec<u64>>> {
        if coeffs.len() != self.n {
            return Err(format!(
                "poly length {} ≠ RNS degree {}",
                coeffs.len(),
                self.n
            ));
        }
        let k = self.limb_count();
        let mut limbs = vec![vec![0u64; self.n]; k];
        for (j, &c) in coeffs.iter().enumerate() {
            let r = self.decompose_coeff(c);
            for i in 0..k {
                limbs[i][j] = r[i];
            }
        }
        Ok(limbs)
    }

    /// Decompose a length-`N` `i128` polynomial into RNS limbs.
    pub fn decompose_poly_i128(&self, coeffs: &[i128]) -> RnsResult<Vec<Vec<u64>>> {
        if coeffs.len() != self.n {
            return Err(format!(
                "poly length {} ≠ RNS degree {}",
                coeffs.len(),
                self.n
            ));
        }
        let k = self.limb_count();
        let mut limbs = vec![vec![0u64; self.n]; k];
        for (j, &c) in coeffs.iter().enumerate() {
            let r = self.decompose_coeff_i128(c);
            for i in 0..k {
                limbs[i][j] = r[i];
            }
        }
        Ok(limbs)
    }

    /// Decompose `Zq` coefficients via direct `c % q_i` (no i128 centering).
    pub fn decompose_poly_zq(&self, coeffs: &[Zq]) -> RnsResult<Vec<Vec<u64>>> {
        if coeffs.len() != self.n {
            return Err(format!(
                "poly length {} ≠ RNS degree {}",
                coeffs.len(),
                self.n
            ));
        }
        let k = self.limb_count();
        let mut limbs = vec![vec![0u64; self.n]; k];
        for (j, &c) in coeffs.iter().enumerate() {
            let r = self.decompose_coeff_zq(c);
            for i in 0..k {
                limbs[i][j] = r[i];
            }
        }
        Ok(limbs)
    }

    /// Inverse CRT for one coefficient → residue in `[0, Q)`.
    #[inline(always)]
    pub fn recombine_coeff_zq(&self, residues: &[u64]) -> RnsResult<Zq> {
        if residues.len() != self.limb_count() {
            return Err("residue count ≠ limb count".into());
        }
        let q = self.modulus;
        let mut acc = Zq::ZERO;
        for i in 0..self.limb_count() {
            let ci = Zq::from(residues[i]);
            let t = mul_mod_zq(ci, self.crt_weight[i], q);
            acc = add_mod_zq(acc, t, q);
        }
        Ok(acc)
    }

    /// Inverse CRT → centered representative in `(-Q/2, Q/2]` as `i128`.
    ///
    /// For ring-engine / small-Q paths (`k ≤ 4`). Wide mid-pipeline CT coeffs
    /// must stay as [`Zq`] via [`Self::recombine_coeff_zq`].
    #[inline(always)]
    pub fn recombine_coeff(&self, residues: &[u64]) -> RnsResult<i128> {
        let u = self.recombine_coeff_zq(residues)?;
        zq_to_center_i128(u, self.modulus)
    }

    /// Recombine `k` limbs (`limb-major`) into a centered `i128` polynomial.
    /// Coefficients are processed in parallel across Rayon.
    pub fn recombine_poly(&self, limbs: &[Vec<u64>]) -> RnsResult<Vec<i128>> {
        require_limb_layout(self, limbs)?;
        let k = self.limb_count();
        let n = self.n;
        (0..n)
            .into_par_iter()
            .map(|j| {
                let mut tmp = vec![0u64; k];
                for i in 0..k {
                    tmp[i] = limbs[i][j];
                }
                self.recombine_coeff(&tmp)
            })
            .collect()
    }

    /// Recombine `k` limbs into `Zq` residues in `[0, Q)` (no centering).
    pub fn recombine_poly_zq(&self, limbs: &[Vec<u64>]) -> RnsResult<Vec<Zq>> {
        require_limb_layout(self, limbs)?;
        let k = self.limb_count();
        let n = self.n;
        (0..n)
            .into_par_iter()
            .map(|j| {
                let mut tmp = vec![0u64; k];
                for i in 0..k {
                    tmp[i] = limbs[i][j];
                }
                self.recombine_coeff_zq(&tmp)
            })
            .collect()
    }
}

fn require_limb_layout(basis: &RnsBasis, limbs: &[Vec<u64>]) -> RnsResult<()> {
    if limbs.len() != basis.limb_count() {
        return Err("limb count mismatch".into());
    }
    if limbs.iter().any(|limb| limb.len() != basis.n) {
        return Err("limb degree mismatch".into());
    }
    Ok(())
}

fn prime_ok(n: usize, q: u64) -> bool {
    q < (1u64 << 31) && (q & 1) == 1 && NegacyclicPlan::new(n, q).is_some()
}

fn validate_primes(n: usize, primes: &[u64]) -> RnsResult<()> {
    if primes.is_empty() {
        return Err("empty RNS prime list".into());
    }
    if primes.len() > 7 {
        return Err(format!("RNS limb count {} exceeds 7", primes.len()));
    }
    for &q in primes {
        if !prime_ok(n, q) {
            return Err(format!("RNS prime {q} invalid for N={n}"));
        }
    }
    Ok(())
}

fn product_u128(primes: &[u64]) -> RnsResult<u128> {
    let mut modulus = 1u128;
    for &q in primes {
        modulus = modulus
            .checked_mul(q as u128)
            .ok_or_else(|| "RNS modulus Q overflowed u128".to_string())?;
    }
    Ok(modulus)
}

/// Largest-first search for `k` distinct Metal-eligible NTT primes.
pub fn find_rns_primes(n: usize, k: usize) -> Option<Vec<u64>> {
    let mut primes = Vec::with_capacity(k);
    let mut ceiling = 1u64 << 31;
    for _ in 0..k {
        let q = find_ntt_modulus_below(n, ceiling)?;
        primes.push(q);
        ceiling = q;
    }
    Some(primes)
}

/// Auxiliary special-modulus basis \(P = \prod p_j\) for hybrid key-switching.
///
/// Primes are Metal-eligible, co-prime to every limb of `Q`, and used for
/// modulus-raising / mod-down. `P` itself fits in `u64` when `k_p = 1`.
#[derive(Debug, Clone)]
pub struct AuxiliaryModulus {
    pub primes: Vec<u64>,
    /// \(P = \prod p_j\).
    pub modulus: u128,
    /// \(P^{-1} \bmod q_i\) for each Q-limb prime (same order as `q_primes`).
    pub p_inv_mod_q: Vec<u64>,
}

impl AuxiliaryModulus {
    /// Pick `k_p` aux primes below \(2^{31}\), excluding those already in `q_primes`.
    pub fn generate(n: usize, k_p: usize, q_primes: &[u64]) -> RnsResult<Self> {
        if k_p == 0 {
            return Err("k_p must be ≥ 1".into());
        }
        let mut exclude: std::collections::HashSet<u64> = q_primes.iter().copied().collect();
        let mut primes = Vec::with_capacity(k_p);
        let mut ceiling = 1u64 << 31;
        for _ in 0..k_p {
            let mut found = None;
            while let Some(p) = find_ntt_modulus_below(n, ceiling) {
                ceiling = p;
                if exclude.insert(p) {
                    found = Some(p);
                    break;
                }
            }
            let p = found.ok_or_else(|| format!("could not find {k_p} aux primes for N={n}"))?;
            primes.push(p);
        }
        let modulus = product_u128(&primes)?;
        let mut p_inv_mod_q = Vec::with_capacity(q_primes.len());
        for &q in q_primes {
            let p_mod_q = (modulus % q as u128) as u64;
            let inv = mod_inv(p_mod_q, q);
            if inv == 0 {
                return Err(format!("P has no inverse mod q={q}"));
            }
            p_inv_mod_q.push(inv);
        }
        Ok(Self {
            primes,
            modulus,
            p_inv_mod_q,
        })
    }

    /// Exact RNS mod-down: given residues of \(x\) on \(Q \cup P\) (Q limbs
    /// first, then P limbs), return residues of \(\lfloor x / P \rceil\) on \(Q\).
    ///
    /// Uses the standard formula \(y_i = (x_i - x_P) \cdot P^{-1} \bmod q_i\)
    /// with \(x_P\) reconstructed from the P-limb block (exact when
    /// \|x\| < PQ/2, which holds after hybrid KS with small noise).
    pub fn mod_down_coeff(
        &self,
        q_limbs: &[u64],
        p_limbs: &[u64],
        q_primes: &[u64],
    ) -> RnsResult<Vec<u64>> {
        if q_limbs.len() != q_primes.len() || q_limbs.len() != self.p_inv_mod_q.len() {
            return Err("mod_down Q limb count mismatch".into());
        }
        if p_limbs.len() != self.primes.len() {
            return Err("mod_down P limb count mismatch".into());
        }
        // Reconstruct x_P ∈ (-P/2, P/2] from P-limb residues via small CRT.
        let x_p = if self.primes.len() == 1 {
            let p = self.primes[0];
            let r = p_limbs[0] % p;
            if r > p / 2 {
                r as i128 - p as i128
            } else {
                r as i128
            }
        } else {
            // Multi-limb P: recombine with a temporary basis.
            let p_basis = RnsBasis::from_primes(1, self.primes.clone())?;
            let centered = p_basis.recombine_coeff(p_limbs)?;
            centered
        };
        let mut out = Vec::with_capacity(q_primes.len());
        for i in 0..q_primes.len() {
            let q = q_primes[i];
            let xi = q_limbs[i] % q;
            // x_P mod q
            let xp_mod = to_residue(x_p, q);
            let diff = if xi >= xp_mod {
                xi - xp_mod
            } else {
                xi + q - xp_mod
            };
            let y = ((diff as u128 * self.p_inv_mod_q[i] as u128) % q as u128) as u64;
            out.push(y);
        }
        Ok(out)
    }

    /// Limb-major poly mod-down: `q_limbs[limb][coeff]`, `p_limbs[limb][coeff]`.
    pub fn mod_down_poly(
        &self,
        q_limbs: &[Vec<u64>],
        p_limbs: &[Vec<u64>],
        q_primes: &[u64],
        n: usize,
    ) -> RnsResult<Vec<Vec<u64>>> {
        let k_q = q_primes.len();
        let mut out = vec![vec![0u64; n]; k_q];
        for j in 0..n {
            let ql: Vec<u64> = (0..k_q).map(|i| q_limbs[i][j]).collect();
            let pl: Vec<u64> = (0..self.primes.len()).map(|i| p_limbs[i][j]).collect();
            let y = self.mod_down_coeff(&ql, &pl, q_primes)?;
            for i in 0..k_q {
                out[i][j] = y[i];
            }
        }
        Ok(out)
    }
}

/// Default aux limb count for hybrid KS (\(k_P = 1\) → \|P\| ≈ 31 bits).
pub const DEFAULT_AUX_LIMBS: usize = 1;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn rns_roundtrip_small_coeffs() {
        let basis = RnsBasis::generate(64, 4).unwrap();
        assert_eq!(basis.limb_count(), 4);
        for &c in &[-1000i64, -1, 0, 1, 42, 1_000_000, 1i64 << 40] {
            let limbs = basis.decompose_coeff(c);
            let back = basis.recombine_coeff(&limbs).unwrap();
            assert_eq!(back, c as i128, "c={c}");
        }
    }

    #[test]
    fn rns_poly_roundtrip() {
        let n = 64usize;
        let basis = RnsBasis::generate(n, 3).unwrap();
        let coeffs: Vec<i64> = (0..n as i64).map(|i| (i - 32) * 1_000_000).collect();
        let limbs = basis.decompose_poly(&coeffs).unwrap();
        let back = basis.recombine_poly(&limbs).unwrap();
        for i in 0..n {
            assert_eq!(back[i], coeffs[i] as i128);
        }
    }

    #[test]
    fn primes_are_metal_friendly() {
        let basis = RnsBasis::generate(16384, 4).unwrap();
        for &q in &basis.primes {
            assert!(q < (1u64 << 31));
            assert_eq!((q - 1) % (2 * 16384), 0);
        }
        assert!(basis.modulus_bits() >= 100);
    }

    #[test]
    fn generate_seven_limbs_heprs_budget() {
        let basis = RnsBasis::generate(8192, 7).unwrap();
        assert_eq!(basis.limb_count(), 7);
        for &q in &basis.primes {
            assert!(q < (1u64 << 31));
            assert_eq!((q - 1) % (2 * 8192), 0);
        }
        let bits = basis.modulus_bits();
        assert!(
            (195..=218).contains(&bits),
            "7-limb |Q| bits={bits}, expected ≈202 in 195..=218"
        );
    }

    #[test]
    fn fhe_basis_matches_heprs_qp_budget() {
        let basis = RnsBasis::generate(8192, FHE_RNS_LIMBS).unwrap();
        assert_eq!(FHE_RNS_LIMBS, 7);
        assert_eq!(basis.limb_count(), 7);
        let bits = basis.modulus_bits();
        eprintln!("[fhe] N=8192 FHE_RNS_LIMBS=7 |Q| bits={bits} (HEPRS QP≈202, ceiling 218)");
        assert!(
            (195..=218).contains(&bits),
            "FHE basis |Q| bits={bits}, HEPRS QP≈202 / HE ceiling 218"
        );
    }

    #[test]
    fn mul_mod_u128_matches_naive() {
        let m2 = 97u128 * 89u128;
        assert_eq!(mul_mod_u128(123, 456, m2), (123u128 * 456) % m2);
        let m = 1_000_000_007u128 * 1_000_000_009u128;
        let a = m - 123;
        let b = m - 456;
        let got = mul_mod_u128(a, b, m);
        assert!(got < m);
        // Cross-check via big-int style: (a%m)*(b%m)%m with known small case
        assert_eq!(mul_mod_u128(3, 5, 7), 1);
        assert_eq!(mul_mod_u128(u128::MAX, u128::MAX, 97), (u128::MAX % 97) * (u128::MAX % 97) % 97);
    }

    /// Task 1 gate: CRT recombination of a production-size poly is single-digit ms.
    #[test]
    fn crt_recombine_is_fast_at_n16384() {
        let n = 16384usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let coeffs: Vec<i64> = (0..n as i64)
            .map(|i| ((i * 1_000_000) % 1_000_000_007) - 500_000_000)
            .collect();
        let limbs = basis.decompose_poly(&coeffs).unwrap();
        // Warmup
        let _ = basis.recombine_poly(&limbs).unwrap();
        let t0 = Instant::now();
        let back = basis.recombine_poly(&limbs).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        for i in 0..n {
            assert_eq!(back[i], coeffs[i] as i128);
        }
        assert!(
            ms < 10.0,
            "CRT recombine took {ms:.2} ms — must be < 10 ms (Task 1)"
        );
        eprintln!("CRT recombine N={n}: {ms:.3} ms");
    }
}
