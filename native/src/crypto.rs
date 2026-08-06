//! RLWE / CKKS cryptographic primitives (Phase 3).
//!
//! - CSPRNG via `getrandom`
//! - Ternary secret keys \(s \leftarrow \{-1,0,1\}^N\)
//! - Centered-binomial error \(e \leftarrow \mathrm{CBD}(\eta)\)
//! - Uniform \(a \leftarrow U(R_Q)\)
//! - KeyGen / Encrypt / Decrypt over the RNS modulus \(Q\)
//! - Ciphertext–plaintext multiply and Galois key-switching for rotate-and-sum
//!
//! Ciphertext coefficients live in [`crate::zq::Zq`] (`ethnum::U256`) so the
//! HEPRS-matched seven-limb product (\(|Q|\approx 202\)) fits without mid-pipeline
//! i128 centering.

use crate::ckks_rotate::automorphism_zq;
use crate::ntt::{
    forward_ntt_negacyclic, inverse_ntt_negacyclic, mul_mod, NegacyclicPlan,
};
use crate::rns::{AuxiliaryModulus, RnsBasis, DEFAULT_AUX_LIMBS};
use crate::zq::{
    add_mod_zq, center_i128_to_zq, center_i64_to_zq, mul_mod_zq, neg_mod_zq, zq_from_u128,
    zq_ilog2, zq_to_center_i128, Zq,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub type CryptoResult<T> = Result<T, String>;

/// An RNS basis plus its per-limb NTT plans, memoised by `(n, limbs, first_prime)`.
type BasisEntry = (Arc<RnsBasis>, Arc<Vec<NegacyclicPlan>>);
type BasisKey = (usize, usize, u64);

/// Registry enabling the O(N log N) ring multiply from call sites that only
/// carry `Q`.
///
/// `encrypt` and `decrypt` take a modulus rather than a basis, so without this
/// they fall back to the schoolbook O(N²) path. At N=16384 that path dominates
/// everything else in the encrypted pipeline. `U256` is not `Hash`, so the map
/// key is `(n, limb_count, first_prime)` and lookup by `Q` scans for a match.
fn basis_registry() -> &'static Mutex<HashMap<BasisKey, BasisEntry>> {
    static REG: OnceLock<Mutex<HashMap<BasisKey, BasisEntry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn basis_key(basis: &RnsBasis) -> BasisKey {
    (
        basis.n,
        basis.limb_count(),
        *basis.primes.first().unwrap_or(&0),
    )
}

/// Memoise `basis` and its NTT plans so ring multiplies under this `(N, Q)` can
/// use the transform path. Idempotent; safe to call on every key operation.
pub fn register_basis(basis: &RnsBasis) -> CryptoResult<()> {
    let key = basis_key(basis);
    {
        let map = basis_registry()
            .lock()
            .map_err(|_| "basis registry poisoned".to_string())?;
        if map.contains_key(&key) {
            return Ok(());
        }
    }
    let plans = Arc::new(limb_plans(basis)?);
    let mut map = basis_registry()
        .lock()
        .map_err(|_| "basis registry poisoned".to_string())?;
    map.entry(key)
        .or_insert_with(|| (Arc::new(basis.clone()), plans));
    Ok(())
}

fn lookup_basis(n: usize, q: Zq) -> Option<BasisEntry> {
    let map = basis_registry().lock().ok()?;
    map.values()
        .find(|(b, _)| b.n == n && b.modulus == q)
        .cloned()
}

/// Cached per-limb NTT plans for an already-registered basis.
pub fn cached_limb_plans(basis: &RnsBasis) -> CryptoResult<Arc<Vec<NegacyclicPlan>>> {
    let key = basis_key(basis);
    {
        let map = basis_registry()
            .lock()
            .map_err(|_| "basis registry poisoned".to_string())?;
        if let Some((_, plans)) = map.get(&key) {
            return Ok(Arc::clone(plans));
        }
    }
    register_basis(basis)?;
    let map = basis_registry()
        .lock()
        .map_err(|_| "basis registry poisoned".to_string())?;
    map.get(&key)
        .map(|(_, plans)| Arc::clone(plans))
        .ok_or_else(|| "basis registration failed".to_string())
}

/// Centered binomial parameter: stddev \(\sigma = \sqrt{\eta/2}\).
///
/// η=20 gives σ ≈ 3.16, matching the σ ≈ 3.2 assumed by the Homomorphic
/// Encryption Security Standard (2018) parameter tables.
pub const DEFAULT_NOISE_ETA: usize = 20;
/// Digit base for key-switching: \(B = 2^{\texttt{KS_DIGIT_BITS}}\).
pub const KS_DIGIT_BITS: u32 = 20;
pub const KS_DIGIT_BASE: u128 = 1u128 << KS_DIGIT_BITS;

/// Maximum `log2(q)` at each degree for classical 128-bit security with a
/// uniform ternary secret, from the Homomorphic Encryption Security Standard
/// (2018). Valid for σ ≈ 3.2; see `DEFAULT_NOISE_ETA`.
const HE_STANDARD_128: [(usize, u32); 6] = [
    (1024, 27),
    (2048, 54),
    (4096, 109),
    (8192, 218),
    (16384, 438),
    (32768, 881),
];

/// Whether `(N, log2 Q)` sits inside the published 128-bit security envelope.
pub fn is_128bit_secure(n: usize, log_q: u32) -> bool {
    match HE_STANDARD_128.iter().find(|(deg, _)| *deg == n) {
        Some((_, max_log_q)) => log_q <= *max_log_q,
        None => HE_STANDARD_128
            .iter()
            .filter(|(deg, _)| *deg <= n)
            .next_back()
            .is_some_and(|(_, max_log_q)| log_q <= *max_log_q),
    }
}

/// Smallest tabulated degree that admits `log_q` at 128-bit classical security.
pub fn min_degree_for_128bit(log_q: u32) -> Option<usize> {
    HE_STANDARD_128
        .iter()
        .find(|(_, max_log_q)| log_q <= *max_log_q)
        .map(|(deg, _)| *deg)
}

/// Cryptographically secure byte fill.
pub fn secure_bytes(out: &mut [u8]) -> CryptoResult<()> {
    getrandom::getrandom(out).map_err(|e| format!("CSPRNG failed: {e}"))
}

/// Pull a `u64` from the CSPRNG.
fn secure_u64() -> CryptoResult<u64> {
    let mut buf = [0u8; 8];
    secure_bytes(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Ternary coefficient: −1, 0, or +1 with equal probability.
pub fn sample_ternary_coeff() -> CryptoResult<i64> {
    let r = secure_u64()? % 3;
    Ok(r as i64 - 1)
}

/// Secret key \(s \leftarrow \{-1,0,1\}^N\).
pub fn sample_ternary_poly(n: usize) -> CryptoResult<Vec<i64>> {
    (0..n).map(|_| sample_ternary_coeff()).collect()
}

/// Centered binomial CBD(η): \(\sum_{i=1}^{η}(b_i - b'_i)\).
pub fn sample_cbd_coeff(eta: usize) -> CryptoResult<i64> {
    let nbytes = (2 * eta + 7) / 8;
    let mut buf = vec![0u8; nbytes];
    secure_bytes(&mut buf)?;
    let mut acc = 0i64;
    for i in 0..eta {
        let bit_a = (buf[i / 8] >> (i % 8)) & 1;
        let j = i + eta;
        let bit_b = (buf[j / 8] >> (j % 8)) & 1;
        acc += bit_a as i64 - bit_b as i64;
    }
    Ok(acc)
}

pub fn sample_cbd_poly(n: usize, eta: usize) -> CryptoResult<Vec<i64>> {
    (0..n).map(|_| sample_cbd_coeff(eta)).collect()
}

/// Uniform residue in `[0, Q)` via rejection sampling over 256-bit draws.
pub fn sample_uniform_coeff(q: Zq) -> CryptoResult<Zq> {
    if q == Zq::ZERO {
        return Err("modulus Q must be positive".into());
    }
    // Largest multiple of q below 2^256.
    let bound = Zq::MAX - (Zq::MAX % q);
    loop {
        let mut buf = [0u8; 32];
        secure_bytes(&mut buf)?;
        let r = Zq::from_le_bytes(buf);
        if r < bound {
            return Ok(r % q);
        }
    }
}

pub fn sample_uniform_poly(n: usize, q: Zq) -> CryptoResult<Vec<Zq>> {
    (0..n).map(|_| sample_uniform_coeff(q)).collect()
}

fn center_to_zq(c: i64, q: Zq) -> Zq {
    center_i64_to_zq(c, q)
}

fn zq_to_center(u: Zq, q: Zq) -> CryptoResult<i128> {
    zq_to_center_i128(u, q)
}

fn i128_to_zq(c: i128, q: Zq) -> Zq {
    center_i128_to_zq(c, q)
}

/// Ciphertext addition: \((c_0, c_1) + (d_0, d_1) = (c_0+d_0,\, c_1+d_1)\).
pub fn add_ct(a: &Ciphertext, b: &Ciphertext, q: Zq) -> CryptoResult<Ciphertext> {
    if a.degree() != b.degree() {
        return Err("ciphertext degree mismatch".into());
    }
    Ok(Ciphertext {
        c0: add_poly_mod(&a.c0, &b.c0, q),
        c1: add_poly_mod(&a.c1, &b.c1, q),
    })
}

fn add_poly_mod(a: &[Zq], b: &[Zq], q: Zq) -> Vec<Zq> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| add_mod_zq(x, y, q))
        .collect()
}

fn neg_poly_mod(a: &[Zq], q: Zq) -> Vec<Zq> {
    a.iter().map(|&x| neg_mod_zq(x, q)).collect()
}

/// Negacyclic schoolbook multiply in `Z_Q[X]/(X^N+1)` (used for small polys / tests).
pub fn mul_poly_negacyclic(a: &[Zq], b: &[Zq], q: Zq) -> Vec<Zq> {
    let n = a.len();
    let mut out = vec![Zq::ZERO; n];
    for i in 0..n {
        for j in 0..n {
            let mut prod = mul_mod_zq(a[i], b[j], q);
            let mut k = i + j;
            if k >= n {
                k -= n;
                prod = neg_mod_zq(prod, q);
            }
            out[k] = add_mod_zq(out[k], prod, q);
        }
    }
    out
}

/// Fast negacyclic mul via per-limb NTT when an RNS basis is available.
///
/// Decomposes with [`RnsBasis::decompose_poly_zq`] (direct `% q_i`, no i128
/// centering) and recombines with [`RnsBasis::recombine_poly_zq`].
pub fn mul_poly_ntt_rns(
    a: &[Zq],
    b: &[Zq],
    basis: &RnsBasis,
    plans: &[NegacyclicPlan],
) -> CryptoResult<Vec<Zq>> {
    let n = basis.n;
    if a.len() != n || b.len() != n {
        return Err("poly degree mismatch".into());
    }
    let a_limbs = basis.decompose_poly_zq(a)?;
    let b_limbs = basis.decompose_poly_zq(b)?;
    let k = basis.limb_count();
    let mut prod_limbs = vec![vec![0u64; n]; k];
    for limb in 0..k {
        let mut fa = a_limbs[limb].clone();
        let mut fb = b_limbs[limb].clone();
        forward_ntt_negacyclic(&mut fa, &plans[limb])
            .map_err(|_| "forward NTT failed".to_string())?;
        forward_ntt_negacyclic(&mut fb, &plans[limb])
            .map_err(|_| "forward NTT failed".to_string())?;
        let mut fc: Vec<u64> = fa
            .iter()
            .zip(fb.iter())
            .map(|(&x, &y)| mul_mod(x, y, plans[limb].q))
            .collect();
        inverse_ntt_negacyclic(&mut fc, &plans[limb])
            .map_err(|_| "inverse NTT failed".to_string())?;
        prod_limbs[limb] = fc;
    }
    basis.recombine_poly_zq(&prod_limbs)
}

fn limb_plans(basis: &RnsBasis) -> CryptoResult<Vec<NegacyclicPlan>> {
    basis
        .primes
        .iter()
        .map(|&p| {
            NegacyclicPlan::new(basis.n, p).ok_or_else(|| format!("failed limb plan q={p}"))
        })
        .collect()
}

/// Ternary secret key.
#[derive(Debug, Clone)]
pub struct SecretKey {
    pub s: Vec<i64>,
}

/// Public key \(pk = (b, a)\) with \(b = [-as + e]_Q\).
#[derive(Debug, Clone)]
pub struct PublicKey {
    pub b: Vec<Zq>,
    pub a: Vec<Zq>,
    pub q: Zq,
}

/// RLWE ciphertext \((c_0, c_1)\).
#[derive(Debug, Clone)]
pub struct Ciphertext {
    pub c0: Vec<Zq>,
    pub c1: Vec<Zq>,
}

impl Ciphertext {
    pub fn degree(&self) -> usize {
        self.c0.len()
    }
}

/// Galois evaluation key for automorphism \(X \mapsto X^k\):
/// digit-decomposed encryptions of \(B^j \cdot \varphi_k(s)\).
#[derive(Debug, Clone)]
pub struct GaloisKey {
    pub k: usize,
    /// `evk[digit] = (b, a)` encrypting `B^digit * φ_k(s)` under `s`.
    pub digits: Vec<(Vec<Zq>, Vec<Zq>)>,
    pub q: Zq,
    /// Content-derived identity of this key's evaluation material.
    pub fingerprint: u64,
}

/// FNV-1a over the evaluation-key material, salted with `k` and `q`.
fn evk_fingerprint(k: usize, q: Zq, digits: &[(Vec<Zq>, Vec<Zq>)]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    let mut eat = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(PRIME);
    };
    eat(k as u64);
    let q_bytes = q.to_le_bytes();
    for chunk in q_bytes.chunks_exact(8) {
        eat(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    eat(digits.len() as u64);
    for (b, a) in digits {
        for poly in [b, a] {
            eat(poly.len() as u64);
            for &c in poly.iter() {
                let cb = c.to_le_bytes();
                for chunk in cb.chunks_exact(8) {
                    eat(u64::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
        }
    }
    h
}

/// Relinearization key: digit-decomposed encryptions of \(B^j \cdot s^2\).
#[derive(Debug, Clone)]
pub struct RelinKey {
    pub inner: GaloisKey,
}

/// Synthetic `k` used to cache the relinearization EVK without colliding with
/// any rotate-and-sum Galois step (`k = 5^{2^i} mod 2N` is always odd).
pub const RELIN_K: usize = 0;

/// Full key material for encrypted PRS evaluation.
#[derive(Debug, Clone)]
pub struct EvaluationKeys {
    pub sk: SecretKey,
    pub pk: PublicKey,
    pub galois: Vec<GaloisKey>,
    /// Relinearization key for ciphertext × ciphertext multiplication.
    pub relin: RelinKey,
    pub basis: RnsBasis,
    /// Auxiliary special modulus \(P\) for hybrid KS / mod-down.
    pub aux: AuxiliaryModulus,
}

/// Wall-clock split matching HEPRS `-print` stages (encrypt / eval / decrypt).
#[derive(Debug, Clone, Copy, Default)]
pub struct EncryptedStageMs {
    pub encrypt_ms: f64,
    pub eval_ms: f64,
    pub decrypt_ms: f64,
}

impl EncryptedStageMs {
    pub fn total_ms(self) -> f64 {
        self.encrypt_ms + self.eval_ms + self.decrypt_ms
    }
}

/// `B^d mod Q` without overflowing intermediate `u128` for large digit indices.
fn digit_scale(d: usize, q: Zq) -> Zq {
    let base = zq_from_u128(KS_DIGIT_BASE);
    let mut s = Zq::ONE;
    for _ in 0..d {
        s = mul_mod_zq(s, base, q);
    }
    s
}

/// `a * s` in `Z_Q[X]/(X^N+1)` with `s` ternary.
fn poly_mul_as(a: &[Zq], s: &[i64], q: Zq) -> Vec<Zq> {
    if let Some((basis, plans)) = lookup_basis(a.len(), q) {
        let s_u: Vec<Zq> = s.iter().map(|&c| center_to_zq(c, q)).collect();
        if let Ok(out) = mul_poly_ntt_rns(a, &s_u, &basis, &plans) {
            return out;
        }
    }
    poly_mul_as_schoolbook(a, s, q)
}

/// Reference O(N²) expansion. Retained as the correctness oracle and as the
/// fallback for unregistered bases.
fn poly_mul_as_schoolbook(a: &[Zq], s: &[i64], q: Zq) -> Vec<Zq> {
    let n = a.len();
    let mut out = vec![Zq::ZERO; n];
    for (j, &sj) in s.iter().enumerate() {
        if sj == 0 {
            continue;
        }
        for i in 0..n {
            let mut k = i + j;
            let mut term = a[i];
            if k >= n {
                k -= n;
                term = neg_mod_zq(term, q);
            }
            if sj > 0 {
                out[k] = add_mod_zq(out[k], term, q);
            } else {
                out[k] = add_mod_zq(out[k], neg_mod_zq(term, q), q);
            }
        }
    }
    out
}

/// KeyGen: sample \(s\), \(a\), \(e\); set \(b = [-as+e]_Q\).
pub fn keygen(basis: &RnsBasis, eta: usize) -> CryptoResult<(SecretKey, PublicKey)> {
    register_basis(basis)?;
    let n = basis.n;
    let q = basis.modulus;
    let s = sample_ternary_poly(n)?;
    let a = sample_uniform_poly(n, q)?;
    let e = sample_cbd_poly(n, eta)?;
    let as_poly = poly_mul_as(&a, &s, q);
    let e_u: Vec<Zq> = e.iter().map(|&c| center_to_zq(c, q)).collect();
    let neg_as = neg_poly_mod(&as_poly, q);
    let b = add_poly_mod(&neg_as, &e_u, q);
    Ok((SecretKey { s }, PublicKey { b, a, q }))
}

/// Encrypt a centered message polynomial under `pk`.
pub fn encrypt(pk: &PublicKey, message: &[i128], eta: usize) -> CryptoResult<Ciphertext> {
    let n = pk.a.len();
    if message.len() != n {
        return Err("message degree mismatch".into());
    }
    let q = pk.q;
    let v = sample_ternary_poly(n)?;
    let e0 = sample_cbd_poly(n, eta)?;
    let e1 = sample_cbd_poly(n, eta)?;
    let vb = poly_mul_as(&pk.b, &v, q);
    let va = poly_mul_as(&pk.a, &v, q);
    let m_u: Vec<Zq> = message.iter().map(|&c| i128_to_zq(c, q)).collect();
    let e0_u: Vec<Zq> = e0.iter().map(|&c| center_to_zq(c, q)).collect();
    let e1_u: Vec<Zq> = e1.iter().map(|&c| center_to_zq(c, q)).collect();
    let c0 = add_poly_mod(&add_poly_mod(&vb, &m_u, q), &e0_u, q);
    let c1 = add_poly_mod(&va, &e1_u, q);
    Ok(Ciphertext { c0, c1 })
}

/// Decrypt: \(m \approx c_0 + c_1 \cdot s \pmod Q\).
pub fn decrypt(sk: &SecretKey, ct: &Ciphertext, q: Zq) -> CryptoResult<Vec<i128>> {
    if ct.c0.len() != sk.s.len() || ct.c1.len() != sk.s.len() {
        return Err("ciphertext/secret degree mismatch".into());
    }
    let c1s = poly_mul_as(&ct.c1, &sk.s, q);
    let phase = add_poly_mod(&ct.c0, &c1s, q);
    phase.iter().map(|&u| zq_to_center(u, q)).collect()
}

/// Ciphertext–plaintext multiply: \((c_0, c_1) \cdot m = (c_0 m, c_1 m)\).
pub fn mul_ct_pt(ct: &Ciphertext, pt: &[Zq], basis: &RnsBasis) -> CryptoResult<Ciphertext> {
    let plans = limb_plans(basis)?;
    let c0 = mul_poly_ntt_rns(&ct.c0, pt, basis, &plans)?;
    let c1 = mul_poly_ntt_rns(&ct.c1, pt, basis, &plans)?;
    Ok(Ciphertext { c0, c1 })
}

/// Generate a relinearization key encrypting digit-scaled \(s^2\) under \(s\).
pub fn gen_relin_key(sk: &SecretKey, basis: &RnsBasis, eta: usize) -> CryptoResult<RelinKey> {
    register_basis(basis)?;
    let n = basis.n;
    let q = basis.modulus;
    let n_digits = digit_count(q);
    let s_u: Vec<Zq> = sk.s.iter().map(|&c| center_to_zq(c, q)).collect();
    let plans = limb_plans(basis)?;
    let s2 = mul_poly_ntt_rns(&s_u, &s_u, basis, &plans)?;
    let mut digits = Vec::with_capacity(n_digits);
    for d in 0..n_digits {
        let scale = digit_scale(d, q);
        let m_u: Vec<Zq> = s2
            .iter()
            .map(|&u| mul_mod_zq(u % q, scale, q))
            .collect();
        let a = sample_uniform_poly(n, q)?;
        let e = sample_cbd_poly(n, eta)?;
        let as_poly = poly_mul_as(&a, &sk.s, q);
        let e_u: Vec<Zq> = e.iter().map(|&c| center_to_zq(c, q)).collect();
        let b = add_poly_mod(&neg_poly_mod(&as_poly, q), &add_poly_mod(&m_u, &e_u, q), q);
        digits.push((b, a));
    }
    let fingerprint = evk_fingerprint(RELIN_K, q, &digits);
    Ok(RelinKey {
        inner: GaloisKey {
            k: RELIN_K,
            digits,
            q,
            fingerprint,
        },
    })
}

/// Ciphertext × ciphertext with relinearization (HEPRS `MulRelinNew`).
pub fn mul_ct_ct(
    a: &Ciphertext,
    b: &Ciphertext,
    relin: &RelinKey,
    basis: &RnsBasis,
    prefer_metal: bool,
) -> CryptoResult<Ciphertext> {
    if a.degree() != b.degree() || a.degree() != basis.n {
        return Err("ciphertext degree mismatch".into());
    }
    let plans = limb_plans(basis)?;
    let d00 = mul_poly_ntt_rns(&a.c0, &b.c0, basis, &plans)?;
    let d01 = mul_poly_ntt_rns(&a.c0, &b.c1, basis, &plans)?;
    let d10 = mul_poly_ntt_rns(&a.c1, &b.c0, basis, &plans)?;
    let d11 = mul_poly_ntt_rns(&a.c1, &b.c1, basis, &plans)?;
    let q = basis.modulus;
    let d0 = d00;
    let d1 = add_poly_mod(&d01, &d10, q);
    let d2 = d11;
    let switched = key_switch_with_backend(
        &Ciphertext { c0: d0, c1: d2 },
        &relin.inner,
        basis,
        prefer_metal,
    )?;
    Ok(Ciphertext {
        c0: switched.c0,
        c1: add_poly_mod(&switched.c1, &d1, q),
    })
}

/// Apply ring automorphism \(\varphi_k\) coefficient-wise to a ciphertext.
pub fn automorphism_ct(ct: &Ciphertext, k: usize, q: Zq) -> Ciphertext {
    Ciphertext {
        c0: automorphism_zq(&ct.c0, k, q),
        c1: automorphism_zq(&ct.c1, k, q),
    }
}

pub fn digit_count(q: Zq) -> usize {
    if q <= Zq::ONE {
        return 1;
    }
    let bits = zq_ilog2(q) + 1;
    ((bits + KS_DIGIT_BITS - 1) / KS_DIGIT_BITS) as usize
}

fn decompose_digits(coeff: Zq, q: Zq, n_digits: usize) -> Vec<Zq> {
    let mut x = coeff % q;
    let base = zq_from_u128(KS_DIGIT_BASE);
    let mut digits = Vec::with_capacity(n_digits);
    for _ in 0..n_digits {
        digits.push(x % base);
        x /= base;
    }
    digits
}

/// Decompose every coefficient of `c1` into base-`B` digit polynomials.
pub fn decompose_c1_digits(c1: &[Zq], q: Zq, n_digits: usize) -> Vec<Vec<Zq>> {
    let n = c1.len();
    let mut digit_polys = vec![vec![Zq::ZERO; n]; n_digits];
    for (i, &c) in c1.iter().enumerate() {
        let digs = decompose_digits(c, q, n_digits);
        for d in 0..n_digits {
            digit_polys[d][i] = digs[d];
        }
    }
    digit_polys
}

/// Generate a Galois evaluation key for automorphism \(k\).
pub fn gen_galois_key(
    sk: &SecretKey,
    k: usize,
    basis: &RnsBasis,
    eta: usize,
) -> CryptoResult<GaloisKey> {
    register_basis(basis)?;
    let n = basis.n;
    let q = basis.modulus;
    let n_digits = digit_count(q);
    let s_u: Vec<Zq> = sk.s.iter().map(|&c| center_to_zq(c, q)).collect();
    let s_rot = automorphism_zq(&s_u, k, q);
    let mut digits = Vec::with_capacity(n_digits);
    for d in 0..n_digits {
        let scale = digit_scale(d, q);
        // Multiply in Z_Q — never form centered*B^d in i128 (overflows for
        // large digit indices at |Q|≈202).
        let m_u: Vec<Zq> = s_rot
            .iter()
            .map(|&u| mul_mod_zq(u % q, scale, q))
            .collect();
        let a = sample_uniform_poly(n, q)?;
        let e = sample_cbd_poly(n, eta)?;
        let as_poly = poly_mul_as(&a, &sk.s, q);
        let e_u: Vec<Zq> = e.iter().map(|&c| center_to_zq(c, q)).collect();
        let b = add_poly_mod(&neg_poly_mod(&as_poly, q), &add_poly_mod(&m_u, &e_u, q), q);
        digits.push((b, a));
    }
    let fingerprint = evk_fingerprint(k, q, &digits);
    Ok(GaloisKey {
        k,
        digits,
        q,
        fingerprint,
    })
}

/// Key-switch a ciphertext from \(\varphi_k(s)\) back to \(s\).
pub fn key_switch(ct: &Ciphertext, gk: &GaloisKey, basis: &RnsBasis) -> CryptoResult<Ciphertext> {
    key_switch_with_backend(ct, gk, basis, false)
}

/// Key-switch with optional Metal acceleration for limb NTT + KS MAC.
pub fn key_switch_with_backend(
    ct: &Ciphertext,
    gk: &GaloisKey,
    basis: &RnsBasis,
    prefer_metal: bool,
) -> CryptoResult<Ciphertext> {
    if prefer_metal {
        return crate::metal_ks::key_switch_accelerated(ct, gk, basis, true);
    }
    let n = ct.degree();
    let q = gk.q;
    let n_digits = gk.digits.len();
    let plans = limb_plans(basis)?;

    let mut acc0 = ct.c0.clone();
    let mut acc1 = vec![Zq::ZERO; n];
    let digit_polys = decompose_c1_digits(&ct.c1, q, n_digits);

    for d in 0..n_digits {
        let (ref b, ref a) = gk.digits[d];
        let t0 = mul_poly_ntt_rns(&digit_polys[d], b, basis, &plans)?;
        let t1 = mul_poly_ntt_rns(&digit_polys[d], a, basis, &plans)?;
        acc0 = add_poly_mod(&acc0, &t0, q);
        acc1 = add_poly_mod(&acc1, &t1, q);
    }
    Ok(Ciphertext {
        c0: acc0,
        c1: acc1,
    })
}

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

/// Encrypted rotate-and-sum with Galois key-switching after each automorphism.
pub fn rotate_and_sum_encrypted(
    ct: &Ciphertext,
    galois_keys: &[GaloisKey],
    basis: &RnsBasis,
) -> CryptoResult<Ciphertext> {
    rotate_and_sum_encrypted_backend(ct, galois_keys, basis, false)
}

/// Encrypted rotate-and-sum; `prefer_metal` routes KS through the Metal pipeline.
pub fn rotate_and_sum_encrypted_backend(
    ct: &Ciphertext,
    galois_keys: &[GaloisKey],
    basis: &RnsBasis,
    prefer_metal: bool,
) -> CryptoResult<Ciphertext> {
    let n = ct.degree();
    let slots = n / 2;
    let q = basis.modulus;
    let mut acc = ct.clone();
    let mut step = 1usize;
    while step < slots {
        let k = mod_pow_usize(5, step, 2 * n);
        let rotated = automorphism_ct(&acc, k, q);
        let gk = galois_keys
            .iter()
            .find(|g| g.k == k)
            .ok_or_else(|| format!("missing Galois key for k={k}"))?;
        let switched = key_switch_with_backend(&rotated, gk, basis, prefer_metal)?;
        acc = Ciphertext {
            c0: add_poly_mod(&acc.c0, &switched.c0, q),
            c1: add_poly_mod(&acc.c1, &switched.c1, q),
        };
        step <<= 1;
    }
    Ok(acc)
}

/// Build sk, pk, and Galois keys for every rotate-and-sum step.
pub fn setup_evaluation_keys(basis: &RnsBasis, eta: usize) -> CryptoResult<EvaluationKeys> {
    let (sk, pk) = keygen(basis, eta)?;
    let n = basis.n;
    let slots = n / 2;
    let aux = AuxiliaryModulus::generate(n, DEFAULT_AUX_LIMBS, &basis.primes)
        .map_err(|e| format!("aux modulus P: {e}"))?;
    let mut galois = Vec::new();
    let mut step = 1usize;
    while step < slots {
        let k = mod_pow_usize(5, step, 2 * n);
        galois.push(gen_galois_key(&sk, k, basis, eta)?);
        step <<= 1;
    }
    let relin = gen_relin_key(&sk, basis, eta)?;
    Ok(EvaluationKeys {
        sk,
        pk,
        galois,
        relin,
        basis: basis.clone(),
        aux,
    })
}

/// Encrypt an already-encoded (centered i64) message polynomial.
pub fn encrypt_encoded(pk: &PublicKey, coeffs: &[i64], eta: usize) -> CryptoResult<Ciphertext> {
    let msg: Vec<i128> = coeffs.iter().map(|&c| c as i128).collect();
    encrypt(pk, &msg, eta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckks_encode::{decode_real_slots_i128, encode_real_slots};

    #[test]
    fn ternary_and_cbd_samplers_are_bounded() {
        for _ in 0..100 {
            let t = sample_ternary_coeff().unwrap();
            assert!(t >= -1 && t <= 1);
            let e = sample_cbd_coeff(8).unwrap();
            assert!(e >= -8 && e <= 8);
        }
        let u = sample_uniform_coeff(Zq::from(97u64)).unwrap();
        assert!(u < Zq::from(97u64));
    }

    #[test]
    fn encrypt_decrypt_roundtrip_small() {
        let n = 64usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).unwrap();
        let scale = (20u32 as f64).exp2();
        let slots: Vec<f64> = (0..n / 2).map(|i| 0.01 * (i as f64 - 10.0)).collect();
        let coeffs = encode_real_slots(&slots, n, scale).unwrap();
        let ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).unwrap();
        let phase = decrypt(&sk, &ct, basis.modulus).unwrap();
        let back = decode_real_slots_i128(&phase, scale).unwrap();
        for i in 0..slots.len() {
            assert!(
                (slots[i] - back[i]).abs() < 0.05,
                "slot {i}: {} vs {}",
                slots[i],
                back[i]
            );
        }
    }

    #[test]
    fn ct_ct_mul_scales_with_degree() {
        for &n in &[64usize, 256, 1024, 4096, 8192] {
            let basis = RnsBasis::generate(n, 4).unwrap();
            let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA).unwrap();
            let scale = (40u32 as f64).exp2();
            let a: Vec<f64> = (0..n / 2).map(|i| 0.01 * ((i % 5) as f64 - 2.0)).collect();
            let b: Vec<f64> = (0..n / 2).map(|i| 0.02 * ((i % 7) as f64 - 3.0)).collect();
            let ca = encode_real_slots(&a, n, scale).unwrap();
            let cb = encode_real_slots(&b, n, scale).unwrap();
            let cta = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA).unwrap();
            let ctb = encrypt_encoded(&keys.pk, &cb, DEFAULT_NOISE_ETA).unwrap();
            let prod = mul_ct_ct(&cta, &ctb, &keys.relin, &basis, false).unwrap();
            let phase = decrypt(&keys.sk, &prod, basis.modulus).unwrap();
            let got = decode_real_slots_i128(&phase, scale * scale).unwrap();
            let err = a
                .iter()
                .zip(b.iter())
                .zip(got.iter())
                .map(|((x, y), g)| (x * y - g).abs())
                .fold(0.0f64, f64::max);
            eprintln!("[ct_ct] N={n} max abs err={err:.3e}");
            assert!(
                err < 1e-3,
                "N={n}: ct×ct err {err:.3e} exceeds budget — relin/noise broken at this degree"
            );
        }
    }

    /// Production path: ct×ct products + accumulate + rotate-and-sum at N=8192.
    #[test]
    fn ct_ct_prs_fold_at_secure_degree() {
        let n = 8192usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA).unwrap();
        let scale = (40u32 as f64).exp2();
        let slots = n / 2;
        let mut want = 0.0f64;
        let mut acc: Option<Ciphertext> = None;
        for c in 0..2usize {
            let a: Vec<f64> = (0..slots).map(|i| ((i + c) % 3) as f64).collect();
            let b: Vec<f64> = (0..slots)
                .map(|i| 1e-4 * (((i * 17 + c) % 11) as f64 - 5.0))
                .collect();
            want += a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
            let ca = encode_real_slots(&a, n, scale).unwrap();
            let cb = encode_real_slots(&b, n, scale).unwrap();
            let cta = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA).unwrap();
            let ctb = encrypt_encoded(&keys.pk, &cb, DEFAULT_NOISE_ETA).unwrap();
            let prod = mul_ct_ct(&cta, &ctb, &keys.relin, &basis, false).unwrap();
            acc = Some(match acc {
                None => prod,
                Some(prev) => add_ct(&prev, &prod, basis.modulus).unwrap(),
            });
        }
        let summed = rotate_and_sum_encrypted(&acc.unwrap(), &keys.galois, &basis).unwrap();
        let phase = decrypt(&keys.sk, &summed, basis.modulus).unwrap();
        let got = decode_real_slots_i128(&phase, scale * scale).unwrap()[0];
        let err = (got - want).abs();
        eprintln!("[ct_ct_prs] N={n} got={got:.6} want={want:.6} err={err:.3e}");
        assert!(
            err < 1e-2,
            "PRS fold err {err:.3e} — ct×ct+rotate path broken at N=8192"
        );
    }

    #[test]
    fn ct_ct_mul_matches_plaintext_product() {
        let n = 64usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA).unwrap();
        let scale = (40u32 as f64).exp2();
        let a: Vec<f64> = (0..n / 2).map(|i| 0.01 * ((i % 5) as f64 - 2.0)).collect();
        let b: Vec<f64> = (0..n / 2).map(|i| 0.02 * ((i % 7) as f64 - 3.0)).collect();
        let ca = encode_real_slots(&a, n, scale).unwrap();
        let cb = encode_real_slots(&b, n, scale).unwrap();
        let cta = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA).unwrap();
        let ctb = encrypt_encoded(&keys.pk, &cb, DEFAULT_NOISE_ETA).unwrap();
        let prod = mul_ct_ct(&cta, &ctb, &keys.relin, &basis, false).unwrap();
        let phase = decrypt(&keys.sk, &prod, basis.modulus).unwrap();
        let got = decode_real_slots_i128(&phase, scale * scale).unwrap();
        for i in 0..a.len() {
            let want = a[i] * b[i];
            assert!(
                (got[i] - want).abs() < 1e-4,
                "slot {i}: got {} want {want}",
                got[i]
            );
        }
    }

    #[test]
    fn ct_pt_mul_matches_plaintext_product() {
        let n = 64usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA).unwrap();
        let scale = (20u32 as f64).exp2();
        let a: Vec<f64> = (0..n / 2).map(|i| 0.05 * ((i % 7) as f64)).collect();
        let b: Vec<f64> = (0..n / 2).map(|i| 0.02 * (i as f64 - 5.0)).collect();
        let ca = encode_real_slots(&a, n, scale).unwrap();
        let cb = encode_real_slots(&b, n, scale).unwrap();
        let ct = encrypt_encoded(&keys.pk, &ca, DEFAULT_NOISE_ETA).unwrap();
        let pt: Vec<Zq> = cb
            .iter()
            .map(|&c| center_to_zq(c, basis.modulus))
            .collect();
        let prod = mul_ct_pt(&ct, &pt, &basis).unwrap();
        let phase = decrypt(&keys.sk, &prod, basis.modulus).unwrap();
        let decoded = decode_real_slots_i128(&phase, scale * scale).unwrap();
        for i in 0..n / 2 {
            let want = a[i] * b[i];
            assert!(
                (decoded[i] - want).abs() < 0.1,
                "slot {i}: got {} want {want}",
                decoded[i]
            );
        }
    }

    /// The transform path must equal the schoolbook oracle *exactly* mod Q.
    #[test]
    fn ring_multiply_ntt_matches_schoolbook() {
        for &n in &[64usize, 256, 1024] {
            let basis = RnsBasis::generate(n, 4).unwrap();
            let q = basis.modulus;
            let a = sample_uniform_poly(n, q).unwrap();
            let s = sample_ternary_poly(n).unwrap();

            let want = poly_mul_as_schoolbook(&a, &s, q);
            register_basis(&basis).unwrap();
            let got = poly_mul_as(&a, &s, q);

            assert_eq!(
                got.len(),
                want.len(),
                "N={n}: transform path returned wrong degree"
            );
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g, w,
                    "N={n}: transform ring multiply differs from schoolbook at coefficient {i}"
                );
            }
        }
    }

    /// The registry must not silently mis-serve a basis for a different modulus.
    #[test]
    fn ring_multiply_falls_back_when_basis_unregistered() {
        let n = 128usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let q = basis.modulus;
        let a = sample_uniform_poly(n, q).unwrap();
        let s = sample_ternary_poly(n).unwrap();
        let bogus_q = q - Zq::from(2u64);
        let want = poly_mul_as_schoolbook(&a, &s, bogus_q);
        let got = poly_mul_as(&a, &s, bogus_q);
        assert_eq!(got, want, "unregistered modulus did not fall back correctly");
    }

    #[test]
    fn galois_keyswitch_preserves_message() {
        let n = 64usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let (sk, pk) = keygen(&basis, DEFAULT_NOISE_ETA).unwrap();
        let k = 5usize;
        let gk = gen_galois_key(&sk, k, &basis, DEFAULT_NOISE_ETA).unwrap();
        let scale = (40u32 as f64).exp2();
        let slots: Vec<f64> = (0..n / 2).map(|i| 0.1 * (i as f64)).collect();
        let coeffs = encode_real_slots(&slots, n, scale).unwrap();
        let ct = encrypt_encoded(&pk, &coeffs, DEFAULT_NOISE_ETA).unwrap();
        let rotated = automorphism_ct(&ct, k, basis.modulus);
        let switched = key_switch(&rotated, &gk, &basis).unwrap();
        let phase = decrypt(&sk, &switched, basis.modulus).unwrap();
        let back = decode_real_slots_i128(&phase, scale).unwrap();
        let s_u: Vec<Zq> = coeffs
            .iter()
            .map(|&c| center_to_zq(c, basis.modulus))
            .collect();
        let rot_pt = automorphism_zq(&s_u, k, basis.modulus);
        let rot_c: Vec<i128> = rot_pt
            .iter()
            .map(|&u| zq_to_center(u, basis.modulus).unwrap())
            .collect();
        let expect = decode_real_slots_i128(&rot_c, scale).unwrap();
        for i in 0..slots.len() {
            assert!(
                (back[i] - expect[i]).abs() < 0.05,
                "slot {i}: got {} want {}",
                back[i],
                expect[i]
            );
        }
    }

    #[test]
    fn encrypted_rotate_and_sum_matches_slot_sum() {
        let n = 64usize;
        let basis = RnsBasis::generate(n, 4).unwrap();
        let keys = setup_evaluation_keys(&basis, DEFAULT_NOISE_ETA).unwrap();
        let scale = (44u32 as f64).exp2();
        let slots: Vec<f64> = (0..n / 2).map(|i| 0.05 * (i as f64) + 0.1).collect();
        let expected: f64 = slots.iter().sum();
        let coeffs = encode_real_slots(&slots, n, scale).unwrap();
        let ct = encrypt_encoded(&keys.pk, &coeffs, DEFAULT_NOISE_ETA).unwrap();
        let summed = rotate_and_sum_encrypted(&ct, &keys.galois, &basis).unwrap();
        let phase = decrypt(&keys.sk, &summed, basis.modulus).unwrap();
        let back = decode_real_slots_i128(&phase, scale).unwrap();
        for (i, &v) in back.iter().enumerate() {
            assert!(
                (v - expected).abs() < 0.02,
                "slot {i}: got {v}, want ≈ {expected}"
            );
        }
    }
}
