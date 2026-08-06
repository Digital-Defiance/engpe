//! Negacyclic NTT over `Z_q[X]/(X^N + 1)`, vendored from fhe-evolve and
//! extended for `q < 2^63` via `u128` modular products.
//!
//! Original fhe-evolve path required `q < 2^32` (Montgomery/NEON). CKKS PRS
//! pointwise products at even modest scales need a wider single prime, so this
//! copy lifts that ceiling while keeping the same psi/omega algebra and the
//! negacyclic convolution gate.

/// `(a * b) % q` with `u128` intermediate. Safe for `a, b, q < 2^63`.
#[inline(always)]
pub fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        exp >>= 1;
        base = mul_mod(base, base, modulus);
    }
    result
}

pub fn mod_inv(base: u64, p: u64) -> u64 {
    mod_pow(base, p - 2, p)
}

pub fn find_primitive_2n_root(n: u64, q: u64) -> Option<u64> {
    if n < 2 || q < 3 {
        return None;
    }
    let two_n = n.checked_mul(2)?;
    if (q - 1) % two_n != 0 {
        return None;
    }
    let exponent = (q - 1) / two_n;
    // A random/small generator almost always works when 2N | (q-1) and q is prime.
    for g in 2u64..4096 {
        let psi = mod_pow(g, exponent, q);
        if psi <= 1 {
            continue;
        }
        if mod_pow(psi, two_n, q) == 1 && mod_pow(psi, n, q) == q - 1 {
            return Some(psi);
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct NegacyclicPlan {
    pub n: usize,
    pub q: u64,
    #[allow(dead_code)]
    pub psi: u64,
    pub omega: u64,
    pub omega_inv: u64,
    pub n_inv: u64,
    pub psi_powers: Vec<u64>,
    pub psi_inv_powers: Vec<u64>,
    /// Flat `omega^i` for Metal stage-twiddle construction.
    pub omega_powers: Vec<u64>,
    /// Flat `omega^(-i)` for Metal inverse.
    pub omega_inv_powers: Vec<u64>,
}

impl NegacyclicPlan {
    pub fn new(n: usize, q: u64) -> Option<Self> {
        if n < 2 || (n & (n - 1)) != 0 {
            return None;
        }
        // Wide path: allow up to 2^63 - 1 so CKKS products fit a single prime.
        if q < 3 || q >= (1u64 << 63) {
            return None;
        }

        let n64 = n as u64;
        let psi = find_primitive_2n_root(n64, q)?;
        if mod_pow(psi, 2 * n64, q) != 1 || mod_pow(psi, n64, q) != q - 1 {
            return None;
        }

        let omega = mul_mod(psi, psi, q);
        if mod_pow(omega, n64, q) != 1 || mod_pow(omega, n64 / 2, q) == 1 {
            return None;
        }

        let psi_inv = mod_inv(psi, q);
        if mul_mod(psi, psi_inv, q) != 1 {
            return None;
        }
        let omega_inv = mod_inv(omega, q);
        let n_inv = mod_inv(n64 % q, q);

        let mut psi_powers = Vec::with_capacity(n);
        let mut psi_inv_powers = Vec::with_capacity(n);
        psi_powers.push(1);
        psi_inv_powers.push(1);
        for i in 1..n {
            psi_powers.push(mul_mod(psi_powers[i - 1], psi, q));
            psi_inv_powers.push(mul_mod(psi_inv_powers[i - 1], psi_inv, q));
        }

        let mut omega_powers = Vec::with_capacity(n);
        let mut omega_inv_powers = Vec::with_capacity(n);
        omega_powers.push(1);
        omega_inv_powers.push(1);
        for i in 1..n {
            omega_powers.push(mul_mod(omega_powers[i - 1], omega, q));
            omega_inv_powers.push(mul_mod(omega_inv_powers[i - 1], omega_inv, q));
        }

        Some(Self {
            n,
            q,
            psi,
            omega,
            omega_inv,
            n_inv,
            psi_powers,
            psi_inv_powers,
            omega_powers,
            omega_inv_powers,
        })
    }
}

fn bit_reverse(data: &mut [u64]) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

pub fn forward_ntt(data: &mut [u64], omega: u64, q: u64) -> Result<(), ()> {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(());
    }
    bit_reverse(data);
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let w_step = mod_pow(omega, (n / len) as u64, q);
        let mut i = 0;
        while i < n {
            let mut w: u64 = 1;
            for j in 0..half {
                let u = data[i + j];
                let v = mul_mod(data[i + j + half], w, q);
                data[i + j] = (u + v) % q;
                data[i + j + half] = (u + q - v) % q;
                w = mul_mod(w, w_step, q);
            }
            i += len;
        }
        len <<= 1;
    }
    Ok(())
}

pub fn forward_ntt_negacyclic(data: &mut [u64], plan: &NegacyclicPlan) -> Result<(), ()> {
    if data.len() != plan.n {
        return Err(());
    }
    let q = plan.q;
    for i in 0..plan.n {
        data[i] = mul_mod(data[i], plan.psi_powers[i], q);
    }
    forward_ntt(data, plan.omega, q)
}

pub fn inverse_ntt_negacyclic(data: &mut [u64], plan: &NegacyclicPlan) -> Result<(), ()> {
    if data.len() != plan.n {
        return Err(());
    }
    let q = plan.q;
    forward_ntt(data, plan.omega_inv, q)?;
    for i in 0..plan.n {
        let scaled = mul_mod(data[i], plan.n_inv, q);
        data[i] = mul_mod(scaled, plan.psi_inv_powers[i], q);
    }
    Ok(())
}

fn is_probable_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    // Deterministic Miller–Rabin bases for n < 2^64.
    let witnesses = [2u64, 3, 5, 7, 11, 13, 23];
    let mut d = n - 1;
    let mut s = 0u32;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'witness: for &a in &witnesses {
        if a % n == 0 {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Smallest NTT-friendly prime `q ≡ 1 (mod 2N)` with at least `min_bits` bits.
pub fn find_ntt_modulus(n: usize, min_bits: u32) -> Option<u64> {
    let two_n = (2 * n) as u64;
    let mut q = 1u64 << min_bits.min(62);
    q = q - (q % two_n) + 1;
    if q <= two_n {
        q += two_n;
    }
    while q < (1u64 << 63) {
        if is_probable_prime(q) {
            if NegacyclicPlan::new(n, q).is_some() {
                return Some(q);
            }
        }
        q = q.checked_add(two_n)?;
    }
    None
}

/// Largest NTT-friendly prime strictly below `max_q` (for Metal: `1<<32`).
pub fn find_ntt_modulus_below(n: usize, max_q: u64) -> Option<u64> {
    let two_n = (2 * n) as u64;
    if max_q <= two_n + 1 {
        return None;
    }
    let mut q = max_q - 1;
    q = q - (q % two_n) + 1;
    if q >= max_q {
        q = q.saturating_sub(two_n);
    }
    while q > two_n {
        if is_probable_prime(q) {
            if NegacyclicPlan::new(n, q).is_some() {
                return Some(q);
            }
        }
        if q <= two_n {
            break;
        }
        q -= two_n;
    }
    None
}

pub fn verify_negacyclic_convolution(plan: &NegacyclicPlan) -> bool {
    let n = plan.n;
    let q = plan.q;
    // Sparse operands: only indices 0 and N-1 are nonzero so the schoolbook
    // reference is O(1) rather than O(N²). Index N-1 forces the X^N = -1 wrap.
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    a[0] = 1;
    a[n - 1] = 2 % q;
    b[0] = 3 % q;
    b[n - 1] = 5 % q;

    let nz_a: Vec<usize> = a
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v != 0).then_some(i))
        .collect();
    let nz_b: Vec<usize> = b
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v != 0).then_some(i))
        .collect();

    let mut expected = vec![0u64; n];
    for &i in &nz_a {
        for &j in &nz_b {
            let mut prod = mul_mod(a[i], b[j], q);
            let mut k = i + j;
            if k >= n {
                k -= n;
                prod = if prod == 0 { 0 } else { q - prod };
            }
            expected[k] = (expected[k] + prod) % q;
        }
    }

    let mut fa = a;
    let mut fb = b;
    if forward_ntt_negacyclic(&mut fa, plan).is_err() {
        return false;
    }
    if forward_ntt_negacyclic(&mut fb, plan).is_err() {
        return false;
    }
    let mut fc: Vec<u64> = (0..n).map(|i| mul_mod(fa[i], fb[i], q)).collect();
    if inverse_ntt_negacyclic(&mut fc, plan).is_err() {
        return false;
    }
    fc == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_and_convolution_small() {
        let plan = NegacyclicPlan::new(16, 97).unwrap();
        assert!(verify_negacyclic_convolution(&plan));
    }

    #[test]
    fn finds_wide_modulus_for_1024() {
        let q = find_ntt_modulus(1024, 40).unwrap();
        assert!(q >= (1u64 << 40));
        let plan = NegacyclicPlan::new(1024, q).unwrap();
        assert!(verify_negacyclic_convolution(&plan));
    }
}
