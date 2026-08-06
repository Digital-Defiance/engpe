//! Wide residue arithmetic for RNS products that exceed `u128`.
//!
//! HEPRS `PN13QP202pq` targets \(\log_2(QP)\approx 202\). Metal NTT limbs are
//! `< 2^{31}`, so we match that budget with seven ~29-bit primes
//! (\(|Q|\approx 200\)–\(210\)) and store coefficients in `ethnum::U256`.

use ethnum::U256;

/// Element of \(Z_Q\) with \(Q < 2^{256}\).
pub type Zq = U256;

#[inline]
pub fn zq_from_u128(x: u128) -> Zq {
    Zq::from(x)
}

#[inline]
pub fn zq_from_u64(x: u64) -> Zq {
    Zq::from(x)
}

#[inline]
pub fn add_mod_zq(a: Zq, b: Zq, m: Zq) -> Zq {
    debug_assert!(m != Zq::ZERO);
    let a = a % m;
    let b = b % m;
    let sum = a + b;
    if sum >= m {
        sum - m
    } else {
        sum
    }
}

#[inline]
pub fn sub_mod_zq(a: Zq, b: Zq, m: Zq) -> Zq {
    let a = a % m;
    let b = b % m;
    if a >= b {
        a - b
    } else {
        m - (b - a)
    }
}

/// `(a * b) mod m`.
///
/// Uses a `u128` widening path when everything fits, otherwise double-and-add
/// on the **smaller** factor (CRT residues are `u64`, so recombine stays fast).
#[inline]
pub fn mul_mod_zq(a: Zq, b: Zq, m: Zq) -> Zq {
    if m == Zq::ZERO {
        return Zq::ZERO;
    }
    let a = a % m;
    let b = b % m;
    if a == Zq::ZERO || b == Zq::ZERO {
        return Zq::ZERO;
    }
    // Fast path: ring / plaintext Q fits in u128 (≤ 4 limbs of ~31-bit primes).
    if a.high() == &0 && b.high() == &0 && m.high() == &0 {
        return Zq::from(mul_mod_u128_inline(*a.low(), *b.low(), *m.low()));
    }
    // Double-and-add on the smaller operand.
    let (mut x, mut y) = if a < b { (b, a) } else { (a, b) };
    let mut acc = Zq::ZERO;
    while y != Zq::ZERO {
        if (y & Zq::ONE) != Zq::ZERO {
            acc = add_mod_zq(acc, x, m);
        }
        x = add_mod_zq(x, x, m);
        y >>= 1;
    }
    acc
}

#[inline(always)]
fn mul_mod_u128_inline(a: u128, b: u128, m: u128) -> u128 {
    if m == 0 {
        return 0;
    }
    let a = a % m;
    let b = b % m;
    if a == 0 || b == 0 {
        return 0;
    }
    // Widening 128×128 → reduce with (hi, lo).
    let (hi, lo) = {
        let a0 = a as u64 as u128;
        let a1 = a >> 64;
        let b0 = b as u64 as u128;
        let b1 = b >> 64;
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let (mid, c1) = p01.overflowing_add(p10);
        let c1 = if c1 { 1u128 } else { 0 };
        let lo_hi = (p00 >> 64) + (mid & 0xffff_ffff_ffff_ffff);
        let lo = (p00 & 0xffff_ffff_ffff_ffff) | ((lo_hi & 0xffff_ffff_ffff_ffff) << 64);
        let hi = p11 + (mid >> 64) + (lo_hi >> 64) + (c1 << 64);
        (hi, lo)
    };
    rem_u256_inline(hi, lo, m)
}

#[inline(always)]
fn rem_u256_inline(hi: u128, lo: u128, m: u128) -> u128 {
    if m == 0 {
        return 0;
    }
    if hi == 0 {
        return lo % m;
    }
    let two128 = {
        let t = u128::MAX % m;
        if t == m - 1 {
            0
        } else {
            t + 1
        }
    };
    let hi_term = if two128 == 0 {
        0
    } else {
        let a = hi % m;
        let b = two128;
        // a*b may still be 256-bit; one recursive rem is enough (a,b < m ≤ 2^128).
        let (h2, l2) = {
            let a0 = a as u64 as u128;
            let a1 = a >> 64;
            let b0 = b as u64 as u128;
            let b1 = b >> 64;
            let p00 = a0 * b0;
            let p01 = a0 * b1;
            let p10 = a1 * b0;
            let p11 = a1 * b1;
            let (mid, c1) = p01.overflowing_add(p10);
            let c1 = if c1 { 1u128 } else { 0 };
            let lo_hi = (p00 >> 64) + (mid & 0xffff_ffff_ffff_ffff);
            let lo = (p00 & 0xffff_ffff_ffff_ffff) | ((lo_hi & 0xffff_ffff_ffff_ffff) << 64);
            let hi = p11 + (mid >> 64) + (lo_hi >> 64) + (c1 << 64);
            (hi, lo)
        };
        if h2 == 0 {
            l2 % m
        } else {
            rem_u256_inline(h2, l2, m)
        }
    };
    let lo_m = lo % m;
    let sum = hi_term + lo_m;
    if sum >= m {
        sum - m
    } else {
        sum
    }
}

#[inline]
pub fn neg_mod_zq(a: Zq, m: Zq) -> Zq {
    let a = a % m;
    if a == Zq::ZERO {
        Zq::ZERO
    } else {
        m - a
    }
}

/// Centered representative in \((-Q/2, Q/2]\) as `i128` when it fits.
///
/// Fresh noise and CKKS messages at \(\Delta\le 2^{40}\) fit; arbitrary
/// ciphertext coefficients do **not** — those stay as `Zq`.
pub fn zq_to_center_i128(u: Zq, q: Zq) -> Result<i128, String> {
    let u = u % q;
    let half = q >> 1;
    if u > half {
        let neg = q - u; // positive distance below 0
        let v = u128::try_from(neg).map_err(|_| "centered residue exceeds i128")?;
        if v > (i128::MAX as u128) {
            return Err("centered residue exceeds i128".into());
        }
        Ok(-(v as i128))
    } else {
        let v = u128::try_from(u).map_err(|_| "centered residue exceeds i128")?;
        if v > (i128::MAX as u128) {
            return Err("centered residue exceeds i128".into());
        }
        Ok(v as i128)
    }
}

pub fn center_i128_to_zq(c: i128, q: Zq) -> Zq {
    if c >= 0 {
        zq_from_u128(c as u128) % q
    } else {
        let abs = (-c) as u128;
        sub_mod_zq(Zq::ZERO, zq_from_u128(abs), q)
    }
}

pub fn center_i64_to_zq(c: i64, q: Zq) -> Zq {
    center_i128_to_zq(c as i128, q)
}

/// \(Q = \prod q_i\) as `U256`.
pub fn product_zq(primes: &[u64]) -> Result<Zq, String> {
    let mut modulus = Zq::ONE;
    for &p in primes {
        let next = modulus.checked_mul(Zq::from(p));
        modulus = next.ok_or_else(|| "RNS modulus Q overflowed U256".to_string())?;
    }
    if modulus == Zq::ZERO {
        return Err("empty product".into());
    }
    Ok(modulus)
}

pub fn zq_ilog2(q: Zq) -> u32 {
    if q == Zq::ZERO {
        0
    } else {
        255 - q.leading_zeros()
    }
}
