//! CKKS plaintext encode / decode via the negacyclic canonical embedding.
//!
//! # Algorithm
//! Embeddings are evaluations at `ζ_j = exp(iπ(2j+1)/N)`, `j = 0..N-1`.
//! Twisting coefficients by `ψ^k` with `ψ = exp(iπ/N)` reduces this to a
//! length-`N` FFT, so ring multiplication in `Z[X]/(X^N+1)` is slot-wise on
//! the first `N/2` (real) slots after conjugate-symmetric packing.
//!
//! Steps for encode of `N/2` reals:
//! 1. Pack scaled slots into conjugate-symmetric embedding targets.
//! 2. Inverse FFT → twisted coefficients `d_k`.
//! 3. Untwist `c_k = d_k / ψ^k`, take real part, round.
//!
//! # Precision
//! Rounding injects ≤ `1/(2Δ)` per coefficient. After one multiply the message
//! scale is `Δ²`. Expect absolute PRS error ≪ `1e-2` at `Δ = 2^16` for
//! `N ≤ 16384` on unit-scale inputs. `Δ = 2^40` is supported via the RNS path
//! (`ckks_rns` + CRT) which reconstitutes wide coeffs before decode.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

pub type EncodeResult<T> = Result<T, String>;

pub fn scale_from_bits(scale_bits: u32) -> f64 {
    (scale_bits as f64).exp2()
}

fn psi_power(k: usize, n: usize) -> Complex<f64> {
    // ψ^k = exp(i π k / N)
    Complex::from_polar(1.0, PI * (k as f64) / (n as f64))
}

fn require_degree(n: usize) -> EncodeResult<()> {
    if n >= 2 && (n & (n - 1)) == 0 {
        Ok(())
    } else {
        Err(format!("degree {n} must be a power of 2 ≥ 2"))
    }
}

/// Pack `N/2` real slots into length-`N` conjugate-symmetric embeddings.
fn pack_targets(slots: &[f64], n: usize, scale: f64) -> Vec<Complex<f64>> {
    let m = n / 2;
    let mut targets = vec![Complex::new(0.0, 0.0); n];
    for j in 0..m {
        let z = Complex::new(slots[j] * scale, 0.0);
        targets[j] = z;
        targets[n - 1 - j] = z.conj();
    }
    targets
}

/// Encode `N/2` real slots → degree-`N` integer polynomial.
pub fn encode_real_slots(slots: &[f64], n: usize, scale: f64) -> EncodeResult<Vec<i64>> {
    require_degree(n)?;
    let m = n / 2;
    if slots.len() != m {
        return Err(format!("expected {m} slots for degree {n}, got {}", slots.len()));
    }
    if scale <= 0.0 || !scale.is_finite() {
        return Err("scale must be positive and finite".into());
    }

    let mut targets = pack_targets(slots, n, scale);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut targets);

    // emb = IFFT(d) ⇒ d = FFT(emb)/N. Untwist c_k = d_k / ψ^k.
    let mut out = vec![0i64; n];
    for k in 0..n {
        let d = targets[k] / (n as f64);
        let c = d / psi_power(k, n);
        out[k] = c.re.round() as i64;
    }
    Ok(out)
}

/// Decode degree-`N` coefficients → `N/2` real slots.
pub fn decode_real_slots(coeffs: &[i64], scale: f64) -> EncodeResult<Vec<f64>> {
    let wide: Vec<i128> = coeffs.iter().map(|&c| c as i128).collect();
    decode_real_slots_i128(&wide, scale)
}

/// Decode wide (`i128`) coefficients with automatic rescaling so the FFT stays
/// inside the `f64` mantissa. Used after CRT recombination of RNS products.
pub fn decode_real_slots_i128(coeffs: &[i128], scale: f64) -> EncodeResult<Vec<f64>> {
    let n = coeffs.len();
    require_degree(n)?;
    if scale <= 0.0 || !scale.is_finite() {
        return Err("scale must be positive and finite".into());
    }
    let shift = rescale_shift(coeffs);
    let scale_adj = scale / (shift as f64).exp2();
    let mut buf: Vec<Complex<f64>> = (0..n)
        .map(|k| Complex::new(coeff_to_f64(coeffs[k], shift), 0.0) * psi_power(k, n))
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buf);

    let m = n / 2;
    let mut out = vec![0.0; m];
    for j in 0..m {
        out[j] = buf[j].re / scale_adj;
    }
    Ok(out)
}

fn rescale_shift(coeffs: &[i128]) -> u32 {
    let max_abs = coeffs.iter().map(|c| c.unsigned_abs()).max().unwrap_or(0);
    if max_abs <= 1 {
        return 0;
    }
    let bits = max_abs.ilog2() + 1;
    bits.saturating_sub(50)
}

fn coeff_to_f64(c: i128, shift: u32) -> f64 {
    if shift == 0 {
        return c as f64;
    }
    let sign = if c < 0 { -1.0 } else { 1.0 };
    let u = c.unsigned_abs();
    let hi = (u >> shift) as f64;
    let mask = (1u128 << shift) - 1;
    let lo = (u & mask) as f64 / (shift as f64).exp2();
    sign * (hi + lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip_small() {
        let n = 32;
        let slots: Vec<f64> = (0..n / 2).map(|i| (i as f64) * 0.1 - 0.5).collect();
        let scale = (16u32 as f64).exp2();
        let coeffs = encode_real_slots(&slots, n, scale).unwrap();
        let back = decode_real_slots(&coeffs, scale).unwrap();
        for i in 0..slots.len() {
            assert!(
                (slots[i] - back[i]).abs() < 1e-3,
                "slot {i}: {} vs {}",
                slots[i],
                back[i]
            );
        }
    }

    #[test]
    fn ring_mul_is_slotwise() {
        let n = 64usize;
        let scale = (14u32 as f64).exp2();
        let a: Vec<f64> = (0..n / 2).map(|i| 0.05 * (i as f64 - 10.0)).collect();
        let b: Vec<f64> = (0..n / 2).map(|i| 0.02 * ((i % 7) as f64)).collect();
        let ca = encode_real_slots(&a, n, scale).unwrap();
        let cb = encode_real_slots(&b, n, scale).unwrap();

        // Schoolbook negacyclic convolution over Z (no mod) for the identity check.
        let mut prod = vec![0i64; n];
        for i in 0..n {
            for j in 0..n {
                let mut k = i + j;
                let mut term = ca[i] * cb[j];
                if k >= n {
                    k -= n;
                    term = -term;
                }
                prod[k] += term;
            }
        }
        let decoded = decode_real_slots(&prod, scale * scale).unwrap();
        for i in 0..n / 2 {
            let expected = a[i] * b[i];
            assert!(
                (decoded[i] - expected).abs() < 5e-2,
                "slot {i}: got {} want {expected}",
                decoded[i]
            );
        }
    }
}
