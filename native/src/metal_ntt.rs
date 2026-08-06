//! Metal GPU batch negacyclic NTT bindings + correctness gate.
//!
//! Wraps the validated node-fhe-accelerate Metal kernels (Montgomery R=2^32).
//! Falls back to the Rayon CPU path when Metal is unavailable or `q ≥ 2^32`.

use std::path::PathBuf;
use std::sync::OnceLock;

use crate::ntt::{mul_mod, NegacyclicPlan};

#[cfg(engpe_metal)]
#[link(name = "engpe_metal", kind = "static")]
extern "C" {
    fn engpe_metal_init(metallib_path: *const std::os::raw::c_char) -> bool;
    fn engpe_metal_available() -> bool;
    fn engpe_metal_batch_ntt_forward(
        coeffs: *mut u64,
        degree: usize,
        batch_size: usize,
        modulus: u64,
        omega_powers: *const u64,
        psi_powers: *const u64,
    ) -> bool;
    fn engpe_metal_batch_ntt_inverse(
        coeffs: *mut u64,
        degree: usize,
        batch_size: usize,
        modulus: u64,
        omega_inv_powers: *const u64,
        psi_inv_powers: *const u64,
    ) -> bool;
    fn engpe_metal_batch_modmul(
        a: *const u64,
        b: *const u64,
        out: *mut u64,
        count: usize,
        modulus: u64,
    ) -> bool;
}

/// True when Metal NTT pipelines are live for this process.
pub fn metal_available() -> bool {
    ensure_metal_init();
    #[cfg(engpe_metal)]
    unsafe {
        engpe_metal_available()
    }
    #[cfg(not(engpe_metal))]
    {
        false
    }
}

fn ensure_metal_init() {
    static INIT: OnceLock<bool> = OnceLock::new();
    INIT.get_or_init(|| init_metal_once());
}

fn init_metal_once() -> bool {
    #[cfg(engpe_metal)]
    {
        let c = std::ffi::CString::new(metallib_path()).unwrap_or_default();
        // SAFETY: path is a NUL-terminated CString; ObjC bridge owns lifetime.
        unsafe { engpe_metal_init(c.as_ptr()) }
    }
    #[cfg(not(engpe_metal))]
    {
        false
    }
}

fn metallib_path() -> String {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("fhe_shaders.metallib"),
        manifest.join("dist/shaders/fhe_shaders.metallib"),
        PathBuf::from("native/fhe_shaders.metallib"),
        PathBuf::from("fhe_shaders.metallib"),
    ];
    candidates
        .into_iter()
        .find(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fhe_shaders.metallib".into())
}

/// Metal is eligible only for odd moduli below 2^32.
pub fn metal_modulus_ok(q: u64) -> bool {
    q >= 3 && (q & 1) == 1 && q < (1u64 << 32)
}

fn gate_operands(n: usize, q: u64) -> (Vec<u64>, Vec<u64>) {
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    a[0] = 1;
    a[n - 1] = 2 % q;
    b[0] = 3 % q;
    b[n - 1] = 5 % q;
    (a, b)
}

fn schoolbook_negacyclic(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut expected = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            accumulate_term(&mut expected, i, j, a[i], b[j], n, q);
        }
    }
    expected
}

fn accumulate_term(out: &mut [u64], i: usize, j: usize, ai: u64, bj: u64, n: usize, q: u64) {
    let mut prod = mul_mod(ai, bj, q);
    let mut k = i + j;
    if k >= n {
        k -= n;
        prod = neg_mod(prod, q);
    }
    out[k] = (out[k] + prod) % q;
}

fn neg_mod(prod: u64, q: u64) -> u64 {
    if prod == 0 {
        0
    } else {
        q - prod
    }
}

fn metal_mul_via_ntt(a: &[u64], b: &[u64], plan: &NegacyclicPlan) -> Option<Vec<u64>> {
    let n = plan.n;
    let mut batch = Vec::with_capacity(2 * n);
    batch.extend_from_slice(a);
    batch.extend_from_slice(b);
    if !metal_forward_batch(&mut batch, 2, plan) {
        return None;
    }
    let mut prod = vec![0u64; n];
    for i in 0..n {
        prod[i] = mul_mod(batch[i], batch[n + i], plan.q);
    }
    if !metal_inverse_batch(&mut prod, 1, plan) {
        return None;
    }
    Some(prod)
}

/// Negacyclic convolution identity on the GPU (paper §6.5/6.6 gate).
/// Populates index `N-1` in both operands so `X^N = -1` is exercised.
pub fn verify_metal_negacyclic_convolution(plan: &NegacyclicPlan) -> bool {
    if !metal_available() || !metal_modulus_ok(plan.q) {
        return false;
    }
    let (a, b) = gate_operands(plan.n, plan.q);
    let expected = schoolbook_negacyclic(&a, &b, plan.q);
    match metal_mul_via_ntt(&a, &b, plan) {
        Some(prod) => prod == expected,
        None => false,
    }
}

pub(crate) fn metal_forward_batch(coeffs: &mut [u64], batch: usize, plan: &NegacyclicPlan) -> bool {
    #[cfg(engpe_metal)]
    {
        ensure_metal_init();
        // SAFETY: coeffs length is batch*degree; twiddles match plan.n.
        unsafe {
            engpe_metal_batch_ntt_forward(
                coeffs.as_mut_ptr(),
                plan.n,
                batch,
                plan.q,
                plan.omega_powers.as_ptr(),
                plan.psi_powers.as_ptr(),
            )
        }
    }
    #[cfg(not(engpe_metal))]
    {
        let _ = (coeffs, batch, plan);
        false
    }
}

pub(crate) fn metal_inverse_batch(coeffs: &mut [u64], batch: usize, plan: &NegacyclicPlan) -> bool {
    #[cfg(engpe_metal)]
    {
        ensure_metal_init();
        // SAFETY: coeffs length is batch*degree; inv twiddles match plan.n.
        unsafe {
            engpe_metal_batch_ntt_inverse(
                coeffs.as_mut_ptr(),
                plan.n,
                batch,
                plan.q,
                plan.omega_inv_powers.as_ptr(),
                plan.psi_inv_powers.as_ptr(),
            )
        }
    }
    #[cfg(not(engpe_metal))]
    {
        let _ = (coeffs, batch, plan);
        false
    }
}

fn metal_modmul(a: &[u64], b: &[u64], out: &mut [u64], q: u64) -> bool {
    #[cfg(engpe_metal)]
    {
        // SAFETY: a, b, out share length; bridge writes exactly count limbs.
        unsafe {
            engpe_metal_batch_modmul(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), out.len(), q)
        }
    }
    #[cfg(not(engpe_metal))]
    {
        let _ = (a, b, out, q);
        false
    }
}

fn lengths_ok(a: &[u64], b: &[u64], n: usize) -> bool {
    a.len() == n && b.len() == n
}

fn batch_lens_ok(cts: &[u64], pts: &[u64], cipher_count: usize, n: usize) -> bool {
    cts.len() == cipher_count * n && pts.len() == cipher_count * n
}

fn metal_ready(plan: &NegacyclicPlan) -> bool {
    metal_available() && metal_modulus_ok(plan.q)
}

fn forward_pair_batch(a: &[u64], b: &[u64], plan: &NegacyclicPlan) -> Option<Vec<u64>> {
    let n = plan.n;
    let mut batch = Vec::with_capacity(2 * n);
    batch.extend_from_slice(a);
    batch.extend_from_slice(b);
    metal_forward_batch(&mut batch, 2, plan).then_some(batch)
}

/// Pointwise product via Metal forward → modmul → inverse.
/// Returns `None` if Metal refuses; caller should fall back to CPU.
pub fn metal_ntt_pointwise_product(
    a: &[u64],
    b: &[u64],
    plan: &NegacyclicPlan,
) -> Option<Vec<u64>> {
    if !(metal_ready(plan) && lengths_ok(a, b, plan.n)) {
        return None;
    }
    let n = plan.n;
    let batch = forward_pair_batch(a, b, plan)?;
    let mut prod = vec![0u64; n];
    metal_modmul(&batch[..n], &batch[n..], &mut prod, plan.q)
        .then(|| ())
        .and_then(|_| metal_inverse_batch(&mut prod, 1, plan).then_some(prod))
}

fn finish_batch_mul(
    joined: &mut [u64],
    cts: &mut [u64],
    cipher_count: usize,
    plan: &NegacyclicPlan,
) -> Option<()> {
    let n = plan.n;
    let (fa, fb) = joined.split_at_mut(cipher_count * n);
    let mut prod = vec![0u64; cipher_count * n];
    metal_modmul(fa, fb, &mut prod, plan.q)
        .then(|| ())
        .and_then(|_| metal_inverse_batch(&mut prod, cipher_count, plan).then(|| ()))
        .map(|_| cts.copy_from_slice(&prod))
}

/// Batch Metal multiply of `cipher_count` poly pairs as two flat `[cipher][N]`
/// buffers. Overwrites `cts` with the inverse-NTT products.
pub fn metal_batch_mul_pairs(
    cts: &mut [u64],
    pts: &[u64],
    cipher_count: usize,
    plan: &NegacyclicPlan,
) -> Option<()> {
    if !(metal_ready(plan) && batch_lens_ok(cts, pts, cipher_count, plan.n)) {
        return None;
    }
    let n = plan.n;
    let mut joined = Vec::with_capacity(2 * cipher_count * n);
    joined.extend_from_slice(cts);
    joined.extend_from_slice(pts);
    metal_forward_batch(&mut joined, 2 * cipher_count, plan)
        .then(|| ())
        .and_then(|_| finish_batch_mul(&mut joined, cts, cipher_count, plan))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::NegacyclicPlan;

    #[test]
    fn metal_gate_or_skip() {
        // q = 12289 works for N=64 and is < 2^32.
        let plan = NegacyclicPlan::new(64, 12289).unwrap();
        if !metal_available() {
            eprintln!("Metal unavailable — skipping GPU gate");
            return;
        }
        assert!(
            verify_metal_negacyclic_convolution(&plan),
            "Metal negacyclic convolution gate failed"
        );
    }
}
