//! CKKS Polygenic Risk Score FFI types and unpack helpers.
//!
//! Header `Uint32Array` length 7:
//!   `[poly_degree, snp_count, slot_count, cipher_count, scale_bits,
//!     patient_count, disease_count]`
//!
//! Slots `Float64Array` length `(patient_count + disease_count) * per_side`:
//!   `[0 .. M * per_side)`                    — patient genotypes (patient-major)
//!   `[M * per_side .. (M+D) * per_side)`     — D disease weight panels

use napi::bindgen_prelude::*;

/// Number of u32 fields in the packed PRS header.
pub const PRS_HEADER_FIELD_COUNT: usize = 7;

/// Unpacked CKKS / PRS parameters after FFI validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CkksPrsParams {
    pub poly_degree: u32,
    pub snp_count: u32,
    pub slot_count: u32,
    pub cipher_count: u32,
    pub scale_bits: u32,
    pub patient_count: u32,
    pub disease_count: u32,
}

impl CkksPrsParams {
    pub fn per_side(&self) -> usize {
        self.cipher_count as usize * self.slot_count as usize
    }

    pub fn expected_slots_len(&self) -> usize {
        (self.patient_count as usize + self.disease_count as usize) * self.per_side()
    }
}

/// Views into the flat multi-patient / multi-disease slot buffers.
#[derive(Debug)]
pub struct PrsSlotViews<'a> {
    pub params: CkksPrsParams,
    /// Length = `patient_count * per_side` (patient-major).
    pub genotype_slots: &'a [f64],
    /// Length = `disease_count * per_side` (disease-major).
    pub weight_slots: &'a [f64],
}

impl<'a> PrsSlotViews<'a> {
    /// Genotype slice for patient `p` (length `per_side`).
    pub fn patient_geno(&self, p: usize) -> &'a [f64] {
        let n = self.params.per_side();
        &self.genotype_slots[p * n..(p + 1) * n]
    }

    /// Weight slice for disease `d` (length `per_side`).
    pub fn disease_weights(&self, d: usize) -> &'a [f64] {
        let n = self.params.per_side();
        &self.weight_slots[d * n..(d + 1) * n]
    }
}

fn require_power_of_two_degree(poly_degree: u32) -> Result<()> {
    if poly_degree >= 2 && (poly_degree & (poly_degree - 1)) == 0 {
        Ok(())
    } else {
        Err(Error::new(
            Status::InvalidArg,
            format!("poly_degree must be a power of 2 ≥ 2; got {poly_degree}"),
        ))
    }
}

fn require_slot_layout(poly_degree: u32, slot_count: u32) -> Result<()> {
    let expected = poly_degree / 2;
    if slot_count == expected {
        Ok(())
    } else {
        Err(Error::new(
            Status::InvalidArg,
            format!("slot_count {slot_count} ≠ poly_degree/2 ({expected})"),
        ))
    }
}

fn require_cipher_count(snp_count: u32, slot_count: u32, cipher_count: u32) -> Result<()> {
    let expected = (snp_count as u64).div_ceil(slot_count as u64) as u32;
    if cipher_count == expected {
        Ok(())
    } else {
        Err(Error::new(
            Status::InvalidArg,
            format!(
                "cipher_count {cipher_count} ≠ ceil(snp_count/slot_count) = {expected}"
            ),
        ))
    }
}

/// Unpack and validate the FFI header.
pub fn unpack_header(header: &[u32]) -> Result<CkksPrsParams> {
    if header.len() != PRS_HEADER_FIELD_COUNT {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Header length {} is not {}",
                header.len(),
                PRS_HEADER_FIELD_COUNT
            ),
        ));
    }

    let poly_degree = header[0];
    let snp_count = header[1];
    let slot_count = header[2];
    let cipher_count = header[3];
    let scale_bits = header[4];
    let patient_count = header[5];
    let disease_count = header[6];

    require_power_of_two_degree(poly_degree)?;
    require_slot_layout(poly_degree, slot_count)?;
    require_cipher_count(snp_count, slot_count, cipher_count)?;
    if patient_count < 1 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("patient_count must be ≥ 1; got {patient_count}"),
        ));
    }
    if disease_count < 1 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("disease_count must be ≥ 1; got {disease_count}"),
        ));
    }

    Ok(CkksPrsParams {
        poly_degree,
        snp_count,
        slot_count,
        cipher_count,
        scale_bits,
        patient_count,
        disease_count,
    })
}

/// Split the flat slots buffer into genotype and multi-disease weight views.
pub fn unpack_slots<'a>(params: CkksPrsParams, slots: &'a [f64]) -> Result<PrsSlotViews<'a>> {
    let expected = params.expected_slots_len();
    if slots.len() != expected {
        return Err(Error::new(
            Status::InvalidArg,
            format!("slots length {} ≠ expected {expected}", slots.len()),
        ));
    }
    let mid = params.patient_count as usize * params.per_side();
    Ok(PrsSlotViews {
        params,
        genotype_slots: &slots[..mid],
        weight_slots: &slots[mid..],
    })
}

/// Plaintext oracle for (patient, disease): Σ genotype[i] · weight_d[i].
pub fn plaintext_prs_oracle_pair(
    views: &PrsSlotViews<'_>,
    patient: usize,
    disease: usize,
) -> f64 {
    let n = views.params.snp_count as usize;
    let geno = views.patient_geno(patient);
    let weights = views.disease_weights(disease);
    let mut sum = 0.0;
    for i in 0..n {
        sum += geno[i] * weights[i];
    }
    sum
}

/// Plaintext oracle for patient `p`, disease 0 (compat).
pub fn plaintext_prs_oracle_patient(views: &PrsSlotViews<'_>, patient: usize) -> f64 {
    plaintext_prs_oracle_pair(views, patient, 0)
}

/// Plaintext oracle for patient 0, disease 0 (compat).
#[allow(dead_code)]
pub fn plaintext_prs_oracle(views: &PrsSlotViews<'_>) -> f64 {
    plaintext_prs_oracle_pair(views, 0, 0)
}

/// Flat oracles, patient-major × disease-minor: index = p * D + d.
pub fn plaintext_prs_oracles(views: &PrsSlotViews<'_>) -> Vec<f64> {
    let m = views.params.patient_count as usize;
    let d = views.params.disease_count as usize;
    let mut out = Vec::with_capacity(m * d);
    for p in 0..m {
        for disease in 0..d {
            out.push(plaintext_prs_oracle_pair(views, p, disease));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header(snp: u32, n: u32, patients: u32, diseases: u32) -> Vec<u32> {
        let slots = n / 2;
        let ciphers = snp.div_ceil(slots);
        vec![n, snp, slots, ciphers, 16, patients, diseases]
    }

    #[test]
    fn unpack_header_110k_at_n16384() {
        let h = sample_header(110_000, 16_384, 1, 1);
        let p = unpack_header(&h).unwrap();
        assert_eq!(p.slot_count, 8192);
        assert_eq!(p.cipher_count, 14);
        assert_eq!(p.scale_bits, 16);
        assert_eq!(p.patient_count, 1);
        assert_eq!(p.disease_count, 1);
    }

    #[test]
    fn unpack_rejects_bad_slot_count() {
        let mut h = sample_header(100, 16, 1, 1);
        h[2] = 7;
        assert!(unpack_header(&h).is_err());
    }

    #[test]
    fn oracle_matches_manual_dot() {
        let h = sample_header(4, 8, 1, 1);
        let params = unpack_header(&h).unwrap();
        let mut slots = vec![0.0; params.expected_slots_len()];
        slots[0] = 0.0;
        slots[1] = 1.0;
        slots[2] = 2.0;
        slots[3] = 1.0;
        let mid = params.per_side();
        slots[mid] = 0.5;
        slots[mid + 1] = -0.25;
        slots[mid + 2] = 0.1;
        slots[mid + 3] = 0.0;
        let views = unpack_slots(params, &slots).unwrap();
        let expected = 0.0 * 0.5 + 1.0 * -0.25 + 2.0 * 0.1 + 1.0 * 0.0;
        assert!((plaintext_prs_oracle(&views) - expected).abs() < 1e-12);
    }

    #[test]
    fn multi_disease_slot_layout() {
        let h = sample_header(4, 8, 2, 2);
        let params = unpack_header(&h).unwrap();
        assert_eq!(params.expected_slots_len(), 4 * params.per_side());
        let mut slots = vec![0.0; params.expected_slots_len()];
        let per = params.per_side();
        slots[0] = 1.0; // patient 0
        slots[per] = 2.0; // patient 1
        slots[2 * per] = 0.5; // disease 0
        slots[3 * per] = 0.25; // disease 1
        let views = unpack_slots(params, &slots).unwrap();
        assert!((plaintext_prs_oracle_pair(&views, 0, 0) - 0.5).abs() < 1e-12);
        assert!((plaintext_prs_oracle_pair(&views, 0, 1) - 0.25).abs() < 1e-12);
        assert!((plaintext_prs_oracle_pair(&views, 1, 0) - 1.0).abs() < 1e-12);
        assert!((plaintext_prs_oracle_pair(&views, 1, 1) - 0.5).abs() < 1e-12);
    }
}
