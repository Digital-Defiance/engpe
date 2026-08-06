/**
 * Core types for the Edge-Native Genomic Privacy Engine (ENGPE).
 *
 * Multi-disease SIMD (Task 7): M patient genotypes × D disease weight panels
 * in one FFI call. Genotypes are encoded once and broadcast-multiplied across
 * all D plaintext weight vectors.
 */

/** Target SNP panel size for the clinical PRS model. */
export const SNP_COUNT = 110_000;

/** Default number of concurrent disease models in a SIMD sweep. */
export const DEFAULT_DISEASE_COUNT = 5;

/**
 * Default CKKS polynomial degree. Matches the largest tier characterised in
 * the fhe-evolve M4 Max study (N ∈ {1024…16384}). Slot count = N / 2.
 */
export const DEFAULT_POLY_DEGREE = 16_384;

/** CKKS complex slots available at a given polynomial degree. */
export function ckksSlotCount(polyDegree: number): number {
  return polyDegree / 2;
}

/**
 * Number of CKKS ciphertexts (and matching plaintexts) needed to hold
 * `snpCount` real values at the given degree.
 */
export function cipherCountFor(snpCount: number, polyDegree: number): number {
  const slots = ckksSlotCount(polyDegree);
  if (slots < 1 || (polyDegree & (polyDegree - 1)) !== 0) {
    throw new Error(`polyDegree must be a power of 2 ≥ 2; got ${polyDegree}`);
  }
  return Math.ceil(snpCount / slots);
}

/** CKKS encoding / evaluation parameters shared across the FFI boundary. */
export interface CkksPrsParams {
  /** Polynomial degree N (power of 2). */
  polyDegree: number;
  /** Number of SNPs in the panel (≤ packed slot capacity). */
  snpCount: number;
  /** CKKS scale Δ used when encoding reals into coefficient form. */
  scale: number;
  /** Patients packed into one FFI call (default 1). */
  patientCount?: number;
  /** Disease weight panels packed into one FFI call (default 1). */
  diseaseCount?: number;
}

/**
 * Synthetic (or loaded) PRS inputs before CKKS packing.
 * Genotypes are allelic dosages in {0, 1, 2}; weights are clinical effect sizes.
 */
export interface PrsVectors {
  /** Length = snpCount; each entry ∈ {0, 1, 2}. */
  patientGenotype: Uint8Array;
  /** Length = snpCount; floating-point clinical weights. */
  medicalWeights: Float64Array;
}

/**
 * Multi-patient × multi-disease cohort.
 * Genotypes encoded once; D weight panels share the same SNP layout.
 */
export interface PrsCohort {
  /** Length = patientCount; each genotype length = snpCount. */
  genotypes: Uint8Array[];
  /**
   * Disease weight panels. Prefer `diseaseWeights` (D panels).
   * `medicalWeights` is a single-disease compat alias (disease 0).
   */
  diseaseWeights: Float64Array[];
  /** Compat: first disease panel (same reference as diseaseWeights[0]). */
  medicalWeights: Float64Array;
}

/**
 * Flat FFI payload ready for a single napi call.
 *
 * Slot layout (Task 7):
 *   [0 .. M * perSide)                 — patient genotypes (patient-major)
 *   [M * perSide .. (M+D) * perSide)   — D disease weight panels
 * where perSide = cipherCount * slotCount.
 *
 * Header Uint32Array:
 *   [polyDegree, snpCount, slotCount, cipherCount, scaleBits, patientCount, diseaseCount]
 */
export interface PackedPrsBatch {
  /** Packed metadata header — see layout above. */
  header: Uint32Array;
  /**
   * Concatenated genotype(s) + weight slot vectors.
   * Length = (patientCount + diseaseCount) * cipherCount * slotCount.
   */
  slots: Float64Array;
  /** Derived convenience mirrors of the header (not sent over FFI). */
  meta: {
    polyDegree: number;
    snpCount: number;
    slotCount: number;
    cipherCount: number;
    scale: number;
    patientCount: number;
    diseaseCount: number;
  };
}

/** Number of u32 fields in the packed PRS header. */
export const PRS_HEADER_FIELD_COUNT = 7;

/** Header field indices. */
export const PrsHeaderField = {
  PolyDegree: 0,
  SnpCount: 1,
  SlotCount: 2,
  CipherCount: 3,
  ScaleBits: 4,
  PatientCount: 5,
  DiseaseCount: 6,
} as const;

/** One (patient, disease) decoded PRS vs plaintext oracle. */
export interface PatientPrsScore {
  plaintextScore: number;
  decodedScore: number;
  absError: number;
  patientIndex: number;
  diseaseIndex: number;
}

/**
 * Result of a homomorphic PRS evaluation (CPU or Metal CKKS path).
 * `scores` is patient-major, disease-minor: index = p * D + d.
 */
export interface PrsEvaluationResult {
  patientCount: number;
  diseaseCount: number;
  /** Plaintext reference for patient 0, disease 0 (compat). */
  plaintextScore: number;
  /** Decoded score for patient 0, disease 0 (compat). */
  decodedScore: number;
  batchMultiplies: number;
  cipherCount: number;
  absError: number;
  accepted: boolean;
  backend: 'cpu' | 'metal';
  nttMs: number;
  /** Length = patientCount * diseaseCount (patient-major). */
  scores: PatientPrsScore[];
  /** Convenience: scores[p][d]. */
  matrix: PatientPrsScore[][];
}
