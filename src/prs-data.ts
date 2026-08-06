/**
 * Synthetic PRS panel generation and zero-overhead FFI packing.
 *
 * Multi-disease layout (Task 7): M genotype panels + D disease weight panels
 * in one Float64Array. Genotypes are encoded once and broadcast across diseases.
 */

import {
  SNP_COUNT,
  DEFAULT_POLY_DEGREE,
  DEFAULT_DISEASE_COUNT,
  PRS_HEADER_FIELD_COUNT,
  PrsHeaderField,
  ckksSlotCount,
  cipherCountFor,
  type CkksPrsParams,
  type PackedPrsBatch,
  type PrsCohort,
  type PrsVectors,
} from './types.js';

/** Default CKKS scale (2^16) for the single-prime CPU NTT path. */
export const DEFAULT_CKKS_SCALE = 2 ** 16;

export interface GenerateOptions {
  snpCount?: number;
  seed?: number;
  weightBound?: number;
}

export interface GenerateCohortOptions extends GenerateOptions {
  patientCount?: number;
  /** Number of distinct disease weight panels (default 1). */
  diseaseCount?: number;
}

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function fillGenotype(out: Uint8Array, rand: () => number): void {
  for (let i = 0; i < out.length; i++) {
    const u = rand();
    out[i] = u < 0.45 ? 0 : u < 0.9 ? 1 : 2;
  }
}

function fillWeights(
  out: Float64Array,
  rand: () => number,
  weightBound: number,
): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = (rand() * 2 - 1) * weightBound;
  }
}

/** Generate a synthetic 110k-SNP patient genotype + one medical weight panel. */
export function generateSyntheticPrs(options: GenerateOptions = {}): PrsVectors {
  const snpCount = options.snpCount ?? SNP_COUNT;
  const seed = (options.seed ?? 0xc0ffee) >>> 0;
  const weightBound = options.weightBound ?? 0.05;
  const rand = mulberry32(seed);
  const patientGenotype = new Uint8Array(snpCount);
  const medicalWeights = new Float64Array(snpCount);
  fillGenotype(patientGenotype, rand);
  fillWeights(medicalWeights, rand, weightBound);
  return { patientGenotype, medicalWeights };
}

/**
 * Generate M patient genotypes + D disease weight panels (deterministic seed).
 */
export function generateSyntheticCohort(options: GenerateCohortOptions = {}): PrsCohort {
  const snpCount = options.snpCount ?? SNP_COUNT;
  const patientCount = options.patientCount ?? 1;
  const diseaseCount = options.diseaseCount ?? 1;
  if (patientCount < 1) {
    throw new Error(`patientCount must be ≥ 1; got ${patientCount}`);
  }
  if (diseaseCount < 1) {
    throw new Error(`diseaseCount must be ≥ 1; got ${diseaseCount}`);
  }
  const seed = (options.seed ?? 0xc0ffee) >>> 0;
  const weightBound = options.weightBound ?? 0.05;
  const rand = mulberry32(seed);

  const diseaseWeights: Float64Array[] = [];
  for (let d = 0; d < diseaseCount; d++) {
    const w = new Float64Array(snpCount);
    fillWeights(w, rand, weightBound);
    diseaseWeights.push(w);
  }

  const genotypes: Uint8Array[] = [];
  for (let p = 0; p < patientCount; p++) {
    const g = new Uint8Array(snpCount);
    fillGenotype(g, rand);
    genotypes.push(g);
  }
  return {
    genotypes,
    diseaseWeights,
    medicalWeights: diseaseWeights[0]!,
  };
}

/** Plaintext reference PRS — single patient × single disease. */
export function plaintextPrs(vectors: PrsVectors): number {
  return plaintextDot(vectors.patientGenotype, vectors.medicalWeights);
}

/** Plaintext oracle for every patient (single disease / disease 0). */
export function plaintextPrsCohort(cohort: PrsCohort): number[] {
  const w = cohort.diseaseWeights[0] ?? cohort.medicalWeights;
  return cohort.genotypes.map((g) => plaintextDot(g, w));
}

/**
 * Plaintext oracle matrix: scores[p][d] = ⟨geno_p, weights_d⟩.
 * Shape M × D.
 */
export function plaintextPrsMatrix(cohort: PrsCohort): number[][] {
  const weights = cohort.diseaseWeights;
  return cohort.genotypes.map((g) => weights.map((w) => plaintextDot(g, w)));
}

/** Flat patient-major oracle vector: index = p * D + d. */
export function plaintextPrsFlat(cohort: PrsCohort): number[] {
  const matrix = plaintextPrsMatrix(cohort);
  return matrix.flat();
}

function plaintextDot(genotype: Uint8Array, weights: Float64Array): number {
  if (genotype.length !== weights.length) {
    throw new Error(
      `Length mismatch: genotype ${genotype.length} vs weights ${weights.length}`,
    );
  }
  let sum = 0;
  for (let i = 0; i < genotype.length; i++) {
    sum += genotype[i]! * weights[i]!;
  }
  return sum;
}

function packOnePanel(
  dest: Float64Array,
  values: ArrayLike<number>,
  snpCount: number,
): void {
  for (let i = 0; i < snpCount; i++) {
    dest[i] = values[i]!;
  }
}

/** Pack a single patient (compat wrapper around cohort packing). */
export function packPrsForFfi(
  vectors: PrsVectors,
  params: Partial<CkksPrsParams> = {},
): PackedPrsBatch {
  return packCohortForFfi(
    {
      genotypes: [vectors.patientGenotype],
      diseaseWeights: [vectors.medicalWeights],
      medicalWeights: vectors.medicalWeights,
    },
    params,
  );
}

/**
 * Pack M genotypes + D disease weight panels into the flat FFI layout.
 */
export function packCohortForFfi(
  cohort: PrsCohort,
  params: Partial<CkksPrsParams> = {},
): PackedPrsBatch {
  const patientCount = cohort.genotypes.length;
  const diseaseCount = cohort.diseaseWeights.length;
  if (patientCount < 1) {
    throw new Error('cohort must contain at least one genotype');
  }
  if (diseaseCount < 1) {
    throw new Error('cohort must contain at least one disease weight panel');
  }
  const polyDegree = params.polyDegree ?? DEFAULT_POLY_DEGREE;
  const snpCount = params.snpCount ?? cohort.diseaseWeights[0]!.length;
  const scale = params.scale ?? DEFAULT_CKKS_SCALE;

  requireCohortLengths(cohort, snpCount, patientCount, diseaseCount);
  if (scale <= 0 || !Number.isFinite(scale)) {
    throw new Error(`CKKS scale must be a positive finite number; got ${scale}`);
  }

  const slotCount = ckksSlotCount(polyDegree);
  const cipherCount = cipherCountFor(snpCount, polyDegree);
  const perSide = cipherCount * slotCount;
  const slots = new Float64Array((patientCount + diseaseCount) * perSide);

  for (let p = 0; p < patientCount; p++) {
    const view = slots.subarray(p * perSide, (p + 1) * perSide);
    packOnePanel(view, cohort.genotypes[p]!, snpCount);
  }
  for (let d = 0; d < diseaseCount; d++) {
    const view = slots.subarray(
      (patientCount + d) * perSide,
      (patientCount + d + 1) * perSide,
    );
    packOnePanel(view, cohort.diseaseWeights[d]!, snpCount);
  }

  const scaleBits = Math.round(Math.log2(scale));
  const header = new Uint32Array(PRS_HEADER_FIELD_COUNT);
  header[PrsHeaderField.PolyDegree] = polyDegree;
  header[PrsHeaderField.SnpCount] = snpCount;
  header[PrsHeaderField.SlotCount] = slotCount;
  header[PrsHeaderField.CipherCount] = cipherCount;
  header[PrsHeaderField.ScaleBits] = scaleBits >>> 0;
  header[PrsHeaderField.PatientCount] = patientCount;
  header[PrsHeaderField.DiseaseCount] = diseaseCount;

  return {
    header,
    slots,
    meta: {
      polyDegree,
      snpCount,
      slotCount,
      cipherCount,
      scale,
      patientCount,
      diseaseCount,
    },
  };
}

function requireCohortLengths(
  cohort: PrsCohort,
  snpCount: number,
  patientCount: number,
  diseaseCount: number,
): void {
  if (cohort.genotypes.length !== patientCount) {
    throw new Error('internal patientCount mismatch');
  }
  if (cohort.diseaseWeights.length !== diseaseCount) {
    throw new Error('internal diseaseCount mismatch');
  }
  for (let p = 0; p < patientCount; p++) {
    if (cohort.genotypes[p]!.length !== snpCount) {
      throw new Error(
        `Genotype[${p}] length ${cohort.genotypes[p]!.length} ≠ snpCount ${snpCount}`,
      );
    }
  }
  for (let d = 0; d < diseaseCount; d++) {
    if (cohort.diseaseWeights[d]!.length !== snpCount) {
      throw new Error(
        `Weights[${d}] length ${cohort.diseaseWeights[d]!.length} ≠ snpCount ${snpCount}`,
      );
    }
  }
}

/** Validate a packed batch without crossing the FFI boundary. */
export function validatePackedPrs(batch: PackedPrsBatch): void {
  if (batch.header.length !== PRS_HEADER_FIELD_COUNT) {
    throw new Error(
      `Header length ${batch.header.length} ≠ ${PRS_HEADER_FIELD_COUNT}`,
    );
  }
  const polyDegree = batch.header[PrsHeaderField.PolyDegree]!;
  const snpCount = batch.header[PrsHeaderField.SnpCount]!;
  const slotCount = batch.header[PrsHeaderField.SlotCount]!;
  const cipherCount = batch.header[PrsHeaderField.CipherCount]!;
  const patientCount = batch.header[PrsHeaderField.PatientCount] ?? 1;
  const diseaseCount = batch.header[PrsHeaderField.DiseaseCount] ?? 1;

  if (polyDegree < 2 || (polyDegree & (polyDegree - 1)) !== 0) {
    throw new Error(`polyDegree must be a power of 2 ≥ 2; got ${polyDegree}`);
  }
  if (slotCount !== polyDegree / 2) {
    throw new Error(
      `slotCount ${slotCount} inconsistent with polyDegree ${polyDegree}`,
    );
  }
  if (cipherCount !== Math.ceil(snpCount / slotCount)) {
    throw new Error(
      `cipherCount ${cipherCount} inconsistent with snpCount/slotCount`,
    );
  }
  if (patientCount < 1) {
    throw new Error(`patientCount must be ≥ 1; got ${patientCount}`);
  }
  if (diseaseCount < 1) {
    throw new Error(`diseaseCount must be ≥ 1; got ${diseaseCount}`);
  }
  const expected = (patientCount + diseaseCount) * cipherCount * slotCount;
  if (batch.slots.length !== expected) {
    throw new Error(
      `slots length ${batch.slots.length} ≠ expected ${expected}`,
    );
  }
}

export { DEFAULT_DISEASE_COUNT };
