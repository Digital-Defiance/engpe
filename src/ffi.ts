/**
 * FFI boundary for CKKS Polygenic Risk Score evaluation (CPU + Metal).
 *
 * Output Float64Array (Task 7 multi-disease layout):
 *   [patientCount, diseaseCount, backend, nttMs, cipherCount, accepted,
 *    o0, d0, e0, …]  // patient-major × disease-minor
 * backend: 0 = CPU Rayon, 1 = Metal GPU
 */

import { createRequire } from 'node:module';
import {
  plaintextPrs,
  plaintextPrsFlat,
  validatePackedPrs,
} from './prs-data.js';
import type {
  PackedPrsBatch,
  PatientPrsScore,
  PrsCohort,
  PrsEvaluationResult,
  PrsVectors,
} from './types.js';

export type NttBackendName = 'cpu' | 'metal';

export interface NativeModule {
  evaluatePrsStub(header: Uint32Array, slots: Float64Array): Float64Array;
  evaluatePrs(header: Uint32Array, slots: Float64Array): Float64Array;
  evaluatePrsPipelineNapi?(
    header: Uint32Array,
    slots: Float64Array,
    preferMetal: boolean,
  ): Float64Array;
  validatePrsBatch(header: Uint32Array, slots: Float64Array): number;
  echoPrsGenotypeSlots(
    header: Uint32Array,
    slots: Float64Array,
    count: number,
  ): Float64Array;
  isMetalAvailable?(): boolean;
  /** Same engine entry point as the Tauri `evaluate_clinic_sweep` command. */
  evaluateClinicSweep?(
    patientCount: number,
    diseaseCount: number,
    seed: number,
  ): Float64Array;
  evaluatePrsEncryptedNapi?(
    header: Uint32Array,
    slots: Float64Array,
    preferMetal: boolean,
  ): Float64Array;
  /** Like encrypted, but indices 6–8 are encrypt/eval/decrypt ms before scores. */
  evaluatePrsEncryptedStagedNapi?(
    header: Uint32Array,
    slots: Float64Array,
    preferMetal: boolean,
  ): Float64Array;
  /** Streamed patient-packed ct×ct (zero Galois KS). Staged layout. */
  evaluatePrsPatientPackedNapi?(
    geno: Float64Array,
    weights: Float64Array,
    patients: number,
    snpCount: number,
    polyDegree: number,
    scaleBits: number,
    preferMetal: boolean,
  ): Float64Array;
  evaluatePrsAsyncNapi?(
    header: Uint32Array,
    slots: Float64Array,
    preferMetal: boolean,
  ): Float64Array;
  evaluateAsyncJobSweepNapi?(
    patientCount: number,
    diseaseCount: number,
    snp: number,
    polyDegree: number,
    scaleBits: number,
    nJobs: number,
    preferMetal: boolean,
  ): Float64Array;
  benchmarkCrtRecombineMs?(polyDegree: number): number;
}

export function loadNativeModule(): NativeModule {
  const require = createRequire(import.meta.url);
  const possiblePaths = [
    '../native.node',
    '../native/target/release/libengpe_native.dylib',
    '../engpe-native.darwin-arm64.node',
    '../native/index.node',
  ];
  for (const relPath of possiblePaths) {
    try {
      return require(relPath) as NativeModule;
    } catch {
      // try next
    }
  }
  throw new Error(
    'Native module not found. Run `npm run build:native` to compile the Rust crate.',
  );
}

const HEADER_LEN = 6;

function parseScores(
  out: Float64Array,
  patientCount: number,
  diseaseCount: number,
): PatientPrsScore[] {
  const scores: PatientPrsScore[] = [];
  const pairs = patientCount * diseaseCount;
  for (let i = 0; i < pairs; i++) {
    const base = HEADER_LEN + 3 * i;
    scores.push({
      plaintextScore: out[base]!,
      decodedScore: out[base + 1]!,
      absError: out[base + 2]!,
      patientIndex: Math.floor(i / diseaseCount),
      diseaseIndex: i % diseaseCount,
    });
  }
  return scores;
}

function toMatrix(
  scores: PatientPrsScore[],
  patientCount: number,
  diseaseCount: number,
): PatientPrsScore[][] {
  const matrix: PatientPrsScore[][] = [];
  for (let p = 0; p < patientCount; p++) {
    matrix.push(scores.slice(p * diseaseCount, (p + 1) * diseaseCount));
  }
  return matrix;
}

export function parseNativeResult(out: Float64Array): PrsEvaluationResult {
  const patientCount = out[0]! | 0;
  const diseaseCount = out[1]! | 0;
  const backend: NttBackendName = out[2] === 1.0 ? 'metal' : 'cpu';
  const scores = parseScores(out, patientCount, diseaseCount);
  const primary = scores[0]!;
  return {
    patientCount,
    diseaseCount,
    plaintextScore: primary.plaintextScore,
    decodedScore: primary.decodedScore,
    batchMultiplies: out[4]!,
    cipherCount: out[4]!,
    absError: primary.absError,
    accepted: out[5]! === 1.0,
    backend,
    nttMs: out[3] ?? 0,
    scores,
    matrix: toMatrix(scores, patientCount, diseaseCount),
  };
}

function tsOracleResult(
  oracles: number[],
  patientCount: number,
  diseaseCount: number,
  cipherCount: number,
): PrsEvaluationResult {
  const scores = oracles.map((o, i) => ({
    plaintextScore: o,
    decodedScore: o,
    absError: 0,
    patientIndex: Math.floor(i / diseaseCount),
    diseaseIndex: i % diseaseCount,
  }));
  const primary = scores[0]!;
  return {
    patientCount,
    diseaseCount,
    plaintextScore: primary.plaintextScore,
    decodedScore: primary.decodedScore,
    batchMultiplies: cipherCount,
    cipherCount,
    absError: 0,
    accepted: true,
    backend: 'cpu',
    nttMs: 0,
    scores,
    matrix: toMatrix(scores, patientCount, diseaseCount),
  };
}

function callNative(
  batch: PackedPrsBatch,
  native: NativeModule,
  preferMetal: boolean,
): PrsEvaluationResult {
  if (native.evaluatePrsPipelineNapi) {
    return parseNativeResult(
      native.evaluatePrsPipelineNapi(batch.header, batch.slots, preferMetal),
    );
  }
  const fn = preferMetal ? native.evaluatePrs : native.evaluatePrsStub;
  return parseNativeResult(fn.call(native, batch.header, batch.slots));
}

/** Evaluate a single-patient packed batch (compat). */
export function evaluatePrs(
  vectors: PrsVectors,
  batch: PackedPrsBatch,
  native?: NativeModule | null,
  preferMetal = true,
): PrsEvaluationResult {
  validatePackedPrs(batch);
  if (native) {
    return callNative(batch, native, preferMetal);
  }
  return tsOracleResult(
    [plaintextPrs(vectors)],
    1,
    1,
    batch.meta.cipherCount,
  );
}

/** Evaluate an M×D cohort packed into one FFI call. */
export function evaluatePrsCohort(
  cohort: PrsCohort,
  batch: PackedPrsBatch,
  native?: NativeModule | null,
  preferMetal = true,
): PrsEvaluationResult {
  validatePackedPrs(batch);
  if (batch.meta.patientCount !== cohort.genotypes.length) {
    throw new Error(
      `patientCount ${batch.meta.patientCount} ≠ cohort size ${cohort.genotypes.length}`,
    );
  }
  if (batch.meta.diseaseCount !== cohort.diseaseWeights.length) {
    throw new Error(
      `diseaseCount ${batch.meta.diseaseCount} ≠ disease panels ${cohort.diseaseWeights.length}`,
    );
  }
  if (native) {
    return callNative(batch, native, preferMetal);
  }
  return tsOracleResult(
    plaintextPrsFlat(cohort),
    cohort.genotypes.length,
    cohort.diseaseWeights.length,
    batch.meta.cipherCount,
  );
}
