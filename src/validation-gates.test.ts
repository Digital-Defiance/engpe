/**
 * Task 10 validation gates (TypeScript side).
 *
 * `plaintextPrsMatrix` is the sole ground truth. Nothing here compares one FHE
 * backend against another; every assertion is against the plaintext oracle.
 * Environment-dependent gates use `skipIf` so a missing GPU is reported as a
 * skip rather than silently passing.
 */

import { describe, it, expect } from 'vitest';
import {
  generateSyntheticCohort,
  packCohortForFfi,
  plaintextPrsFlat,
  plaintextPrsMatrix,
} from './prs-data.js';
import { evaluatePrsCohort, loadNativeModule } from './ffi.js';
import type { NativeModule } from './ffi.js';
import type { PrsCohort } from './types.js';

/** Per-coordinate tolerance mandated by the directive. */
const EPSILON = 1e-4;

const PATIENT_COUNT = 4;
const DISEASE_COUNT = 5;
const SNP_COUNT = 512;
const POLY_DEGREE = 1024;
const SCALE = 2 ** 40;
const WEIGHT_BOUND = 0.02;
const SEED = 0xc0ffee;

let native: NativeModule | null = null;
try {
  native = loadNativeModule();
} catch {
  native = null;
}
const hasNative = native !== null;
const hasMetal = hasNative && native!.isMetalAvailable?.() === true;

function seededCohort(seed = SEED): PrsCohort {
  return generateSyntheticCohort({
    snpCount: SNP_COUNT,
    patientCount: PATIENT_COUNT,
    diseaseCount: DISEASE_COUNT,
    weightBound: WEIGHT_BOUND,
    seed,
  });
}

function packSeeded(cohort: PrsCohort) {
  return packCohortForFfi(cohort, {
    snpCount: SNP_COUNT,
    polyDegree: POLY_DEGREE,
    scale: SCALE,
  });
}

/** Assert every (p, d) coordinate of an FHE run is within EPSILON of the oracle. */
function expectMatrixWithinEpsilon(cohort: PrsCohort, preferMetal: boolean): void {
  const batch = packSeeded(cohort);
  const result = evaluatePrsCohort(cohort, batch, native, preferMetal);
  const oracle = plaintextPrsMatrix(cohort);

  expect(result.accepted).toBe(true);
  expect(result.backend).toBe(preferMetal ? 'metal' : 'cpu');
  expect(result.patientCount).toBe(PATIENT_COUNT);
  expect(result.diseaseCount).toBe(DISEASE_COUNT);
  expect(result.matrix).toHaveLength(PATIENT_COUNT);

  for (let p = 0; p < PATIENT_COUNT; p++) {
    expect(result.matrix[p]).toHaveLength(DISEASE_COUNT);
    for (let d = 0; d < DISEASE_COUNT; d++) {
      const cell = result.matrix[p]![d]!;
      const want = oracle[p]![d]!;
      expect(cell.patientIndex).toBe(p);
      expect(cell.diseaseIndex).toBe(d);
      // The engine's own oracle must agree with the TS ground truth first,
      // otherwise absError is measured against the wrong target.
      expect(cell.plaintextScore).toBeCloseTo(want, 10);
      expect(Math.abs(cell.decodedScore - want)).toBeLessThan(EPSILON);
    }
  }
}

describe('deterministic seeded cohort generator', () => {
  it('reproduces M patients and D disease panels for a fixed seed', () => {
    const a = seededCohort();
    const b = seededCohort();
    expect(a.genotypes).toHaveLength(PATIENT_COUNT);
    expect(a.diseaseWeights).toHaveLength(DISEASE_COUNT);
    for (let p = 0; p < PATIENT_COUNT; p++) {
      expect(Array.from(a.genotypes[p]!)).toEqual(Array.from(b.genotypes[p]!));
    }
    for (let d = 0; d < DISEASE_COUNT; d++) {
      expect(Array.from(a.diseaseWeights[d]!)).toEqual(
        Array.from(b.diseaseWeights[d]!),
      );
    }
    expect(plaintextPrsFlat(a)).toEqual(plaintextPrsFlat(b));
  });

  it('diverges across seeds', () => {
    expect(plaintextPrsFlat(seededCohort(SEED))).not.toEqual(
      plaintextPrsFlat(seededCohort(SEED + 1)),
    );
  });

  it('gives each disease a distinct weight panel', () => {
    const { diseaseWeights } = seededCohort();
    for (let d = 1; d < DISEASE_COUNT; d++) {
      expect(Array.from(diseaseWeights[d]!)).not.toEqual(
        Array.from(diseaseWeights[0]!),
      );
    }
  });

  it('produces a full M×D oracle matrix', () => {
    const matrix = plaintextPrsMatrix(seededCohort());
    expect(matrix).toHaveLength(PATIENT_COUNT);
    for (const row of matrix) {
      expect(row).toHaveLength(DISEASE_COUNT);
      for (const cell of row) {
        expect(Number.isFinite(cell)).toBe(true);
      }
    }
    // A matrix of zeros would make the precision gate vacuous.
    expect(matrix.flat().some((v) => Math.abs(v) > 1e-6)).toBe(true);
  });
});

describe('GATE 1: multi-disease SIMD precision (4×5)', () => {
  it.skipIf(!hasNative)('CPU RNS matrix matches the oracle within 1e-4', () => {
    expectMatrixWithinEpsilon(seededCohort(), false);
  });

  it.skipIf(!hasMetal)('Metal RNS matrix matches the oracle within 1e-4', () => {
    expectMatrixWithinEpsilon(seededCohort(), true);
  });

  it.skipIf(!hasMetal)('holds across independent seeds', () => {
    for (const seed of [1, 2, 3]) {
      expectMatrixWithinEpsilon(seededCohort(seed), true);
    }
  });
});
