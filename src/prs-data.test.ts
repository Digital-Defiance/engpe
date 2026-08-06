import { describe, it, expect } from 'vitest';
import {
  SNP_COUNT,
  DEFAULT_POLY_DEGREE,
  PRS_HEADER_FIELD_COUNT,
  ckksSlotCount,
  cipherCountFor,
} from './types.js';
import {
  DEFAULT_CKKS_SCALE,
  generateSyntheticCohort,
  generateSyntheticPrs,
  packCohortForFfi,
  packPrsForFfi,
  plaintextPrs,
  plaintextPrsCohort,
  plaintextPrsMatrix,
  validatePackedPrs,
} from './prs-data.js';
import { evaluatePrs, evaluatePrsCohort } from './ffi.js';
import { runSyntheticPrs } from './index.js';

describe('CKKS slot / cipher sizing', () => {
  it('computes slots as N/2', () => {
    expect(ckksSlotCount(16_384)).toBe(8192);
    expect(ckksSlotCount(1024)).toBe(512);
  });

  it('needs 14 ciphertexts for 110k SNPs at N=16384', () => {
    expect(cipherCountFor(SNP_COUNT, DEFAULT_POLY_DEGREE)).toBe(14);
  });

  it('rejects non-power-of-two degrees', () => {
    expect(() => cipherCountFor(100, 1000)).toThrow(/power of 2/);
  });
});

describe('generateSyntheticPrs', () => {
  it('produces a 110k panel with dosages in {0,1,2}', () => {
    const v = generateSyntheticPrs({ seed: 1 });
    expect(v.patientGenotype.length).toBe(SNP_COUNT);
    expect(v.medicalWeights.length).toBe(SNP_COUNT);
    for (let i = 0; i < 1000; i++) {
      expect([0, 1, 2]).toContain(v.patientGenotype[i]);
    }
  });

  it('is deterministic for a fixed seed', () => {
    const a = generateSyntheticPrs({ snpCount: 1000, seed: 42 });
    const b = generateSyntheticPrs({ snpCount: 1000, seed: 42 });
    expect(Array.from(a.patientGenotype)).toEqual(Array.from(b.patientGenotype));
    expect(Array.from(a.medicalWeights)).toEqual(Array.from(b.medicalWeights));
  });

  it('diverges across seeds', () => {
    const a = generateSyntheticPrs({ snpCount: 100, seed: 1 });
    const b = generateSyntheticPrs({ snpCount: 100, seed: 2 });
    expect(Array.from(a.patientGenotype)).not.toEqual(Array.from(b.patientGenotype));
  });
});

describe('plaintextPrs', () => {
  it('computes the dot product', () => {
    const vectors = {
      patientGenotype: Uint8Array.from([0, 1, 2, 1]),
      medicalWeights: Float64Array.from([0.5, -0.25, 0.1, 0.0]),
    };
    expect(plaintextPrs(vectors)).toBeCloseTo(-0.05, 12);
  });
});

describe('packPrsForFfi', () => {
  it('packs the full 110k panel into 14×8192 slot chunks', () => {
    const vectors = generateSyntheticPrs({ seed: 7 });
    const batch = packPrsForFfi(vectors);
    expect(batch.header.length).toBe(PRS_HEADER_FIELD_COUNT);
    expect(batch.meta.cipherCount).toBe(14);
    expect(batch.meta.slotCount).toBe(8192);
    expect(batch.meta.patientCount).toBe(1);
    expect(batch.slots.length).toBe(2 * 14 * 8192);
    expect(batch.meta.scale).toBe(DEFAULT_CKKS_SCALE);
    validatePackedPrs(batch);
  });

  it('packs M patients + shared weights', () => {
    const m = 4;
    const snpCount = 20;
    const polyDegree = 64;
    const cohort = generateSyntheticCohort({ snpCount, patientCount: m, seed: 5 });
    const batch = packCohortForFfi(cohort, { snpCount, polyDegree });
    expect(batch.meta.patientCount).toBe(m);
    expect(batch.meta.diseaseCount).toBe(1);
    const perSide = batch.meta.cipherCount * batch.meta.slotCount;
    expect(batch.slots.length).toBe((m + 1) * perSide);
    for (let p = 0; p < m; p++) {
      for (let i = 0; i < snpCount; i++) {
        expect(batch.slots[p * perSide + i]).toBe(cohort.genotypes[p]![i]);
      }
    }
    for (let i = 0; i < snpCount; i++) {
      expect(batch.slots[m * perSide + i]).toBe(cohort.medicalWeights[i]);
    }
    validatePackedPrs(batch);
  });

  it('packs M patients × D disease weight panels', () => {
    const m = 2;
    const d = 5;
    const snpCount = 20;
    const polyDegree = 64;
    const cohort = generateSyntheticCohort({
      snpCount,
      patientCount: m,
      diseaseCount: d,
      seed: 8,
    });
    const batch = packCohortForFfi(cohort, { snpCount, polyDegree });
    expect(batch.meta.patientCount).toBe(m);
    expect(batch.meta.diseaseCount).toBe(d);
    const perSide = batch.meta.cipherCount * batch.meta.slotCount;
    expect(batch.slots.length).toBe((m + d) * perSide);
    const matrix = plaintextPrsMatrix(cohort);
    expect(matrix).toHaveLength(m);
    expect(matrix[0]).toHaveLength(d);
    validatePackedPrs(batch);
  });

  it('zero-pads the trailing slots of the last ciphertext', () => {
    const snpCount = 10;
    const polyDegree = 16; // slots = 8 → 2 ciphers
    const vectors = generateSyntheticPrs({ snpCount, seed: 3 });
    const batch = packPrsForFfi(vectors, { snpCount, polyDegree });
    expect(batch.meta.cipherCount).toBe(2);
    const perSide = 2 * 8;
    // Last 6 genotype slots (indices 10..15) must be padding zeros
    for (let i = snpCount; i < perSide; i++) {
      expect(batch.slots[i]).toBe(0);
      expect(batch.slots[perSide + i]).toBe(0);
    }
  });

  it('preserves genotype and weight values in slot order', () => {
    const vectors = generateSyntheticPrs({ snpCount: 20, seed: 9 });
    const batch = packPrsForFfi(vectors, { snpCount: 20, polyDegree: 64 });
    const perSide = batch.meta.cipherCount * batch.meta.slotCount;
    for (let i = 0; i < 20; i++) {
      expect(batch.slots[i]).toBe(vectors.patientGenotype[i]);
      expect(batch.slots[perSide + i]).toBe(vectors.medicalWeights[i]);
    }
  });
});

describe('evaluatePrs (TS oracle path)', () => {
  it('returns the plaintext oracle and cipher batch size', () => {
    const vectors = generateSyntheticPrs({ snpCount: 500, seed: 11 });
    const batch = packPrsForFfi(vectors, { snpCount: 500, polyDegree: 1024 });
    const result = evaluatePrs(vectors, batch, null);
    expect(result.accepted).toBe(true);
    expect(result.patientCount).toBe(1);
    expect(result.cipherCount).toBe(batch.meta.cipherCount);
    expect(result.batchMultiplies).toBe(batch.meta.cipherCount);
    expect(result.plaintextScore).toBeCloseTo(plaintextPrs(vectors), 10);
    expect(result.absError).toBe(0);
    expect(result.scores).toHaveLength(1);
  });

  it('returns M plaintext oracles for a cohort', () => {
    const cohort = generateSyntheticCohort({
      snpCount: 50,
      patientCount: 3,
      seed: 12,
    });
    const batch = packCohortForFfi(cohort, { snpCount: 50, polyDegree: 128 });
    const result = evaluatePrsCohort(cohort, batch, null);
    const oracles = plaintextPrsCohort(cohort);
    expect(result.patientCount).toBe(3);
    expect(result.scores).toHaveLength(3);
    for (let i = 0; i < 3; i++) {
      expect(result.scores[i]!.plaintextScore).toBeCloseTo(oracles[i]!, 10);
      expect(result.scores[i]!.absError).toBe(0);
    }
  });
});

describe('evaluatePrs (native CPU CKKS)', () => {
  it('decoded score matches plaintext oracle within tolerance', async () => {
    let native;
    try {
      const { loadNativeModule } = await import('./ffi.js');
      native = loadNativeModule();
    } catch {
      return; // native addon not built — skip
    }
    const snpCount = 100;
    const polyDegree = 128;
    const vectors = generateSyntheticPrs({ snpCount, seed: 21 });
    const batch = packPrsForFfi(vectors, {
      snpCount,
      polyDegree,
      scale: 2 ** 14,
    });
    const result = evaluatePrs(vectors, batch, native, false);
    expect(result.accepted).toBe(true);
    expect(result.patientCount).toBe(1);
    expect(result.plaintextScore).toBeCloseTo(plaintextPrs(vectors), 10);
    expect(result.absError).toBeLessThan(1e-2);
    expect(result.decodedScore).toBeCloseTo(result.plaintextScore, 1);
    expect(result.backend).toBe('cpu');
  });

  it('cohort scores match plaintext oracles', async () => {
    let native;
    try {
      const { loadNativeModule } = await import('./ffi.js');
      native = loadNativeModule();
    } catch {
      return;
    }
    const snpCount = 80;
    const polyDegree = 128;
    const patientCount = 4;
    const cohort = generateSyntheticCohort({ snpCount, patientCount, seed: 33 });
    const batch = packCohortForFfi(cohort, {
      snpCount,
      polyDegree,
      scale: 2 ** 14,
    });
    const result = evaluatePrsCohort(cohort, batch, native, false);
    const oracles = plaintextPrsCohort(cohort);
    expect(result.accepted).toBe(true);
    expect(result.backend).toBe('cpu');
    expect(result.patientCount).toBe(patientCount);
    for (let i = 0; i < patientCount; i++) {
      expect(result.scores[i]!.plaintextScore).toBeCloseTo(oracles[i]!, 10);
      expect(result.scores[i]!.absError).toBeLessThan(1e-2);
    }
  });
  it('CPU RNS at 2^40 matches oracle within 1e-4', async () => {
    let native;
    try {
      const { loadNativeModule } = await import('./ffi.js');
      native = loadNativeModule();
    } catch {
      return;
    }
    const snpCount = 512;
    const polyDegree = 1024;
    const vectors = generateSyntheticPrs({
      snpCount,
      seed: 55,
      weightBound: 0.02,
    });
    const batch = packPrsForFfi(vectors, {
      snpCount,
      polyDegree,
      scale: 2 ** 40,
    });
    const result = evaluatePrs(vectors, batch, native, false);
    expect(result.accepted).toBe(true);
    expect(result.backend).toBe('cpu');
    expect(result.absError).toBeLessThan(1e-4);
  });
});

describe('evaluatePrs (native Metal RNS CKKS)', () => {
  it('decoded score matches oracle within 1e-4 at N=16384', async () => {
    let native;
    try {
      const { loadNativeModule } = await import('./ffi.js');
      native = loadNativeModule();
    } catch {
      return;
    }
    if (!native.isMetalAvailable?.()) {
      return;
    }
    const snpCount = 2048;
    const polyDegree = 16384;
    const vectors = generateSyntheticPrs({ snpCount, seed: 42, weightBound: 0.02 });
    const batch = packPrsForFfi(vectors, {
      snpCount,
      polyDegree,
      scale: 2 ** 40,
    });
    const result = evaluatePrs(vectors, batch, native, true);
    expect(result.accepted).toBe(true);
    expect(result.backend).toBe('metal');
    expect(result.absError).toBeLessThan(1e-4);
  });

  it('cohort Metal RNS matches M oracles within 1e-4', async () => {
    let native;
    try {
      const { loadNativeModule } = await import('./ffi.js');
      native = loadNativeModule();
    } catch {
      return;
    }
    if (!native.isMetalAvailable?.()) {
      return;
    }
    const snpCount = 512;
    const polyDegree = 1024;
    const patientCount = 4;
    const cohort = generateSyntheticCohort({
      snpCount,
      patientCount,
      seed: 44,
      weightBound: 0.02,
    });
    const batch = packCohortForFfi(cohort, {
      snpCount,
      polyDegree,
      scale: 2 ** 40,
    });
    const result = evaluatePrsCohort(cohort, batch, native, true);
    expect(result.accepted).toBe(true);
    expect(result.backend).toBe('metal');
    expect(result.patientCount).toBe(patientCount);
    for (const s of result.scores) {
      expect(s.absError).toBeLessThan(1e-4);
    }
  });

  /** VALIDATION GATE (Task 7): every (patient, disease) within ε < 1e-4. */
  it('multi-disease Metal matrix matches 2D oracle within 1e-4', async () => {
    let native;
    try {
      const { loadNativeModule } = await import('./ffi.js');
      native = loadNativeModule();
    } catch {
      return;
    }
    if (!native.isMetalAvailable?.()) {
      return;
    }
    const snpCount = 512;
    const polyDegree = 1024;
    const patientCount = 2;
    const diseaseCount = 5;
    const cohort = generateSyntheticCohort({
      snpCount,
      patientCount,
      diseaseCount,
      seed: 77,
      weightBound: 0.02,
    });
    const batch = packCohortForFfi(cohort, {
      snpCount,
      polyDegree,
      scale: 2 ** 40,
    });
    const result = evaluatePrsCohort(cohort, batch, native, true);
    const oracle = plaintextPrsMatrix(cohort);
    expect(result.accepted).toBe(true);
    expect(result.backend).toBe('metal');
    expect(result.patientCount).toBe(patientCount);
    expect(result.diseaseCount).toBe(diseaseCount);
    expect(result.matrix).toHaveLength(patientCount);
    for (let p = 0; p < patientCount; p++) {
      for (let d = 0; d < diseaseCount; d++) {
        const cell = result.matrix[p]![d]!;
        expect(cell.plaintextScore).toBeCloseTo(oracle[p]![d]!, 10);
        expect(cell.absError).toBeLessThan(1e-4);
      }
    }
  });
});

describe('runSyntheticPrs', () => {
  it('runs end-to-end in tsOnly mode for the full panel', () => {
    const { batch, result, oracle } = runSyntheticPrs({ tsOnly: true, seed: 99 });
    expect(batch.meta.snpCount).toBe(SNP_COUNT);
    expect(batch.meta.cipherCount).toBe(14);
    expect(result.accepted).toBe(true);
    expect(result.plaintextScore).toBeCloseTo(oracle, 10);
  });
});
