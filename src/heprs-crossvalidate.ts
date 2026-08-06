/**
 * Cross-implementation validation against the published HEPRS example dataset.
 *
 * Knight et al. [1] released a synthetic HAPGEN2 cohort (10,001 features ×
 * 50 individuals), the ridge-regression weights, and — critically — their own
 * plaintext PRS predictions. That last file makes it possible to check `ENGPE`
 * against an *independent* implementation rather than only against its own
 * internal oracle, which is the strongest correctness evidence available
 * without controlled-access clinical data.
 *
 * Three quantities are compared per individual:
 *   1. `engpeEncrypted`  — ENGPE's RLWE result on their genotypes and weights.
 *   2. `plaintextDot`    — the dot product recomputed here in double precision.
 *   3. `heprsPublished`  — their released prediction, minus the model intercept.
 *
 * (1) vs (2) measures FHE fidelity. (2) vs (3) measures agreement between two
 * independent PRS implementations, and is bounded below by the 6-decimal
 * rounding of their released weight file.
 *
 * [1] Knight et al., Cell Reports Methods 6:101271 (2026);
 *     github.com/gersteinlab/HEPRS, MIT licensed.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadNativeModule, parseNativeResult } from './ffi.js';
import { packCohortForFfi } from './prs-data.js';
import type { PrsCohort } from './types.js';

const DATA_DIR = join(process.cwd(), 'heprs-validation');
const SCALE = 2 ** 40;
/**
 * N=8192 matches the ring dimension HEPRS used (log2 N = 13) and is the
 * smallest degree ENGPE permits for encrypted evaluation.
 */
const POLY_DEGREE = 8192;

function readCsvMatrix(name: string): number[][] {
  return readFileSync(join(DATA_DIR, name), 'utf8')
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => line.split(',').map(Number));
}

function readCsvColumn(name: string): number[] {
  return readFileSync(join(DATA_DIR, name), 'utf8')
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map(Number);
}

function mean(xs: readonly number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function pearson(a: readonly number[], b: readonly number[]): number {
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]! - ma;
    const y = b[i]! - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  return num / Math.sqrt(da * db);
}

function main(): void {
  const genotypes = readCsvMatrix('genotype_10kSNP_50individual.csv');
  const beta = readCsvMatrix('beta_10kSNP_phenotype0.csv')[0]!;
  const published = readCsvColumn('phenotype0_pred_10kSNP_50individual.csv');

  const patients = genotypes.length;
  const snpCount = beta.length;
  if (genotypes.some((g) => g.length !== snpCount)) {
    throw new Error('genotype/beta width mismatch');
  }

  // Their released predictions carry a ridge intercept that is not part of the
  // encrypted dot product and is not present in the distributed files.
  const dots = genotypes.map((g) => g.reduce((acc, v, i) => acc + v * beta[i]!, 0));
  const intercept = mean(published.map((p, i) => p - dots[i]!));

  const cohort: PrsCohort = {
    genotypes: genotypes.map((g) => Uint8Array.from(g)),
    diseaseWeights: [Float64Array.from(beta)],
    medicalWeights: Float64Array.from(beta),
  };

  const native = loadNativeModule();
  if (!native.evaluatePrsEncryptedNapi) {
    throw new Error('evaluatePrsEncryptedNapi missing — rebuild native');
  }
  const preferMetal = native.isMetalAvailable?.() ?? false;
  const batch = packCohortForFfi(cohort, {
    snpCount,
    polyDegree: POLY_DEGREE,
    scale: SCALE,
  });

  const t0 = performance.now();
  const result = parseNativeResult(
    native.evaluatePrsEncryptedNapi(batch.header, batch.slots, preferMetal),
  );
  const elapsedMs = performance.now() - t0;

  const encrypted = result.scores.map((s) => s.decodedScore);

  const errVsDot = encrypted.map((e, i) => Math.abs(e - dots[i]!));
  const errVsPublished = encrypted.map((e, i) =>
    Math.abs(e + intercept - published[i]!),
  );
  const dotVsPublished = dots.map((d, i) => Math.abs(d + intercept - published[i]!));

  console.log(
    JSON.stringify(
      {
        source: 'gersteinlab/HEPRS example_data (MIT)',
        patients,
        features: snpCount,
        polyDegree: POLY_DEGREE,
        backend: result.backend,
        preferMetal,
        totalMs: elapsedMs,
        msPerPatient: elapsedMs / patients,
        interceptRecovered: intercept,
        // FHE fidelity: encrypted vs double-precision dot product.
        maxAbsErrEncryptedVsPlaintextDot: Math.max(...errVsDot),
        meanAbsErrEncryptedVsPlaintextDot: mean(errVsDot),
        // Cross-implementation: ENGPE encrypted vs HEPRS published prediction.
        maxAbsErrEncryptedVsHeprsPublished: Math.max(...errVsPublished),
        pearsonEncryptedVsHeprsPublished: pearson(encrypted, published),
        // Floor imposed by their 6-decimal weight/prediction rounding.
        maxAbsErrPlaintextDotVsHeprsPublished: Math.max(...dotVsPublished),
        sample: encrypted.slice(0, 5).map((e, i) => ({
          engpeEncrypted: e + intercept,
          heprsPublished: published[i],
        })),
      },
      null,
      2,
    ),
  );
}

main();
