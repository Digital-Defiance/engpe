/**
 * Multi-patient encrypted cohort timing (HEPRS-scale M≈1146).
 *
 * Full 110k × 1146 at ~2.6 s/patient is ~50 min; this script measures:
 *   A) 110k SNPs at several M, warm amortized rate → project 1146
 *   B) optional full 1146 × 10k (example-scale panel) wall-clock
 *
 * Usage: npx tsx src/bench-cohort.ts [mode]
 *   mode = rate (default) | example1146 | full1146
 */

import { loadNativeModule, parseNativeResult } from './ffi.js';
import { generateSyntheticCohort, packCohortForFfi } from './prs-data.js';
import { DEFAULT_POLY_DEGREE, SNP_COUNT } from './types.js';

const SCALE = 2 ** 40;
const HEPRS_COHORT = 1146;

function mib(bytes: number): number {
  return Math.round((bytes / (1024 * 1024)) * 10) / 10;
}

function runEncrypted(
  native: ReturnType<typeof loadNativeModule>,
  snpCount: number,
  patientCount: number,
  polyDegree: number,
  preferMetal: boolean,
  warmReps: number,
): {
  coldMs: number;
  warmMedianMs: number;
  msPerPatient: number;
  maxAbsError: number;
  rssAfterMiB: number;
  backend: string;
} {
  const cohort = generateSyntheticCohort({
    snpCount,
    patientCount,
    seed: 0xc0ffee ^ patientCount,
  });
  const batch = packCohortForFfi(cohort, { snpCount, polyDegree, scale: SCALE });
  const samples: number[] = [];
  let backend = '';
  let maxAbsError = 0;
  const before = process.memoryUsage();
  for (let i = 0; i < 1 + warmReps; i++) {
    const t0 = performance.now();
    const result = parseNativeResult(
      native.evaluatePrsEncryptedNapi!(batch.header, batch.slots, preferMetal),
    );
    samples.push(performance.now() - t0);
    backend = result.backend;
    maxAbsError = Math.max(...result.scores.map((s) => s.absError), 0);
  }
  const after = process.memoryUsage();
  const cold = samples[0]!;
  const warm = samples.slice(1).sort((a, b) => a - b);
  const warmMedian =
    warm.length === 0
      ? cold
      : warm[Math.floor(warm.length / 2)]!;
  return {
    coldMs: cold,
    warmMedianMs: warmMedian,
    msPerPatient: warmMedian / patientCount,
    maxAbsError,
    rssAfterMiB: mib(after.rss),
    backend,
  };
}

function main(): void {
  const mode = process.argv[2] ?? 'rate';
  const native = loadNativeModule();
  if (!native.evaluatePrsEncryptedNapi) {
    throw new Error('evaluatePrsEncryptedNapi missing');
  }
  const preferMetal = native.isMetalAvailable?.() ?? false;

  if (mode === 'rate') {
    const rows = [];
    for (const m of [1, 4, 16] as const) {
      const r = runEncrypted(
        native,
        SNP_COUNT,
        m,
        DEFAULT_POLY_DEGREE,
        preferMetal,
        m === 1 ? 2 : 1,
      );
      rows.push({ patients: m, snpCount: SNP_COUNT, polyDegree: DEFAULT_POLY_DEGREE, ...r });
    }
    const rate = rows[rows.length - 1]!.msPerPatient;
    console.log(
      JSON.stringify(
        {
          mode: 'rate',
          preferMetal,
          rows,
          project1146: {
            patients: HEPRS_COHORT,
            snpCount: SNP_COUNT,
            projectedWarmMs: rate * HEPRS_COHORT,
            projectedWarmMin: (rate * HEPRS_COHORT) / 60_000,
            basis: `amortized ms/patient from M=${rows[rows.length - 1]!.patients}`,
          },
        },
        null,
        2,
      ),
    );
    return;
  }

  if (mode === 'example1146') {
    const r = runEncrypted(native, 10_001, HEPRS_COHORT, 8192, preferMetal, 0);
    console.log(
      JSON.stringify(
        {
          mode: 'example1146',
          note: '1146 patients × 10k SNPs at N=8192 (example-scale panel, HEPRS cohort size)',
          preferMetal,
          ...r,
          totalMin: r.coldMs / 60_000,
        },
        null,
        2,
      ),
    );
    return;
  }

  if (mode === 'full1146') {
    const r = runEncrypted(native, SNP_COUNT, HEPRS_COHORT, DEFAULT_POLY_DEGREE, preferMetal, 0);
    console.log(
      JSON.stringify(
        {
          mode: 'full1146',
          note: 'full 110k × 1146 — long run',
          preferMetal,
          ...r,
          totalMin: r.coldMs / 60_000,
        },
        null,
        2,
      ),
    );
    return;
  }

  throw new Error(`unknown mode ${mode}`);
}

main();
