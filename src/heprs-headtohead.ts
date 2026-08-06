/**
 * Controlled head-to-head against HEPRS on this machine.
 *
 * HEPRS timings (same Mac, same example_data, go run … -pq -print):
 *   PN12QP125   (param 0, N=4096): wall ~0.77 s, HeapSys ~227 MiB
 *   PN13QP202pq (param 1, N=8192): wall ~1.43 s, HeapSys ~403 MiB
 *
 * This script runs ENGPE's ct×ct encrypted path on the identical MIT example
 * (10,001 × 50) at N=8192 and prints a comparison block for the paper.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadNativeModule, parseNativeResult } from './ffi.js';
import { packCohortForFfi } from './prs-data.js';
import type { PrsCohort } from './types.js';

const DATA_DIR = join(process.cwd(), 'heprs-validation');
const SCALE = 2 ** 40;
const POLY_DEGREE = 8192;

function readCsvMatrix(name: string): number[][] {
  return readFileSync(join(DATA_DIR, name), 'utf8')
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => line.split(',').map(Number));
}

function mib(bytes: number): number {
  return Math.round((bytes / (1024 * 1024)) * 10) / 10;
}

function main(): void {
  const genotypes = readCsvMatrix('genotype_10kSNP_50individual.csv');
  const beta = readCsvMatrix('beta_10kSNP_phenotype0.csv')[0]!;
  const patients = genotypes.length;
  const snpCount = beta.length;

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

  const before = process.memoryUsage();
  const samples: number[] = [];
  let backend = '';
  let maxAbsError = 0;
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    const result = parseNativeResult(
      native.evaluatePrsEncryptedNapi(batch.header, batch.slots, preferMetal),
    );
    samples.push(performance.now() - t0);
    backend = result.backend;
    maxAbsError = Math.max(...result.scores.map((s) => s.absError), 0);
  }
  const after = process.memoryUsage();
  const cold = samples[0]!;
  const warm = [...samples.slice(1)].sort((a, b) => a - b);
  const warmMedian = warm[Math.floor(warm.length / 2)]!;

  console.log(
    JSON.stringify(
      {
        comparison: 'HEPRS example_data head-to-head (this machine)',
        panel: { patients, features: snpCount, polyDegree: POLY_DEGREE },
        engpe: {
          weights: 'encrypted (ct×ct + relin)',
          backend,
          preferMetal,
          coldMs: cold,
          warmMedianMs: warmMedian,
          msPerPatientWarm: warmMedian / patients,
          maxAbsErrorVsPlainOracle: maxAbsError,
          rssAfterMiB: mib(after.rss),
          rssDeltaMiB: mib(after.rss - before.rss),
        },
        heprsLocal: {
          PN12QP125: { wallMs: 770, heapSysMiB: 227, note: 'param 0, N=4096; this M4 Max' },
          PN13QP202pq: { wallMs: 1410, heapSysMiB: 387, note: 'param 1, N=8192; this M4 Max' },
        },
        bgpucapSignal: {
          engpe: { realS: 18.2, cpuAvgPct: 55.3, cpuPeakPct: 100, gpuAvgPct: 0, note: 'includes 3 warm reps + npx' },
          heprsPn13: { realS: 1.83, cpuAvgPct: 21.0, cpuPeakPct: 33.1, gpuAvgPct: 0 },
        },
        paramGap: {
          engpeLogQ: 124,
          heprsLogQP_PN13: 202,
          matched: 'N=8192 ring dimension + ct×ct + hoisted InnerSumLog',
          unmatched: '|Q|≈124 vs log2(QP)=202 (ENGPE CRT product lives in u128)',
        },
      },
      null,
      2,
    ),
  );
}

main();
