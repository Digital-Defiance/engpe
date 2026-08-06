/**
 * Peak-memory probe for the encrypted full-panel lane.
 *
 * Reports the resident-set high-water mark for a single fully encrypted
 * 110,000-SNP evaluation, so ENGPE's footprint can be compared against
 * published server-side HE-PRS memory figures. Run under `/usr/bin/time -l`
 * to cross-check the process maximum against Node's own accounting.
 */

import { loadNativeModule, parseNativeResult } from './ffi.js';
import { generateSyntheticCohort, packCohortForFfi } from './prs-data.js';
import { DEFAULT_POLY_DEGREE, SNP_COUNT } from './types.js';

const SCALE = 2 ** 40;

function mib(bytes: number): number {
  return Math.round((bytes / (1024 * 1024)) * 10) / 10;
}

function main(): void {
  const native = loadNativeModule();
  if (!native.evaluatePrsEncryptedNapi) {
    throw new Error('evaluatePrsEncryptedNapi missing — rebuild native');
  }
  const preferMetal = native.isMetalAvailable?.() ?? false;

  const cohort = generateSyntheticCohort({
    snpCount: SNP_COUNT,
    patientCount: 1,
    seed: 0xa11d,
  });
  const batch = packCohortForFfi(cohort, {
    snpCount: SNP_COUNT,
    polyDegree: DEFAULT_POLY_DEGREE,
    scale: SCALE,
  });

  // The first call pays one-time RLWE KeyGen plus 13 Galois evaluation keys at
  // N=16384; later calls reuse the cached key set. Reporting only one of the
  // two would misstate the cost, so both are measured in the same process.
  const before = process.memoryUsage();
  const samples: number[] = [];
  let backend = '';
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    const result = parseNativeResult(
      native.evaluatePrsEncryptedNapi(batch.header, batch.slots, preferMetal),
    );
    samples.push(performance.now() - t0);
    backend = result.backend;
  }
  const after = process.memoryUsage();

  const cold = samples[0]!;
  const warm = samples.slice(1);
  const warmMedian = [...warm].sort((a, b) => a - b)[Math.floor(warm.length / 2)]!;

  console.log(
    JSON.stringify(
      {
        lane: 'encrypted-full-m1-memory',
        preferMetal,
        backend,
        snpCount: SNP_COUNT,
        polyDegree: DEFAULT_POLY_DEGREE,
        patientCount: 1,
        coldMs: cold,
        warmMs: warm,
        warmMedianMs: warmMedian,
        keySetupMs: cold - warmMedian,
        rssBeforeMiB: mib(before.rss),
        rssAfterMiB: mib(after.rss),
        rssDeltaMiB: mib(after.rss - before.rss),
        heapUsedAfterMiB: mib(after.heapUsed),
        externalAfterMiB: mib(after.external),
      },
      null,
      2,
    ),
  );
}

main();
