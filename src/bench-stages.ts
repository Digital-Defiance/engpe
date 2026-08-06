/**
 * HEPRS-style stage split on the SNP-packed encrypted path.
 * Usage: npx tsx src/bench-stages.ts
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadNativeModule } from './ffi.js';
import { packCohortForFfi } from './prs-data.js';
import type { PrsCohort } from './types.js';

const DATA_DIR = join(process.cwd(), 'heprs-validation');
const SCALE = 2 ** 40;
const POLY = 8192;

function readCsvMatrix(name: string): number[][] {
  return readFileSync(join(DATA_DIR, name), 'utf8')
    .split('\n')
    .filter((l) => l.trim().length > 0)
    .map((l) => l.split(',').map(Number));
}

function main(): void {
  const genotypes = readCsvMatrix('genotype_10kSNP_50individual.csv');
  const beta = readCsvMatrix('beta_10kSNP_phenotype0.csv')[0]!;
  const cohort: PrsCohort = {
    genotypes: genotypes.map((g) => Uint8Array.from(g)),
    diseaseWeights: [Float64Array.from(beta)],
    medicalWeights: Float64Array.from(beta),
  };
  const native = loadNativeModule() as {
    isMetalAvailable?(): boolean;
    evaluatePrsEncryptedStagedNapi?(
      h: Uint32Array,
      s: Float64Array,
      m: boolean,
    ): Float64Array;
  };
  if (!native.evaluatePrsEncryptedStagedNapi) {
    throw new Error('evaluatePrsEncryptedStagedNapi missing — rebuild native');
  }
  const preferMetal = native.isMetalAvailable?.() ?? false;
  const batch = packCohortForFfi(cohort, {
    snpCount: beta.length,
    polyDegree: POLY,
    scale: SCALE,
  });

  // Cold + warm
  const samples = [];
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    const out = native.evaluatePrsEncryptedStagedNapi(
      batch.header,
      batch.slots,
      preferMetal,
    );
    samples.push({
      wallMs: performance.now() - t0,
      encryptMs: out[6],
      evalMs: out[7],
      decryptMs: out[8],
      backend: out[2] === 1 ? 'metal' : 'cpu',
    });
  }
  console.log(
    JSON.stringify(
      {
        panel: 'HEPRS example 10k×50',
        polyDegree: POLY,
        preferMetal,
        samples,
        warm: samples[2],
      },
      null,
      2,
    ),
  );
}

main();
