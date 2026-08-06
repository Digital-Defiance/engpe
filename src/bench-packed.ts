/**
 * Streamed patient-packed memory proof (synthetic genotypes inside Rust).
 *
 * Runs two SNP counts at fixed M to show RSS does not grow with panel size.
 * Usage: npx tsx src/bench-packed.ts
 */

import { loadNativeModule } from './ffi.js';

function mib(b: number): number {
  return Math.round((b / (1024 * 1024)) * 10) / 10;
}

interface Native {
  isMetalAvailable?(): boolean;
  benchPatientPackedSyntheticNapi?(
    patients: number,
    snpCount: number,
    polyDegree: number,
    scaleBits: number,
    preferMetal: boolean,
  ): Float64Array;
}

function run(
  native: Native,
  patients: number,
  snpCount: number,
  polyDegree: number,
  preferMetal: boolean,
) {
  const before = process.memoryUsage();
  const t0 = performance.now();
  const out = native.benchPatientPackedSyntheticNapi!(
    patients,
    snpCount,
    polyDegree,
    40,
    preferMetal,
  );
  const wallMs = performance.now() - t0;
  const after = process.memoryUsage();
  let maxErr = 0;
  for (let i = 0; i < patients; i++) {
    maxErr = Math.max(maxErr, out[9 + 3 * i + 2]!);
  }
  return {
    patients,
    snpCount,
    polyDegree,
    wallMs,
    msPerPatient: wallMs / patients,
    encryptMs: out[6],
    evalMs: out[7],
    decryptMs: out[8],
    ciphertexts: out[4],
    maxAbsError: maxErr,
    rssAfterMiB: mib(after.rss),
    rssDeltaMiB: mib(after.rss - before.rss),
  };
}

function main(): void {
  const native = loadNativeModule() as Native;
  if (!native.benchPatientPackedSyntheticNapi) {
    throw new Error('benchPatientPackedSyntheticNapi missing — rebuild native');
  }
  const preferMetal = native.isMetalAvailable?.() ?? false;
  const patients = 1146;
  const polyDegree = 8192;
  // Two panel sizes: RSS should stay in the same ballpark if streaming works.
  const rows = [512, 2000].map((snp) =>
    run(native, patients, snp, polyDegree, preferMetal),
  );
  console.log(
    JSON.stringify(
      {
        mode: 'patient-packed-synthetic-stream',
        preferMetal,
        note: 'Genotypes generated inside Rust; only accumulator + 2 live CTs.',
        rows,
        rssGrewWithSnps: rows[1]!.rssAfterMiB - rows[0]!.rssAfterMiB,
      },
      null,
      2,
    ),
  );
}

main();
