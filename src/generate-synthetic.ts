/**
 * CLI: emit a synthetic 110k-SNP panel and report packing stats.
 * Does not write binary payloads by default — use for smoke checks.
 */

import { generateSyntheticPrs, packPrsForFfi, plaintextPrs } from './prs-data.js';
import { DEFAULT_POLY_DEGREE, SNP_COUNT } from './types.js';

const snpCount = Number(process.argv[2] ?? SNP_COUNT);
const polyDegree = Number(process.argv[3] ?? DEFAULT_POLY_DEGREE);
const seed = Number(process.argv[4] ?? 0xc0ffee);

const vectors = generateSyntheticPrs({ snpCount, seed });
const batch = packPrsForFfi(vectors, { snpCount, polyDegree });
const oracle = plaintextPrs(vectors);

console.log(
  JSON.stringify(
    {
      snpCount: batch.meta.snpCount,
      polyDegree: batch.meta.polyDegree,
      slotCount: batch.meta.slotCount,
      cipherCount: batch.meta.cipherCount,
      scale: batch.meta.scale,
      header: Array.from(batch.header),
      slotsLength: batch.slots.length,
      slotsBytes: batch.slots.byteLength,
      plaintextPrs: oracle,
    },
    null,
    2,
  ),
);
