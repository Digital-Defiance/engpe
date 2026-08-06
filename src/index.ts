/**
 * CKKS PRS orchestration entry — generate synthetic panel, pack for FFI,
 * invoke the CPU CKKS encode → NTT mul → rotate-and-sum path.
 */

import { evaluatePrs, loadNativeModule } from './ffi.js';
import {
  DEFAULT_CKKS_SCALE,
  generateSyntheticPrs,
  packPrsForFfi,
  plaintextPrs,
} from './prs-data.js';
import {
  DEFAULT_POLY_DEGREE,
  SNP_COUNT,
  type PrsEvaluationResult,
} from './types.js';

export {
  SNP_COUNT,
  DEFAULT_POLY_DEGREE,
  ckksSlotCount,
  cipherCountFor,
  type CkksPrsParams,
  type PackedPrsBatch,
  type PrsCohort,
  type PrsVectors,
  type PrsEvaluationResult,
} from './types.js';

export {
  DEFAULT_CKKS_SCALE,
  generateSyntheticPrs,
  generateSyntheticCohort,
  packPrsForFfi,
  packCohortForFfi,
  plaintextPrs,
  plaintextPrsCohort,
  plaintextPrsMatrix,
  plaintextPrsFlat,
  validatePackedPrs,
} from './prs-data.js';

export {
  evaluatePrs,
  evaluatePrsCohort,
  loadNativeModule,
  type NativeModule,
} from './ffi.js';

export interface RunSyntheticPrsOptions {
  snpCount?: number;
  polyDegree?: number;
  scale?: number;
  seed?: number;
  /** If true, skip native load and use the TS oracle only. */
  tsOnly?: boolean;
}

/**
 * End-to-end run: synthetic panel → packed FFI → CPU CKKS evaluate.
 */
export function runSyntheticPrs(
  options: RunSyntheticPrsOptions = {},
): {
  vectors: ReturnType<typeof generateSyntheticPrs>;
  batch: ReturnType<typeof packPrsForFfi>;
  result: PrsEvaluationResult;
  oracle: number;
} {
  const snpCount = options.snpCount ?? SNP_COUNT;
  const polyDegree = options.polyDegree ?? DEFAULT_POLY_DEGREE;
  const scale = options.scale ?? DEFAULT_CKKS_SCALE;

  const vectors = generateSyntheticPrs({
    snpCount,
    seed: options.seed,
  });
  const batch = packPrsForFfi(vectors, { polyDegree, snpCount, scale });
  const oracle = plaintextPrs(vectors);

  let native = null;
  if (!options.tsOnly) {
    try {
      native = loadNativeModule();
    } catch {
      native = null;
    }
  }

  const result = evaluatePrs(vectors, batch, native);
  return { vectors, batch, result, oracle };
}

// CLI when executed directly
const isMain =
  typeof process !== 'undefined' &&
  process.argv[1] &&
  (process.argv[1].endsWith('index.js') || process.argv[1].endsWith('index.ts'));

if (isMain) {
  const { batch, result, oracle } = runSyntheticPrs();
  console.log(
    JSON.stringify(
      {
        snpCount: batch.meta.snpCount,
        polyDegree: batch.meta.polyDegree,
        slotCount: batch.meta.slotCount,
        cipherCount: batch.meta.cipherCount,
        slotsBytes: batch.slots.byteLength,
        oracle,
        result,
      },
      null,
      2,
    ),
  );
}
