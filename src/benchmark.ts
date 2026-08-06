/**
 * Multi-patient throughput sweep + Phase 3–4 lanes:
 *   - CPU / Metal RNS (plaintext ring path, CRT-optimized)
 *   - metal-gpu-async (double-buffered host∥GPU)
 *   - cpu/metal-encrypted (full RLWE + Galois KS; Metal KS when available)
 *   - CRT recombination microbench
 */

import { evaluatePrsCohort, loadNativeModule, type NativeModule } from './ffi.js';
import { generateSyntheticCohort, packCohortForFfi } from './prs-data.js';
import { DEFAULT_POLY_DEGREE, SNP_COUNT } from './types.js';
import { parseNativeResult } from './ffi.js';

/** Shared RNS scale for CPU and Metal. */
const RNS_SCALE = 2 ** 40;
const PATIENT_SWEEP = [1, 4, 16, 64, 128] as const;
/**
 * Small encrypted smoke panel (CI-friendly).
 *
 * N=8192 is the smallest degree at which the fixed 4-limb ~124-bit modulus is
 * still inside the 128-bit security envelope, and the encrypted entry point
 * now rejects anything below it. The earlier N=1024 lane was ~97 bits over the
 * ceiling for its degree and was never a deployable configuration.
 */
const ENC_SNP = 512;
const ENC_POLY = 8192;
const ENC_SCALE = 2 ** 40;

function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  if ((s.length & 1) === 0) {
    return ((s[mid - 1] ?? 0) + (s[mid] ?? 0)) / 2;
  }
  return s[mid] ?? 0;
}

function maxAbsError(scores: readonly { absError: number }[]): number {
  return Math.max(...scores.map((s) => s.absError), 0);
}

interface RunOut {
  ms: number;
  amortizedMs: number;
  maxAbsError: number;
  backend: string;
  patientCount: number;
}

type LaneMode = 'sync' | 'async' | 'encrypted';

interface BenchOpts {
  label: string;
  preferMetal: boolean;
  scale: number;
  patientCount: number;
  runs: number;
  mode: LaneMode;
  snpCount: number;
  polyDegree: number;
}

function runOnce(opts: BenchOpts, seed: number, native: NativeModule): RunOut {
  const cohort = generateSyntheticCohort({
    snpCount: opts.snpCount,
    patientCount: opts.patientCount,
    seed,
  });
  const batch = packCohortForFfi(cohort, {
    snpCount: opts.snpCount,
    polyDegree: opts.polyDegree,
    scale: opts.scale,
  });
  const t0 = performance.now();
  let result;
  if (opts.mode === 'encrypted') {
    if (!native.evaluatePrsEncryptedNapi) {
      throw new Error('evaluatePrsEncryptedNapi missing — rebuild native');
    }
    result = parseNativeResult(
      native.evaluatePrsEncryptedNapi(batch.header, batch.slots, opts.preferMetal),
    );
  } else if (opts.mode === 'async') {
    if (!native.evaluatePrsAsyncNapi) {
      throw new Error('evaluatePrsAsyncNapi missing — rebuild native');
    }
    result = parseNativeResult(
      native.evaluatePrsAsyncNapi(batch.header, batch.slots, opts.preferMetal),
    );
  } else {
    result = evaluatePrsCohort(cohort, batch, native, opts.preferMetal);
  }
  const ms = performance.now() - t0;
  return {
    ms,
    amortizedMs: ms / opts.patientCount,
    maxAbsError: maxAbsError(result.scores),
    backend: result.backend,
    patientCount: opts.patientCount,
  };
}

function logLane(opts: BenchOpts, times: number[], amortized: number[], last: RunOut): void {
  console.log(
    JSON.stringify(
      {
        lane: opts.label,
        mode: opts.mode,
        preferMetal: opts.preferMetal,
        scale: opts.scale,
        patientCount: opts.patientCount,
        snpCount: opts.snpCount,
        polyDegree: opts.polyDegree,
        runs: opts.runs,
        medianTotalMs: median(times),
        medianAmortizedPerPatientMs: median(amortized),
        minTotalMs: Math.min(...times),
        maxTotalMs: Math.max(...times),
        // Spread as a fraction of the median, so a single-sample headline
        // figure can never be quoted without its variability.
        spreadPct:
          median(times) > 0
            ? ((Math.max(...times) - Math.min(...times)) / median(times)) * 100
            : 0,
        samplesMs: times,
        lastBackend: last.backend,
        lastMaxAbsError: last.maxAbsError,
        withinEps1e4: last.maxAbsError < 1e-4,
      },
      null,
      2,
    ),
  );
}

function benchLane(opts: BenchOpts, native: NativeModule): void {
  const times: number[] = [];
  const amortized: number[] = [];
  let last = runOnce(opts, 1, native);
  last = runOnce(opts, 1, native);
  for (let i = 0; i < opts.runs; i++) {
    last = runOnce(opts, 100 + i, native);
    times.push(last.ms);
    amortized.push(last.amortizedMs);
  }
  logLane(opts, times, amortized, last);
}

function sweepPatients(metal: boolean, runs: number, native: NativeModule): void {
  for (const m of PATIENT_SWEEP) {
    benchLane(
      {
        label: `cpu-rns-m${m}`,
        preferMetal: false,
        scale: RNS_SCALE,
        patientCount: m,
        runs,
        mode: 'sync',
        snpCount: SNP_COUNT,
        polyDegree: DEFAULT_POLY_DEGREE,
      },
      native,
    );
    if (metal) {
      benchLane(
        {
          label: `metal-gpu-m${m}`,
          preferMetal: true,
          scale: RNS_SCALE,
          patientCount: m,
          runs,
          mode: 'sync',
          snpCount: SNP_COUNT,
          polyDegree: DEFAULT_POLY_DEGREE,
        },
        native,
      );
    }
  }
}

function benchAsyncAndEncrypted(metal: boolean, runs: number, native: NativeModule): void {
  // Multi-job async sweep — host prep of job N+1 overlaps Metal eval of job N.
  if (metal && native.evaluateAsyncJobSweepNapi) {
    const nJobs = 4;
    const samples: number[] = [];
    let lastErr = 0;
    let lastBackend = 'metal';
    for (let i = 0; i < runs; i++) {
      const out = native.evaluateAsyncJobSweepNapi(
        16,
        1,
        SNP_COUNT,
        DEFAULT_POLY_DEGREE,
        40,
        nJobs,
        true,
      );
      samples.push(out[0]!);
      lastErr = out[1]!;
      lastBackend = out[2] === 1 ? 'metal' : 'cpu';
    }
    console.log(
      JSON.stringify(
        {
          lane: 'metal-gpu-async-m16x4',
          mode: 'async',
          preferMetal: true,
          patientCount: 16,
          nJobs,
          snpCount: SNP_COUNT,
          polyDegree: DEFAULT_POLY_DEGREE,
          runs,
          medianTotalMs: median(samples),
          medianAmortizedPerPatientMs: median(samples) / (16 * nJobs),
          minTotalMs: Math.min(...samples),
          maxTotalMs: Math.max(...samples),
          lastBackend,
          lastMaxAbsError: lastErr,
          withinEps1e4: lastErr < 1e-4,
        },
        null,
        2,
      ),
    );
  }
  // Fully encrypted — small smoke + full 110k M=1 CPU vs Metal KS.
  for (const m of [1, 2] as const) {
    benchLane(
      {
        label: `encrypted-m${m}`,
        preferMetal: metal,
        scale: ENC_SCALE,
        patientCount: m,
        runs: Math.min(runs, 2),
        mode: 'encrypted',
        snpCount: ENC_SNP,
        polyDegree: ENC_POLY,
      },
      native,
    );
  }
  // Directive-11: full-panel encrypted M=1, CPU baseline then Metal KS.
  for (const preferMetal of metal ? [false, true] : [false]) {
    benchLane(
      {
        label: preferMetal ? 'metal-encrypted-full-m1' : 'cpu-encrypted-full-m1',
        preferMetal,
        scale: RNS_SCALE,
        patientCount: 1,
        // The O(N log N) ring multiply brought this lane from ~22 s to ~3 s,
        // so the headline FHE figure can now afford real repetitions.
        runs: Math.max(runs, 3),
        mode: 'encrypted',
        snpCount: SNP_COUNT,
        polyDegree: DEFAULT_POLY_DEGREE,
      },
      native,
    );
  }
}

function benchCrt(native: NativeModule): void {
  if (!native.benchmarkCrtRecombineMs) {
    console.log(JSON.stringify({ skipped: 'crt-recombine', reason: 'symbol missing' }));
    return;
  }
  const samples: number[] = [];
  for (let i = 0; i < 5; i++) {
    samples.push(native.benchmarkCrtRecombineMs(DEFAULT_POLY_DEGREE));
  }
  console.log(
    JSON.stringify(
      {
        lane: 'crt-recombine',
        polyDegree: DEFAULT_POLY_DEGREE,
        medianMs: median(samples),
        minMs: Math.min(...samples),
        maxMs: Math.max(...samples),
        targetMs: 10,
        withinTarget: median(samples) < 10,
      },
      null,
      2,
    ),
  );
}

function main(): void {
  const runs = Number(process.argv[2] ?? 3);
  let native: NativeModule;
  try {
    native = loadNativeModule();
  } catch (e) {
    console.error(String(e));
    process.exit(1);
    return;
  }
  const metal = native.isMetalAvailable?.() ?? false;
  console.log(
    JSON.stringify(
      { metalAvailable: metal, runs, patientSweep: PATIENT_SWEEP },
      null,
      2,
    ),
  );
  benchCrt(native);
  sweepPatients(metal, runs, native);
  benchAsyncAndEncrypted(metal, runs, native);
  if (!metal) {
    console.log(JSON.stringify({ skipped: 'metal-gpu', reason: 'Metal unavailable' }));
  }
}

main();
