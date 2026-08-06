/**
 * GATE 3 (Task 10): Tauri IPC air-gap mock.
 *
 * Invokes the `evaluate_clinic_sweep` handler through the native addon — the
 * exact function body `src-tauri/src/lib.rs` wraps in `#[tauri::command]` —
 * bypassing the React UI. Every outbound network primitive is trapped first, so
 * any socket, DNS lookup or fetch attempted by the engine fails the test.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import dns from 'node:dns';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import { loadNativeModule } from '../../src/ffi.js';
import type { NativeModule } from '../../src/ffi.js';

const EPSILON = 1e-4;
const PATIENT_COUNT = 3;
const DISEASE_COUNT = 4;
const SEED = 0xc0ffee;

/** `[patients, diseases, backend, nttMs, maxAbsError, airgapped]` */
const HEADER_LEN = 6;
/** `[patient, disease, plaintext, decoded, absError]` */
const CELL_LEN = 5;

interface SweepCell {
  patient: number;
  disease: number;
  plaintext: number;
  decoded: number;
  absError: number;
}

function parseSweep(out: Float64Array) {
  const patientCount = out[0]!;
  const diseaseCount = out[1]!;
  const cellCount = (out.length - HEADER_LEN) / CELL_LEN;
  const cells: SweepCell[] = [];
  for (let i = 0; i < cellCount; i++) {
    const b = HEADER_LEN + i * CELL_LEN;
    cells.push({
      patient: out[b]!,
      disease: out[b + 1]!,
      plaintext: out[b + 2]!,
      decoded: out[b + 3]!,
      absError: out[b + 4]!,
    });
  }
  return {
    patientCount,
    diseaseCount,
    backend: out[2] === 1 ? 'metal' : 'cpu',
    nttMs: out[3]!,
    maxAbsError: out[4]!,
    airgapped: out[5] === 1,
    cells,
  };
}

const networkAttempts: string[] = [];
const restores: Array<() => void> = [];

function trap<T extends object, K extends keyof T>(host: T, key: K, label: string): void {
  const original = host[key];
  if (typeof original !== 'function') {
    return;
  }
  host[key] = (() => {
    networkAttempts.push(label);
    throw new Error(`air-gap violation: ${label} called`);
  }) as T[K];
  restores.push(() => {
    host[key] = original;
  });
}

let native: NativeModule | null = null;
let loadError: unknown = null;

beforeAll(() => {
  try {
    native = loadNativeModule();
  } catch (err) {
    loadError = err;
  }
  trap(globalThis as { fetch?: unknown }, 'fetch', 'fetch');
  trap(net.Socket.prototype, 'connect', 'net.Socket.connect');
  trap(http, 'request', 'http.request');
  trap(https, 'request', 'https.request');
  trap(dns, 'lookup', 'dns.lookup');
});

afterAll(() => {
  for (const restore of restores.reverse()) {
    restore();
  }
});

describe('GATE 3: evaluate_clinic_sweep air-gap', () => {
  it('loads the engine without a network', () => {
    expect(loadError, 'native addon missing — run `npm run build:native`').toBe(null);
    expect(native?.evaluateClinicSweep).toBeTypeOf('function');
  });

  it('returns correct risk scores for a synthetic cohort', () => {
    const out = native!.evaluateClinicSweep!(PATIENT_COUNT, DISEASE_COUNT, SEED);
    const sweep = parseSweep(out);

    expect(sweep.airgapped).toBe(true);
    expect(sweep.patientCount).toBe(PATIENT_COUNT);
    expect(sweep.diseaseCount).toBe(DISEASE_COUNT);
    expect(sweep.cells).toHaveLength(PATIENT_COUNT * DISEASE_COUNT);
    expect(sweep.maxAbsError).toBeLessThan(EPSILON);

    for (const [i, cell] of sweep.cells.entries()) {
      expect(cell.patient).toBe(Math.floor(i / DISEASE_COUNT));
      expect(cell.disease).toBe(i % DISEASE_COUNT);
      expect(Number.isFinite(cell.decoded)).toBe(true);
      expect(cell.absError).toBeCloseTo(Math.abs(cell.decoded - cell.plaintext), 12);
      expect(cell.absError).toBeLessThan(EPSILON);
    }
    // Guard against a vacuous pass on an all-zero matrix.
    expect(sweep.cells.some((c) => Math.abs(c.plaintext) > 1e-6)).toBe(true);
  });

  it('is deterministic for a fixed seed', () => {
    const a = parseSweep(native!.evaluateClinicSweep!(2, 3, 7));
    const b = parseSweep(native!.evaluateClinicSweep!(2, 3, 7));
    expect(a.cells.map((c) => c.decoded)).toEqual(b.cells.map((c) => c.decoded));

    const c = parseSweep(native!.evaluateClinicSweep!(2, 3, 8));
    expect(c.cells.map((x) => x.plaintext)).not.toEqual(
      a.cells.map((x) => x.plaintext),
    );
  });

  it('rejects an empty cohort instead of returning junk', () => {
    expect(() => native!.evaluateClinicSweep!(0, 3, SEED)).toThrow();
    expect(() => native!.evaluateClinicSweep!(3, 0, SEED)).toThrow();
  });

  it('attempted zero network operations', () => {
    expect(networkAttempts).toEqual([]);
  });

  // Runs last: proves the traps above would have caught a real violation,
  // so the preceding empty-log assertion is meaningful rather than a no-op.
  it('traps a deliberate network attempt', () => {
    expect(() => new net.Socket().connect(80, 'example.com')).toThrow(
      /air-gap violation/,
    );
    expect(networkAttempts).toEqual(['net.Socket.connect']);
  });
});
