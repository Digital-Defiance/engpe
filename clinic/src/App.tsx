import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type ScoreCell = {
  patient: number;
  disease: number;
  plaintext: number;
  decoded: number;
  absError: number;
};

type SweepResult = {
  patientCount: number;
  diseaseCount: number;
  backend: string;
  nttMs: number;
  maxAbsError: number;
  scores: ScoreCell[];
  airgapped: boolean;
};

export function App() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SweepResult | null>(null);
  const [patients, setPatients] = useState(2);
  const [diseases, setDiseases] = useState(5);

  async function loadCohort() {
    setBusy(true);
    setError(null);
    try {
      const out = await invoke<SweepResult>('evaluate_clinic_sweep', {
        patientCount: patients,
        diseaseCount: diseases,
        seed: 0xc0ffee,
      });
      setResult(out);
    } catch (e) {
      setError(String(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 960, margin: '0 auto', padding: '2.5rem 1.5rem 4rem' }}>
      <p style={{ color: 'var(--muted)', margin: 0, fontSize: '0.85rem' }}>
        Offline · Stateless · Apple Silicon unified memory
      </p>
      <h1 style={{ fontSize: '2.4rem', margin: '0.4rem 0 0.6rem' }}>ENGPE Clinic</h1>
      <p style={{ maxWidth: 36rem, color: 'var(--muted)', lineHeight: 1.5 }}>
        Load a synthetic multi-disease patient cohort. All CKKS math stays on-device;
        nothing leaves this air-gapped session.
      </p>

      <section
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          alignItems: 'end',
          margin: '2rem 0 1.5rem',
        }}
      >
        <label style={{ display: 'grid', gap: 4 }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Patients (M)</span>
          <input
            type="number"
            min={1}
            max={16}
            value={patients}
            onChange={(e) => setPatients(Number(e.target.value))}
            style={{ width: 88, padding: '0.45rem 0.5rem' }}
          />
        </label>
        <label style={{ display: 'grid', gap: 4 }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Diseases (D)</span>
          <input
            type="number"
            min={1}
            max={8}
            value={diseases}
            onChange={(e) => setDiseases(Number(e.target.value))}
            style={{ width: 88, padding: '0.45rem 0.5rem' }}
          />
        </label>
        <button type="button" disabled={busy} onClick={loadCohort}>
          {busy ? 'Evaluating…' : 'Load Patient Cohort'}
        </button>
      </section>

      {error && (
        <p style={{ color: 'var(--warn)' }} role="alert">
          {error}
        </p>
      )}

      {result && (
        <section>
          <h2 style={{ fontSize: '1.25rem' }}>Multi-disease risk scores</h2>
          <p style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>
            Backend {result.backend} · {result.nttMs.toFixed(1)} ms · max |ε|{' '}
            {result.maxAbsError.toExponential(2)}
            {result.airgapped ? ' · air-gap verified' : ''}
          </p>
          <table>
            <thead>
              <tr>
                <th>Patient</th>
                <th>Disease</th>
                <th>Oracle</th>
                <th>Decoded</th>
                <th>|ε|</th>
              </tr>
            </thead>
            <tbody>
              {result.scores.map((s) => (
                <tr key={`${s.patient}-${s.disease}`}>
                  <td>{s.patient}</td>
                  <td>{s.disease}</td>
                  <td>{s.plaintext.toFixed(6)}</td>
                  <td>{s.decoded.toFixed(6)}</td>
                  <td>{s.absError.toExponential(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </main>
  );
}
