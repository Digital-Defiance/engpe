# ENGPE Stateless Clinic (Task 9)

Offline-first Tauri desktop UI over the ENGPE Rust FHE engine.

## Run

```bash
cd clinic
npm install
npm run tauri dev
```

Requires Rust + Xcode (Metal) on Apple Silicon. The `evaluate_clinic_sweep` command never opens network sockets; all CKKS math stays in-process on unified memory.

## Air-gap gate

```bash
cd ../native && cargo test --release airgap_full_patient_sweep
```
