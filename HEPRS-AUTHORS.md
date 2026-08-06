# Note for Knight / Gerstein lab — ENGPE as an on-device evolution of HEPRS

**Intent.** Share a working edge-native CKKS PRS stack that matches the *cryptographic job* of HEPRS (encrypted genotypes **and** weights, `MulRelin`-style ct×ct, one `InnerSumLog`-style fold per individual) **and** the `PN13QP202pq` depth budget under Metal’s prime-width constraint — measured on the same machine as a rebuild of your public MIT example.

## What matches

| Item | Status |
| --- | --- |
| Your MIT example CSVs (10,001 × 50) | Same inputs |
| Machine for head-to-head | Apple M4 Max (both stacks) |
| Ring dimension on that example | $N = 8192$ |
| Weights encrypted | Yes (ct×ct + relinearization) |
| Accumulate then one fold / individual | Yes |
| Modulus **budget** | HEPRS $\log_2(QP)\approx 202$; ENGPE $\|Q\| = 217$ bits (7 Metal limbs $< 2^{31}$) |
| Predictions vs your `phenotype0_pred_…` | $r \approx 1$; FHE err $\sim 3\times10^{-6}$ |

## What differs (stated up front)

| Item | HEPRS | ENGPE |
| --- | --- | --- |
| Primes | Lattigo literal (some $>2^{31}$) | Metal-eligible only — **budget match, not bit-identical** |
| Scale | $2^{27}$ | $2^{40}$ |
| 10k×50 wall (this M4 Max) | **1.41 s** | **~10.5 s warm** / ~15.9 s cold |
| Library | Lattigo (Go) | Custom RNS-CKKS + Metal KS (Rust/TS, `U256` coeffs) |
| Deployment | Three-party evaluator | On-device clinic (Tauri) |

## Numbers we are standing behind

- **Matched FHE basis:** `FHE_RNS_LIMBS = 7`, $\|Q\| = 217$ bits at $N = 8192$ (HE ceiling 218). Gate: `fhe_basis_matches_heprs_qp_budget`. Details: `PARAMETERS.md`.
- **10k × 50 MIT example:** warm **~10.5 s**, RSS ≈ 2.7 GB (`results-headtohead-matched.txt`); HEPRS PN13 **1.41 s**.
- **110k × 1 (7-limb):** warm **5.41 s**, RSS **1.11 GB**, key setup 12.7 s (`results-ctct-memory-7limb.txt`).
- **Patient-packed streamed evaluator:** SNPs streamed, zero Galois KS; ~388 MiB @ 1146×512 / ~1.4 GB @ 1146×2k (`results-packed-memproof.txt`). RSS still creeps with SNP count (caches) — **bounding deferred**.

## How to reproduce (<15 min)

```bash
npm run build:native
npx tsx src/heprs-crossvalidate.ts
npx tsx src/heprs-headtohead.ts
npx tsx src/bench-stages.ts
npx tsx src/bench-packed.ts
cd native && cargo test --release fhe_basis -- --nocapture

cd heprs-upstream
go run main.go example_data/genotype_10kSNP_50individual.csv \
  example_data/beta_10kSNP_phenotype0.csv phenotype0 1 1 50 -pq -print
```

## Ask for you

1. Is **budget match** (217-bit Metal $Q$ vs your 202-bit $QP$) acceptable, or do you need bit-identical Lattigo primes (which would drop Metal KS)?
2. Does the published **~65 GB** figure include a fully materialised cohort of ciphertexts?
3. Priority for flat packed RSS at 110k SNPs vs further wall-clock work on the 10k×50 panel?

## Positioning one-liner

*Same PRS circuit and Metal-compatible ≈202-bit modulus budget as HEPRS; slower on the public example; designed so the evaluator fits a laptop — packing shipped, cache-bound RSS polish deferred.*
