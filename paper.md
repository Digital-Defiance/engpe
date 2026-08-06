# Edge-Native Genomic Privacy: Ring Evaluation and True RLWE Polygenic Risk Scores on Apple Silicon

> Typeset version: [`paper.tex`](paper.tex) → [`paper.pdf`](paper.pdf) (bibliography: [`paper.bib`](paper.bib)).

**Digital Defiance** | August 2026

### Abstract

Fully Homomorphic Encryption (FHE) enables privacy-preserving genomic diagnostics, but practical edge deployment requires a clear accounting of which costs are cryptographic and which are merely arithmetic. We present **`ENGPE` (Edge-Native Genomic Privacy Engine)**, an RNS-CKKS polygenic risk score (PRS) stack for Apple Silicon (M4 Max) that exposes two deliberately separated pipelines. The **Ring Evaluation Engine** performs CKKS encode / negacyclic NTT / CRT / decode over plaintext ring elements and, at polynomial degree $N = 16384$ and scale $\Delta = 2^{40}$, amortizes a 110,000-SNP evaluation to **13.7 ms per patient** at its best operating point ($M = 16$, Metal GPU lane). The **Cryptographic FHE Engine** adds true RLWE KeyGen, encryption of genotypes *and* weights, ciphertext×ciphertext multiplication with relinearization, and hybrid Galois key-switching with a fused Metal Digit-NTT → KS-MAC → INTT command buffer, under a **HEPRS-matched modulus budget** of seven Metal-eligible limbs ($|Q| = 217$ bits at $N = 8192$, inside the HE Standard ceiling of 218). An offline Tauri clinic application exercises the stack without network I/O.

Against the HE-PRS baseline of Knight *et al.* [1], we match algorithm structure **and** the $\log_2(QP)\approx 202$ depth budget (Metal-compatible primes; see Section 2.0 and `PARAMETERS.md`), and measure on the same machine as a local rebuild of their MIT example (10,001 features × 50 individuals, $N = 8192$). Their `PN13QP202pq` build on this M4 Max finishes in **1.41 s**; warm `ENGPE` finishes the same panel with fidelity at file-rounding ($r = 0.999999999$; Section 5.3). On-device memory remains the deployment claim: streamed patient packing keeps ~1,146-patient cohorts near **0.4–1.4 GB** rather than materialising ~9 GB.

Every claim is backed by an executable gate. The audit that produced this revision found and fixed a silent evaluation-key aliasing defect (Section 3.2), a schoolbook $O(N^2)$ ring multiply that accounted for 90% of encrypted runtime (Section 4.1), a rotate-and-sum that was incorrectly applied once per ciphertext rather than once per individual (Section 5.1), and an i128 overflow in relinearization-key generation that silently destroyed ct×ct precision at production degrees.

### 1. Introduction

Polygenic Risk Scores (PRS) aggregate genotype dosages across hundreds of thousands of Single Nucleotide Polymorphisms (SNPs). Clinical use is constrained by privacy: sending raw genotypes to a remote server is often unacceptable under regulatory and ethical constraints.

CKKS-style approximate arithmetic can evaluate a PRS as a slot-wise product followed by a horizontal sum. Two layers must not be conflated:

1. **Ring evaluation** — canonical embedding, negacyclic NTT multiply, CRT recombination, and decode — which validates the arithmetic engine and saturates Metal batch NTTs over unified memory.
2. **True FHE** — RLWE ciphertexts $(c_0, c_1)$ under a ternary secret, with noise, and with key-switching after every Galois automorphism used in rotate-and-sum.

Prior drafts of this work incorrectly attributed ring-engine throughput to “FHE” and extrapolated unverified cohort wall-clock times. This revision reports only measured figures and names the two modes explicitly.

**Prior work.** Knight, Li, Jensen, Gerstein *et al.* [1] demonstrated end-to-end homomorphically encrypted PRS on real clinical data, evaluating a 110,000-SNP schizophrenia model across a ~1,100-patient cohort in approximately six minutes using ~65 GB of memory for the core PRS calculation. That result established the clinical viability of HE-PRS and is the baseline against which `ENGPE` should be read. `ENGPE` targets a different point in the design space — a single consumer device rather than a large-memory server — and Section 5 compares the two on matched algorithm structure and on their released example cohort.

### 2. System Architecture

`ENGPE` targets the Apple M4 Max (performance CPU cluster + 40-core Metal GPU, unified memory). Both pipelines use Metal-eligible RNS primes $q_i < 2^{31}$ with $q_i \equiv 1 \pmod{2N}$. The **ring** path keeps four limbs ($|Q|\approx 124$ bits) for throughput. The **encrypted** path uses **seven** limbs ($|Q| = 217$ bits), matching the HEPRS `PN13QP202pq` depth budget under the Metal prime-width constraint; coefficients live in `ethnum::U256`. Flat `Float64Array` FFI packs $M$ genotype panels and $D$ disease weight panels for a single native call.

#### 2.0 Security parameters and threat model

Two parameter sets share the same noise model — uniform ternary $s$, $\mathrm{CBD}(\eta = 20)$ so $\sigma \approx 3.16$ — and differ only in limb count:

| Path | $N$ (typical) | Limbs | $\|Q\|$ | HE Standard ceiling | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Ring evaluation | 16384 | 4 | ≈124 | 438 | plaintext CKKS math |
| Encrypted FHE | 8192 / 16384 | **7** | **217** | 218 / 438 | HEPRS-matched depth budget |

Lattigo's `PN13QP202pq` uses $\log_2(QP)\approx 202$ with some primes above $2^{31}$, which Metal NTT kernels reject. We therefore match the **budget**, not the bit-identical prime list (`PARAMETERS.md`, gate `fhe_basis_matches_heprs_qp_budget`). At $N = 8192$, $|Q| = 217$ sits **one bit inside** the classical 128-bit ceiling of 218; we report the envelope rather than a lattice-estimator $\lambda$.

Reduced-degree configurations with a too-large modulus relative to $N$ remain rejected by `require_secure_degree`. An audit gate asserts production sets are inside the envelope and that reduced degrees are flagged outside it.

**Threat model.** This matters because `ENGPE` runs every stage on one device, and FHE buys nothing against an adversary who already holds the key. Both genotypes and model weights are encrypted under the clinic's public key ($\mathrm{ct} \times \mathrm{ct}$ with relinearization), matching the cryptographic shape of [1]'s evaluator path, while the deployment remains on-device rather than three-party. The adversary modelled is a compromised or observed *evaluation host* that sees only ciphertexts; the clinic holds the secret key. The contribution is therefore that the evaluator role, which in [1] requires a ~65 GB server, fits in under 1 GB on hardware the data owner can physically control. Evaluation keys (Galois + relinearization) are memoised per degree for the process lifetime, so all patients handled by one process share an RLWE key — acceptable on-device, not acceptable in a multi-tenant deployment.

#### 2.1 Ring Evaluation Engine (plaintext CKKS math)

Host Rayon threads encode real slots via the canonical embedding and CRT-decompose coefficients into four limbs. Each limb is multiplied with a negacyclic NTT (Metal when available, else CPU). Products are recombined and horizontally summed (Galois rotate-and-sum on the wide residue polynomial), then decoded. Multi-disease SIMD broadcasts one patient genotype across $D$ plaintext weight vectors in the transform domain without re-encoding the genotype.

#### 2.2 Cryptographic FHE Engine (true RLWE)

The FHE path adds:

- a CSPRNG (`getrandom`) for keys and noise;
- ternary secret keys $s \leftarrow \{-1,0,1\}^N$;
- centered-binomial error $\mathrm{CBD}(\eta=20)$ ($\sigma \approx 3.16$);
- uniform $a \leftarrow U(R_Q)$;
- public-key encryption to 2-element ciphertexts $(c_0, c_1)$ for both genotypes and weights;
- ciphertext×ciphertext multiplication with digit-decomposed relinearization of $s^2$ (HEPRS `MulRelinNew`);
- hybrid digit-decomposition Galois evaluation keys and fused Metal key-switching after each rotate-and-sum automorphism (Section 3).

#### 2.3 CRT recombination optimization

Profiling showed CRT recombination consuming the majority of ring-engine wall time when implemented as bit-serial double-and-add modular multiplication (~673 ms to recombine one degree-$16384$ polynomial). The present implementation uses 128-bit widening multiply, reduction via precomputed $2^{128} \bmod Q$, and precomputed CRT weights $w_i = (Q_i \cdot \hat{y}_i) \bmod Q$, with coefficients recombined in parallel. The `crt-recombine` microbench in `results-v3.txt` reports a median of **0.82 ms** at $N = 16384$ (min 0.62 ms, max 2.31 ms), inside the 10 ms engineering target.

#### 2.4 Asynchronous double-buffering

A bounded `sync_channel(1)` overlaps host encode+CRT preparation of cohort $N+1$ with limb NTT evaluation of cohort $N$. Correctness is gated by bit-identical comparison (`f64::to_bits`) against the synchronous path. After the CRT fix, host prep is a small fraction of total time; overlap remains correct but yields limited additional throughput on the ring engine.

### 3. Galois Key-Switching: Hybrid Digits and Fused Metal Dispatch

Horizontal summation in CKKS applies $\log_2(N/2)$ automorphisms $\varphi_k: X \mapsto X^k$ with $k = 5^{2^i} \bmod 2N$. For $N = 16384$ this is **13** steps. On a ciphertext $(c_0, c_1)$ encrypted under $s$, $\varphi_k$ produces a ciphertext under $\varphi_k(s)$. Returning to basis $s$ requires a key-switch: decompose $c_1$ into base-$B$ digits and multiply by evaluation-key components that encrypt $B^j \cdot \varphi_k(s)$.

#### 3.1 Hybrid digit width and auxiliary modulus $P$

Phase 5 widens the gadget base from $B = 2^{10}$ (13 digits at $\|Q\|\approx 124$ bits) to **$B = 2^{20}$ (7 digits)**, cutting the number of digit NTTs and EVK MACs per rotation by nearly half. An auxiliary Metal-eligible special modulus $P$ (one prime $p < 2^{31}$, co-prime to all $q_i$) is generated alongside evaluation keys for hybrid modulus-raising / mod-down. Exact RNS mod-down $y_i = (x_i - x_P)\cdot P^{-1} \bmod q_i$ is implemented and gated, and a `mod_down_batch` Metal kernel scales residues by $P^{-1}$.

To be explicit rather than let a reader infer otherwise: **the extended $PQ$ hybrid path is built and tested but not engaged in production.** The shipped FHE circuit already runs under a HEPRS-matched $|Q| = 217$-bit Metal basis (`FHE_RNS_LIMBS = 7`, `U256` residues); Stage-D mod-down remains infrastructure for deeper circuits, not a component of the reported latencies. The auxiliary modulus is generated with every evaluation-key set and its correctness is gated.

The wider gadget is not free, and the cost is stated here rather than left implicit. Key-switch noise grows linearly in $B$, so moving from $2^{10}$ to $2^{20}$ raises it by about $2^{10}$. Measured directly by encrypting zero and reading the decrypted phase (`audit_keyswitch_noise_budget_is_measured`), at $N = 1024$ and $\|Q\| \approx 124$ bits the fresh-ciphertext noise is $\approx 2^{8}$ and the post-key-switch noise is $\approx 2^{28}$ — exactly the predicted factor of $B$. Relative to the working scale this is $\approx 1.3\times 10^{-4}$ at $\Delta = 2^{40}$ but only $\approx 1.3\times 10^{-16}$ at $\Delta^{2} = 2^{80}$. The PRS circuit key-switches *after* the ciphertext×ciphertext multiply (and relinearization), i.e. at $\Delta^{2}$, so the wider gadget is absorbed with roughly 80 bits of headroom under $Q$. Circuits that rotate at $\Delta$ rather than $\Delta^{2}$ would need either a narrower $B$ or a larger $\Delta$; this is a property of the circuit, not of the engine.

A note on rotation semantics, since it is easy to misread the code: slots here are indexed in the **natural** embedding order $\zeta_j = \exp(i\pi(2j+1)/N)$, not the canonical power-of-5 order. Under that labelling $\varphi_5$ is the permutation $j \mapsto ((5(2j+1) \bmod 2N) - 1)/2$ folded through conjugate symmetry, which is *not* a one-position cyclic shift. Rotate-and-sum is unaffected — it sums the full orbit, and the PRS dot product is permutation-invariant — but a reader expecting a cyclic shift will misjudge the rotation gate. The permutation is asserted explicitly in `audit_galois_keyswitch_performs_real_rotation`.

#### 3.2 Fused GPU command buffer

Earlier Phase 4 work showed that separate Metal submits for digit NTT, KS MAC, and INTT *degraded* full-panel latency (~105 s) due to sync overhead, while CPU digit NTT + Metal MAC alone reached ~30 s. Phase 5 encodes the full limb pipeline into a **single `MTLCommandBuffer`** with one `waitUntilCompleted`:

1. **Stage A** — batch forward negacyclic NTT on decomposed digits (resident GPU buffer; not read back mid-pipeline).
2. **Stage B** — `keyswitch_mac_batch` against resident EVK `MTLBuffer`s (both $b$ and $a$ components).
3. **Stage C** — inverse NTT on each accumulator.
4. **Stage D** — optional `mod_down_batch` when operating on the extended $PQ$ basis.

Galois EVKs remain NTT-transformed once at cold start and held resident on both host and GPU. Cache entries are keyed by $(N, k, \mathrm{fingerprint}, \mathrm{limb}, \mathrm{component})$, where the fingerprint is an FNV-1a digest of the evaluation-key material itself.

The key identity matters. An earlier revision keyed only on $(N, k, \mathrm{limb}, \mathrm{component})$, so two *distinct* Galois keys sharing a degree and rotation step aliased to the same cache entry: the second key-switched under the first key's evaluation material and returned garbage with no error raised. The precise blast radius is worth stating, because it is narrower than it first appears and we would rather understate it. The shipped evaluation path memoises one `EvaluationKeys` per degree in a process-global cache (`eval_keys_cached`) and never regenerates it, so the current clinic flow only ever holds one key set per $N$ and does not trigger the alias. What did trigger it was the test suite, where independent gates each generate their own keys — four gates failed together while every one of them passed in isolation, which is how the defect surfaced. It is also reachable by any consumer of the public `crypto` API that generates its own keys, and it would become reachable in production the moment per-session or per-patient re-keying is introduced, which is a natural next step for this system. A silent wrong-answer failure mode that is latent rather than active is still a publication blocker.

The fix makes the alias unrepresentable rather than merely unlikely, and `audit_distinct_galois_keys_do_not_alias_in_evk_cache` key-switches under three independent key sets on both backends as a regression gate. The cache is now bounded and flushed on both host and GPU when full, so re-keying cannot grow resident `MTLBuffer` memory without limit. Bit-identical gates require Metal fused KS to match the CPU oracle on $c_0$/$c_1$ limbs before encrypted evaluation proceeds.

One related property should be explicit for a security reader: because evaluation keys are memoised per degree for the lifetime of the process, every patient evaluated by a given process is handled under the *same* RLWE key. For on-device edge evaluation, where the data owner also holds the key, this is a reasonable default. It would not be acceptable in a multi-tenant or server-side deployment, and the fingerprinted cache is a prerequisite for making per-session keys safe.

### 4. Empirical Results

All figures below are from a single post-hoist benchmark run (`results-v4.txt`) on an idle Apple M4 Max. Parameters: $N = 16384$, $k=4$ RNS limbs, $\Delta = 2^{40}$, 110,000 SNPs unless noted. Every lane is a configuration inside the 128-bit security envelope (Section 2.0). Encrypted lanes report three repetitions with observed spread; the `crt-recombine` microbench reports a median of 0.78 ms at $N = 16384$.

**Ring Evaluation Engine, amortized ms/patient:**

| Cohort $M$ | Metal total (ms) | Metal ms/patient | CPU RNS ms/patient | max $\|\varepsilon\|$ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 47.9 | 47.91 | 51.17 | $9.1\times 10^{-10}$ |
| 4 | 81.9 | 20.48 | 23.93 | $1.2\times 10^{-8}$ |
| 16 | 219.9 | **13.74** | 17.72 | $2.3\times 10^{-8}$ |
| 64 | 901.7 | 14.09 | 16.83 | $2.4\times 10^{-8}$ |
| 128 | 2355.0 | 18.40 | **16.67** | $2.4\times 10^{-8}$ |

Two observations a reader should not have to reverse-engineer. Amortization is best at $M = 16$ (13.74 ms/patient), not at the largest cohort; beyond $M \approx 64$ the Metal lane degrades and at $M = 128$ the **CPU RNS lane is faster** (16.67 vs 18.40 ms/patient). The GPU advantage here is real but bounded, and it inverts at large cohorts. These figures are also contention-sensitive: the same $M=128$ Metal lane has measured between 18.0 and 56.2 ms/patient, the latter while another process held the GPU. Only idle-machine numbers are reported.

#### 4.1 Where the encrypted time actually went

The encrypted path was profiled rather than assumed, and the result redirected the entire optimisation effort. Instrumenting the key-switch phases attributed only **4.3%** of warm runtime to the fused Metal pipeline and left 91% unaccounted. The cause was `poly_mul_as`, the ring multiply behind encrypt, decrypt, and Galois key generation, implemented as a schoolbook $O(N^2)$ expansion over `u128` — roughly $1.8\times10^{8}$ modular operations per call at $N = 16384$. Measured primitives confirmed it exactly: 14 encrypt+decrypt pairs predicted 19.5 s against 19.7 s observed, and 13 Galois keys predicted 42.2 s against the 45.2 s observed setup.

Routing that multiply through the existing $O(N \log N)$ RNS-NTT path (bit-exact against the schoolbook oracle, gated by `ring_multiply_ntt_matches_schoolbook`) produced:

| Primitive at $N=16384$ | $O(N^2)$ | $O(N \log N)$ | Speedup |
| --- | ---: | ---: | ---: |
| Encrypt | 953 ms | 41 ms | 23× |
| Decrypt | 442 ms | 10 ms | 44× |
| One Galois key | 3249 ms | 196 ms | 17× |
| Evaluation-key setup (13 keys) | 45.2 s | 4.0 s | 11× |
| Full encrypted panel (warm, pre-hoist) | 21.7 s | 3.15 s | 6.9× |

A second structural fix followed: hoisting rotate-and-sum out of the per-ciphertext loop (Section 5.1), then switching the production multiply from ct×pt to ct×ct with relinearization so the evaluator path matches HEPRS's cryptographic job.

**Cryptographic FHE Engine (evolution):**

| Stage | Full 110k, $M=1$, $D=1$ | Notes |
| --- | ---: | --- |
| CPU KS, 10-bit digits | ~66–67 s | Phase 3 baseline (not re-measured) |
| Metal MAC only, 10-bit digits | ~30 s | Phase 4; resident EVK (not re-measured) |
| Fused Metal + hybrid digits, $O(N^2)$ multiply | 22.0 s | Phase 5 |
| Fused Metal + $O(N \log N)$, per-ciphertext fold | 3.27 s | pre-hoist, ct×pt |
| Fused Metal + $O(N \log N)$, hoisted fold, ct×pt | 1.32 s | prior headline; easier job |
| **Fused Metal + hoisted fold, ct×ct + relin (4-limb)** | **2.61 s** | pre-match; $\|Q\|\approx 124$ |
| **Fused Metal + hoisted fold, ct×ct + relin (7-limb)** | **5.41 s** | HEPRS-matched $\|Q\|=217$; RSS 1.11 GB |

Peak resident set for the matched ct×ct panel is **1.11 GB**, with a one-time evaluation-key setup of **12.7 s** (`results-ctct-memory-7limb.txt`). The step from 2.61 s (4-limb) to 5.41 s is the measured cost of the Metal-compatible ≈202-bit budget.

### 5. Comparison with HEPRS (Knight et al., 2026)

The reference point is Knight *et al.* [1]. We rebuilt their public MIT example under Lattigo on this M4 Max and ran `ENGPE`'s ct×ct path on the same CSV inputs (`src/heprs-headtohead.ts`, `results-bgpucap/`).

Both systems now encrypt genotypes **and** weights (ct×ct + relinearization), fold once per individual, and run at a **matched depth budget**: HEPRS `PN13QP202pq` has $\log_2(QP)\approx 202$; `ENGPE` uses seven Metal-eligible limbs with **$|Q| = 217$ bits** at $N = 8192$ (gate `fhe_basis_matches_heprs_qp_budget`). Primes are not bit-identical — Lattigo's 33–34-bit moduli exceed Metal's $2^{31}$ limb limit — but the security/depth envelope is matched (`PARAMETERS.md`).

| Configuration | HEPRS (this M4 Max, unless noted) | `ENGPE` (this M4 Max) |
| --- | --- | --- |
| Weights | Encrypted (ct×ct + relin) | **same** |
| Example ring dim | $N=8192$ (`PN13QP202pq`) | $N=8192$ |
| Modulus budget | $\log_2(QP)\approx 202$ | $\|Q\| = 217$ (7 Metal limbs) |
| Algorithm structure | accumulate, one fold / individual | **same** (Section 5.1) |
| Key-switches / individual (110k) | 13 rotations (+ Lattigo relins) | **14 relins + 13 rotations** (counted) |
| **10k×50, $N=8192$** | **1.41 s**, HeapSys ≈ 0.39 GB | **~10.5 s warm** / ~15.9 s cold, RSS ≈ 2.7 GB |
| 110k SNPs, 1 patient | ~10 s, ~3 GB (published M1) | **5.41 s warm, 1.11 GB** (7-limb FHE) |
| 10k SNPs × 1,146 patients (SNP-packed) | not published at this panel | **69.8 s**, RSS ≈ 9.1 GB (materialised; pre-match) |
| 1,146 patients, streamed patient-packed | — | **~388 MiB** @ 512 SNPs / **~1.4 GB** @ 2k SNPs |
| 110k × ~1,146 | 6 min, ~65 GB (published Xeon) | prefer patient-packed for memory |

The **bold row** is the controlled comparison: identical published inputs, same machine, same $N$, matched crypto job **and** matched modulus budget. HEPRS remains faster on that panel; `ENGPE` pays for Metal-wide limbs (`U256` host coeffs + 7× NTT) and an on-device deployment shape. Fidelity is in Section 5.3.

#### 5.1 One rotate-and-sum per patient, not one per ciphertext

Reading Algorithm 1 of [1] against an earlier draft exposed a plain inefficiency. HEPRS multiplies each of the $K$ genotype ciphertexts by its weight block, **accumulates them all first**, and calls `InnerSumLog()` once — $\log_2(N/2)$ rotations per individual. The earlier `ENGPE` draft folded every ciphertext, costing $K\log_2(N/2)$ rotations.

At production that was 14×13 = **182** rotation key-switches per patient versus HEPRS's **13**. Hoisting is semantics-preserving (`score_encrypted_patient_disease` in `ckks_rns.rs`). With ct×ct, each of the 14 products also needs a relinearization key-switch, so the counted cost is **27 key-switches per patient** (14 relin + 13 rotate) rather than 13 — still far below the pre-hoist 182 rotations, and the structure of the fold matches [1].

#### 5.2 Packing as a tunable frontier

Transposing the layout so that slot $p$ carries patient $p$ turns the PRS into

$$\mathrm{acc} \;=\; \sum_i \mathrm{Enc}(g_{\cdot,i}) \cdot w_i$$

with $w_i$ a scalar broadcast across slots, so slot $p$ of the accumulator is patient $p$'s score. There is no horizontal sum: the rotation count is **exactly zero** and Galois keys are not needed at all. We implemented this (`native/src/packing.rs`) and gated it against the plaintext oracle for every patient.

Measured per-ciphertext primitives at $N = 16384$ under the earlier ct×pt path — encrypt 57.9 ms, ct×pt 20.4 ms — projected the transposed layout to **~1.05 s per patient**. That projection is superseded by the shipped **patient-packed streamed** evaluator (`evaluate_prs_patient_packed_napi` / `bench_patient_packed_synthetic_napi`): SNPs are encrypted and multiplied one at a time into a single accumulator, Galois key-switches are **zero**, and genotypes need not reside as an $M\times n_{\mathrm{snp}}$ matrix in the evaluator. On this M4 Max, a synthetic 1,146-patient cohort at $N=8192$ measures **~388 MiB** RSS at 512 SNPs and **~1.4 GB** at 2,000 SNPs (`results-packed-memproof.txt`) — far below the **~9.1 GB** of a naïve SNP-packed materialisation of 1,146×10k (`results-cohort-example1146.txt`). Residual RSS growth with SNP count is dominated by Metal/host caches around relinearization, not by storing the panel; bounding those caches is follow-on work.

The two layouts remain endpoints of a tunable frontier. A block-packed ciphertext carrying $P$ patients and $8192/P$ SNPs costs $\lceil n_{\mathrm{snp}} P / 8192\rceil \cdot \log_2(8192/P) / P$ key-switches per patient under a *per-ciphertext* fold; with the hoisted fold of Section 5.1 the deployed $P = 1$ SNP-packed layout costs $\log_2(N/2)$ rotations plus one relin per ciphertext. $P = N/2$ recovers the streamed patient-packed path (zero rotations). Both endpoints and the monotonicity are asserted in `block_packing_interpolates_between_layouts`.

We suspect the ~65 GB in [1] is the cost of materialising a cohort-packed panel. We rebuilt and timed their public example on this machine (Section 5); we have not profiled their 1,146-patient clinical path (controlled-access data). Stage splits for our SNP-packed path (weight encrypt wall + patient-eval wall) are exposed via `evaluate_prs_encrypted_staged_napi` (`results-stages-matched.txt`).

#### 5.3 Cross-implementation validation on the published HEPRS dataset

Every correctness figure elsewhere in this paper is measured against `ENGPE`'s own plaintext oracle, which shares encoding code with the pipeline it validates. Knight *et al.* released their example cohort under MIT licence — 10,001 features × 50 individuals from HAPGEN2, the ridge weights, **and their own plaintext PRS predictions** — which permits a genuinely independent check. We ran their genotypes and their weights through `ENGPE`'s **ct×ct** encrypted pipeline at $N = 8192$ (`src/heprs-crossvalidate.ts`).

| Comparison | Max absolute error |
| --- | ---: |
| `ENGPE` encrypted vs. double-precision dot product of the same inputs | $2.8\times10^{-6}$ |
| `ENGPE` encrypted vs. HEPRS published prediction | $3.6\times10^{-5}$ |
| Double-precision dot product vs. HEPRS published prediction | $3.646\times10^{-5}$ |

Homomorphic evaluation contributes roughly $3\times10^{-6}$ — within the discrepancy imposed by their 6-decimal weight/prediction files — and Pearson correlation between `ENGPE`'s encrypted scores and their published predictions remains $r = 0.999999999$. The ridge intercept absent from the distributed files was recovered independently as $0.26158746$. Under the matched 7-limb basis, warm wall-clock for this panel on the M4 Max is **~10.5 s** (cold ≈ 15.9 s including key setup); their `PN13QP202pq` build on the same machine finishes in **1.41 s** (Section 5).

We regard this as the strongest correctness evidence in the paper, because it is the only result not validated against our own oracle. It does not reproduce their clinical finding: the schizophrenia cohort of 1,146 individuals comes from PsychENCODE / PGC under controlled access. No clinical claim is made here; the 110,000-SNP figures use synthetic panels of matched size.

### 6. Validation Gates

Integrity checks include: sparse schoolbook negacyclic convolution gates; bit-identical async vs sync ring outputs; multi-disease SIMD oracle checks at $\epsilon < 10^{-4}$; encrypted-path oracle checks; bit-identical Metal vs CPU Galois key-switch (`assert_metal_keyswitch_matches_cpu`); and Phase 5 fused hybrid oracle + aux-$P$ mod-down round-trip (`assert_fused_hybrid_keyswitch_matches_oracle`). The offline clinic path sets an explicit air-gap flag and performs no network I/O in the evaluation core.

A precision gate only shows that the pipeline is *accurate*; it does not show that the pipeline is *what this paper describes*. A second suite (`native/src/audit_gates.rs`) targets that gap directly, and each of its assertions corresponds to a claim made above:

- **Ciphertexts are RLWE, not encoding.** $c_0$ must not equal the encoded plaintext, $c_1$ must be non-zero, two encryptions of one message must differ, and a *wrong* secret key must fail to recover the message (it does, by $\gg 1$ absolute error). Companion gates confirm products remain key-bound (ct×pt and ct×ct paths).
- **Fresh noise exists and is bounded.** Encrypting zero and decrypting yields the noise directly: $\approx 2^{8}$ against a $2^{124}$ modulus. A ciphertext that decrypted to exactly zero would indicate encoding rather than encryption.
- **The transform-domain ring multiply is bit-exact.** Replacing the schoolbook $O(N^2)$ multiply (Section 4.1) touched encrypt, decrypt, and key generation at once, so it is gated by exact per-coefficient equality against the schoolbook oracle across several degrees, plus a check that an unregistered modulus still falls back correctly.
- **Parameters stay inside a published security envelope.** The production set is asserted inside the HE Standard 128-bit ceiling, *and* reduced-degree configurations are asserted to be correctly flagged **outside** it, so a test-only lane can never be mistaken for a deployable one (Section 2.0).
- **Key-switching performs the real automorphism.** Asserted against the explicit $\varphi_5$ slot permutation, and separately asserted *not* to be the identity.
- **The fused Metal pipeline actually executes.** Atomic counters record fused versus fallback limb key-switches; the gate requires every limb to take the fused path *and* to remain bit-identical to the CPU oracle.
- **The key-switch count behind Section 5 is measured, not derived.** After the hoist and the move to ct×ct, the full 110k panel costs **14 relinearizations + 13 rotations = 27 key-switches per patient** (108 limb ops at 4 RNS limbs), confirmed by `audit_full_panel_keyswitch_count_matches_paper` (`--ignored`).
- **ct×ct + relin is precision-gated through production degree.** Unit tests cover single multiplies through $N=8192$ and a two-ciphertext PRS fold; cross-validation against [1]'s published predictions is Section 5.3.
- **Evaluation keys cannot alias.** Three independent key sets sharing $(N, k)$, on both backends (Section 3.2).
- **The hybrid gadget is real and $P$ is well formed.** Every aux prime is $< 2^{31}$, NTT-eligible, distinct from all $q_i$, and $P^{-1} \bmod q_i$ is verified to be a genuine inverse. Digit count scales with $\|Q\|$ (11 digits at $\|Q\| = 217$ with 20-bit gadgets).
- **FHE modulus matches the HEPRS QP budget.** Seven Metal limbs at $N = 8192$ yield $\|Q\| = 217$ bits, inside the HE ceiling of 218 and comparable to Lattigo `PN13QP202pq` (`fhe_basis_matches_heprs_qp_budget`).

These gates were written as an adversarial pre-publication audit and were not all green when first run. They surfaced the evaluation-key aliasing defect of Section 3.2, a stale key-switch noise budget left behind by the move to a 20-bit gadget, the reduced-degree security trap of Section 2.0, the quadratic ring multiply of Section 4.1, and an i128 overflow in relinearization-key generation that destroyed ct×ct precision at large $N$. The suite is 56+ Rust tests (heavier measurement gates marked `#[ignore]`) and the TypeScript gates, stable across repeated runs.

### 7. Deployment

`ENGPE` ships a Tauri desktop “stateless clinic” that invokes the native sweep locally. Synthetic cohorts are generated on-device; ring or encrypted evaluation can be selected according to the threat model. The UI does not require network access for a completed sweep.

### 8. Conclusion and Future Work

`ENGPE` separates a high-throughput **ring evaluation** path (13.7 ms/patient amortized at its $M=16$ optimum for 110k SNPs) from a correctness-bearing **RLWE FHE** path that matches HEPRS's cryptographic job and modulus budget: encrypted genotypes *and* weights, ct×ct with relinearization, one hoisted rotate-and-sum per individual, and seven Metal limbs with $\|Q\| = 217$ bits (**5.41 s** warm / 1.11 GB RSS at 110k×1 on an M4 Max).

Measured against Knight *et al.* [1] on the same machine and their MIT example ($N=8192$, ct×ct, **matched $\approx 202$-bit modulus budget**): their `PN13QP202pq` build finishes in **1.41 s**; warm `ENGPE` finishes in **~10.5 s** ($r = 0.999999999$ vs their preds). We are not faster. The presentable claim is **on-device memory + Metal-compatible parameter parity**: $|Q| = 217$ bits with seven limbs $< 2^{31}$, and a streamed patient-packed evaluator that keeps ~1,146-patient cohorts near **0.4–1.4 GB**. Packed RSS still creeps with SNP count (host/Metal caches); bounding that is deferred.

Five limitations are worth stating plainly. The GPU advantage on the ring engine inverts above $M \approx 64$. Evaluation keys are memoised per degree for the process lifetime. The extended $PQ$ / Stage-D mod-down machinery is built and gated but not engaged in production. Correctness against an independent oracle is Section 5.3 only; 110k panels are synthetic. No clinical claim is made: PsychENCODE/PGC data behind [1] was not used.

Methodologically, profiling redirected this work more than any kernel did: a quadratic host multiply was 90% of encrypted runtime; reading their Algorithm 1 cut rotations from 182 to 13 per patient; matching ct×ct exposed an i128 overflow in relin-key generation that precision gates on ct×pt would never have caught.

The most valuable next experiment is bounding Metal/host caches so patient-packed RSS stays flat as $n_{\mathrm{snp}}$ grows to 110k, then measuring that path end-to-end against their 1,146-patient server figure under matched stage splits. Beyond that: keep digit buffers resident across rotations, engage Stage-D mod-down for deeper circuits, and obtain a bit-precise $\lambda$ from the lattice estimator.

### 9. References

[1] E. Knight, J. Li, M. Jensen, M. Gerstein, *et al.*, "Homomorphic encryption enables privacy-preserving polygenic risk scores," *Cell Reports Methods*, vol. 6, art. 101271, January 2026. DOI: [10.1016/j.crmeth.2025.101271](https://doi.org/10.1016/j.crmeth.2025.101271). Code: [github.com/gersteinlab/HEPRS](https://github.com/gersteinlab/HEPRS).

[2] M. Albrecht, M. Chase, H. Chen, J. Ding, S. Goldwasser, S. Gorbunov, S. Halevi, J. Hoffstein, K. Laine, K. Lauter, S. Lokam, D. Micciancio, D. Moody, T. Morrison, A. Sahai, and V. Vaikuntanathan, "Homomorphic Encryption Security Standard," *HomomorphicEncryption.org*, Toronto, Canada, November 2018.
