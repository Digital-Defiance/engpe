# Parameter note: ENGPE vs HEPRS `PN13QP202pq`

## What HEPRS uses (Lattigo v3)

```text
PN13QP202pq:
  LogN = 13          (N = 8192)
  LogSlots = 12
  Q = [33-bit] + 5×[27-bit]     ≈ 168 bits
  P = [34-bit]                  ≈ 34 bits
  log₂(QP) ≈ 202
  DefaultScale = 2^27
```

## What ENGPE ships (measured)

Apple Metal NTT limbs must satisfy `q_i < 2^31` and `q_i ≡ 1 (mod 2N)`.
Lattigo’s 33–34-bit primes are **not** Metal-eligible, so we match the
**security / depth budget**, not the bit-identical prime list.

| Quantity | HEPRS | ENGPE FHE path |
| --- | --- | --- |
| Ring degree | $N = 8192$ | $N = 8192$ (example) / $16384$ (110k) |
| Limb count | 6 Q + 1 P | **7** Q limbs (`FHE_RNS_LIMBS`) |
| $\|Q\|$ / $\log_2(QP)$ | QP ≈ 202 | **$\|Q\| = 217$ bits** (measured) |
| HE ceiling @ $N=8192$ | — | 218 (one bit of margin) |
| Scale | $2^{27}$ | $2^{40}$ |
| Coeff type | Lattigo RNS | `ethnum::U256` host + `u64` limbs |
| Circuit | depth-1 ct×ct + InnerSumLog | **same** |

Ring / plaintext path keeps `RING_RNS_LIMBS = 4` ($\|Q\|\approx 124$) for throughput.

Gate: `rns::tests::fhe_basis_matches_heprs_qp_budget` prints
`[fhe] N=8192 FHE_RNS_LIMBS=7 |Q| bits=217`.

## Estimator caveat

Envelope membership ≠ a bit-precise $\lambda$. Co-author review should run the
lattice estimator on both the Lattigo literal and this 7-limb set.

## Packed RSS (deferred)

Patient-packed streaming cut cohort RSS from ~9 GB (materialised SNP-pack) to
~0.4–1.4 GB, but RSS still creeps with SNP count (Metal/host caches around
relin). **Bounding that cache is deferred** — parameter parity was the higher
bar for calling the stacks equivalent.
