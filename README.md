# Edge-Native Genomic Privacy Engine (ENGPE)

CKKS-based Polygenic Risk Score (PRS) calculator for Apple Silicon, built on the
hardware-in-the-loop FHE sandbox characterised in [`fhe-evolve`](./fhe-evolve)
(see `fhe-evolve/paper/main.pdf`).

## Status

- **Task 1** — synthetic 110k panel, flat FFI packing, Metal-mapped stubs
- **Task 2** — CPU CKKS encode (canonical embedding), negacyclic NTT multiply
  (vendored/extended from fhe-evolve), Galois rotate-and-sum, decoded vs
  `plaintextPrs` oracle

Default CPU scale is `2^16` (single-prime NTT). Scale `2^40` needs RNS limbs.

## Layout

```
src/           TypeScript orchestration + packing
native/src/
  ntt.rs           Wide-modulus negacyclic NTT (from fhe-evolve, q < 2^63)
  ckks_encode.rs   Canonical embedding encode/decode
  ckks_rotate.rs   X ↦ X^{5^k} automorphism + rotate-and-sum
  ckks_eval.rs     Rayon batch pipeline
  ckks_prs.rs      FFI unpack + oracle
  lib.rs           napi exports
```

## Commands

```bash
npm install
npm test
npm run build:native
npm run generate:data
cd native && cargo test
```
