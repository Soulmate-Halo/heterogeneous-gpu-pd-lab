# Changelog

[简体中文](CHANGELOG_ZH.md)

## v2.5

- Named the final asynchronous layered-PD form **Dense Acceleration** and defined its concurrent compute window as the **Dense Region**.
- Published the final Dense Acceleration checkpoint at 2129.69 tok/s, while keeping the v2.3-only 37.16 tok/s decode result explicitly separated.
- Clarified that the Dense Region is micro-batch scheduling overlap, not duplicate same-layer compute, tensor parallelism, or duplicated weights.
- Added the separate-envelope 27B IQ3 validation: 658.52 tok/s at pp4096, 319.10 tok/s at pp65536, 225.10 tok/s at pp98304, and 19.57 tok/s decode.
- Made English the default repository language and added a complete Chinese mirror.
- Added the Soulmate spirit mark and AI Max+ 395 acceleration to the project title.
- Added two source-linked DGX Spark community controls with explicit non-comparability metadata.
- Removed public layer-allocation ratios and replaced them with revision labels.

## v2.3

- Published the balance-refinement checkpoint at 1999.51 tok/s prefill and 37.16 tok/s decode.

## v2.2

- Published the overlap-refinement checkpoint at 1893.87 tok/s prefill.

## v2.1

- Published the first fused-pipeline checkpoint at 1865.08 tok/s prefill.

## v1.0

- Published independent heterogeneous PD with four measured handoff nodes.
- Documented the single-host 9B validation boundary.
