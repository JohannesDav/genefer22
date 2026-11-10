# genefer22: experimental CUDA GPU optimizations

The modifications in this fork are entirely AI-generated and should not be considered safe or stable.

Based on [Yves Gallot’s genefer22](https://github.com/galloty/genefer22). This fork adds CUDA optimizations for PrimeGrid’s GFN-16, GFN-17-Mega, GFN-18, GFN-19, GFN-20, GFN-21 and GFN-22 subprojects. The [tested NVIDIA GPUs](doc/calibrated-gpus.md) span several generations. Testing was performed on Linux x86-64. Other GPUs use the original OpenCL implementation; CPU builds remain separate.

One GPU executable selects tested GPU settings, then architecture or subproject defaults for unfamiliar cards. It checks the arithmetic bounds for each candidate and retains original arithmetic where an optimized configuration is unavailable. No calibration or Python installation is required to run it.

The optimized kernels pack several digits into each coefficient where useful, allowing shorter transforms. Smaller prime moduli make modular arithmetic cheaper; enough are used to keep reconstruction exact. Transform layouts vary with the GPU and GFN size, and CUDA graphs reduce kernel launch overhead.

## Performance

Complete runs including proof generation, with optional extra checks disabled. Each cell is **upstream → this fork in seconds (time saved)**. The reference is compiled upstream commit `56905b2`, not PrimeGrid's distributed binary.

| GPU | GFN-16 | GFN-17 | GFN-18 | GFN-19 |
|---|---:|---:|---:|---:|
| GTX 1080 Ti | 118.3 → 92.2 (22.0%) | 332.0 → 257.8 (22.4%) | 1612.9 → 1070.7 (33.6%) | 5369.4 → 3946.7 (26.5%) |
| RTX 2080 Ti | 50.3 → 32.5 (35.5%) | 126.3 → 94.3 (25.3%) | 366.9 → 247.8 (32.5%) | 1519.5 → 815.8 (46.3%) |
| RTX 3060 12GB | 59.2 → 41.2 (30.3%) | 196.5 → 134.1 (31.8%) | 663.6 → 375.0 (43.5%) | 2830.1 → 1594.6 (43.7%) |
| RTX 3090 | 41.5 → 25.3 (39.1%) | 109.9 → 69.7 (36.6%) | 372.4 → 222.6 (40.2%) | 1464.2 → 746.2 (49.0%) |
| RTX 4060 Ti 16GB | 38.6 → 25.8 (33.2%) | 111.7 → 76.0 (32.0%) | 354.9 → 215.4 (39.3%) | 1183.2 → 683.9 (42.2%) |
| RTX 4090 | 42.0 → 18.1 (57.0%) | 82.7 → 41.9 (49.3%) | 166.7 → 92.4 (44.6%) | 451.5 → 263.3 (41.7%) |
| RTX 5070 | 35.1 → 19.5 (44.4%) | 76.1 → 55.8 (26.7%) | 241.6 → 159.4 (34.0%) | 844.0 → 499.9 (40.8%) |
| RTX 5090 | 44.6 → 17.7 (60.3%) | 87.6 → 39.3 (55.1%) | 159.7 → 86.9 (45.6%) | 361.3 → 234.5 (35.1%) |

| GPU | GFN-20 | GFN-21 | GFN-22 |
|---|---:|---:|---:|
| GTX 1080 Ti | 19710.0 → 12210.6 (38.0%) | — | — |
| RTX 2080 Ti | 6336.9 → 3167.4 (50.0%) | — | — |
| RTX 3060 12GB | 9483.9 → 5087.8 (46.4%) | — | — |
| RTX 3090 | 4431.6 → 2135.7 (51.8%) | 31500.4 → 12147.2 (61.4%) | — |
| RTX 4060 Ti 16GB | 4419.9 → 2222.5 (49.7%) | 22186.9 → 8769.6 (60.5%) | — |
| RTX 4090 | 1580.7 → 854.8 (45.9%) | 5605.7 → 2735.0 (51.2%) | 19801.8 → 8631.3 (56.4%) |
| RTX 5070 | 3083.0 → 1616.4 (47.6%) | 13223.3 → 5761.2 (56.4%) | — |
| RTX 5090 | 1122.6 → 669.2 (40.4%) | 5304.4 → 2306.2 (56.5%) | 14611.3 → 6840.8 (53.2%) |

— means no complete benchmark was run because of testing time/cost; the combination remains supported.

Bases for GFN-16 through GFN-22: 620,000,000, 520,000,000, 100,000,000, 30,000,000, 8,000,000, 2,800,000, 660,000.

All 65 complete comparisons produced identical proofs and passed upstream certificate checks. [Full results and methodology](doc/benchmarks.md).

## Build and use

Build instructions, optional checks and the standalone calibration tool are documented in [doc/cuda.md](doc/cuda.md). The GPU executable is named `geneferg`, matching upstream. Its normal command-line and BOINC interfaces are retained.

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

This remains experimental software. Proof compatibility and arithmetic tests are not a substitute for independent review or broader deployment testing.
