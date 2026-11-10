# genefer22: experimental CUDA GPU optimizations

The modifications in this fork are entirely AI-generated and should not be considered safe or stable.

Based on [Yves Gallot’s genefer22](https://github.com/galloty/genefer22). This fork speeds up PrimeGrid’s GFN-16, GFN-17-Mega, GFN-18, GFN-19, GFN-20, GFN-21 and GFN-22 tests on NVIDIA GPUs. It has been tested on Linux x86-64 across [38 GPU and memory configurations](doc/calibrated-gpus.md).

The same executable works across GPU models and selects its calculation settings automatically. It checks worst-case arithmetic bounds for each candidate and falls back to the original arithmetic where an optimization cannot be used.

The main changes are shorter transforms that process several digits together, faster modular arithmetic using smaller primes, and transform layouts chosen for the GPU and test size. Enough primes are used to reconstruct results exactly. CUDA graphs also reduce the overhead of launching GPU kernels.

## Performance

Each cell shows **Gallot’s version → this fork, in seconds (percentage of time saved)**. These are complete tests including proof generation. The comparison uses compiled upstream commit `56905b2`, rather than PrimeGrid’s distributed binary. The fork’s optional additional checks are disabled.

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

— means we did not run a complete benchmark because of the time and cost involved. Those GPU/subproject combinations are still supported.

All GPUs used the same benchmark candidate for each subproject. The bases tested were **620,000,000 for GFN-16**, **520,000,000 for GFN-17**, **100,000,000 for GFN-18**, **30,000,000 for GFN-19**, **8,000,000 for GFN-20**, **2,800,000 for GFN-21** and **660,000 for GFN-22**. These are benchmark inputs, not limits on supported bases.

In all 65 complete comparisons, this fork and Gallot’s version produced identical proof files. Those proofs also passed Gallot’s certificate checks. See the [full results and test method](doc/benchmarks.md).

## Build and use

See [build and usage instructions](doc/cuda.md). The executable is named `geneferg` and accepts the same command-line options as Gallot’s GPU version, with two optional verification flags added. For example, to test a GFN-19 candidate with base 28,000,000 and save its proof:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

This remains experimental software. Proof compatibility and arithmetic tests are not a substitute for independent review or broader deployment testing.
