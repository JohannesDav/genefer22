# genefer22: experimental CUDA speedups for NVIDIA GPUs

**Warning:** all changes in this fork were written by AI and have not been independently reviewed. Treat it as experimental software.

This is a fork of [Yves Gallot’s genefer22](https://github.com/galloty/genefer22), the program PrimeGrid uses to test numbers of the form b<sup>2<sup>n</sup></sup> + 1 (generalized Fermat numbers) for primality. It changes only the GPU version, and only on NVIDIA cards. It targets PrimeGrid’s GFN-16 to GFN-22 subprojects, where n is 16 to 22.

In our measurements, a complete test took about 20–60% less time than with Gallot’s version, depending on the GPU and the GFN size. The results are the same: in all 65 comparisons, both versions wrote identical proof files.

## What changed

Most of a test is spent squaring a very large number millions of times. Genefer stores the number as a list of base-b digits and squares it with number-theoretic transforms (an exact, integer counterpart of the FFT) computed modulo two or three primes. It then combines those results into the exact answer with the Chinese remainder theorem.

This fork changes how that squaring runs on NVIDIA GPUs:

- **Packed squaring.** Several consecutive base-b digits (up to 16) are combined into each transform element, so the transform is that many times shorter. The elements are larger, so more primes are needed to recover the exact result. The number of primes is chosen for each candidate from a worst-case bound.
- **Alternative three-prime arithmetic.** On some GPUs, mostly for GFN-16 and GFN-17, it is faster to keep Gallot’s transform but use three primes just below 2<sup>30</sup>. These leave room in 32-bit words to postpone modular reductions.
- **Settings for each GPU.** The GPU code has several possible layouts. The program contains the fastest settings measured on [38 GPU models and memory sizes](doc/calibrated-gpus.md) and picks them automatically, so one executable serves every supported card.
- **Less launch overhead.** Work is submitted to the GPU in batches using CUDA graphs, instead of one kernel launch at a time.

## Exactness and fallbacks

None of these changes trade accuracy for speed. Before using an optimization for a candidate, the program checks worst-case bounds showing that every intermediate value stays within the range the arithmetic can represent exactly. Two-prime arithmetic is only used below Gallot’s own limit for it.

If a check fails, the program uses Gallot’s arithmetic instead. It also runs Gallot’s original code, unchanged, on non-NVIDIA GPUs, for GFN sizes outside 16–22, or if CUDA cannot start.

Every test still runs Gallot’s Gerbicz–Li error check. Two optional checks can also verify the finished result with his original code; see [optional extra checks](doc/cuda.md#optional-extra-checks).

## Performance

Times are for one complete test, including proof generation, in seconds: **Gallot’s version → this fork (time saved)**. Gallot’s version was compiled from commit `56905b2`.

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

A dash means we skipped that comparison because of the time and cost involved, not because the combination is unsupported.

Each subproject used the same candidate on every GPU, with a round base near PrimeGrid’s search range in November 2026: 620,000,000 for GFN-16, 520,000,000 for GFN-17-Mega, 100,000,000 for GFN-18, 30,000,000 for GFN-19, 8,000,000 for GFN-20, 2,800,000 for GFN-21 and 660,000 for GFN-22.

Every proof from these comparisons also passed Gallot’s certificate check. The [full results](doc/benchmarks.md) include seven more GPUs and describe the method.

## Building and running

See [building and running the CUDA version](doc/cuda.md). The executable, `geneferg`, takes the same options as Gallot’s GPU version, plus two optional check flags. For example, this tests 28000000<sup>2<sup>19</sup></sup> + 1 and writes the proof to `candidate.proof`:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

Only Linux x86-64 has been tested. PrimeGrid does not distribute this version, so using it for PrimeGrid work means setting up BOINC to run it yourself.
