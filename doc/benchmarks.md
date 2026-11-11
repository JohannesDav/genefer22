# Benchmark results

Each comparison runs a complete test of the same candidate on the same machine, once with the original genefer22 and once with this fork. Times include proof generation.

- The original version was compiled from Gallot’s commit `56905b2339b3fed056fc0c24d2e331d588d028bb`.
- Both versions ran Gallot’s Gerbicz–Li error check. This fork’s optional `--check-certificate` and `--check-tail` checks were off.
- Each subproject used the same candidate on every GPU, with a round base near PrimeGrid’s search range in November 2026, according to its [project status page](https://www.primegrid.com/server_status_subprojects.php): 620,000,000 for GFN-16, 520,000,000 for GFN-17-Mega, 100,000,000 for GFN-18, 30,000,000 for GFN-19, 8,000,000 for GFN-20, 2,800,000 for GFN-21 and 660,000 for GFN-22.

In every comparison, the proof files were identical and passed Gallot’s certificate check.

Each cell shows seconds with the original version → this fork (time saved). A dash means the comparison was not run because of testing time and cost; the combination is still supported.

| GPU | GFN-16 | GFN-17 | GFN-18 | GFN-19 |
|---|---:|---:|---:|---:|
| RTX 2080 Ti | 40.5 → 33.1 (18.3%) | 121.1 → 93.6 (22.7%) | 375.9 → 253.9 (32.5%) | 1501.3 → 842.3 (43.9%) |
| RTX 4060 Ti | 38.8 → 32.4 (16.6%) | 113.7 → 89.9 (20.9%) | 353.6 → 232.4 (34.3%) | 1209.6 → 720.1 (40.5%) |
| RTX 5070 | 45.1 → 35.5 (21.3%) | 86.8 → 69.0 (20.4%) | 240.7 → 166.9 (30.7%) | 834.8 → 500.1 (40.1%) |
| RTX 5090 | — | — | — | — |

| GPU | GFN-20 | GFN-21 | GFN-22 |
|---|---:|---:|---:|
| RTX 2080 Ti | — | — | — |
| RTX 4060 Ti | 4376.0 → 2282.6 (47.8%) | — | — |
| RTX 5070 | 3017.9 → 1594.2 (47.2%) | — | — |
| RTX 5090 | 1378.7 → 870.9 (36.8%) | 5308.9 → 2547.7 (52.0%) | 15157.8 → 7961.4 (47.5%) |

## Comparison with CUDA

Each cell shows how much of CUDA's time saving this OpenCL version achieves. **100% means equal savings; higher is better.** For example, if CUDA saves 40% of upstream runtime and OpenCL saves 30%, the table shows 75%. Times include proof generation.

The OpenCL and CUDA measurements used different rentals, drivers and power settings, so this is an approximate comparison.

| GPU | GFN-16 | GFN-17 | GFN-18 | GFN-19 |
|---|---:|---:|---:|---:|
| RTX 2080 Ti | 51% | 90% | 100% | 95% |
| RTX 4060 Ti | 50% | 65% | 87% | 96% |
| RTX 5070 | 48% | 77% | 90% | 98% |
| RTX 5090 | — | — | — | — |

| GPU | GFN-20 | GFN-21 | GFN-22 |
|---|---:|---:|---:|
| RTX 2080 Ti | — | — | — |
| RTX 4060 Ti | 96% | — | — |
| RTX 5070 | 99% | — | — |
| RTX 5090 | 91% | 92% | 89% |
