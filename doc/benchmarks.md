# Complete-run benchmarks

Each row compares the time taken by Gallot’s version and this fork to test the same candidate on the same machine. Durations are in seconds and include proof generation. Both versions use the standard Gerbicz–Li error check; this fork’s optional `--check-certificate` and `--check-tail` checks are disabled. “Upstream” means code compiled from Gallot’s commit `56905b2339b3fed056fc0c24d2e331d588d028bb`, rather than PrimeGrid’s distributed executable. GPU clocks, power limits, drivers and the rest of the machine can affect the times.

The “Base” column identifies the candidate tested: for GFN size `n` and base `b`, the number is `b^(2^n) + 1`. We chose round benchmark bases near the ranges PrimeGrid was searching on 2026-11-10, according to its [project status page](https://www.primegrid.com/server_status_subprojects.php). These inputs do not define the program’s supported base range.

Every pair of runs produced byte-identical proof files. After timing the runs, we used Gallot’s original implementation to generate a certificate from each compared proof and check that certificate. Those separate checks passed and are not included in the timings.

The table contains all 65 completed comparisons, including cards omitted from the shorter README table. Short tests used to choose GPU settings are not included here; the [calibrated GPU list](calibrated-gpus.md) describes that broader testing.

| GPU | GFN | Base | Upstream (s) | This fork (s) | Time saved |
|---|---:|---:|---:|---:|---:|
| GTX 1080 Ti | 16 | 620,000,000 | 118.3 | 92.2 | 22.0% |
| GTX 1080 Ti | 17 | 520,000,000 | 332.0 | 257.8 | 22.4% |
| GTX 1080 Ti | 18 | 100,000,000 | 1612.9 | 1070.7 | 33.6% |
| GTX 1080 Ti | 19 | 30,000,000 | 5369.4 | 3946.7 | 26.5% |
| GTX 1080 Ti | 20 | 8,000,000 | 19710.0 | 12210.6 | 38.0% |
| GTX TITAN X | 16 | 620,000,000 | 189.9 | 149.1 | 21.5% |
| GTX TITAN X | 17 | 520,000,000 | 576.0 | 458.7 | 20.4% |
| RTX 2060 | 16 | 620,000,000 | 61.6 | 46.8 | 24.1% |
| RTX 2060 | 17 | 520,000,000 | 193.8 | 150.4 | 22.4% |
| RTX 2080 Ti | 16 | 620,000,000 | 50.3 | 32.5 | 35.5% |
| RTX 2080 Ti | 17 | 520,000,000 | 126.3 | 94.3 | 25.3% |
| RTX 2080 Ti | 18 | 100,000,000 | 366.9 | 247.8 | 32.5% |
| RTX 2080 Ti | 19 | 30,000,000 | 1519.5 | 815.8 | 46.3% |
| RTX 2080 Ti | 20 | 8,000,000 | 6336.9 | 3167.4 | 50.0% |
| RTX 3060 | 16 | 620,000,000 | 59.2 | 41.2 | 30.3% |
| RTX 3060 | 17 | 520,000,000 | 196.5 | 134.1 | 31.8% |
| RTX 3060 | 18 | 100,000,000 | 663.6 | 375.0 | 43.5% |
| RTX 3060 | 19 | 30,000,000 | 2830.1 | 1594.6 | 43.7% |
| RTX 3060 | 20 | 8,000,000 | 9483.9 | 5087.8 | 46.4% |
| RTX 3070 | 16 | 620,000,000 | 57.3 | 32.2 | 43.7% |
| RTX 3070 | 17 | 520,000,000 | 139.5 | 92.7 | 33.6% |
| RTX 3070 | 18 | 100,000,000 | 442.6 | 251.0 | 43.3% |
| RTX 3080 | 16 | 620,000,000 | 45.4 | 30.5 | 32.9% |
| RTX 3080 | 17 | 520,000,000 | 130.3 | 89.6 | 31.2% |
| RTX 3080 | 20 | 8,000,000 | 7412.5 | 3771.5 | 49.1% |
| RTX 3090 | 16 | 620,000,000 | 41.5 | 25.3 | 39.1% |
| RTX 3090 | 17 | 520,000,000 | 109.9 | 69.7 | 36.6% |
| RTX 3090 | 18 | 100,000,000 | 372.4 | 222.6 | 40.2% |
| RTX 3090 | 19 | 30,000,000 | 1464.2 | 746.2 | 49.0% |
| RTX 3090 | 20 | 8,000,000 | 4431.6 | 2135.7 | 51.8% |
| RTX 3090 | 21 | 2,800,000 | 31500.4 | 12147.2 | 61.4% |
| RTX 4060 Ti | 16 | 620,000,000 | 38.6 | 25.8 | 33.2% |
| RTX 4060 Ti | 17 | 520,000,000 | 111.7 | 76.0 | 32.0% |
| RTX 4060 Ti | 18 | 100,000,000 | 354.9 | 215.4 | 39.3% |
| RTX 4060 Ti | 19 | 30,000,000 | 1183.2 | 683.9 | 42.2% |
| RTX 4060 Ti | 20 | 8,000,000 | 4419.9 | 2222.5 | 49.7% |
| RTX 4060 Ti | 21 | 2,800,000 | 22186.9 | 8769.6 | 60.5% |
| RTX 4070 | 16 | 620,000,000 | 41.0 | 23.9 | 41.7% |
| RTX 4070 | 17 | 520,000,000 | 103.7 | 66.9 | 35.5% |
| RTX 4070 | 19 | 30,000,000 | 945.8 | 542.3 | 42.7% |
| RTX 4090 | 16 | 620,000,000 | 42.0 | 18.1 | 57.0% |
| RTX 4090 | 17 | 520,000,000 | 82.7 | 41.9 | 49.3% |
| RTX 4090 | 18 | 100,000,000 | 166.7 | 92.4 | 44.6% |
| RTX 4090 | 19 | 30,000,000 | 451.5 | 263.3 | 41.7% |
| RTX 4090 | 20 | 8,000,000 | 1580.7 | 854.8 | 45.9% |
| RTX 4090 | 21 | 2,800,000 | 5605.7 | 2735.0 | 51.2% |
| RTX 4090 | 22 | 660,000 | 19801.8 | 8631.3 | 56.4% |
| RTX 5070 | 16 | 620,000,000 | 35.1 | 19.5 | 44.4% |
| RTX 5070 | 17 | 520,000,000 | 76.1 | 55.8 | 26.7% |
| RTX 5070 | 18 | 100,000,000 | 241.6 | 159.4 | 34.0% |
| RTX 5070 | 19 | 30,000,000 | 844.0 | 499.9 | 40.8% |
| RTX 5070 | 20 | 8,000,000 | 3083.0 | 1616.4 | 47.6% |
| RTX 5070 | 21 | 2,800,000 | 13223.3 | 5761.2 | 56.4% |
| RTX 5080 | 16 | 620,000,000 | 51.6 | 24.1 | 53.3% |
| RTX 5080 | 17 | 520,000,000 | 109.6 | 60.0 | 45.2% |
| RTX 5080 | 21 | 2,800,000 | 9268.6 | 3894.3 | 58.0% |
| RTX 5090 | 16 | 620,000,000 | 44.6 | 17.7 | 60.3% |
| RTX 5090 | 17 | 520,000,000 | 87.6 | 39.3 | 55.1% |
| RTX 5090 | 18 | 100,000,000 | 159.7 | 86.9 | 45.6% |
| RTX 5090 | 19 | 30,000,000 | 361.3 | 234.5 | 35.1% |
| RTX 5090 | 20 | 8,000,000 | 1122.6 | 669.2 | 40.4% |
| RTX 5090 | 21 | 2,800,000 | 5304.4 | 2306.2 | 56.5% |
| RTX 5090 | 22 | 660,000 | 14611.3 | 6840.8 | 53.2% |
| Tesla V100 | 16 | 620,000,000 | 61.3 | 40.4 | 34.1% |
| Tesla V100 | 17 | 520,000,000 | 145.3 | 95.3 | 34.4% |

## Comparison with the earlier GFN-19-only version

This separate test compares the earlier version tuned for GFN-19 on the RTX 3090 and RTX 4060 Ti with the version that supports GFN-16 through GFN-22. Both tested base 28,000,000. We measured two complete runs of each version on each card after warming the driver’s compilation cache, alternating the versions in earlier/new/new/earlier order. All proof files matched and the certificate checks passed.

The machines differed from those used in the table above. These figures are useful for comparing the two versions within each row, rather than comparing absolute times between tables.

| GPU | Earlier GFN-19-only version (s) | This version (s) | Change in duration |
|---|---:|---:|---:|
| RTX 3090 | 792.0 | 791.0 | -0.1% |
| RTX 4060 Ti 16GB | 762.6 | 754.1 | -1.1% |

The small differences are consistent with normal timing variation; these runs did not show a performance loss from adding support for other GPUs and GFN sizes.
