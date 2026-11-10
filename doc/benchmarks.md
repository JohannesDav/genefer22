# Benchmark results

Each comparison runs a complete test of the same candidate on the same machine, once with the original genefer22 and once with this fork. Times include proof generation.

- The original version was compiled from Gallot’s commit `56905b2339b3fed056fc0c24d2e331d588d028bb`.
- Both versions ran Gallot’s Gerbicz–Li error check. This fork’s optional `--check-certificate` and `--check-tail` checks were off.
- Each subproject used the same candidate on every GPU, with a round base near PrimeGrid’s search range in November 2026, according to its [project status page](https://www.primegrid.com/server_status_subprojects.php): 620,000,000 for GFN-16, 520,000,000 for GFN-17-Mega, 100,000,000 for GFN-18, 30,000,000 for GFN-19, 8,000,000 for GFN-20, 2,800,000 for GFN-21 and 660,000 for GFN-22.

In every comparison, the two versions wrote identical proof files, and each proof passed Gallot’s certificate check.

Each cell shows the time in seconds with the original version → with this fork (time saved). A dash means we skipped that comparison because of the time and cost involved, not because the combination is unsupported.

| GPU | GFN-16 | GFN-17 | GFN-18 | GFN-19 |
|---|---:|---:|---:|---:|
| GTX 1080 Ti | 118.3 → 92.2 (22.0%) | 332.0 → 257.8 (22.4%) | 1612.9 → 1070.7 (33.6%) | 5369.4 → 3946.7 (26.5%) |
| GTX TITAN X | 189.9 → 149.1 (21.5%) | 576.0 → 458.7 (20.4%) | — | — |
| RTX 2060 | 61.6 → 46.8 (24.1%) | 193.8 → 150.4 (22.4%) | — | — |
| RTX 2080 Ti | 50.3 → 32.5 (35.5%) | 126.3 → 94.3 (25.3%) | 366.9 → 247.8 (32.5%) | 1519.5 → 815.8 (46.3%) |
| RTX 3060 12GB | 59.2 → 41.2 (30.3%) | 196.5 → 134.1 (31.8%) | 663.6 → 375.0 (43.5%) | 2830.1 → 1594.6 (43.7%) |
| RTX 3070 | 57.3 → 32.2 (43.7%) | 139.5 → 92.7 (33.6%) | 442.6 → 251.0 (43.3%) | — |
| RTX 3080 | 45.4 → 30.5 (32.9%) | 130.3 → 89.6 (31.2%) | — | — |
| RTX 3090 | 41.5 → 25.3 (39.1%) | 109.9 → 69.7 (36.6%) | 372.4 → 222.6 (40.2%) | 1464.2 → 746.2 (49.0%) |
| RTX 4060 Ti 16GB | 38.6 → 25.8 (33.2%) | 111.7 → 76.0 (32.0%) | 354.9 → 215.4 (39.3%) | 1183.2 → 683.9 (42.2%) |
| RTX 4070 | 41.0 → 23.9 (41.7%) | 103.7 → 66.9 (35.5%) | — | 945.8 → 542.3 (42.7%) |
| RTX 4090 | 42.0 → 18.1 (57.0%) | 82.7 → 41.9 (49.3%) | 166.7 → 92.4 (44.6%) | 451.5 → 263.3 (41.7%) |
| RTX 5070 | 35.1 → 19.5 (44.4%) | 76.1 → 55.8 (26.7%) | 241.6 → 159.4 (34.0%) | 844.0 → 499.9 (40.8%) |
| RTX 5080 | 51.6 → 24.1 (53.3%) | 109.6 → 60.0 (45.2%) | — | — |
| RTX 5090 | 44.6 → 17.7 (60.3%) | 87.6 → 39.3 (55.1%) | 159.7 → 86.9 (45.6%) | 361.3 → 234.5 (35.1%) |
| Tesla V100 | 61.3 → 40.4 (34.1%) | 145.3 → 95.3 (34.4%) | — | — |

| GPU | GFN-20 | GFN-21 | GFN-22 |
|---|---:|---:|---:|
| GTX 1080 Ti | 19710.0 → 12210.6 (38.0%) | — | — |
| RTX 2080 Ti | 6336.9 → 3167.4 (50.0%) | — | — |
| RTX 3060 12GB | 9483.9 → 5087.8 (46.4%) | — | — |
| RTX 3080 | 7412.5 → 3771.5 (49.1%) | — | — |
| RTX 3090 | 4431.6 → 2135.7 (51.8%) | 31500.4 → 12147.2 (61.4%) | — |
| RTX 4060 Ti 16GB | 4419.9 → 2222.5 (49.7%) | 22186.9 → 8769.6 (60.5%) | — |
| RTX 4090 | 1580.7 → 854.8 (45.9%) | 5605.7 → 2735.0 (51.2%) | 19801.8 → 8631.3 (56.4%) |
| RTX 5070 | 3083.0 → 1616.4 (47.6%) | 13223.3 → 5761.2 (56.4%) | — |
| RTX 5080 | — | 9268.6 → 3894.3 (58.0%) | — |
| RTX 5090 | 1122.6 → 669.2 (40.4%) | 5304.4 → 2306.2 (56.5%) | 14611.3 → 6840.8 (53.2%) |
