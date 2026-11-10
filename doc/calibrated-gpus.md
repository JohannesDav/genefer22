# Calibrated NVIDIA GPUs

All listed configurations were calibrated for GFN-16 through GFN-22, using GMP arithmetic checks and repeated calculation-loop measurements. Complete proof comparisons are listed separately in the [benchmark results](benchmarks.md); their coverage varies by GFN.

| GPU (OpenCL device name) | Memory | Architecture | Validation coverage |
|---|---:|---|---|
| NVIDIA GeForce GTX 1080 Ti | 11 GB | Pascal | Calibration and complete proof comparisons |
| NVIDIA GeForce GTX TITAN X | 12 GB | Maxwell | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 2060 | 12 GB | Turing | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 2080 Ti | 11 GB | Turing | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 3060 | 12 GB | Ampere | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 3070 | 8 GB | Ampere | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 3080 | 10 GB | Ampere | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 3090 | 24 GB | Ampere | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 4060 Ti | 16 GB | Ada | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 4070 | 12 GB | Ada | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 4090 | 24 GB | Ada | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 5070 | 12 GB | Blackwell | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 5080 | 16 GB | Blackwell | Calibration and complete proof comparisons |
| NVIDIA GeForce RTX 5090 | 32 GB | Blackwell | Calibration and complete proof comparisons |
| Tesla V100-SXM2-32GB | 32 GB | Volta | Calibration and complete proof comparisons |
| NVIDIA A100-PCIE-40GB | 40 GB | Ampere (SM80) | Calibration and GMP checks |
| Tesla T4 | 16 GB | Turing | Calibration and GMP checks |
| NVIDIA GeForce RTX 4060 | 8 GB | Ada | Calibration and GMP checks |
| NVIDIA GeForce GTX 1070 | 8 GB | Pascal | Calibration and GMP checks |
| NVIDIA GeForce RTX 3060 Ti | 8 GB | Ampere | Calibration and GMP checks |
| NVIDIA GeForce GTX 1660 Ti | 6 GB | Turing | Calibration and GMP checks |
| NVIDIA GeForce RTX 4080 SUPER | 16 GB | Ada | Calibration and GMP checks |
| Tesla P100-PCIE-16GB | 16 GB | Pascal | Calibration and GMP checks |
| NVIDIA GeForce RTX 3080 Ti | 12 GB | Ampere | Calibration and GMP checks |
| NVIDIA GeForce RTX 3070 Ti | 8 GB | Ampere | Calibration and GMP checks |
| NVIDIA GeForce RTX 4060 Ti | 8 GB | Ada | Calibration and GMP checks |
| NVIDIA H100 PCIe | 80 GB | Hopper | Calibration and GMP checks |
| NVIDIA RTX A4000 | 16 GB | Ampere | Calibration and GMP checks |
| NVIDIA GeForce RTX 4080 | 16 GB | Ada | Calibration and GMP checks |
| NVIDIA GeForce GTX 1070 Ti | 8 GB | Pascal | Calibration and GMP checks |
| NVIDIA GeForce RTX 2070 SUPER | 8 GB | Turing | Calibration and GMP checks |
| NVIDIA GeForce RTX 2070 | 8 GB | Turing | Calibration and GMP checks |
| NVIDIA GeForce RTX 3050 | 8 GB | Ampere | Calibration and GMP checks |
| NVIDIA GeForce GTX 1660 SUPER | 6 GB | Turing | Calibration and GMP checks |
| NVIDIA TITAN Xp | 12 GB | Pascal | Calibration and GMP checks |
| NVIDIA RTX A5000 | 24 GB | Ampere | Calibration and GMP checks |
| NVIDIA TITAN V | 12 GB | Volta | Calibration and GMP checks |
| NVIDIA GeForce RTX 4070 Ti | 12 GB | Ada | Calibration and GMP checks |

Other NVIDIA models use architecture or GFN defaults, subject to the same arithmetic and device-resource checks. Driver and ECC reservations can reduce the memory reported by OpenCL; nominal capacities distinguish variants such as the 8GB and 16GB RTX 4060 Ti.
