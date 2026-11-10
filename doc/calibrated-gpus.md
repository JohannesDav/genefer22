# Calibrated NVIDIA GPUs

The executable includes settings tested on each GPU and memory configuration below, for every size from GFN-16 through GFN-22. To choose these settings, we compared different calculation layouts in short, repeated tests and checked their arithmetic against GMP. This is what we call calibration.

For some cards, we also ran complete primality tests and compared their proof files with Gallot’s version. The last column distinguishes that additional testing from calibration alone. It does not mean every GFN size received a complete test on that card; the [benchmark results](benchmarks.md) list the exact combinations tested.

| GPU (name reported by the driver) | Memory | GPU generation | Tests completed |
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

On an NVIDIA model not listed here, the executable first tries settings chosen for other cards of the same generation. If the generation is unknown, it tries general settings for the requested GFN size. It checks the arithmetic bounds and GPU requirements before using either choice. See [how settings are selected](cuda.md#how-settings-are-selected).

Memory capacities in the table are the advertised capacities. The driver can report less usable memory because some is reserved, including for ECC. Separate entries distinguish cards such as the 8GB and 16GB RTX 4060 Ti.
