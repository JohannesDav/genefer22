# GPU profiles

OpenCL calibration tested alternative layouts against GMP on the cards marked below for GFN-16 through GFN-22. The program can retain an existing layout when the new one offers no clear gain. Memory capacity distinguishes variants of the same card.

Profiles for other models use layouts measured on the CUDA branch. The program still checks the arithmetic bounds and device resources before using those layouts. See the [complete OpenCL benchmarks](benchmarks.md) for comparisons that include proof generation.

| GPU | Memory | OpenCL calibration |
|---|---:|---|
| NVIDIA A100-PCIE-40GB | 40 GB | Yes |
| NVIDIA GeForce GTX 1070 | 8 GB | Yes |
| NVIDIA GeForce GTX 1070 Ti | 8 GB | Yes |
| NVIDIA GeForce GTX 1080 Ti | 11 GB | Yes |
| NVIDIA GeForce GTX 1660 SUPER | 6 GB | — |
| NVIDIA GeForce GTX 1660 Ti | 6 GB | Yes |
| NVIDIA GeForce GTX TITAN X | 12 GB | Yes |
| NVIDIA GeForce RTX 2060 | 12 GB | Yes |
| NVIDIA GeForce RTX 2070 | 8 GB | Yes |
| NVIDIA GeForce RTX 2070 SUPER | 8 GB | Yes |
| NVIDIA GeForce RTX 2080 Ti | 11 GB | Yes |
| NVIDIA GeForce RTX 3050 | 8 GB | — |
| NVIDIA GeForce RTX 3060 | 12 GB | Yes |
| NVIDIA GeForce RTX 3060 Ti | 8 GB | Yes |
| NVIDIA GeForce RTX 3070 | 8 GB | Yes |
| NVIDIA GeForce RTX 3070 Ti | 8 GB | Yes |
| NVIDIA GeForce RTX 3080 | 10 GB | Yes |
| NVIDIA GeForce RTX 3080 Ti | 12 GB | Yes |
| NVIDIA GeForce RTX 3090 | 24 GB | Yes |
| NVIDIA GeForce RTX 4060 | 8 GB | Yes |
| NVIDIA GeForce RTX 4060 Ti | 8 GB | Yes |
| NVIDIA GeForce RTX 4060 Ti | 16 GB | Yes |
| NVIDIA GeForce RTX 4070 | 12 GB | Yes |
| NVIDIA GeForce RTX 4070 Ti | 12 GB | Yes |
| NVIDIA GeForce RTX 4080 | 16 GB | Yes |
| NVIDIA GeForce RTX 4080 SUPER | 16 GB | Yes |
| NVIDIA GeForce RTX 4090 | 24 GB | — |
| NVIDIA GeForce RTX 5060 | 8 GB | Yes |
| NVIDIA GeForce RTX 5060 Ti | 16 GB | Yes |
| NVIDIA GeForce RTX 5070 | 12 GB | Yes |
| NVIDIA GeForce RTX 5070 Ti | 16 GB | Yes |
| NVIDIA GeForce RTX 5080 | 16 GB | Yes |
| NVIDIA GeForce RTX 5090 | 32 GB | Yes |
| NVIDIA H100 PCIe | 80 GB | Yes |
| NVIDIA RTX A4000 | 16 GB | Yes |
| NVIDIA RTX A5000 | 24 GB | Yes |
| NVIDIA RTX A6000 | 48 GB | Yes |
| NVIDIA TITAN V | 12 GB | — |
| NVIDIA TITAN Xp | 12 GB | — |
| Tesla P100-PCIE-16GB | 16 GB | Yes |
| Tesla T4 | 16 GB | Yes |
| Tesla V100-SXM2-32GB | 32 GB | Yes |

When there is no model profile for the GFN size, the program tries a profile for the GPU architecture and then a default for the GFN size. See [how the program chooses its arithmetic](opencl.md#how-the-program-chooses-its-arithmetic).
