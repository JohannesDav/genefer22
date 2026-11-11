# Calibrated GPUs

The executable contains GPU settings measured on each GPU below, for every GFN size from 16 to 22. On each card, we timed the alternative kernel layouts in short runs, kept the fastest, and checked the arithmetic against GMP.

| GPU | Memory | Architecture |
|---|---:|---|
| NVIDIA GeForce GTX 1080 Ti | 11 GB | Pascal |
| NVIDIA GeForce GTX TITAN X | 12 GB | Maxwell |
| NVIDIA GeForce RTX 2060 | 12 GB | Turing |
| NVIDIA GeForce RTX 2080 Ti | 11 GB | Turing |
| NVIDIA GeForce RTX 3060 | 12 GB | Ampere |
| NVIDIA GeForce RTX 3070 | 8 GB | Ampere |
| NVIDIA GeForce RTX 3080 | 10 GB | Ampere |
| NVIDIA GeForce RTX 3090 | 24 GB | Ampere |
| NVIDIA GeForce RTX 4060 Ti | 16 GB | Ada |
| NVIDIA GeForce RTX 4070 | 12 GB | Ada |
| NVIDIA GeForce RTX 4090 | 24 GB | Ada |
| NVIDIA GeForce RTX 5070 | 12 GB | Blackwell |
| NVIDIA GeForce RTX 5080 | 16 GB | Blackwell |
| NVIDIA GeForce RTX 5090 | 32 GB | Blackwell |
| Tesla V100-SXM2-32GB | 32 GB | Volta |
| NVIDIA A100-PCIE-40GB | 40 GB | Ampere (A100 class) |
| Tesla T4 | 16 GB | Turing |
| NVIDIA GeForce RTX 4060 | 8 GB | Ada |
| NVIDIA GeForce GTX 1070 | 8 GB | Pascal |
| NVIDIA GeForce RTX 3060 Ti | 8 GB | Ampere |
| NVIDIA GeForce GTX 1660 Ti | 6 GB | Turing |
| NVIDIA GeForce RTX 4080 SUPER | 16 GB | Ada |
| Tesla P100-PCIE-16GB | 16 GB | Pascal |
| NVIDIA GeForce RTX 3080 Ti | 12 GB | Ampere |
| NVIDIA GeForce RTX 3070 Ti | 8 GB | Ampere |
| NVIDIA GeForce RTX 4060 Ti | 8 GB | Ada |
| NVIDIA H100 PCIe | 80 GB | Hopper |
| NVIDIA RTX A4000 | 16 GB | Ampere |
| NVIDIA GeForce RTX 4080 | 16 GB | Ada |
| NVIDIA GeForce GTX 1070 Ti | 8 GB | Pascal |
| NVIDIA GeForce RTX 2070 SUPER | 8 GB | Turing |
| NVIDIA GeForce RTX 2070 | 8 GB | Turing |
| NVIDIA GeForce RTX 3050 | 8 GB | Ampere |
| NVIDIA GeForce GTX 1660 SUPER | 6 GB | Turing |
| NVIDIA TITAN Xp | 12 GB | Pascal |
| NVIDIA RTX A5000 | 24 GB | Ampere |
| NVIDIA TITAN V | 12 GB | Volta |
| NVIDIA GeForce RTX 4070 Ti | 12 GB | Ada |

On an NVIDIA GPU that is not listed, the program uses settings chosen for other cards of the same architecture, or general settings for the GFN size. See [how the program chooses its arithmetic](cuda.md#how-the-program-chooses-its-arithmetic).
