# Calibrated GPUs

The executable contains GPU settings measured with this OpenCL version on each GPU below, for every GFN size from 16 to 22. On each card, we timed the alternative kernel layouts in short runs, kept the fastest, and checked the arithmetic against GMP. Where the layout measured with the [CUDA version](https://github.com/JohannesDav/genefer22) was about as fast, we kept it.

| GPU | Memory | Architecture |
|---|---:|---|
| NVIDIA A100-PCIE-40GB | 40 GB | Ampere (A100 class) |
| NVIDIA GeForce GTX 1070 | 8 GB | Pascal |
| NVIDIA GeForce GTX 1070 Ti | 8 GB | Pascal |
| NVIDIA GeForce GTX 1080 Ti | 11 GB | Pascal |
| NVIDIA GeForce GTX 1660 Ti | 6 GB | Turing |
| NVIDIA GeForce GTX TITAN X | 12 GB | Maxwell |
| NVIDIA GeForce RTX 2060 | 12 GB | Turing |
| NVIDIA GeForce RTX 2070 | 8 GB | Turing |
| NVIDIA GeForce RTX 2070 SUPER | 8 GB | Turing |
| NVIDIA GeForce RTX 2080 Ti | 11 GB | Turing |
| NVIDIA GeForce RTX 3060 | 12 GB | Ampere |
| NVIDIA GeForce RTX 3060 Ti | 8 GB | Ampere |
| NVIDIA GeForce RTX 3070 | 8 GB | Ampere |
| NVIDIA GeForce RTX 3070 Ti | 8 GB | Ampere |
| NVIDIA GeForce RTX 3080 | 10 GB | Ampere |
| NVIDIA GeForce RTX 3080 Ti | 12 GB | Ampere |
| NVIDIA GeForce RTX 3090 | 24 GB | Ampere |
| NVIDIA GeForce RTX 4060 | 8 GB | Ada |
| NVIDIA GeForce RTX 4060 Ti | 8 GB | Ada |
| NVIDIA GeForce RTX 4060 Ti | 16 GB | Ada |
| NVIDIA GeForce RTX 4070 | 12 GB | Ada |
| NVIDIA GeForce RTX 4070 Ti | 12 GB | Ada |
| NVIDIA GeForce RTX 4080 | 16 GB | Ada |
| NVIDIA GeForce RTX 4080 SUPER | 16 GB | Ada |
| NVIDIA GeForce RTX 5060 | 8 GB | Blackwell |
| NVIDIA GeForce RTX 5060 Ti | 16 GB | Blackwell |
| NVIDIA GeForce RTX 5070 | 12 GB | Blackwell |
| NVIDIA GeForce RTX 5070 Ti | 16 GB | Blackwell |
| NVIDIA GeForce RTX 5080 | 16 GB | Blackwell |
| NVIDIA GeForce RTX 5090 | 32 GB | Blackwell |
| NVIDIA H100 PCIe | 80 GB | Hopper |
| NVIDIA RTX A4000 | 16 GB | Ampere |
| NVIDIA RTX A5000 | 24 GB | Ampere |
| NVIDIA RTX A6000 | 48 GB | Ampere |
| Tesla P100-PCIE-16GB | 16 GB | Pascal |
| Tesla T4 | 16 GB | Turing |
| Tesla V100-SXM2-32GB | 32 GB | Volta |

Other NVIDIA GPUs use settings measured with the CUDA version where those exist, or else settings chosen for other cards of the same architecture, or general settings for the GFN size. See [how the program chooses its arithmetic](opencl.md#how-the-program-chooses-its-arithmetic).
