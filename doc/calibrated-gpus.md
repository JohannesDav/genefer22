# Calibrated GPUs

The executable contains GPU settings measured on each GPU and memory size below, for every GFN size from 16 to 22. We call this calibration: on each card, we timed the alternative kernel layouts in short runs, kept the fastest, and checked the arithmetic against GMP.

The last column shows the cards where we also ran complete tests and compared the proof files with Gallot’s version. The [benchmark results](benchmarks.md) list which GFN sizes.

| GPU (name reported by the driver) | Memory | Architecture | Complete tests compared |
|---|---:|---|---|
| NVIDIA GeForce GTX 1080 Ti | 11 GB | Pascal | Yes |
| NVIDIA GeForce GTX TITAN X | 12 GB | Maxwell | Yes |
| NVIDIA GeForce RTX 2060 | 12 GB | Turing | Yes |
| NVIDIA GeForce RTX 2080 Ti | 11 GB | Turing | Yes |
| NVIDIA GeForce RTX 3060 | 12 GB | Ampere | Yes |
| NVIDIA GeForce RTX 3070 | 8 GB | Ampere | Yes |
| NVIDIA GeForce RTX 3080 | 10 GB | Ampere | Yes |
| NVIDIA GeForce RTX 3090 | 24 GB | Ampere | Yes |
| NVIDIA GeForce RTX 4060 Ti | 16 GB | Ada | Yes |
| NVIDIA GeForce RTX 4070 | 12 GB | Ada | Yes |
| NVIDIA GeForce RTX 4090 | 24 GB | Ada | Yes |
| NVIDIA GeForce RTX 5070 | 12 GB | Blackwell | Yes |
| NVIDIA GeForce RTX 5080 | 16 GB | Blackwell | Yes |
| NVIDIA GeForce RTX 5090 | 32 GB | Blackwell | Yes |
| Tesla V100-SXM2-32GB | 32 GB | Volta | Yes |
| NVIDIA A100-PCIE-40GB | 40 GB | Ampere (A100 class) | No |
| Tesla T4 | 16 GB | Turing | No |
| NVIDIA GeForce RTX 4060 | 8 GB | Ada | No |
| NVIDIA GeForce GTX 1070 | 8 GB | Pascal | No |
| NVIDIA GeForce RTX 3060 Ti | 8 GB | Ampere | No |
| NVIDIA GeForce GTX 1660 Ti | 6 GB | Turing | No |
| NVIDIA GeForce RTX 4080 SUPER | 16 GB | Ada | No |
| Tesla P100-PCIE-16GB | 16 GB | Pascal | No |
| NVIDIA GeForce RTX 3080 Ti | 12 GB | Ampere | No |
| NVIDIA GeForce RTX 3070 Ti | 8 GB | Ampere | No |
| NVIDIA GeForce RTX 4060 Ti | 8 GB | Ada | No |
| NVIDIA H100 PCIe | 80 GB | Hopper | No |
| NVIDIA RTX A4000 | 16 GB | Ampere | No |
| NVIDIA GeForce RTX 4080 | 16 GB | Ada | No |
| NVIDIA GeForce GTX 1070 Ti | 8 GB | Pascal | No |
| NVIDIA GeForce RTX 2070 SUPER | 8 GB | Turing | No |
| NVIDIA GeForce RTX 2070 | 8 GB | Turing | No |
| NVIDIA GeForce RTX 3050 | 8 GB | Ampere | No |
| NVIDIA GeForce GTX 1660 SUPER | 6 GB | Turing | No |
| NVIDIA TITAN Xp | 12 GB | Pascal | No |
| NVIDIA RTX A5000 | 24 GB | Ampere | No |
| NVIDIA TITAN V | 12 GB | Volta | No |
| NVIDIA GeForce RTX 4070 Ti | 12 GB | Ada | No |

On an NVIDIA GPU that is not listed, the program uses settings chosen for other cards of the same architecture, or general settings for the GFN size. See [how the program chooses its arithmetic](cuda.md#how-the-program-chooses-its-arithmetic).
