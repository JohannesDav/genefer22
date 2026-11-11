# Calibrated GPUs

The executable contains GPU settings measured with this OpenCL version on each GPU below, for every GFN size from 16 to 22. On each card, we timed the alternative kernel layouts in short runs, kept the fastest, and checked the arithmetic against GMP. Where the layout measured with the [CUDA version](https://github.com/JohannesDav/genefer22) was about as fast, we kept it.

Other NVIDIA GPUs use settings measured with the CUDA version where those exist, or else settings chosen for other cards of the same architecture, or general settings for the GFN size. See [how the program chooses its arithmetic](opencl.md#how-the-program-chooses-its-arithmetic).

| GPU | Memory | Architecture |
|---|---:|---|
| NVIDIA GeForce RTX 5090 | 32 GB | Blackwell |
| NVIDIA GeForce RTX 5090 D | 32 GB | Blackwell |
| NVIDIA GeForce RTX 5080 | 16 GB | Blackwell |
| NVIDIA GeForce RTX 5070 Ti | 16 GB | Blackwell |
| NVIDIA GeForce RTX 5070 | 12 GB | Blackwell |
| NVIDIA GeForce RTX 5060 Ti | 16 GB | Blackwell |
| NVIDIA GeForce RTX 5060 Ti | 8 GB | Blackwell |
| NVIDIA GeForce RTX 5060 | 8 GB | Blackwell |
| NVIDIA GeForce RTX 4090 | 48 GB | Ada |
| NVIDIA GeForce RTX 4090 | 24 GB | Ada |
| NVIDIA GeForce RTX 4080 SUPER | 32 GB | Ada |
| NVIDIA GeForce RTX 4080 SUPER | 16 GB | Ada |
| NVIDIA GeForce RTX 4080 | 16 GB | Ada |
| NVIDIA GeForce RTX 4080 Laptop GPU | 12 GB | Ada |
| NVIDIA GeForce RTX 4070 Ti SUPER | 16 GB | Ada |
| NVIDIA GeForce RTX 4070 Ti | 12 GB | Ada |
| NVIDIA GeForce RTX 4070 SUPER | 12 GB | Ada |
| NVIDIA GeForce RTX 4070 | 12 GB | Ada |
| NVIDIA GeForce RTX 4060 Ti | 16 GB | Ada |
| NVIDIA GeForce RTX 4060 Ti | 8 GB | Ada |
| NVIDIA GeForce RTX 4060 | 8 GB | Ada |
| NVIDIA GeForce RTX 4060 Laptop GPU | 8 GB | Ada |
| NVIDIA GeForce RTX 3090 Ti | 24 GB | Ampere |
| NVIDIA GeForce RTX 3090 | 24 GB | Ampere |
| NVIDIA GeForce RTX 3080 Ti | 12 GB | Ampere |
| NVIDIA GeForce RTX 3080 | 20 GB | Ampere |
| NVIDIA GeForce RTX 3080 | 10 GB | Ampere |
| NVIDIA GeForce RTX 3070 Ti | 8 GB | Ampere |
| NVIDIA GeForce RTX 3070 Ti Laptop GPU | 8 GB | Ampere |
| NVIDIA GeForce RTX 3070 | 8 GB | Ampere |
| NVIDIA GeForce RTX 3060 Ti | 8 GB | Ampere |
| NVIDIA GeForce RTX 3060 | 12 GB | Ampere |
| NVIDIA GeForce RTX 3060 Laptop GPU | 12 GB | Ampere |
| NVIDIA GeForce RTX 3060 Laptop GPU | 6 GB | Ampere |
| NVIDIA GeForce RTX 3050 | 8 GB | Ampere |
| NVIDIA GeForce RTX 3050 | 6 GB | Ampere |
| NVIDIA GeForce RTX 2080 Ti | 11 GB | Turing |
| NVIDIA GeForce RTX 2080 | 8 GB | Turing |
| NVIDIA GeForce RTX 2070 SUPER | 8 GB | Turing |
| NVIDIA GeForce RTX 2070 | 8 GB | Turing |
| NVIDIA GeForce RTX 2060 SUPER | 8 GB | Turing |
| NVIDIA GeForce RTX 2060 | 12 GB | Turing |
| NVIDIA GeForce GTX 1660 Ti | 6 GB | Turing |
| NVIDIA GeForce GTX 1660 SUPER | 6 GB | Turing |
| NVIDIA GeForce GTX 1660 | 6 GB | Turing |
| NVIDIA GeForce GTX 1650 | 4 GB | Turing |
| NVIDIA GeForce GTX 1080 Ti | 11 GB | Pascal |
| NVIDIA GeForce GTX 1080 | 8 GB | Pascal |
| NVIDIA GeForce GTX 1070 Ti | 8 GB | Pascal |
| NVIDIA GeForce GTX 1070 | 8 GB | Pascal |
| NVIDIA TITAN RTX | 24 GB | Turing |
| NVIDIA TITAN V | 12 GB | Volta |
| NVIDIA TITAN Xp | 12 GB | Pascal |
| NVIDIA GeForce GTX TITAN X | 12 GB | Maxwell |
| NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 96 GB | Blackwell |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 96 GB | Blackwell |
| NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 96 GB | Blackwell |
| NVIDIA RTX PRO 5000 Blackwell | 48 GB | Blackwell |
| NVIDIA RTX PRO 4500 Blackwell | 32 GB | Blackwell |
| NVIDIA RTX PRO 4000 Blackwell | 24 GB | Blackwell |
| NVIDIA RTX PRO 4000 Blackwell SFF Edition | 24 GB | Blackwell |
| NVIDIA RTX 6000 Ada Generation | 48 GB | Ada |
| NVIDIA RTX 5880 Ada Generation | 48 GB | Ada |
| NVIDIA RTX 5000 Ada Generation | 32 GB | Ada |
| NVIDIA RTX 4500 Ada Generation | 24 GB | Ada |
| NVIDIA RTX 4000 Ada Generation | 20 GB | Ada |
| NVIDIA RTX 2000 Ada Generation | 16 GB | Ada |
| NVIDIA RTX A6000 | 48 GB | Ampere |
| NVIDIA RTX A5000 | 24 GB | Ampere |
| NVIDIA RTX A4000 | 16 GB | Ampere |
| NVIDIA RTX A3000 12GB Laptop GPU | 12 GB | Ampere |
| NVIDIA RTX A2000 12GB | 12 GB | Ampere |
| NVIDIA RTX A2000 | 6 GB | Ampere |
| Quadro RTX 8000 | 48 GB | Turing |
| Quadro RTX 6000 | 24 GB | Turing |
| Quadro RTX 5000 | 16 GB | Turing |
| Quadro P4000 | 8 GB | Pascal |
| NVIDIA H200 | 141 GB | Hopper |
| NVIDIA H100 NVL | 96 GB | Hopper |
| NVIDIA H100 80GB HBM3 | 80 GB | Hopper |
| NVIDIA H100 PCIe | 80 GB | Hopper |
| NVIDIA L40 | 48 GB | Ada |
| NVIDIA L40S | 48 GB | Ada |
| NVIDIA L4 | 24 GB | Ada |
| NVIDIA A100 80GB PCIe | 80 GB | Ampere (A100 class) |
| NVIDIA A100-SXM4-80GB | 80 GB | Ampere (A100 class) |
| NVIDIA A100-PCIE-40GB | 40 GB | Ampere (A100 class) |
| NVIDIA A100-SXM4-40GB | 40 GB | Ampere (A100 class) |
| NVIDIA A40 | 48 GB | Ampere |
| NVIDIA A16 | 16 GB | Ampere |
| NVIDIA A10 | 24 GB | Ampere |
| NVIDIA A2 | 16 GB | Ampere |
| Tesla T4 | 16 GB | Turing |
| Tesla PG500-216 | 32 GB | Volta |
| Tesla V100-PCIE-32GB | 32 GB | Volta |
| Tesla V100-SXM2-32GB | 32 GB | Volta |
| Tesla V100-SXM2-16GB | 16 GB | Volta |
| Tesla P100-PCIE-16GB | 16 GB | Pascal |
| Tesla P40 | 24 GB | Pascal |
