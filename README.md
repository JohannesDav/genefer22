# genefer22: experimental GFN-19 GPU optimizations

All modifications in this fork are AI-generated and should not be considered safe or stable.

Based on [Yves Gallot's genefer22](https://github.com/galloty/genefer22). This fork targets only PrimeGrid's **GFN-19** subproject on the **RTX 3090** and **RTX 4060 Ti 16GB**, on Linux x86-64. Other GPUs and GFN sizes have not been optimized or validated here.

Measured time for one candidate, including proof generation and independent verification:

| GPU | Base | Upstream reference | Modified | Speedup |
|---|---:|---:|---:|---:|
| RTX 3090, 300W | 30,000,000 | 1,112 s | 481 s | 2.31× |
| RTX 4060 Ti 16GB, 165W | 28,000,000 | 1,198 s | 538 s | 2.23× |

The main optimization techniques are:

- Use two prime moduli where the reference normally needs three, with carry arithmetic specialized for GFN-19.
- Combine transform, squaring and carry steps to reduce intermediate memory traffic and synchronization.
- Tune root-table storage, register use and GPU cache settings separately for each card.
- Reuse CUDA graphs to submit several successive squarings of one candidate together, reducing CPU launch overhead.

Build for the RTX 3090 and test a candidate:

```sh
python3 build.py --gpu 3090 --output build/3090
python3 build/3090/run_checked.py -b 28000000 -f /absolute/path/gproof
```

For the 4060 Ti, use `--gpu 4060` and `build/4060` instead. Both GPU versions come from one codebase, with different kernels and settings for each card.

The optimized prover accepts even base values from 1,024 to 30,000,000, except powers of two. Only setups with one GPU available to the program have been tested. See the [build and BOINC instructions](doc/gfn19.md) for dependencies and setup.
