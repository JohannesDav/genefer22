# genefer22: experimental GFN-19 GPU optimizations (CUDA)

All modifications in this fork are AI-generated and should not be considered safe or stable.

Based on [Yves Gallot’s genefer22](https://github.com/galloty/genefer22). This fork targets only PrimeGrid’s **GFN-19** subproject on the **RTX 3090** and **RTX 4060 Ti 16GB**, on Linux x86-64.

Exact integer arithmetic is the default, with coefficient and carry bounds through base **528,220,156**. The faster **`--two-prime`** option is experimental and limited to base **30,000,000**: it lacks a worst-case arithmetic guarantee, and the risk of an undetected error has not been quantified. Both modes require additional local checks.

Upstream times cover proof generation; modified times also include the extra local checks. Each run tests one candidate.

| GPU | Base | Upstream | Exact (default) | Two primes (experimental) |
|---|---:|---:|---:|---:|
| RTX 3090 | 28,000,000 | 1,108 s | 632 s (1.75×) | 492 s (2.25×) |
| RTX 4060 Ti 16GB | 28,000,000 | 1,185 s | 713 s (1.66×) | 547 s (2.16×) |

The main squaring loop packs eight digits into each coefficient, reducing the transform length by a factor of eight. It uses enough prime moduli to reconstruct every coefficient exactly. General multiplication, final proof construction and checkpoints use three primes. Combining transform stages reduces memory traffic, and CUDA graphs reduce launch overhead.

Build for the RTX 3090 and test a candidate:

```sh
python3 build.py --gpu 3090 --output build/3090
python3 build/3090/run_checked.py -b 28000000 -f /absolute/path/gproof
```

Use `--gpu 4060` for the 4060 Ti. Add `--two-prime` to the second command to select the experimental mode. Both cards are built from this codebase.

Supported bases are even and exclude powers of two. Only one-GPU setups have been tested. See the [build and BOINC instructions](doc/gfn19.md) for dependencies, kernel source and mode selection.
