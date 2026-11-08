# genefer22: experimental GFN-19 GPU optimizations (CUDA)

All modifications in this fork are AI-generated and should not be considered safe or stable.

Based on [Yves Gallot’s genefer22](https://github.com/galloty/genefer22). This fork targets only PrimeGrid’s **GFN-19** subproject on the **RTX 3090** and **RTX 4060 Ti 16GB**, on Linux x86-64.

Arithmetic is selected automatically using worst-case bounds. Exact packed squaring supports bases from **1,026 through 528,220,156**. Other operations use two or three prime moduli according to the base. Higher bases use the original implementation through upstream’s limit of **2,000,000,000**.

Each benchmark tests one candidate and includes proof generation. Optional certificate verification and tail recomputation are excluded.

At base **28,000,000**:

| GPU | Upstream | This fork | Time saved |
|---|---:|---:|---:|
| RTX 3090 | 18m19s | 10m05s | 45% |
| RTX 4060 Ti 16GB | 19m36s | 11m22s | 42% |

The main squaring loop packs eight digits into each coefficient, reducing the transform length by a factor of eight. It uses enough prime moduli to reconstruct every coefficient exactly. General multiplication, final proof construction and checkpoints keep the ordinary two- or three-prime representation. Combining transform stages reduces memory traffic, and CUDA graphs reduce launch overhead.

Build for the RTX 3090 and test a candidate:

```sh
python3 build.py --gpu 3090 --output build/3090
python3 build/3090/run_checked.py -b 28000000 -f /absolute/path/gproof
```

Use `--gpu 4060` for the 4060 Ti. Both cards are built from this codebase. Add `--check-certificate` to independently generate and verify a certificate, or `--check-tail` to recompute the final calculation segment. Both checks are off by default.

Supported bases are even and exclude powers of two. Only one-GPU setups have been tested. See the [build and BOINC instructions](doc/gfn19.md) for dependencies, kernel source and arithmetic ranges.
