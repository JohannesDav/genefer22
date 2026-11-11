# genefer22: experimental OpenCL speedups for NVIDIA GPUs

**Warning:** all changes in this fork were written by AI and have not been independently reviewed.

This is a fork of [Yves Gallot’s genefer22](https://github.com/galloty/genefer22), the program PrimeGrid uses to test numbers of the form b<sup>2<sup>n</sup></sup> + 1 (generalized Fermat numbers) for primality. It changes only the GPU version, and only on NVIDIA cards. It targets PrimeGrid’s GFN-16 to GFN-22 subprojects, where n is 16 to 22. The test itself, the proofs and the error checks are all Gallot’s work. This fork only changes how the arithmetic runs on the GPU (see [exactness and fallbacks](#exactness-and-fallbacks)).

The table shows the speedup for one complete test, including proof generation. A dash means the comparison was not run. The [benchmark results](doc/benchmarks.md) give the times in seconds and the bases used.

| GPU | GFN-16 | GFN-17 | GFN-18 | GFN-19 | GFN-20 | GFN-21 | GFN-22 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 2080 Ti | 1.2× | 1.3× | 1.5× | 1.8× | — | — | — |
| RTX 4060 Ti | 1.2× | 1.3× | 1.5× | 1.7× | 1.9× | — | — |
| RTX 5070 | 1.3× | 1.3× | 1.4× | 1.7× | 1.9× | — | — |
| RTX 5090 | — | — | — | — | 1.6× | 2.1× | 1.9× |

## What changed

Most of a test is spent squaring a very large number millions of times. Genefer stores the number as a list of base-b digits and squares it with number-theoretic transforms (an exact, integer counterpart of the FFT) computed modulo two or three primes. It then combines those results into the exact answer with the Chinese remainder theorem.

This fork changes how that squaring runs on NVIDIA GPUs:

- **Packed squaring.** Several consecutive base-b digits (up to 16) are combined into each transform element, so the transform is that many times shorter. The elements are larger, so more primes are needed to recover the exact result. The number of primes is chosen for each candidate from a worst-case bound.
- **Alternative three-prime arithmetic.** On some GPUs, mostly for GFN-16 and GFN-17, it is faster to keep Gallot’s transform but use three primes just below 2<sup>30</sup>. These leave room in 32-bit words to postpone modular reductions.
- **Settings for each GPU.** The GPU code has several possible layouts. The program contains [settings for many GPU models](doc/calibrated-gpus.md) and picks them automatically, so one executable serves every supported card.
- **Asynchronous execution.** Work is submitted through OpenCL command queues.

## Exactness and fallbacks

None of these changes trade accuracy for speed. Before using an optimization for a candidate, the program checks worst-case bounds showing that every intermediate value stays within the range the arithmetic can represent exactly. Two-prime arithmetic is only used below Gallot’s own limit for it.

The optimizations are designed to preserve exact arithmetic and produce the same results and proof files as the original version. The proof files matched in all 17 benchmark comparisons.

If a check fails, the program uses Gallot’s arithmetic instead. It also runs Gallot’s original code, unchanged, on non-NVIDIA GPUs, for GFN sizes outside 16–22, or if an optimized kernel cannot be initialized.

Every test still runs Gallot’s Gerbicz–Li error check. Two [optional checks](doc/opencl.md#optional-extra-checks) can also verify the finished result with his original code.

## Building and running

See [building and running the OpenCL version](doc/opencl.md). The executable, `geneferg`, takes the same options as Gallot’s GPU version, plus two optional check flags. For example, this tests 28000000<sup>2<sup>19</sup></sup> + 1 and writes the proof to `candidate.proof`:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

Only Linux x86-64 has been tested. PrimeGrid does not distribute this version, so using it for PrimeGrid work means setting up BOINC to run it yourself.

Contributions from people with AMD GPUs or Macs who want to port and test the optimizations are very welcome. These devices currently use Gallot’s original OpenCL implementation.
