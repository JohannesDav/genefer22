# Building and using the GPU version

This guide covers building and running `geneferg`, the GPU executable. It supports all the NVIDIA models and GFN sizes listed in the README without a separate build for each card. The changes have been tested on Linux x86-64.

## Build

Building requires Python 3.8 or newer, a C++17 compiler, GMP development files, the OpenCL development loader, CUDA SDK headers, and, for a BOINC build, the BOINC API development files. To run the executable, the target machine needs the NVIDIA driver with OpenCL support and an OpenCL loader. It does not need Python or the CUDA SDK.

On Debian or Ubuntu, start with:

```sh
sudo apt install g++ python3 libgmp-dev ocl-icd-opencl-dev
```

You also need a CUDA SDK with CUDA 12 headers. In the build command below, `--cuda` points to the SDK directory containing `include/cuda.h`. The build uses these headers to access CUDA through the NVIDIA driver; it does not use `nvcc` or link to the CUDA runtime library.

The default build includes BOINC support and needs its static libraries: `libboinc_api.a`, `libboinc_opencl.a`, and `libboinc.a`. A BOINC source build can provide these. From a BOINC source checkout, run:

```sh
./_autosetup
./configure --disable-server --disable-client --disable-manager
make -C lib
make -C api
```

From this repository’s root, build the GPU executable with the paths to those dependencies:

```sh
python3 build.py --cuda /usr/local/cuda --sdk /path/to/boinc
```

The output is `build/geneferg`. If you only want to run tests directly from the command line, you can omit the BOINC dependency:

```sh
python3 build.py --cuda /usr/local/cuda --without-boinc
```

The GPU kernel sources are in `kernels/src/`. When a calculation starts, the program computes constants for the candidate and uses NVIDIA’s OpenCL compiler to produce PTX, NVIDIA’s intermediate assembly format. It then loads the compiled kernels through the CUDA driver and runs them using CUDA graphs. The compiler and PTX loader are supplied by the installed NVIDIA driver.

Kernel sources and the tested GPU settings in `profiles/nvidia.json` are embedded in the executable, so users do not need separate copies of those files. Rebuild after changing either of them. The Linux x86-64 makefile uses this build script for its GPU target; the CPU target still builds Gallot’s original CPU version separately.

## Running a candidate

For example, this tests the GFN-19 number `28000000^(2^19) + 1` and saves its proof as `candidate.proof`:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

Run `./build/geneferg` without arguments to list the modes and options. A BOINC build uses the same application interface as Gallot’s version. To use it under BOINC, you must also configure BOINC to run the replacement application; building the executable alone does not change your PrimeGrid installation.

The default run retains upstream’s Gerbicz–Li error check. Two additional checks are available separately, both disabled by default:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate --check-certificate
./build/geneferg -n 19 -b 28000000 -p -f candidate --check-tail
```

With `--check-certificate`, `geneferg` uses Gallot’s original OpenCL implementation to generate a certificate from the completed proof and check that certificate. With `--check-tail`, it uses the original implementation to repeat the final part of the calculation from a saved intermediate value and compare the result with the proof. You can enable either check or both. They add time and are excluded from the benchmark timings. If a requested check fails, the program reports a failure rather than a successful result. Temporary certificate files are removed afterward.

## How settings are selected

The program contains a table of calculation settings chosen by testing different transform layouts on real GPUs. Each entry is called a profile. There are separate choices for each GFN size because the best layout can change with the size of the calculation.

It first looks for settings tested on your GPU model and memory capacity. If there is no match, it tries settings selected for that GPU generation, such as Ampere or Ada. If the generation is also unknown, it tries general settings for the GFN size being tested. These choices are built into the executable; startup does not run a performance benchmark. The program prints the selected profile when it initializes the calculation.

A100-class GPUs have separate settings from other Ampere cards because their measured performance favored different transform layouts. The [GPU list](calibrated-gpus.md) shows which models and memory variants have been tested directly.

Before using a profile, the program checks worst-case arithmetic bounds for the candidate. These cover reconstruction from modular results, digit conversion, carries and intermediate integer operations. When several digits are packed together for squaring, it uses enough prime moduli to reconstruct the result exactly. If a layout cannot satisfy those bounds, it tries another. The implementation using three smaller primes is subject to its own worst-case bound. Two-prime calculations are never extended beyond their analytic bound.

At smaller bases, the program uses upstream’s two-prime arithmetic wherever upstream’s bounds allow it, with the threshold calculated separately for each GFN size. It still uses CUDA graphs and can speed up squaring by packing several digits together. Packed squaring may need more primes than the rest of the calculation because its coefficients are larger; its prime count is selected separately using exact bounds.

If none of the optimized layouts can be used, the program runs the original arithmetic, using CUDA graphs where available. It uses Gallot’s original OpenCL implementation on non-NVIDIA GPUs, outside GFN-16 through GFN-22, or when CUDA initialization or device checks fail. Performance on an untested card may differ from that of tested cards in the same generation.

Device checks confirm that the GPU can run the kernels: they check work-group sizes, shared memory, GPU memory capacity and limits reported by the kernel compiler. For troubleshooting or comparison, setting `GENEFER_FORCE_OPENCL=1` runs the original OpenCL implementation instead.

## Calibrating another GPU

The calibration tool measures different calculation layouts on a GPU to help contributors add or improve the settings shipped in `geneferg`. It is a separate developer tool; running it does not change the executable’s settings.

```sh
python3 tools/calibrate.py --cuda /usr/local/cuda --output calibration-results --memory-gib 16
```

The tool builds small test programs, compares transform layouts for GFN-16 through GFN-22, then repeats the most promising choices with longer measurements. It checks arithmetic against GMP and tests multiplication, checkpoints and the operations used during proof generation. Use `--sizes 19` to test only GFN-19, or `--device 1` to select device 1 instead of device 0. `--plan-only` checks arithmetic bounds and generates kernel sources without running a GPU test.

Set `--memory-gib` to the card’s advertised memory capacity, such as `16` for a 16GB RTX 4060 Ti. The driver may report less usable memory because some is reserved. With this option, the tool writes a `profile.json` entry that you can add to the `devices` array in `profiles/nvidia.json` before rebuilding. Without it, the tool records measurements but does not write that entry.

The output directory contains generated sources, commands, logs, hardware information, arithmetic checks and timings. Only the selected profile entry belongs in `profiles/nvidia.json`; the generated files are test records. These short measurements help choose settings, but do not measure a complete primality test. Before publishing performance figures for a new profile, compare complete runs and proof files against Gallot’s version, as in the [benchmark results](benchmarks.md).

## Validation

The tests in `tests/` compare arithmetic against GMP and check that the C++ bound calculations agree with a separate Python implementation. GPU tests also check that saved calculations can be resumed between the optimized and original implementations, and that complete runs produce the same proof files.

You can check arithmetic bounds and automatic profile selection without a GPU. From the repository root, run:

```sh
mkdir -p build/tests
python3 tools/embed_cuda_kernels.py build/tests/generated/cuda/embedded_kernels.h
g++ -std=c++17 -O2 -Isrc tests/cuda_parameters_probe.cpp -lgmp -o build/tests/parameters
python3 tests/test_cuda_parameters.py build/tests/parameters
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated tests/cuda_profiles.cpp src/cuda/profiles.cpp -lgmp -o build/tests/profiles
./build/tests/profiles
```

The BOINC tests run locally using the real BOINC API and a simulated client. They check saving and resuming work, suspension, quitting, aborting, loss of contact with the client, renamed executables and BOINC’s file links. No tasks from this generalized version have been submitted to PrimeGrid for project validation. The local tests do not establish general correctness.
