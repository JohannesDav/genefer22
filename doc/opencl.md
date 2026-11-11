# Building and running the OpenCL version

This page explains how to build `geneferg`, the GPU executable in this fork, how to run it, and how it chooses its arithmetic for each GPU and candidate. One build covers every supported NVIDIA GPU. Only Linux x86-64 has been tested.

## Requirements

To build, you need:

- Python 3.8 or newer and a C++17 compiler.
- GMP development files.
- OpenCL headers and an OpenCL loader.
- The BOINC API libraries, unless you build without BOINC support.

On Debian or Ubuntu, this installs everything except BOINC:

```sh
sudo apt install g++ python3 libgmp-dev ocl-icd-opencl-dev
```

The computer that runs `geneferg` only needs an OpenCL driver for its GPU and an OpenCL loader. It does not need Python.

## Building

For a BOINC build, first build the BOINC libraries (`libboinc_api.a`, `libboinc_opencl.a` and `libboinc.a`). From a BOINC source checkout:

```sh
./_autosetup
./configure --disable-server --disable-client --disable-manager
make -C lib
make -C api
```

Then, from this repository’s root:

```sh
python3 build.py --boinc /path/to/boinc
```

`--boinc` is the BOINC source or installation directory.

To run tests from the command line only, you can leave out BOINC:

```sh
python3 build.py --without-boinc
```

Either command writes `build/geneferg`. The GPU kernel sources in `kernels/src/` and the GPU settings in `profiles/nvidia.json` are built into the executable, so rebuild after changing them.

The GPU target of `genefer/makefile_linux_x64` calls this build script. Its CPU target still builds Gallot’s unchanged CPU version.

## Running a test

This tests 28000000<sup>2<sup>19</sup></sup> + 1 and writes the proof to `candidate.proof`:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

Run `./build/geneferg` with no arguments to list all options. They are the same as in Gallot’s GPU version, plus the two checks below.

At the start of each test, the program generates GPU kernels for that candidate and compiles them with the OpenCL compiler built into the NVIDIA driver.

A BOINC build uses the same BOINC interface as Gallot’s version. PrimeGrid does not distribute this executable, so to use it for PrimeGrid work you have to configure BOINC to run it in place of the official application, for example through BOINC’s anonymous platform (`app_info.xml`).

## Optional extra checks

Every test runs Gallot’s Gerbicz–Li check, which detects computation errors while the test runs. Two optional checks use Gallot’s original OpenCL code to verify the finished result independently of this fork’s arithmetic:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate --check-certificate
./build/geneferg -n 19 -b 28000000 -p -f candidate --check-tail
```

- `--check-certificate` converts the proof into a certificate and checks it. It uses Gallot’s certificate-generation and checking routines.
- `--check-tail` repeats the last 1/128 of the squarings, starting from a saved intermediate value, and compares the result with the proof.

You can use either check or both. Each repeats only a small part of the test. If a check fails, the test is reported as failed.

## How the program chooses its arithmetic

A *profile* is a set of GPU kernel settings for one GFN size, chosen by timing the alternatives on real cards (see [calibrating another GPU](#calibrating-another-gpu)). The executable contains profiles for specific GPU models and memory sizes, profiles for each GPU architecture (such as Pascal, Ampere or Ada), and a default profile for each GFN size.

When a test starts:

1. The program runs Gallot’s original OpenCL code, unchanged, if the GPU is not an NVIDIA card, the GFN size is outside 16–22, the optimized code cannot be initialized, or the GPU lacks a required feature (32-thread warps, 1024-thread blocks and 48 KB of shared memory). Setting the environment variable `GENEFER_FORCE_OPENCL=1` also forces this.
2. Otherwise, it considers up to three profiles in order: one for this exact GPU model and memory size, one for its architecture, and the default for the GFN size.
3. It uses the first profile that passes two checks: the worst-case arithmetic bounds for this base, and the GPU’s memory and resource limits.
4. If no profile passes, it runs Gallot’s arithmetic with his original kernels.

The program prints the profile it selected.

Whatever the profile, everything except squaring uses Gallot’s arithmetic: multiplication, checkpoints and proof generation. It uses two primes exactly where Gallot’s version does, and three primes elsewhere. The one exception is the alternative three-prime arithmetic, which replaces Gallot’s three-prime arithmetic. It is only allowed where Gallot also uses three primes, and for bases up to 943,718,398.

Packed squaring is added on top. Its number of primes is chosen for each base so that the largest value any transform element can reach is reconstructed exactly. The bounds also cover the carries and the conversion back to ordinary digits, including that the converted digits stay within the limits Gallot’s arithmetic relies on.

## Calibrating another GPU

Calibration measures the alternative kernel layouts on a GPU to create or improve its profile:

```sh
python3 tools/calibrate.py --output calibration-results --memory-gib 16
```

The tool builds small test programs and times each layout for GFN-16 to GFN-22 with short runs, then times the fastest candidates again with longer runs. It also checks the arithmetic against GMP, including multiplication, checkpoints and the operations used to generate proofs.

Useful options:

- `--sizes 19` tests only GFN-19.
- `--device 1` uses device 1 instead of device 0.
- `--plan-only` only checks the arithmetic bounds and generates the kernel sources, without using a GPU.
- `--memory-gib` gives the card’s advertised memory in GiB, such as `16` for a 16 GB RTX 4060 Ti.

With `--memory-gib`, the tool writes a `profile.json` entry. To use it, add the entry to the `devices` list in `profiles/nvidia.json` and rebuild. The rest of the output directory is a record of the run.

Before publishing performance figures for a new profile, compare complete runs and proof files against Gallot’s version, as in the [benchmark results](benchmarks.md).

## Tests

Three checks run on the CPU, without a GPU. They need the GMP development files. The first compares the C++ bound calculations with an independent Python implementation. The second checks profile selection for known GPU models, memory variants and unfamiliar GPUs. The third checks packed conversion bounds at the two-prime base limits. From the repository root:

```sh
mkdir -p build/tests
python3 tools/embed_gpu_kernels.py build/tests/generated/gpu/embedded_kernels.h
g++ -std=c++17 -O2 -Isrc tests/gpu_parameters_probe.cpp -lgmp -o build/tests/parameters
python3 tests/test_gpu_parameters.py build/tests/parameters
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated tests/gpu_profiles.cpp src/gpu/profiles.cpp -lgmp -o build/tests/profiles
./build/tests/profiles
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated tests/gpu_handoff.cpp src/gpu/kernel.cpp -lgmp -o build/tests/handoff
./build/tests/handoff
```

The other tests in `tests/` need an NVIDIA GPU. They compare the arithmetic with GMP, check that a saved test can be resumed by either the optimized or the original code, and check that complete runs produce the same proof files as Gallot’s version. The BOINC tests use the real BOINC API with a simulated client. They cover saving and resuming, suspending, quitting, aborting, losing contact with the client, renamed executables and BOINC’s file links.
