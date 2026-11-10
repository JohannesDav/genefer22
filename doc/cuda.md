# Building and using the GPU version

This fork builds one `geneferg` executable for NVIDIA GPUs and keeps Gallot’s original OpenCL implementation as a fallback. The CPU program remains a separate build. The supported and tested platform for these changes is Linux x86-64.

## Build

Building requires Python 3.8 or newer, a C++17 compiler, GMP development files, the OpenCL development loader, CUDA SDK headers, and the BOINC API development files. CUDA is used through the installed NVIDIA driver; the program does not link to the CUDA runtime library. Python and the CUDA SDK are build tools, not requirements for running the resulting executable.

On Debian or Ubuntu, install the ordinary build dependencies with:

```sh
sudo apt install g++ python3 libgmp-dev ocl-icd-opencl-dev
```

Install a CUDA SDK with CUDA 12 headers. The SDK directory supplied below must contain `include/cuda.h`. No GPU-specific compiler flags or `nvcc` invocation are required.

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

The output is `build/geneferg`. To build without the BOINC interface instead, use:

```sh
python3 build.py --cuda /usr/local/cuda --without-boinc
```

Kernel sources are in `kernels/src/`. At startup, the program computes candidate-specific constants and uses NVIDIA’s OpenCL compiler to produce PTX. The CUDA driver loads that code and executes it through CUDA graphs. The OpenCL compiler and PTX loader are part of the installed NVIDIA driver; `nvcc` is not used.

The build embeds kernel sources and the profile table into the executable. Do not distribute a separate kernel directory or calibration output alongside it. Changing `profiles/nvidia.json` or a kernel source requires rebuilding. The Linux x86-64 makefile also delegates its GPU target to this build script; its CPU target remains unchanged.

## Running a candidate

Normal upstream command-line options are retained. For example, this tests one GFN-19 candidate and writes its proof using the prefix `candidate`:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

Run `./build/geneferg` without arguments to list the modes and options. A BOINC-enabled build retains the native BOINC interface. Preparing an executable does not register it as a PrimeGrid application: deploying a replacement under BOINC still requires the appropriate project application configuration.

The default run retains upstream’s Gerbicz–Li error check. Two additional checks are available separately, both disabled by default:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate --check-certificate
./build/geneferg -n 19 -b 28000000 -p -f candidate --check-tail
```

`--check-certificate` uses the original OpenCL arithmetic to generate a certificate from the completed proof and then verify that certificate. `--check-tail` uses the original arithmetic to recompute the final segment from a stored intermediate value and compare it with the proof. Both flags may be combined. They add time, and the benchmark tables exclude them. A requested check must pass before the program reports a successful result. Temporary certificate files are removed afterward.

## How settings are selected

The executable first looks for a calibrated GPU model and memory capacity. If it has no matching entry, it tries an architecture default, followed by a GFN-size default. The defaults distinguish A100-style compute capability 8.0 devices from other Ampere cards, since their measured transform choices differ. These are fixed settings compiled into the executable; it does not benchmark the user’s device at startup. The selected profile is printed when the program initializes its arithmetic.

A profile describes a transform layout, not an exemption from the arithmetic bounds. For every candidate, the program calculates the exact bounds needed for reconstruction, digit conversion, carries and intermediate integer operations. It uses enough prime moduli for that candidate or rejects the packed layout. The smaller-prime three-modulus implementation has its own worst-case bound. There is no option to extend a two-prime calculation beyond its analytic bound.

For smaller bases, ordinary arithmetic uses two primes under the same bounds as upstream, calculated separately for each GFN size. It uses CUDA graphs and can also use packed squaring when its bounds allow it. A profile calibrated with three primes at a larger base cannot override this selection. Packed coefficients represent several base digits, so squaring can require more prime channels than ordinary arithmetic; that count is also selected automatically using exact bounds.

If an optimized layout does not fit, the program retains the original arithmetic, with CUDA graph execution where available. Non-NVIDIA devices, unsupported GFN sizes, unavailable CUDA drivers, or incompatible device resources use original OpenCL. A profile for an untested card is a performance prediction, not evidence that the card was tested.

Device checks cover work-group sizes, local memory, available memory, and compiled kernel limits. CUDA initialization and kernel compilation still depend on a working NVIDIA driver. For diagnosis, `GENEFER_FORCE_OPENCL=1` forces the original OpenCL implementation.

The [calibrated GPU list](calibrated-gpus.md) records the tested model and memory variants.

## Calibrating another GPU

Calibration is a developer tool. It is not required to run `geneferg`, and its output does not modify the installed program automatically.

```sh
python3 tools/calibrate.py --cuda /usr/local/cuda --output calibration-results --memory-gib 16
```

The tool builds arithmetic probes, screens transform layouts for GFN-16 through GFN-22, and confirms the leading choices with longer loop measurements. It checks arithmetic against GMP and exercises multiplication, checkpoints, and proof-phase transitions. Use `--sizes 19` to select one size, or `--device 1` to choose a different device. `--plan-only` generates admitted kernels without using a GPU.

Set `--memory-gib` to the card’s nominal capacity; it writes a `profile.json` entry that can be added to the `devices` array in `profiles/nvidia.json` before rebuilding. The explicit capacity avoids confusing ECC reservations with a different memory variant. Omit the option for measurement-only runs.

The output directory contains generated sources, commands, logs, hardware information, checks, and timings. Keep it outside a release commit. Loop measurements help select profiles; they are not complete primality-test benchmarks. A new profile should also be checked with complete proofs against upstream before it is used for headline performance claims. The [benchmark table](benchmarks.md) identifies the complete-run evidence already collected.

## Validation

The tests in `tests/` compare arithmetic and admission bounds against GMP and the independent Python implementation. GPU tests also exercise checkpoint exchange with the original arithmetic and compare complete proofs against upstream.

The arithmetic-bound and profile-selection tests run without a GPU. From the repository root, run:

```sh
mkdir -p build/tests
python3 tools/embed_cuda_kernels.py build/tests/generated/cuda/embedded_kernels.h
g++ -std=c++17 -O2 -Isrc tests/cuda_parameters_probe.cpp -lgmp -o build/tests/parameters
python3 tests/test_cuda_parameters.py build/tests/parameters
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated tests/cuda_profiles.cpp src/cuda/profiles.cpp -lgmp -o build/tests/profiles
./build/tests/profiles
```

The offline BOINC tests use the actual BOINC API with a simulated local client. They cover checkpoint/restart, suspend/resume, quit, abort, heartbeat loss, renamed executables and project-file links. This generalized executable has not been submitted to PrimeGrid; these tests do not establish project validation or general correctness.
