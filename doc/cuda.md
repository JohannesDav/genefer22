# Building and installing

This guide explains how to build `geneferg`, the GPU executable in this fork, and how to run it on PrimeGrid through BOINC. Only Linux x86-64 has been tested.

For a quick setup on Ubuntu, see the [step-by-step guide](#step-by-step-on-ubuntu).

## Requirements

To build `geneferg`, you need the following. The Ubuntu package names are in brackets.

- Git, Python 3.8 or newer and a C++17 compiler (`git`, `python3`, `g++`).
- GMP development files, including the static library `libgmp.a` (`libgmp-dev`).
- An OpenCL loader with its development files (`ocl-icd-opencl-dev`). The OpenCL headers are included in this repository.
- Headers from CUDA 12 or 13. Only `cuda.h` is used, so you don’t need `nvcc` or the CUDA runtime. You can get them from the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), or download just the header package, as in the [Ubuntu steps](#step-by-step-on-ubuntu).
- The BOINC development files (`libboinc-app-dev`), unless you build without BOINC support. You can also build them from the [BOINC source](https://github.com/BOINC/boinc).

To run it, the computer only needs an NVIDIA driver and an OpenCL loader.

## Building

From a copy of this repository:

```sh
python3 build.py --cuda /usr/local/cuda --boinc /usr
```

`--cuda` is the CUDA directory containing `include/cuda.h`. It defaults to `$CUDA_PATH`, or `/usr/local/cuda` if that isn’t set. `--boinc` is the directory where the BOINC development files are installed (`/usr` for Ubuntu’s package) or a BOINC source checkout.

This writes `build/geneferg`. The GPU code and settings are built into it, so it’s the only file you need.

To use `geneferg` from the command line only, build with `--without-boinc` instead of `--boinc /usr`. This tests 28000000<sup>2<sup>19</sup></sup> + 1 and writes the proof to `candidate.proof`:

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

Run `./build/geneferg` with no arguments to list all options.

## Using it with BOINC

PrimeGrid doesn’t distribute this version, so BOINC has to be told to use your own executable instead of the official one. This is done with BOINC’s [anonymous platform](https://github.com/BOINC/boinc/wiki/Anonymous-platform) feature:

1. Attach your BOINC client to PrimeGrid as usual.
2. Stop the client.
3. Copy `build/geneferg` and this repository’s [app_info.xml](boinc-app_info.xml) into the client’s PrimeGrid project directory. Both must be readable by the account BOINC runs as, and `geneferg` must be executable.
4. Start the client again.

The supplied `app_info.xml` covers all the GFN subprojects this version supports, and you still choose which ones to run in your PrimeGrid preferences. While it’s installed, BOINC only runs those applications, so other PrimeGrid subprojects stop on that computer.

Where the project directory is, and how to stop and start the client, depends on how BOINC is installed. The [Ubuntu steps](#step-by-step-on-ubuntu) below give the exact commands for Ubuntu’s `boinc-client` package.

## Step-by-step on Ubuntu

These steps are written for Ubuntu 24.04 and should also work on other versions such as 22.04 or 26.04, and on distributions based on Ubuntu. They assume you already run BOINC from Ubuntu’s `boinc-client` package, attached to PrimeGrid, on a computer with an NVIDIA GPU and its driver.

### 1. Install the build dependencies

```bash
sudo apt install git g++ python3 curl libgmp-dev ocl-icd-opencl-dev libboinc-app-dev
```

Then download NVIDIA’s CUDA header package and unpack it in your home directory. This only extracts the headers: it doesn’t install anything system-wide or touch your driver.

```bash
curl -LO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-cudart-dev-12-8_12.8.90-1_amd64.deb
dpkg-deb -x cuda-cudart-dev-12-8_12.8.90-1_amd64.deb ~/cuda-headers
```

### 2. Download and build

```bash
git clone https://github.com/JohannesDav/genefer22.git ~/genefer22
cd ~/genefer22
python3 build.py --cuda ~/cuda-headers/usr/local/cuda-12.8 --boinc /usr
```

### 3. Install it in BOINC

> [!IMPORTANT]
> With this `app_info.xml`, BOINC only runs the GFN applications it lists. Other PrimeGrid subprojects, such as CPU work, stop on this computer until you remove it. Tasks already in progress may also be lost when you switch.

From the `genefer22` directory, stop BOINC, copy the two files into its PrimeGrid directory, and start it again:

```bash
sudo systemctl stop boinc-client
sudo install -o boinc -g boinc -m 755 build/geneferg /var/lib/boinc/projects/www.primegrid.com/
sudo install -o boinc -g boinc -m 644 doc/boinc-app_info.xml /var/lib/boinc/projects/www.primegrid.com/app_info.xml
sudo systemctl start boinc-client
```

If you already had an `app_info.xml` for PrimeGrid, this replaces it.

BOINC’s event log (in BOINC Manager, Tools > Event Log) should now show `Found app_info.xml; using anonymous platform` for PrimeGrid. New GFN tasks then run with this version.

### Updating

```bash
cd ~/genefer22
git pull
python3 build.py --cuda ~/cuda-headers/usr/local/cuda-12.8 --boinc /usr
sudo systemctl stop boinc-client
sudo install -o boinc -g boinc -m 755 build/geneferg /var/lib/boinc/projects/www.primegrid.com/
sudo systemctl start boinc-client
```

### Going back to the official version

```bash
sudo systemctl stop boinc-client
sudo rm /var/lib/boinc/projects/www.primegrid.com/app_info.xml
sudo systemctl start boinc-client
```

Then reset PrimeGrid in BOINC Manager (select it in the Projects tab and click Reset project) so BOINC downloads the official applications again. Resetting the project discards its current tasks.

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

1. The program runs Gallot’s original OpenCL code, unchanged, if the GPU is not an NVIDIA card, the GFN size is outside 16–22, CUDA cannot be initialized, or the GPU lacks a required feature (32-thread warps, 1024-thread blocks and 48 KB of shared memory). Setting the environment variable `GENEFER_FORCE_OPENCL=1` also forces this.
2. Otherwise, it considers up to three profiles in order: one for this GPU model, one for its architecture, and the default for the GFN size. An exact model and memory-size match takes priority. If none is available for this test, it tries the nearest calibrated memory size of the same model and architecture.
3. It uses the first profile that passes two checks: the worst-case arithmetic bounds for this base, and the GPU’s memory and resource limits.
4. If no profile passes, it runs Gallot’s arithmetic with his original kernels, launched through CUDA graphs.

The program prints the profile it selected.

Whatever the profile, everything except squaring uses Gallot’s arithmetic: multiplication, checkpoints and proof generation. It uses two primes exactly where Gallot’s version does, and three primes elsewhere. The one exception is the alternative three-prime arithmetic, which replaces Gallot’s three-prime arithmetic. It is only allowed where Gallot also uses three primes, and for bases up to 943,718,398.

Packed squaring is added on top. Its number of primes is chosen for each base so that the largest value any transform element can reach is reconstructed exactly. The bounds also cover the carries and the conversion back to ordinary digits, including that the converted digits stay within the limits Gallot’s arithmetic relies on.

## Calibrating another GPU

Calibration measures the alternative kernel layouts on a GPU to create or improve its profile:

```sh
python3 tools/calibrate.py --cuda /usr/local/cuda --output calibration-results --memory-gib 16
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
python3 tools/embed_cuda_kernels.py build/tests/generated/cuda/embedded_kernels.h
g++ -std=c++17 -O2 -Isrc tests/cuda_parameters_probe.cpp -lgmp -o build/tests/parameters
python3 tests/test_cuda_parameters.py build/tests/parameters
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated tests/cuda_profiles.cpp src/cuda/profiles.cpp -lgmp -o build/tests/profiles
./build/tests/profiles
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated tests/cuda_handoff.cpp src/cuda/kernel.cpp -lgmp -o build/tests/handoff
./build/tests/handoff
```

The other tests in `tests/` need an NVIDIA GPU. They compare the arithmetic with GMP, check that a saved test can be resumed by either the optimized or the original code, and check that complete runs produce the same proof files as Gallot’s version. The BOINC tests use the real BOINC API with a simulated client. They cover saving and resuming, suspending, quitting, aborting, losing contact with the client, renamed executables and BOINC’s file links.
