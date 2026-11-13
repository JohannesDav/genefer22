# Building and installing

This guide explains how to build `geneferg`, the GPU executable in this fork, and how to run it on PrimeGrid through BOINC. Follow the steps for [Ubuntu](#step-by-step-on-ubuntu) or [Windows 11 x64](#step-by-step-on-windows-11-x64).

## Requirements

- Git, Python 3.8 or newer and a C++17 compiler. The guides use GCC on Ubuntu and MinGW-w64 GCC from MSYS2 UCRT64 on Windows.
- GMP development files, including the static library `libgmp.a`.
- OpenCL loader development files. The genefer22 sources include OpenCL headers. Building the Windows BOINC libraries also requires the OpenCL headers from MSYS2.
- BOINC application headers and static libraries (`libboinc_api.a`, `libboinc_opencl.a` and `libboinc.a`) for BOINC integration.
- CUDA 12 or 13 headers. The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) includes them, and both platform guides show how to extract just the header package. The build uses `cuda.h` and requires neither `nvcc` nor the CUDA runtime.

The optimized paths and the supplied BOINC configurations require an NVIDIA GPU with a driver that supports OpenCL. For PrimeGrid, BOINC must detect that GPU in its Event Log.

## Building

Run `build.py` as shown in the platform steps below. `--boinc` points to an installed BOINC SDK or a BOINC source checkout with the application libraries already built. `--cuda` points to the directory containing `include/cuda.h`. It defaults to the `CUDA_PATH` environment variable, then `/usr/local/cuda`.

The executable is `build/geneferg` on Linux and `build/geneferg.exe` on Windows. It contains the GPU kernels and profiles. For command-line use alone, you can replace the `--boinc` option and its path with `--without-boinc`.

For example, run the following from the repository root in the build shell to test 28000000<sup>2<sup>19</sup></sup> + 1 and write `candidate.proof`. On Windows, use `./build/geneferg.exe` in the same command.

```sh
./build/geneferg -n 19 -b 28000000 -p -f candidate
```

Run the executable with no arguments to list all options.

## Using it with BOINC

PrimeGrid doesn’t distribute this version, so BOINC has to be told to use your own executable instead of the official one. This is done with BOINC’s [anonymous platform](https://github.com/BOINC/boinc/wiki/Anonymous-platform) feature:

1. Attach your BOINC client to PrimeGrid as usual.
2. Stop the client.
3. Copy the executable and the matching template into the PrimeGrid project directory. Name the template copy `app_info.xml`. Use the [Linux template](boinc-app_info.xml) for `geneferg` or the [Windows template](boinc-app_info-windows.xml) for `geneferg.exe`. The BOINC account needs permission to read both files and run the executable.
4. Start the client again.

The supplied `app_info.xml` covers all the GFN subprojects this version supports, and you still choose which ones to run in your PrimeGrid preferences. While it’s installed, BOINC only runs those applications, so other PrimeGrid subprojects stop on that computer.

Where the project directory is, and how to stop and start the client, depends on how BOINC is installed. See the [Ubuntu steps](#step-by-step-on-ubuntu) or [Windows steps](#step-by-step-on-windows-11-x64) for the default installation paths.

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

## Step-by-step on Windows 11 x64

These steps assume BOINC is installed and attached to PrimeGrid, and its Event Log lists an NVIDIA OpenCL GPU. Use a regular desktop installation of BOINC, since a service installation can prevent GPU detection.

### 1. Install the build dependencies

Install [MSYS2](https://www.msys2.org/) and open **MSYS2 UCRT64** from the Start menu. Run:

```bash
pacman -Syu
```

If the update closes the window, reopen **MSYS2 UCRT64** and run `pacman -Syu` again. Then install the packages:

```bash
pacman -S --needed git make curl unzip mingw-w64-ucrt-x86_64-gcc \
    mingw-w64-ucrt-x86_64-python mingw-w64-ucrt-x86_64-gmp \
    mingw-w64-ucrt-x86_64-opencl-headers mingw-w64-ucrt-x86_64-opencl-icd
```

In the UCRT64 shell, create a build directory and build the application libraries from BOINC 8.2.9. You can skip the BOINC clone and `make` command for a command-line-only build.

```bash
mkdir -p ~/genefer-build
cd ~/genefer-build
git clone --depth 1 --branch client_release/8.2/8.2.9 https://github.com/BOINC/boinc.git
make -C boinc/lib -f Makefile.mingw -j4 libboinc_api.a libboinc.a libboinc_opencl.a
```

Download and unpack NVIDIA’s CUDA 12.8 header archive in the same shell. This leaves the installed GPU driver in place.

```bash
curl -fLO https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip
unzip -q cuda_cudart-windows-x86_64-12.8.90-archive.zip
```

### 2. Download and build

From the same UCRT64 shell:

```bash
git clone --branch main https://github.com/JohannesDav/genefer22.git
cd genefer22
python build.py --boinc ../boinc --cuda ../cuda_cudart-windows-x86_64-12.8.90-archive
./build/geneferg.exe
```

The last command lists the OpenCL devices as well as the command-line options. Check that your NVIDIA GPU appears. The executable can run outside MSYS2 with the installed GPU driver and Windows OpenCL loader.

### 3. Install it in BOINC

> [!IMPORTANT]
> This configuration enables the listed NVIDIA GFN apps for this host. In BOINC Manager, select PrimeGrid, choose **No new tasks**, and let the existing tasks finish and report before switching. Other PrimeGrid subprojects will be unavailable until you restore the official applications.

In BOINC Manager’s Advanced View, choose **File > Shut down connected client**. Check Task Manager’s **Details** tab to make sure `boinc.exe` has exited, then exit BOINC Manager.

From the repository directory in the UCRT64 shell, copy the executable and configuration. Back up any existing `app_info.xml` before replacing it.

```bash
project=/c/ProgramData/BOINC/projects/www.primegrid.com
cp build/geneferg.exe "$project/" &&
    cp doc/boinc-app_info-windows.xml "$project/app_info.xml"
```

Use the project directory under the **Data directory** shown in BOINC’s Event Log if you chose a custom location. If Windows denies access, use File Explorer to copy the files and approve the prompt. Rename the copied `boinc-app_info-windows.xml` to `app_info.xml`.

Reopen BOINC Manager. In **Tools > Event Log**, look for PrimeGrid’s `Found app_info.xml; using anonymous platform` message and confirm there are no missing-file errors. Select the desired GFN subprojects in PrimeGrid preferences, choose **Allow new tasks** for PrimeGrid, and confirm a new GFN task starts.

### Updating

From the same repository directory in the UCRT64 shell:

```bash
git pull
python build.py --boinc ../boinc --cuda ../cuda_cudart-windows-x86_64-12.8.90-archive
```

Stop BOINC, copy the rebuilt `geneferg.exe` into the project directory, and reopen BOINC Manager. If the Windows XML template changes, follow the installation steps again after the current tasks finish and report.

### Going back to the official version

Let the current tasks finish and report. Stop BOINC, remove `app_info.xml` from the PrimeGrid project directory, and reopen BOINC Manager. Select PrimeGrid in the **Projects** tab and choose **Reset project** to download the official applications again. Resetting the project discards any remaining tasks. Choose **Allow new tasks** when ready.

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
