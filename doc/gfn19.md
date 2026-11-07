# Building and running GFN-19

This guide explains how to build the programs in this fork, run a candidate yourself, and use the BOINC application. The examples assume you are running commands from the repository's top-level directory, where `build.py` is located.

There are two ways to use the build. For a standalone calculation, you choose a candidate and run the supplied `run_checked.py` script. For PrimeGrid work, BOINC must launch the supplied `genefer-boinc` application from a properly configured BOINC installation. Building the code does not attach a PrimeGrid account or download any work.

## 1. Prepare the machine

The optimizations target the RTX 3090 and RTX 4060 Ti 16GB on Linux x86-64. Only configurations with one GPU available to the application have been tested. The optimized prover accepts GFN-19 candidates with even bases between 1,024 and 528,220,156, excluding powers of two. For example, the command later in this guide tests `28000000^(2^19) + 1`.

You will need Python 3.9 or later, a compiler supporting C++17, and glibc 2.34 or later. The build also needs GMP for large-integer arithmetic, the OpenCL development files and NVIDIA OpenCL runtime, the NVIDIA CUDA toolkit, and the BOINC application development libraries. The BOINC libraries are required by this build script even if you only intend to run standalone calculations, because it builds the BOINC applications as well.

On Ubuntu, the relevant system packages include `g++`, `libgmp-dev`, `ocl-icd-opencl-dev` and `libboinc-app-dev`. The NVIDIA driver, CUDA toolkit and NVIDIA OpenCL runtime must also be installed; the OpenCL development package alone does not supply NVIDIA's GPU implementation.

The builds were tested on Ubuntu 24.04 with GCC 13.3 and the BOINC 7.24.1 application libraries. The NVIDIA software differed between the two machines:

| GPU | CUDA toolkit | NVIDIA driver |
|---|---|---|
| RTX 3090 | 13.0 | 580.178.04 |
| RTX 4060 Ti 16GB | 12.8 | 570.169 |

These are the tested configurations. Compatibility with other software versions has not been established.

## 2. Build for your GPU

Both cards use the same C++ codebase, but they need different GPU kernels and settings. You select the card when building; there is no automatic choice between the two versions.

For an RTX 3090, run:

```sh
python3 build.py --gpu 3090 --output build/3090
```

For an RTX 4060 Ti 16GB, run:

```sh
python3 build.py --gpu 4060 --output build/4060
```

The `--output` argument names the directory that will hold the runnable build. It must be a new directory: the builder refuses to overwrite an existing one. If you rebuild later, choose another name, such as `build/3090-new`.

By default, the script looks for the CUDA toolkit under `/usr/local/cuda` and the BOINC development files under `/usr`. If your installations are elsewhere, specify their locations explicitly. For example:

```sh
python3 build.py --gpu 4060 --output build/4060 \
    --cuda /path/to/cuda --sdk /path/to/boinc-sdk
```

Replace those paths with your actual installation directories. The BOINC SDK directory must use the layout expected by Ubuntu's development package: headers under `include/boinc/api` and `include/boinc/lib`, and shared libraries under `lib/x86_64-linux-gnu`. It must also contain the libraries' copyright notice, either under `share/doc/libboinc-app-dev/copyright` or in a `licenses` directory with a filename ending in `copyright`.

When the build finishes, the output directory contains the optimized programs, the original OpenCL programs used for verification and fallback, the matching GPU kernels, and the scripts that run them. It also contains copies of the BOINC shared libraries and their notices. GMP, OpenCL and the NVIDIA driver remain dependencies of the machine where you run it.

Keep this whole directory together. Copying just an executable is insufficient: the supplied launchers need its companion programs, kernels and configuration files. The rest of this guide calls this directory the **build directory**.

The GPU source is supplied in `kernels/src/gfn19-packed.cl`, `kernels/src/gfn19-three.cl` and the upstream `ocl/kernel.cl`. The build copies those sources into the build directory. The NVIDIA driver compiles the required kernels when a calculation starts; the optimized applications execute the resulting code through the CUDA driver. There are no precompiled PTX files to regenerate. The packed constant generator uses only Python’s standard library.

## 3. Run a candidate without BOINC

```sh
python3 build/3090/run_checked.py -b 28000000 -f /absolute/path/gproof
```

The script tests `28000000^(2^19)+1`. The `-f` argument chooses the output filename prefix; do not add `.proof`. Replace the example path with a location you can write to. For the 4060 Ti, use `build/4060/run_checked.py` instead.

The script chooses the arithmetic for the base, generates the proof, and saves `gproof.proof` and a JSON result report. Its working files, logs and checkpoints stay in `gproof.work`. Upstream’s own Gerbicz–Li check and proof construction remain part of the calculation.

### Optional independent checks

Two additional checks are available. Neither runs by default:

```sh
python3 build/3090/run_checked.py -b 28000000 -f /absolute/path/gproof \
    --check-certificate --check-tail
```

`--check-certificate` uses a separate executable built from the original OpenCL implementation. It runs Genefer’s server mode to turn the proof into a certificate, then runs the certificate checker. It compares the reported candidate result with the optimized prover’s result and saves `gproof.cert`.

`--check-tail` retains an intermediate checkpoint and uses the original three-prime arithmetic to recompute the final segment of the primality calculation. It compares every final digit. This check runs independently of the certificate check: either flag can be used alone. The checkpoint needed for this check is retained only when the flag is supplied during proof generation.

These are local calculations and do not contact PrimeGrid. Their time is included in the JSON report when requested, but excluded from the README benchmarks. The report records `certificate_verified` and `tail_verified` separately; `validated` is true only when both additional checks passed. A successful default run therefore has both check fields false.

If the optimized prover fails, or a requested check fails, the standalone script retries with the original prover. The report identifies `fast`, `upstream` or `conservative-fallback` execution and records any failure that caused a retry. BOINC handles failures differently, as described below.

### Interrupting and resuming

After Ctrl+C, keep `gproof.work` and run the same command again to resume. Checkpoints are tied to the candidate, GPU and particular build files. After rebuilding or changing a kernel, use a fresh output prefix. The launchers check file hashes and refuse to resume with a changed build.

To independently check an already completed proof, run:

```sh
python3 build/3090/run_checked.py -b 28000000 -f /absolute/path/gproof \
    --verify-existing
```

This explicitly runs certificate generation and checking again. You can also pass `--check-tail` if the original run retained its tail checkpoint. It cannot reconstruct a checkpoint that was not saved.

## Automatic arithmetic selection

Both launchers select the arithmetic from the base. There is no experimental arithmetic option.

| Even, non-power-of-two bases | Calculation |
|---|---|
| 1,026–2,072,558 | Upstream’s 31-bit two-prime arithmetic, executed with CUDA graphs |
| 2,072,560–4,038,248 | Upstream’s 32-bit two-prime arithmetic, executed with CUDA graphs |
| 4,038,250–4,194,302 | Optimized ordinary three-prime arithmetic |
| 4,194,306–528,220,156 | Packed exact squaring, with ordinary three-prime general operations |
| 528,220,158–2,000,000,000 | Original upstream arithmetic |

The two-prime thresholds reproduce upstream’s integer comparison exactly: `b*b < (p1*p2/2)/2^19`, with integer divisions. The prime pairs are `(2130706433, 2113929217)` and `(4194304001, 4076863489)`. This uses the same arithmetic assumptions and margin as upstream, without extending the range through parity information or sampled coefficient distributions. The selection is in `runtime/routing.py` and the C++ factory.

The packed path groups eight base-b digits into each coefficient, reducing the transform length from 524,288 to 65,536. Larger coefficients need more prime moduli; `runtime/packed_kernel.py` selects enough and checks deterministic bounds on reconstruction, intermediate calculations and carries. General multiplication and checkpoints retain the ordinary three-prime representation. Proof construction also uses that representation, avoiding repeated conversions during its short powers and multiplications.

If a packed representation fails an admission bound, ordinary exact three-prime arithmetic is used instead. Its bounds are in `src/gfn19_three/bounds.h`. The upper optimized limit keeps ordinary digits inside that implementation’s signed decoding range. Above it, the original implementation chooses its own three-prime variants, including its switch at base 1,000,000,000.

The number 4,194,304 is a kernel/representation threshold, not upstream’s two-prime safety limit. It is itself a power of two and therefore not a supported candidate. The arithmetic bounds do not establish that the implementation, compiler or GPU is free of errors.

## 4. Use the build with BOINC

BOINC must launch `genefer-boinc` with the correct build directory. This launcher handles one assigned task and does not attach accounts or fetch work.

A locally built application uses BOINC’s anonymous-platform configuration, normally an `app_info.xml` file. This tells BOINC which executable to use for the project application and how to launch it. The repository does not generate this configuration or install it into your BOINC data directory. The application name and version must match PrimeGrid’s setup.

Keep the complete build directory at a fixed absolute path readable by the BOINC account. That account must be able to execute `genefer-boinc` and `/usr/bin/python3`. The `--guard-bundle` option gives the launcher the location of its companion files. For a build installed at `/opt/genefer-3090`, the executable is `/opt/genefer-3090/genefer-boinc`.

Its fixed launch arguments include:

```text
-boinc --guard-bundle /opt/genefer-3090
```

BOINC supplies the candidate-specific base, GFN size, job type and output prefix. Do not add the standalone example’s base to every work unit. To enable the additional local checks for proof-generation tasks, append either or both flags:

```text
-boinc --guard-bundle /opt/genefer-3090 --check-certificate --check-tail
```

For supported GFN-19 tasks on device 0, the launcher chooses the same exact arithmetic ranges as the standalone script. It publishes the proof and result after successful calculation and any requested checks. Certificate-checking tasks always run the original checker; disabling the additional proof checks does not disable those project-assigned tasks.

The application handles BOINC suspend, resume, quit and abort requests, including during optional verification. It checks the checkpoint’s build identity before resuming. A failed calculation or requested check stops the task without reporting success; BOINC does not automatically repeat it with the original prover.

Requests for other GFN sizes or devices retain the upstream application. `--upstream` explicitly selects the original calculation. Optional checks are available only for supported GFN-19 requests on device 0. PrimeGrid’s validation of a returned task remains separate from these local checks.

## 5. Run the supplied tests

Argument handling, automatic routing, independent check flags and malformed input can be tested without a GPU:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tests/packed_reference.py
python3 tests/crt_reference.py
```

For GPU arithmetic and offline BOINC tests, use the build for your card and a new output directory:

```sh
python3 tests/run.py --bundle build/3090 --output build/test-3090
```

The script compares each selected arithmetic representation with GMP at the two-prime boundaries, packed channel boundaries and upper limits. It exercises squaring, scalar multiplication, general products, copies, transfers and proof-phase transitions. `--arithmetic-only` skips the later BOINC lifecycle calculation; `--bases B1 B2 ...` restricts the arithmetic bases.

The BOINC lifecycle test creates synthetic tasks locally. It tests a full base-28,000,000 proof with both optional checks, interrupts and resumes the calculation, tests suspension during verification, rejects a changed generated kernel and checks a certificate. It does not attach an account, contact PrimeGrid or consume work units. If your build used custom `--sdk` or `--cuda` paths, pass the same options to the test script.

A separate complete standalone regression is available:

```sh
python3 tests/standalone.py --bundle build/3090 \
    --output build/standalone-test-3090
```

It generates a base-1,026 proof with both checks, compares it with the saved upstream reference, repeats verification and confirms that changed build files are rejected. It briefly modifies and restores a kernel file for the last test, so do not use the same build directory concurrently. Use the corresponding `build/4060` paths for the 4060 Ti.

## Measurement notes

Benchmarks compare one complete proof-generation run with upstream commit `56905b2` on the same machine and base. They exclude the optional certificate and tail checks for this fork. The original Gerbicz–Li check and proof construction are included in both timings. Driver caches and variable clocks can affect results.

The test machines use a Xeon W-3223 with PCIe 3.0 ×16 for the 3090, and an EPYC 7532 with PCIe 4.0 ×4 for the 4060 Ti. Both small-base two-prime and large-base packed paths must be benchmarked independently; the speedup from one range should not be assumed for another.
