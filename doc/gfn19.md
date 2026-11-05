# Building and running GFN-19

This guide explains how to build the programs in this fork, run a candidate yourself, and use the BOINC application. The examples assume you are running commands from the repository's top-level directory, where `build.py` is located.

There are two ways to use the build. For a standalone calculation, you choose a candidate and run the supplied `run_checked.py` script. For PrimeGrid work, BOINC must launch the supplied `genefer-boinc` application from a properly configured BOINC installation. Building the code does not attach a PrimeGrid account or download any work.

## 1. Prepare the machine

The optimizations target the RTX 3090 and RTX 4060 Ti 16GB on Linux x86-64. Only configurations with one GPU available to the application have been tested. The optimized prover accepts GFN-19 candidates with even bases between 1,024 and 30,000,000, excluding powers of two. For example, the command later in this guide tests `28000000^(2^19) + 1`.

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

## 3. Run a candidate without BOINC

For the RTX 3090 build above, this command tests base 28,000,000:

```sh
python3 build/3090/run_checked.py -b 28000000 -f /absolute/path/gproof
```

Here, `-b` chooses the base of the GFN-19 candidate. You do not need to specify the exponent: this script is restricted to GFN-19. The `-f` argument chooses where to save the results and what to call them. Replace `/absolute/path/gproof` with a location you can write to, such as `/home/yourname/results/test-28000000`. The last part is a filename prefix, so do not add `.proof` to it.

For a 4060 Ti build, change the script path to `build/4060/run_checked.py`. The candidate and output arguments work the same way.

### What the script does

`run_checked.py` first starts the optimized prover. When that calculation finishes, the script starts a separate executable, `bin/genefer-conservative`, built from Gallot's original OpenCL implementation. It uses Genefer's server mode to generate a verification certificate from the proof, then Genefer's certificate-checking mode to check that certificate. These are local calculations; they do not contact PrimeGrid.

If the optimized calculation fails or its proof does not pass verification, the script retries the candidate with the original OpenCL prover. That retry can take much longer. The JSON report records whether the accepted result came from the optimized calculation (`fast`) or the original prover (`conservative-fallback`).

After successful verification, the example command produces these files:

| File | Contents |
|---|---|
| `/absolute/path/gproof.proof` | The completed proof |
| `/absolute/path/gproof.cert` | The certificate used for verification |
| `/absolute/path/gproof.json` | The result, timings and verification information |

During the calculation, the script keeps its working files, logs and checkpoints in `/absolute/path/gproof.work`. Intermediate files can exist there before verification finishes. Use `run_checked.py` for standalone results: running the native CUDA executable directly skips this script's verification and retry procedure.

### Interrupting, resuming and rebuilding

If you interrupt the script with Ctrl+C, keep the working directory. To resume, run the same command with the same build directory and `-f` prefix. The prover can then use its saved checkpoint.

A checkpoint is tied to the candidate and the particular programs, kernels and settings that created it. The scripts record file hashes to detect changes to those components. If you rebuild the programs or change the GPU version, start with a new output prefix rather than reusing the old working directory. Similarly, editing files inside an existing build directory will cause the integrity checks to reject it.

After a run has completed, the script will refuse to overwrite its final output during another ordinary run. To check the existing proof again without repeating the long candidate calculation, add `--verify-existing`:

```sh
python3 build/3090/run_checked.py -b 28000000 \
    -f /absolute/path/gproof --verify-existing
```

This runs the certificate generation and checking steps again and updates the certificate and JSON report.

## 4. Use the build with BOINC

The standalone script above is for calculations you start yourself. To process PrimeGrid work units, BOINC must instead run `genefer-boinc`, the launcher created in the build directory. BOINC supplies the candidate's arguments, input files and output-file locations.

Installing a locally built application uses BOINC's **anonymous-platform** configuration, normally described in an `app_info.xml` file. That configuration tells BOINC which local executable to use for a project application and how to launch it. This repository does not generate that configuration or install it into your BOINC data directory. The application name and version must be configured appropriately for PrimeGrid; do not overwrite a running task's executable or slot files.

Place the complete build directory at a fixed absolute path readable by the account that runs BOINC. That account must be able to execute the launcher and `/usr/bin/python3`, which the launcher uses. When configuring the application, give the launcher the location of its build directory using `--guard-bundle`. Despite its name, that option simply tells the launcher where its companion files are stored.

For example, if the build directory is `/opt/genefer-3090`, the executable is `/opt/genefer-3090/genefer-boinc`, and its additional launch arguments are:

```text
-boinc --guard-bundle /opt/genefer-3090
```

BOINC adds the task-specific arguments, such as the base, GFN size and whether the task is proof generation or certificate checking. Do not append the standalone example's base and output prefix to every BOINC task. The client must supply those for each work unit.

### How BOINC tasks are handled

For a supported GFN-19 proof-generation task, the launcher starts the optimized BOINC application. That application performs the calculation and then uses the original OpenCL implementation for the local certificate verification described above. Only after verification succeeds does it write the final BOINC outputs and report successful completion to the client.

Certificate-checking tasks use the original OpenCL calculation. Requests outside the launcher's supported optimized range are passed to the original BOINC application as well. You can explicitly select that application by adding `--upstream` to the launch arguments. Each invocation handles one BOINC task; it does not combine work units or fetch additional ones.

The modified BOINC application handles the client's suspend, resume, quit and abort requests, including while the separate verification program is running. It also checks the saved checkpoint's identity before resuming. If verification fails, or files in the build directory have changed, it stops without reporting an unchecked result as successful. Unlike the standalone script, it does not automatically repeat a failed optimized calculation with the original prover.

The local verification is separate from PrimeGrid's validation of a returned work unit. Passing the local checks does not mean that PrimeGrid has accepted the task.

## 5. Run the supplied tests

You can test argument handling, rejection of changed build files and malformed proof input without a GPU:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

To test the GPU arithmetic and the BOINC application, use the build for the card in your machine and choose a new test-output directory:

```sh
python3 tests/run.py --bundle build/3090 --output build/test-3090
```

Here, `--bundle` means the build directory to test. The script first compares GPU arithmetic with GMP, an independent large-integer library. It exercises multiplication, copying intermediate calculation results, and transfers between the GPU and CPU. It then runs a full proof for base 28,000,000, interrupts it to create a checkpoint, resumes it, and checks that its completed proof matches the saved reference hash. It also tests suspension during verification and a certificate-checking task.

Those BOINC tests supply synthetic task data locally. They do not attach an account, contact a project or consume PrimeGrid work units. They do perform full GPU calculations, so let them finish without other jobs using the GPU. If you built with custom `--sdk` or `--cuda` paths, pass the same options to `tests/run.py`.

There is a separate test for the complete standalone script:

```sh
python3 tests/standalone.py --bundle build/3090 \
    --output build/standalone-test-3090
```

This generates a full proof for base 1,026 and compares the proof and result with the original reference. It then rechecks that proof and confirms that the script refuses to run when a build file has changed. For that last check, the test briefly modifies `runtime.json` and restores it, so no other calculation should be using the same build directory while this test runs.

For the 4060 Ti, use `build/4060` in place of `build/3090` and choose separate test-output directories.

## Implementation and measurement notes

The optimized C++ transform and CUDA driver interface live in `src/cuda/`. The GPU kernels are supplied in `kernels/` as PTX, NVIDIA's intermediate assembly language. The NVIDIA driver compiles that assembly for the GPU when the application loads it.

`build.py` rebuilds the C++ executables and copies the matching PTX file into the build directory. It does not regenerate the PTX from higher-level GPU source: a generator that reproduces those final assembly files is not included in this repository. The original `ocl/kernel.cl` belongs to Gallot's OpenCL implementation and cannot be substituted for the optimized kernels. Changes to the optimized assembly require new arithmetic and proof validation.

The RTX 3090 kernels require a driver supporting PTX 9.0; the 4060 Ti kernels require PTX 8.7. Their different root-table layouts and GPU cache settings are the reason for selecting a card explicitly when building.

The performance figures in the README include proof generation and the local verification steps for one candidate. Each comparison ran both versions on the same machine, with warmed driver caches and variable GPU clocks. The reference executable was compiled from upstream commit `56905b2`; it was not the binary distributed by PrimeGrid.
