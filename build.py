#!/usr/bin/env python3
"""Build one Linux x86-64 GFN-19 profile and its independent OpenCL verifier."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("3090", "4060"), required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--sdk", type=Path, default=Path("/usr"), help="BOINC SDK prefix"
    )
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument(
        "--kernel", type=Path, help="use PTX regenerated with kernels/build.py"
    )
    args = parser.parse_args()
    output, sdk, cuda = args.output.resolve(), args.sdk.resolve(), args.cuda.resolve()
    if output.exists():
        parser.error(
            "output already exists; use a new directory to preserve checkpoint identities"
        )
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error("output must not contain the source directory")
    # This layout is used by Ubuntu's libboinc-app-dev package.
    libraries = sdk / "lib/x86_64-linux-gnu"
    for path in (
        sdk / "include/boinc/api/boinc_api.h",
        cuda / "include/cuda.h",
        libraries / "libboinc_api.so",
        libraries / "libboinc_opencl.so",
    ):
        if not path.is_file():
            parser.error("missing build dependency: " + str(path))
    output.mkdir(parents=True)
    (output / "bin").mkdir()
    (output / "kernels").mkdir()
    (output / "lib").mkdir()
    (output / "licenses").mkdir()
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    for name in ("run_checked.py", "boinc_guard_verify.py"):
        shutil.copy2(ROOT / "runtime" / name, output / name)
    shutil.copy2(ROOT / "runtime/dispatch.py", output / "genefer-boinc")
    (output / "genefer-boinc").chmod(0o755)
    kernel = (
        "kernels/gfn19-signed-divider-sm"
        + ("86" if args.gpu == "3090" else "89")
        + ".ptx"
    )
    kernel_source = args.kernel.resolve() if args.kernel else ROOT / kernel
    shutil.copy2(kernel_source, output / kernel)
    policy = dict(
        persist=4 if args.gpu == "3090" else 0,
        inner=50 if args.gpu == "3090" else 100,
        bridge=50 if args.gpu == "3090" else 100,
        graph_span=8,
        dup_special=True,
    )
    (output / "runtime.json").write_text(json.dumps(policy, indent=2) + "\n")
    for name in ("boinc", "boinc_api", "boinc_opencl"):
        library = (libraries / ("lib" + name + ".so.7")).resolve(strict=True)
        shutil.copy2(library, output / "lib" / library.name)
        (output / "lib" / ("lib" + name + ".so.7")).symlink_to(library.name)
    notices = [sdk / "share/doc/libboinc-app-dev/copyright"] + list(
        (sdk / "licenses").glob("*copyright")
    )
    notice = next((path for path in notices if path.is_file()), None)
    if notice is None:
        parser.error("BOINC library copyright file missing under SDK prefix")
    shutil.copy2(notice, output / "licenses/boinc-copyright")
    compiler = os.environ.get("CXX", "g++")
    common = [
        compiler,
        "-std=c++17",
        "-O3",
        "-pthread",
        "-D_GNU_SOURCE",
        "-DGPU",
        "-I",
        str(ROOT / "src"),
        "-I",
        str(ROOT / "Khronos"),
    ]
    sources = [str(ROOT / "src/main.cpp"), str(ROOT / "src/transform_ocl.cpp")]
    cuda_flags = [
        "-DGENEFER_CUDA",
        "-DGENEFER_GUARD_" + args.gpu,
        "-I",
        str(cuda / "include"),
        "-L",
        str(cuda / "lib64/stubs"),
    ]
    boinc_flags = [
        "-DBOINC",
        "-I",
        str(sdk / "include/boinc"),
        "-I",
        str(sdk / "include/boinc/api"),
        "-I",
        str(sdk / "include/boinc/lib"),
        "-L",
        str(libraries),
        "-Wl,-rpath,$ORIGIN/../lib",
    ]
    standard_libs = ["-lOpenCL", "-lgmp"]
    boinc_libs = ["-lboinc_api", "-lboinc_opencl", "-lboinc", "-ldl", "-lrt"]
    commands = [
        common
        + sources
        + standard_libs
        + ["-o", str(output / "bin/genefer-conservative")],
        common
        + boinc_flags
        + sources
        + standard_libs
        + boinc_libs
        + ["-o", str(output / "bin/genefer-boinc-stock")],
        common
        + cuda_flags
        + sources
        + ["-lcuda"]
        + standard_libs
        + ["-o", str(output / "bin/genefer-signed-divider")],
        common
        + cuda_flags
        + boinc_flags
        + ["-DGENEFER_BOINC_GUARD"]
        + sources
        + ["-lcuda"]
        + standard_libs
        + boinc_libs
        + ["-o", str(output / "bin/genefer-boinc-checked")],
    ]
    for command in commands:
        print("Building " + Path(command[-1]).name, flush=True)
        subprocess.run(command, check=True)
    required = (
        "bin/genefer-boinc-checked",
        "bin/genefer-conservative",
        kernel,
        "boinc_guard_verify.py",
        "run_checked.py",
        "runtime.json",
    )
    manifest = dict(
        version=1,
        gpu=args.gpu,
        profile="broad",
        kernel=kernel,
        runtime_policy=policy,
        files={name: sha(output / name) for name in required},
    )
    (output / "boinc-guard-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    files = {
        str(path.relative_to(output)): sha(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    (output / "deployment-manifest.json").write_text(
        json.dumps(dict(version=1, gpu=args.gpu, files=files), indent=2) + "\n"
    )
    print("Built " + str(output))


if __name__ == "__main__":
    main()
