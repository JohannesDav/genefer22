#!/usr/bin/env python3
"""Build exact GFN-19 with automatic arithmetic selection."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
CUDA_BACKEND = True
PACKED_ASSETS = (
    "packed_kernel.py",
    "kernels/gfn19-packed.cl",
    "kernels/gfn19-packed-convert.clh",
    "kernels/gfn19-packed-codec.clh",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("3090", "4060"), required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new build directory"
    )
    parser.add_argument(
        "--sdk", type=Path, default=Path("/usr"), help="BOINC SDK prefix"
    )
    if CUDA_BACKEND:
        parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    args = parser.parse_args()
    output, sdk = (args.output.resolve(), args.sdk.resolve())
    if output.exists():
        parser.error("output already exists; choose a new build directory")
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error("output must not contain the source directory")
    libraries = sdk / "lib/x86_64-linux-gnu"
    dependencies = [
        sdk / "include/boinc/api/boinc_api.h",
        libraries / "libboinc_api.so",
        libraries / "libboinc_opencl.so",
    ]
    if CUDA_BACKEND:
        dependencies.append(args.cuda / "include/cuda.h")
    for path in dependencies:
        if not path.is_file():
            parser.error("missing build dependency: " + str(path))
    for folder in ["bin", "kernels", "lib", "licenses"]:
        (output / folder).mkdir(parents=True)
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    for name in ["run_checked.py", "boinc_guard_verify.py", "routing.py"]:
        shutil.copy2(ROOT / "runtime" / name, output / name)
    for name in PACKED_ASSETS:
        source = (
            ROOT / "runtime/packed_kernel.py"
            if name == "packed_kernel.py"
            else ROOT / "kernels/src" / Path(name).name
        )
        shutil.copy2(source, output / name)
    shutil.copy2(ROOT / "runtime/dispatch.py", output / "genefer-boinc")
    (output / "genefer-boinc").chmod(493)
    three_kernel = "kernels/gfn19-three.cl"
    shutil.copy2(ROOT / "kernels/src/gfn19-three.cl", output / three_kernel)
    two_kernel = "kernels/gfn19-two.cl"
    shutil.copy2(ROOT / "ocl/kernel.cl", output / two_kernel)
    for name in ["boinc", "boinc_api", "boinc_opencl"]:
        library = (libraries / ("lib" + name + ".so.7")).resolve(strict=True)
        shutil.copy2(library, output / "lib" / library.name)
        link = output / "lib" / ("lib" + name + ".so.7")
        if link.name != library.name:
            link.symlink_to(library.name)
    notices = [sdk / "share/doc/libboinc-app-dev/copyright"] + list(
        (sdk / "licenses").glob("*copyright")
    )
    notice = next((p for p in notices if p.is_file()), None)
    if notice is None:
        parser.error("BOINC library copyright notice missing under SDK prefix")
    shutil.copy2(notice, output / "licenses/boinc-copyright")
    common = [
        os.environ.get("CXX", "g++"),
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
    libs = ["-lOpenCL", "-lgmp"]
    boinc = [
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
    boinc_libs = ["-lboinc_api", "-lboinc_opencl", "-lboinc", "-ldl", "-lrt"]
    optimized = ["-DGENEFER_GFN19", "-DGENEFER_GUARD_" + args.gpu]
    gpu_libs = []
    if CUDA_BACKEND:
        optimized += [
            "-DGENEFER_CUDA",
            "-I",
            str(args.cuda / "include"),
            "-L",
            str(args.cuda / "lib64/stubs"),
        ]
        gpu_libs = ["-lcuda"]
    commands = [
        (common + sources + libs, "genefer-conservative"),
        (common + boinc + sources + libs + boinc_libs, "genefer-boinc-stock"),
        (common + [str(ROOT / "src/tail_check.cpp")] + libs, "genefer-tail-check"),
    ]
    names = {
        "three": (
            "genefer-gfn19",
            "genefer-boinc-checked",
            three_kernel,
            ["-DGENEFER_THREE_PRIME"],
        ),
        "two": (
            "genefer-gfn19-two",
            "genefer-boinc-two",
            two_kernel,
            ["-DGENEFER_EXACT_TWO"],
        ),
        "upstream": (
            "genefer-gfn19-upstream",
            "genefer-boinc-upstream",
            two_kernel,
            [],
        ),
    }
    for arithmetic, (prover, native, kernel, extra) in names.items():
        flags = optimized + extra
        commands += [
            (common + flags + sources + libs + gpu_libs, prover),
            (
                common
                + flags
                + boinc
                + ["-DGENEFER_BOINC_GUARD"]
                + sources
                + libs
                + gpu_libs
                + boinc_libs,
                native,
            ),
        ]
    for command, name in commands:
        print("Building " + name, flush=True)
        subprocess.run(command + ["-o", str(output / "bin" / name)], check=True)
    for arithmetic, (prover, native, kernel, _) in names.items():
        required = [
            "bin/" + native,
            "bin/genefer-conservative",
            "bin/genefer-tail-check",
            kernel,
            "boinc_guard_verify.py",
            "run_checked.py",
            "routing.py",
        ]
        if arithmetic in ("two", "three"):
            required.extend(PACKED_ASSETS)
        manifest = dict(
            version=2,
            gpu=args.gpu,
            backend="opencl" if arithmetic == "upstream" else "cuda",
            arithmetic=arithmetic,
            profile=arithmetic,
            kernel=kernel,
            prover="bin/" + prover,
            files={name: sha(output / name) for name in required},
        )
        if arithmetic in ("two", "three"):
            manifest["packed_squaring"] = True
        filename = {
            "three": "boinc-guard-manifest.json",
            "two": "boinc-two-prime-manifest.json",
            "upstream": "boinc-upstream-manifest.json",
        }[arithmetic]
        (output / filename).write_text(json.dumps(manifest, indent=2) + "\n")
    files = {
        str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
    }
    (output / "deployment-manifest.json").write_text(
        json.dumps(dict(version=1, gpu=args.gpu, files=files), indent=2) + "\n"
    )
    print("Built " + str(output))


if __name__ == "__main__":
    main()
