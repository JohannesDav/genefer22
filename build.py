#!/usr/bin/env python3
"""Build one GPU executable with CUDA acceleration and original OpenCL fallback."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "main.cpp",
    "transform_ocl.cpp",
    "transform_cuda_small3.cpp",
    "transform_opencl.cpp",
    "cuda/kernel.cpp",
    "cuda/profiles.cpp",
]


def locate(candidates, description):
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError(
        "Missing " + description + ": " + ", ".join(map(str, candidates))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/geneferg")
    parser.add_argument(
        "--cuda",
        type=Path,
        default=Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")),
        help="CUDA SDK prefix, needed only for its headers",
    )
    parser.add_argument(
        "--boinc",
        "--sdk",
        dest="boinc",
        type=Path,
        default=Path("/usr"),
        help="BOINC installation or source prefix",
    )
    parser.add_argument(
        "--without-boinc", action="store_true", help="omit the BOINC interface"
    )
    parser.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    build = output.parent / (output.name + "-build")
    build.mkdir(exist_ok=True)
    generated = build / "generated"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/embed_cuda_kernels.py"),
            str(generated / "cuda/embedded_kernels.h"),
        ],
        check=True,
    )
    cuda = args.cuda.resolve()
    header = locate(
        [cuda / "include/cuda.h", cuda / "targets/x86_64-linux/include/cuda.h"],
        "CUDA headers",
    )
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    common = [
        "-std=c++17",
        "-O3",
        "-pthread",
        "-D_GNU_SOURCE",
        "-DGPU",
        "-DGENEFER_GENERAL_CUDA",
        "-DGENEFER_GENERAL_PACKED",
        "-DGENEFER_GENERAL_SMALL3_AVAILABLE",
        "-DGENEFER_AUTOMATIC_CUDA",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "Khronos"),
        "-I" + str(generated),
        "-I" + str(header.parent),
    ]
    gmp_name = subprocess.check_output(
        compiler + ["-print-file-name=libgmp.a"], text=True
    ).strip()
    gmp = locate([Path(gmp_name)], "static GMP library")
    libraries = [str(gmp)]
    if not args.without_boinc:
        boinc = args.boinc.resolve()
        locate(
            [
                boinc / "include/boinc/api/boinc_api.h",
                boinc / "include/boinc/boinc_api.h",
                boinc / "api/boinc_api.h",
            ],
            "BOINC API headers",
        )
        common += ["-DBOINC"]
        for directory in [
            boinc / "include/boinc",
            boinc / "include/boinc/api",
            boinc / "include/boinc/lib",
            boinc,
            boinc / "api",
            boinc / "lib",
        ]:
            common.append("-I" + str(directory))
        directories = [
            boinc / "lib/x86_64-linux-gnu",
            boinc / "lib64",
            boinc / "lib",
            boinc / "api",
        ]
        libraries += [
            str(
                locate(
                    [d / "libboinc_api.a" for d in directories],
                    "static BOINC API library",
                )
            ),
            str(
                locate(
                    [d / "libboinc_opencl.a" for d in directories],
                    "static BOINC OpenCL library",
                )
            ),
            str(
                locate([d / "libboinc.a" for d in directories], "static BOINC library")
            ),
        ]
    # The CUDA driver is resolved lazily; never link against libcuda or libcudart.
    loader_candidates = [
        Path(
            subprocess.check_output(
                compiler + ["-print-file-name=" + name], text=True
            ).strip()
        )
        for name in ["libOpenCL.so", "libOpenCL.so.1"]
    ]
    opencl = locate(loader_candidates, "OpenCL loader")
    libraries = [
        "-Wl,--start-group",
        *libraries,
        "-Wl,--end-group",
        str(opencl),
        "-ldl",
        "-lrt",
    ]
    stamp = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.h")) + [
        ROOT / "profiles/nvidia.json",
        generated / "cuda/embedded_kernels.h",
    ]:
        stamp.update(path.read_bytes())
    stamp.update(json.dumps(compiler + common).encode())
    headers_hash = stamp.hexdigest()

    def compile_source(name):
        source = ROOT / "src" / name
        obj = build / (name.replace("/", "-") + ".o")
        receipt = obj.with_suffix(".json")
        identity = hashlib.sha256(
            (headers_hash + hashlib.sha256(source.read_bytes()).hexdigest()).encode()
        ).hexdigest()
        if (
            obj.exists()
            and receipt.exists()
            and json.loads(receipt.read_text()).get("identity") == identity
        ):
            return obj
        print("Building " + name, flush=True)
        subprocess.run(
            compiler + common + ["-c", str(source), "-o", str(obj)], check=True
        )
        receipt.write_text(json.dumps(dict(identity=identity)) + "\n")
        return obj

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        objects = list(pool.map(compile_source, SOURCES))
    subprocess.run(
        compiler
        + [
            "-pthread",
            "-static-libgcc",
            "-static-libstdc++",
            *map(str, objects),
            *libraries,
            "-o",
            str(output),
        ],
        check=True,
    )
    metadata = dict(
        boinc=not args.without_boinc,
        executable_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        sources={
            name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        profiles_sha256=hashlib.sha256(
            (ROOT / "profiles/nvidia.json").read_bytes()
        ).hexdigest(),
    )
    output.with_name(output.name + ".build.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Built " + str(output), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
