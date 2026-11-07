#!/usr/bin/env python3
"""Compare automatically selected GPU arithmetic with GMP, then test offline BOINC."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk", type=Path, default=Path("/usr"))
    p.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    p.add_argument("--bases", type=int, nargs="+")
    p.add_argument("--arithmetic-only", action="store_true")
    args = p.parse_args()
    bundle, output, sdk, cuda = (
        x.resolve() for x in (args.bundle, args.output, args.sdk, args.cuda)
    )
    output.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("checked", bundle / "run_checked.py")
    checked = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checked)
    routes = checked.routing_module(bundle)
    compiler = os.environ.get("CXX", "g++")
    binaries = {}
    rows = []
    bases = args.bases or [
        1026,
        2072558,
        2072560,
        4038248,
        4038250,
        4194302,
        4194306,
        28000000,
        31296290,
        31296292,
        114484426,
        114484428,
        418735722,
        418735724,
        528220156,
        528220158,
        1000000000,
        1000000002,
        2000000000,
    ]
    for base in bases:
        manifest = checked.arithmetic_manifest(bundle, base)
        kind = manifest["arithmetic"]
        width = (
            routes.two_prime_width(base)
            if kind == "two"
            else (32 if base > 1000000000 else 31)
        )
        key = (kind, width)
        if key not in binaries:
            binary = output / (kind + str(width))
            flags = ["-DGPU", "-DGENEFER_CUDA", "-DGENEFER_GUARD_" + manifest["gpu"]]
            if kind == "three":
                flags += ["-DGENEFER_THREE_PRIME"]
            else:
                flags += [
                    "-DGENEFER_REFERENCE_FIELDS=" + ("2" if kind == "two" else "3"),
                    "-DGENEFER_REFERENCE_WIDE=" + str(int(width == 32)),
                ]
                if kind == "two":
                    flags += ["-DGENEFER_EXACT_TWO"]
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O3",
                    *flags,
                    "-I",
                    str(ROOT / "src"),
                    "-I",
                    str(ROOT / "Khronos"),
                    "-I",
                    str(cuda / "include"),
                    str(ROOT / "tests/arithmetic.cpp"),
                    "-L",
                    str(cuda / "lib64/stubs"),
                    "-lcuda",
                    "-lOpenCL",
                    "-lgmp",
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            binaries[key] = binary
        cases = [
            {"ROUNDTRIP": "1", "MULTIPLY": "1", "COPY_RESTORE": "1"},
            {
                "GATE_PATTERN": "alternating",
                "GATE_AMPLITUDE": str(base // 2 + 2),
                "SCALAR": "2147483647",
            },
            {
                "GATE_PATTERN": "constant",
                "GATE_AMPLITUDE": str(base // 2 + 2),
                "DUP_ALL8": "1",
                "DUP_OFFSET": "255",
            },
            {
                "GATE_PATTERN": "sparse",
                "MULTIPLY": "1",
                "CURRENT_MULTIPLICAND": "1",
                "SQUARE_AFTER_MULTIPLY": "1",
            },
        ]
        if kind == "three":
            cases += [
                {
                    "PROOF_PHASE": "1",
                    "ROUNDTRIP": "1",
                    "COPY_RESTORE": "1",
                    "MULTIPLY": "1",
                    "CURRENT_MULTIPLICAND": "1",
                    "SQUARE_AFTER_MULTIPLY": "1",
                }
            ]
        for extra in cases:
            env = checked.clean_environment()
            checked.configure_runtime(env, bundle, manifest, base, output)
            log = output / ("case-" + str(len(rows)) + ".log")
            with log.open("w") as f:
                subprocess.run(
                    [str(binaries[key]), str(base), "32"],
                    env=dict(env, **extra),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            rows.append(
                dict(base=base, arithmetic=kind, width=width, case=extra, status="PASS")
            )
            (output / "arithmetic-results.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            print("PASS", base, kind, extra, flush=True)
    if args.arithmetic_only:
        return
    initializer = output / "boinc-init"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(sdk / "include/boinc"),
            "-I",
            str(sdk / "include/boinc/lib"),
            str(ROOT / "tests/boinc_init.cpp"),
            "-L",
            str(sdk / "lib/x86_64-linux-gnu"),
            "-Wl,-rpath," + str(sdk / "lib/x86_64-linux-gnu"),
            "-lboinc",
            "-o",
            str(initializer),
        ],
        check=True,
    )
    subprocess.run(
        [
            "python3",
            str(ROOT / "tests/boinc_lifecycle.py"),
            "--bundle",
            str(bundle),
            "--initializer",
            str(initializer),
            "--output",
            str(output / "boinc"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
