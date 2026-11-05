#!/usr/bin/env python3
"""Compile and run GMP arithmetic tests and an offline BOINC proof lifecycle."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, default=Path("/usr"))
    args = parser.parse_args()
    bundle, output, sdk = (
        p.resolve() for p in (args.bundle, args.output, args.sdk)
    )
    output.mkdir(parents=True)
    manifest = json.loads((bundle / "boinc-guard-manifest.json").read_text())
    compiler = os.environ.get("CXX", "g++")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-DGPU",
        "-DGENEFER_GFN19",
        "-DGENEFER_GUARD_" + manifest["gpu"],
        "-I",
        str(ROOT / "src"),
        "-I",
        str(ROOT / "Khronos"),
        str(ROOT / "tests/arithmetic.cpp"),
        "-lOpenCL",
        "-lgmp",
        "-o",
        str(output / "arithmetic"),
    ]
    subprocess.run(command, check=True)
    env = dict(os.environ, GENEFER_KERNEL=str(bundle / manifest["kernel"]))
    cases = [
        (1026, {}),
        (13427472, {"ROUNDTRIP": "1"}),
        (28000000, {"COPY_RESTORE": "1", "MULTIPLY": "1"}),
        (30000000, {"SCALAR": "3"}),
        (28000000, {"GATE_PATTERN": "sparse"}),
        (28000000, {"GATE_PATTERN": "zero"}),
        (28000000, {"DUP_ALL8": "1", "DUP_OFFSET": "255"}),
    ]
    for index, (base, extra) in enumerate(cases):
        with (output / ("arithmetic-" + str(index) + ".log")).open("w") as log:
            subprocess.run(
                [str(output / "arithmetic"), str(base), "32"],
                env=dict(env, **extra),
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print("Arithmetic case " + str(index) + " passed", flush=True)
    command = [
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
        str(output / "boinc-init"),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "/usr/bin/python3",
            str(ROOT / "tests/boinc_lifecycle.py"),
            "--bundle",
            str(bundle),
            "--initializer",
            str(output / "boinc-init"),
            "--output",
            str(output / "boinc"),
        ],
        check=True,
    )
    print("PASS: arithmetic and offline BOINC tests", flush=True)


if __name__ == "__main__":
    main()
