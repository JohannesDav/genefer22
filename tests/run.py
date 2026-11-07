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
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--two-prime", action="store_true")
    parser.add_argument("--arithmetic-only", action="store_true")
    args = parser.parse_args()
    bundle, output, sdk, cuda = (
        p.resolve() for p in (args.bundle, args.output, args.sdk, args.cuda)
    )
    output.mkdir(parents=True)
    manifest = json.loads(
        (
            bundle
            / (
                "boinc-two-prime-manifest.json"
                if args.two_prime
                else "boinc-guard-manifest.json"
            )
        ).read_text()
    )
    compiler = os.environ.get("CXX", "g++")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-DGPU",
        "-DGENEFER_CUDA",
        "-DGENEFER_GUARD_" + manifest["gpu"],
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
        str(output / "arithmetic"),
    ]
    if not args.two_prime:
        command.insert(1, "-DGENEFER_THREE_PRIME")
    subprocess.run(command, check=True)
    import importlib.util

    spec = importlib.util.spec_from_file_location("checked", bundle / "run_checked.py")
    checked = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checked)
    env = checked.clean_environment()
    checked.configure_runtime(env, bundle, manifest)
    cases = [
        (1026, {}),
        (13427472, {"ROUNDTRIP": "1"}),
        (28000000, {"COPY_RESTORE": "1", "MULTIPLY": "1"}),
        (30000000, {"SCALAR": "3"}),
        (28000000, {"GATE_PATTERN": "sparse"}),
        (28000000, {"GATE_PATTERN": "zero"}),
        (28000000, {"DUP_ALL8": "1", "DUP_OFFSET": "255"}),
    ]
    if not args.two_prime:
        for base in [28000000, 528220156]:
            cases.append(
                (
                    base,
                    {
                        "PROOF_PHASE": "1",
                        "ROUNDTRIP": "1",
                        "COPY_RESTORE": "1",
                        "MULTIPLY": "1",
                        "CURRENT_MULTIPLICAND": "1",
                        "SQUARE_AFTER_MULTIPLY": "1",
                    },
                )
            )
        for base in [28000000, 528220156]:
            cases.append(
                (
                    base,
                    {
                        "MULTIPLY": "1",
                        "CURRENT_MULTIPLICAND": "1",
                        "SQUARE_AFTER_MULTIPLY": "1",
                    },
                )
            )
        for base in [38000000, 100000000, 528220156]:
            cases.extend(
                [
                    (base, {"COPY_RESTORE": "1", "MULTIPLY": "1"}),
                    (
                        base,
                        {
                            "GATE_PATTERN": "alternating",
                            "GATE_AMPLITUDE": str(base + 2),
                            "SCALAR": "2147483647",
                        },
                    ),
                ]
            )
        cases.extend(
            [
                (1026, {"SCALAR": "2147483647"}),
                (
                    528220156,
                    {"GATE_PATTERN": "constant", "GATE_AMPLITUDE": "528220158"},
                ),
            ]
        )
    for index, (base, extra) in enumerate(cases):
        case_environment = dict(env)
        checked.configure_runtime(case_environment, bundle, manifest, base, output)

        with (output / ("arithmetic-" + str(index) + ".log")).open("w") as log:
            subprocess.run(
                [str(output / "arithmetic"), str(base), "32"],
                env=dict(case_environment, **extra),
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print("Arithmetic case " + str(index) + " passed", flush=True)
    if args.arithmetic_only:
        print("PASS: GMP arithmetic tests", flush=True)
        return
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
        ]
        + (["--two-prime"] if args.two_prime else []),
        check=True,
    )
    print("PASS: arithmetic and offline BOINC tests", flush=True)


if __name__ == "__main__":
    main()
