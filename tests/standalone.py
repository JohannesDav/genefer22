#!/usr/bin/env python3
"""Generate and independently verify a standalone proof against a fixed reference."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--two-prime", action="store_true")
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    output.mkdir()
    prefix = output / "candidate"
    command = [
        "/usr/bin/python3",
        str(bundle / "run_checked.py"),
        "-b",
        "1026",
        "-f",
        str(prefix),
    ]
    if args.two_prime: command.append("--two-prime")
    subprocess.run(command, check=True)
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["validated"] and report["mode"] == "fast", report
    assert (
        report["proof_sha256"]
        == "f2baad1af5c6e3a484eff198e3d5d8754dbd2c40872928a7731953de994a5edd"
    )
    assert report["proof_key"] == "AF7EFC0A8E5E7735"
    assert report["residue64"] == "693104B982BC8D48"
    subprocess.run(command + ["--verify-existing"], check=True)
    repeated = json.loads(prefix.with_suffix(".json").read_text())
    assert repeated["validated"] and repeated["mode"] == "verified-existing"
    assert repeated["proof_sha256"] == report["proof_sha256"]
    # Modified bundle files must be refused before running GPU work.
    manifest = json.loads((bundle / ("boinc-two-prime-manifest.json" if args.two_prime else "boinc-guard-manifest.json")).read_text())
    kernel = bundle / manifest["kernel"]
    original = kernel.read_bytes()
    try:
        kernel.write_bytes(original + b" ")
        rejected = subprocess.run(
            command + ["--verify-existing"], capture_output=True, text=True
        )
        assert rejected.returncode != 0 and "Bundle file changed" in rejected.stderr
    finally:
        kernel.write_bytes(original)
    print(
        "PASS: standalone proof, repeat verification and changed-bundle rejection",
        flush=True,
    )


if __name__ == "__main__":
    main()
