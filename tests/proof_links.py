#!/usr/bin/env python3
"""Check optional verification with proof/result files behind relative symlinks."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    slot, project = output / "slot", output / "project"
    slot.mkdir()
    project.mkdir()
    (slot / "candidate.proof").symlink_to("../project/candidate.proof")
    (slot / "results.txt").symlink_to("../project/results.txt")
    with (output / "run.log").open("w") as log:
        subprocess.run(
            [
                str(args.executable.resolve()),
                "-n",
                "16",
                "-b",
                "1026",
                "-p",
                "-f",
                "candidate",
                "--check-certificate",
                "--check-tail",
            ],
            cwd=slot,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    text = (output / "run.log").read_text()
    assert "Extra tail recomputation passed" in text
    assert "Extra certificate verification passed" in text
    digest = hashlib.sha256((project / "candidate.proof").read_bytes()).hexdigest()
    assert digest == "cf499d43d54abc4ce0a33ff13c9882cb2051306bad13484cee719edceb01b986"
    assert (slot / "candidate.proof").is_symlink()
    assert (slot / "results.txt").is_symlink()
    assert not list(slot.glob("candidate.verify-*"))
    (output / "result.json").write_text(
        json.dumps(dict(status="PASS", proof_sha256=digest), indent=2) + "\n"
    )
    print("PASS: relative symlinks and both optional verification flags", flush=True)


if __name__ == "__main__":
    main()
