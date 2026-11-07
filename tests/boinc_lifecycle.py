#!/usr/bin/env python3
"""Offline BOINC proof, checkpoint, suspend/resume and certificate-check test."""

import argparse
import json
from pathlib import Path
import shutil
from boinc_ipc import Episode, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    output.mkdir()
    slot, public = output / "slot", output / "project"
    slot.mkdir()
    public.mkdir()
    shutil.copy2(bundle / "genefer-boinc", slot / "renamed executable")
    links = {"gproof.proof": "output.proof", "results.txt": "result.out"}
    for logical, physical in links.items():
        (slot / logical).write_text(
            "<soft_link>../project/" + physical + "</soft_link>\n"
        )
    originals = {name: (slot / name).read_text() for name in links}

    def private():
        assert not (public / "result.out").exists()
        assert not (public / "output.proof").exists()
        assert not (slot / "boinc_finish_called").exists()

    episodes = []
    episode = Episode(
        slot,
        public,
        bundle,
        args.initializer.resolve(),
        "p",
        "checkpoint",
    )
    try:
        episode.wait(lambda: bool(episode.samples), 30)
        episode.delay(6)
        episode.control("<suspend/>")
        episode.delay(3)
        private()
        episode.control("<resume/>")
        episode.delay(3)
        episode.control("<quit/>")
        episodes.append(episode.finish(90))
    finally:
        if episode.process.poll() is None:
            episode.process.kill()
            episode.process.wait()
    private()
    assert (slot / ".gfn_checked_19_28000000.ctx").is_file()
    assert episodes[-1]["returncode"] == 0
    print("Checkpoint/quit passed", flush=True)
    manifest = json.loads((bundle / "boinc-guard-manifest.json").read_text())
    if manifest.get("packed_squaring"):
        rejected_root = output / "changed-kernel"
        rejected_root.mkdir()
        rejected_slot = rejected_root / "slot"
        rejected_project = rejected_root / "project"
        shutil.copytree(slot, rejected_slot)
        rejected_project.mkdir()
        generated = rejected_slot / ".gfn_checked_19_28000000.packed.cl"
        generated.write_text(generated.read_text() + "\n// changed kernel\n")
        rejected = Episode(
            rejected_slot,
            rejected_project,
            bundle,
            args.initializer.resolve(),
            "p",
            "changed-kernel",
        )
        try:
            # An intentional preparation failure can precede the first periodic
            # progress message. Its error and absence of outputs are checked below.
            outcome = rejected.finish(60, require_status=False)
        finally:
            if rejected.process.poll() is None:
                rejected.process.kill()
                rejected.process.wait()
        assert outcome["returncode"] != 0
        assert not (rejected_project / "output.proof").exists()
        assert not (rejected_project / "result.out").exists()
        assert (
            "generated packed kernel changed"
            in (
                rejected_slot / ".gfn_checked_19_28000000.helper-prepare.log"
            ).read_text()
        )
        print("Changed packed kernel refused before resuming", flush=True)

    episode = Episode(
        slot,
        public,
        bundle,
        args.initializer.resolve(),
        "p",
        "resume-proof",
    )
    try:
        episode.wait(
            lambda: (slot / ".gfn_checked_19_28000000.helper-verify.log").exists(), 1200
        )
        private()
        episode.control("<suspend/>")
        episode.delay(3)
        private()
        episode.control("<resume/>")
        episodes.append(episode.finish(150))
    finally:
        if episode.process.poll() is None:
            episode.process.kill()
            episode.process.wait()
    assert episodes[-1]["returncode"] == 0
    assert (slot / "boinc_finish_called").read_text().strip() == "0"
    assert "Resuming from a checkpoint." in (slot / "stderr.txt").read_text()
    assert (
        sha(public / "output.proof")
        == "d8f472ca5b573809d70b30fa31a2dc386b3a7b2237c7cbb1148ed5e3f6e0cce4"
    )
    assert "pkey = 7FCD90789A28A076" in (public / "result.out").read_text()
    for name, text in originals.items():
        assert (slot / name).read_text() == text
    verified = json.loads((slot / ".gfn_checked_19_28000000.verified.json").read_text())
    print("Full resumed proof matches the reference", flush=True)
    # Exercise certificate checks in a separate slot with an actual OS symlink.
    check_slot, check_public = output / "check-slot", output / "check-project"
    check_slot.mkdir()
    check_public.mkdir()
    shutil.copy2(bundle / "genefer-boinc", check_slot / "renamed executable")
    shutil.copy2(
        verified["verification"]["certificate_path"], check_public / "input.cert"
    )
    (check_slot / "gproof.cert").symlink_to("../check-project/input.cert")
    (check_slot / "results.txt").symlink_to("../check-project/result.out")
    episode = Episode(
        check_slot,
        check_public,
        bundle,
        args.initializer.resolve(),
        "c",
        "certificate",
    )
    try:
        episodes.append(episode.finish(150))
    finally:
        if episode.process.poll() is None:
            episode.process.kill()
            episode.process.wait()
    assert episodes[-1]["returncode"] == 0
    assert (check_slot / "boinc_finish_called").read_text().strip() == "0"
    assert (
        "ckey = " + verified["verification"]["certificate_key"]
        in (check_public / "result.out").read_text()
    )
    assert (check_slot / "gproof.cert").is_symlink()
    assert (check_slot / "results.txt").is_symlink()
    (output / "result.json").write_text(
        json.dumps(dict(status="PASS", episodes=episodes), indent=2) + "\n"
    )
    print("PASS: offline BOINC proof and certificate lifecycle", flush=True)


if __name__ == "__main__":
    main()
