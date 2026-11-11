#!/usr/bin/env python3
"""Offline native BOINC proof, checkpoint and suspend/resume test.

Uses a real BOINC SDK shared-memory fixture, without contacting a project.
"""

import argparse
import json
from pathlib import Path
import shutil
from boinc_ipc import Episode, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    slot, public = output / "slot", output / "project"
    slot.mkdir()
    public.mkdir()
    shutil.copy2(args.executable.resolve(), slot / "renamed executable")
    links = {"gproof.proof": "output.proof", "results.txt": "result.out"}
    for logical, physical in links.items():
        (slot / logical).write_text(
            "<soft_link>../project/" + physical + "</soft_link>\n"
        )
    originals = {name: (slot / name).read_text() for name in links}
    episodes = []

    def launch(label):
        return Episode(
            slot,
            public,
            args.initializer.resolve(),
            "p",
            label,
            candidate=1026,
            check_certificate=True,
            check_tail=True,
        )

    def finish(episode, seconds):
        try:
            return episode.finish(seconds)
        finally:
            if episode.process.poll() is None:
                episode.process.kill()
                episode.process.wait()

    episode = launch("checkpoint")
    try:
        episode.wait(lambda: bool(episode.samples), 60)
        episode.control("<suspend/>")
        episode.delay(3)
        assert not (public / "result.out").exists()
        episode.control("<resume/>")
        episode.delay(3)
        episode.control("<quit/>")
        episodes.append(finish(episode, 120))
    finally:
        if episode.process.poll() is None:
            episode.process.kill()
            episode.process.wait()
    assert episodes[-1]["returncode"] == 0
    assert (slot / "gproof.ctx").is_file()
    assert not (public / "result.out").exists()
    assert not (slot / "boinc_finish_called").exists()
    print("Checkpoint, suspend and quit passed", flush=True)

    episode = launch("resume-proof")
    episodes.append(finish(episode, 1200))
    assert episodes[-1]["returncode"] == 0
    assert (slot / "boinc_finish_called").read_text().strip() == "0"
    stderr = (slot / "stderr.txt").read_text()
    assert "Resuming from a checkpoint." in stderr
    assert "Extra tail recomputation passed" in stderr
    assert "Extra certificate verification passed" in stderr
    assert (
        sha(public / "output.proof")
        == "f2baad1af5c6e3a484eff198e3d5d8754dbd2c40872928a7731953de994a5edd"
    )
    assert "pkey = AF7EFC0A8E5E7735" in (public / "result.out").read_text()
    for name, text in originals.items():
        assert (slot / name).read_text() == text
    # Process control must never publish a successful task result.
    for label in ("abort", "heartbeat-loss"):
        case = output / label
        case.mkdir()
        controlled_slot, controlled_public = case / "slot", case / "project"
        controlled_slot.mkdir()
        controlled_public.mkdir()
        shutil.copy2(args.executable.resolve(), controlled_slot / "renamed executable")
        episode = Episode(
            controlled_slot,
            controlled_public,
            args.initializer.resolve(),
            "p",
            label,
            candidate=28000000,
            check_certificate=False,
            check_tail=False,
            # With a nonzero client PID, the SDK checks that process instead
            # of heartbeat expiry. Zero explicitly exercises heartbeat mode.
            client_pid=0 if label == "heartbeat-loss" else None,
        )
        try:
            episode.wait(lambda: bool(episode.samples), 60)
            if label == "abort":
                episode.control("<abort/>")
            else:
                episode.wait(
                    lambda: episode.process.poll() is not None, 100, heartbeat=False
                )
            episodes.append(finish(episode, 120))
        finally:
            if episode.process.poll() is None:
                episode.process.kill()
                episode.process.wait()
        assert episodes[-1]["returncode"] == 0
        assert not (controlled_slot / "results.txt").exists()
        assert not (controlled_slot / "boinc_finish_called").exists()
        print(label + " passed without reporting success", flush=True)
    (output / "result.json").write_text(
        json.dumps(dict(status="PASS", episodes=episodes), indent=2) + "\n"
    )
    print("PASS: native BOINC lifecycle and both optional checks", flush=True)


if __name__ == "__main__":
    main()
