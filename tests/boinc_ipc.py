#!/usr/bin/env python3
"""Offline BOINC API fixture with synthetic task data.

Tests actual native binary and controls, not a substitute for a real-client test.
"""
import hashlib
import mmap
import os
import subprocess
import time


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Episode:
    def __init__(
        self,
        slot,
        public,
        bundle,
        initializer,
        mode,
        label,
        client_pid=None,
        candidate=28000000,
        two_prime=False,
    ):
        self.slot, self.label = slot, label
        subprocess.run(
            [
                str(initializer),
                str(slot),
                str(os.getpid() if client_pid is None else client_pid),
                str(public),
            ],
            check=True,
        )
        self.file = (slot / "boinc_mmap_file").open("r+b")
        self.mem = mmap.mmap(self.file.fileno(), 8192)
        self.log = (slot.parent / (label + ".launch.log")).open("w")
        self.samples = []
        self.last_heartbeat = 0
        self.process = subprocess.Popen(
            [
                str(slot / "renamed executable"),
                "-boinc",
                "-" + mode,
                "-n",
                "19",
                "-b",
                str(candidate),
                "-f",
                "gproof",
                "--guard-bundle",
                str(bundle),
            ] + (["--two-prime"] if two_prime else []),
            cwd=slot,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started = time.monotonic()

    def send(self, channel, message):
        offset = 1024 * channel
        if self.mem[offset]:
            return False
        data = message.encode() + b"\0"
        assert len(data) < 1024
        self.mem[offset + 1 : offset + 1 + len(data)] = data
        self.mem[offset] = 1
        return True

    def tick(self, heartbeat=True):
        now = time.monotonic()
        if heartbeat and now - self.last_heartbeat >= 0.5:
            if self.send(4, "<heartbeat/>"):
                self.last_heartbeat = now
        if self.mem[5 * 1024]:
            data = bytes(self.mem[5 * 1024 + 1 : 6 * 1024]).split(b"\0")[0].decode()
            self.mem[5 * 1024] = 0
            self.samples.append(dict(elapsed=now - self.started, message=data))
        time.sleep(0.025)

    def wait(self, predicate, seconds=60, heartbeat=True):
        deadline = time.monotonic() + seconds
        while not predicate():
            assert self.process.poll() is None, (
                self.label,
                "unexpected exit",
                self.process.returncode,
            )
            assert time.monotonic() < deadline, (self.label, "deadline")
            self.tick(heartbeat)

    def control(self, message):
        self.wait(lambda: self.send(0, message), 10)

    def delay(self, seconds):
        end = time.monotonic() + seconds
        self.wait(lambda: time.monotonic() >= end, seconds + 5)

    def finish(self, seconds=120, require_status=True):
        deadline = time.monotonic() + seconds
        while self.process.poll() is None:
            assert time.monotonic() < deadline, (self.label, "finish deadline")
            self.tick()
        self.log.close()
        self.mem.close()
        self.file.close()
        stderr = (self.slot / "stderr.txt").read_text()
        assert "running in standalone mode" not in stderr
        if require_status:
            assert self.samples, (self.label, "no actual IPC app-status messages")
        return dict(
            label=self.label,
            returncode=self.process.returncode,
            elapsed=time.monotonic() - self.started,
            status_samples=self.samples,
            stderr_sha256=sha(self.slot / "stderr.txt"),
        )
