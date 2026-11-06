#!/usr/bin/python3 -I
"""Exec one BOINC-aware backend; no IPC, child supervisor or account handling.

The validated optimized coordinator serves its guarded domain. All other
requests retain the unmodified upstream executable's argument/math behavior.
"""
from pathlib import Path
import hashlib
import json
import os
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError("GFN19 dispatcher: " + message)


def number(words, flag):
    for index, word in enumerate(words):
        if word == "-boinc":
            continue
        if word == flag:
            try:
                return int(words[index + 1])
            except (IndexError, ValueError):
                return None
        if word.startswith(flag) and len(word) > len(flag):
            try:
                return int(word[len(flag) :])
            except ValueError:
                return None
    return None


def main():
    words = sys.argv[1:]
    bundle = None
    forward = []
    force = False
    two_prime = False
    index = 0
    while index < len(words):
        word = words[index]
        if word == "--guard-bundle":
            require(bundle is None and index + 1 < len(words), "invalid --guard-bundle")
            bundle = Path(words[index + 1]).resolve()
            index += 2
            continue
        if word == "--two-prime":
            two_prime = True
            index += 1
            continue
        if word == "--upstream":
            force = True
            index += 1
            continue
        forward.append(word)
        index += 1
    require(bundle is not None and bundle.is_dir(), "explicit --guard-bundle required")
    manifest = json.loads((bundle / "deployment-manifest.json").read_text())
    require(
        manifest["version"] == 1 and manifest["gpu"] in ["3090", "4060"],
        "deployment schema",
    )
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, digest in manifest["files"].items():
        path = bundle / name
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and path.resolve().is_relative_to(bundle),
            "unsafe manifest path",
        )
        require(sha(path) == digest, "deployment file changed: " + name)
    require(
        sha(Path(__file__)) == manifest["files"]["genefer-boinc"],
        "copied launcher changed",
    )
    base = number(forward, "-b")
    n = number(forward, "-n")
    device = number(forward, "--device")
    if device is None:
        device = number(forward, "-d")
    if device is None:
        device = 0
    mode = "p" if "-p" in forward else "c" if "-c" in forward else None
    ordinary = base is not None and base % 2 == 0 and base & (base - 1) != 0
    optimized = (
        not force
        and "-boinc" in forward
        and n == 19
        and device == 0
        and ordinary
        and (
            (mode == "p" and 1024 <= base <= 528220156)
            or (mode == "c" and 1024 <= base <= 2000000000)
        )
    )
    arithmetic="two" if two_prime and base is not None and base<=30000000 else "three"
    backend = (
        bundle
        / "bin"
        / ("genefer-boinc-two-prime" if optimized and arithmetic=="two" else "genefer-boinc-checked" if optimized else "genefer-boinc-stock")
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(bundle / "lib") + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    if not optimized:
        for name in list(env):
            if name.startswith(("GENEFER_", "CUDA_")) or name in [
                "LOAD_SOURCE",
                "DUMP_SOURCE",
                "DUMP_COMPILE_SOURCE",
                "DUMP_PTX",
                "PROFILE",
            ]:
                env.pop(name)
    # Diagnostic only: no account/init data, public output or finish marker.
    diagnostic = dict(
        version=1,
        route="guarded" if optimized else "upstream",
        arithmetic=arithmetic if optimized else "upstream",
        base=base,
        n=n,
        device=device,
        backend_sha256=sha(backend),
        deployment_manifest_sha256=sha(bundle / "deployment-manifest.json"),
        pid=os.getpid(),
    )
    temporary = Path(".gfn19-dispatch-" + str(os.getpid()) + ".new")
    temporary.write_text(json.dumps(diagnostic, indent=2) + "\n")
    temporary.replace(".gfn19-dispatch.json")
    arguments = forward + (["--guard-bundle", str(bundle)] if optimized else [])
    os.execve(str(backend), [str(backend)] + arguments, env)


if __name__ == "__main__":
    main()
