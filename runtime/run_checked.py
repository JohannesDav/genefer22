#!/usr/bin/env python3
"""Run an exact GFN19 prover, with optional independent checks.
Standalone workflow; this is not a BOINC application wrapper.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import uuid

PACKED_ASSETS = (
    "packed_kernel.py",
    "kernels/gfn19-packed.cl",
    "kernels/gfn19-packed-convert.clh",
    "kernels/gfn19-packed-codec.clh",
)
BUNDLE = Path(__file__).resolve().parent
CKEY = re.compile(r"\bckey = ([0-9A-Fa-f]{16})\b")
PKEY = re.compile(r"\bpkey = ([0-9A-Fa-f]{16})\b")
RESIDUE = re.compile(r"\bres64 = ([0-9A-Fa-f]{16})\b")


class ValidationError(RuntimeError):
    pass


def proof_header(path, base):
    with path.open("rb") as f:
        header = f.read(16)
    if len(header) != 16:
        raise ValidationError("Proof has a truncated header")
    version, depth, size, stored_base = struct.unpack("<iiII", header)
    if version != 1 or not 1 <= depth <= 20 or size != 1 << 19 or stored_base != base:
        raise ValidationError("Proof parameters do not match the requested candidate")
    expected = 8 + (depth + 1) * (8 + 4 * size) + 4
    if path.stat().st_size != expected:
        raise ValidationError("Proof has an unexpected length")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def clean_environment():
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("GENEFER_", "CUDA_")) or name in (
            "LOAD_SOURCE",
            "DUMP_SOURCE",
            "DUMP_COMPILE_SOURCE",
            "DUMP_PTX",
            "PROFILE",
        ):
            env.pop(name, None)
    return env


def command(binary, mode, base, device, prefix, cwd, log, env):
    args = [
        str(binary),
        "-n",
        "19",
        "-b",
        str(base),
        mode,
        "-d",
        str(device),
        "-f",
        str(prefix),
    ]
    started = time.monotonic()
    print("{}: {}".format(mode, log), flush=True)
    with log.open("wb") as out:
        p = subprocess.Popen(
            args, cwd=str(cwd), env=env, stdout=out, stderr=subprocess.STDOUT
        )
        try:
            code = p.wait()
        except KeyboardInterrupt:
            try:
                p.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.terminate()
                p.wait()
            raise
    text = log.read_text(errors="replace")
    if code:
        raise ValidationError(
            "{} exited with status {}; see {}".format(binary.name, code, log)
        )
    return text, time.monotonic() - started


def verify(proof, base, device, work, conservative):
    proof_header(proof, base)
    directory = work / ("verify-" + uuid.uuid4().hex[:12])
    directory.mkdir()
    prefix = directory / "gproof"
    shutil.copyfile(str(proof), str(prefix) + ".proof")
    env = clean_environment()
    server, st = command(
        conservative,
        "-s",
        base,
        device,
        prefix,
        directory,
        directory / "server.log",
        env,
    )
    candidate = "{}^{{2^19}} + 1 is ".format(base)
    if candidate not in server or (
        "composite" not in server and "a probable prime" not in server
    ):
        raise ValidationError(
            "Conservative server did not produce a result; see " + str(directory)
        )
    if "Experimental GFN-19" in server or "GFN-19 optimized kernels:" in server:
        raise ValidationError(
            "Certificate generation did not use the conservative binary"
        )
    key = CKEY.search(server)
    if key is None or not Path(str(prefix) + ".cert").is_file():
        raise ValidationError("Conservative server did not produce a certificate/key")
    checked, ct = command(
        conservative,
        "-c",
        base,
        device,
        prefix,
        directory,
        directory / "check.log",
        env,
    )
    check_key = CKEY.search(checked)
    if (
        candidate + "checked" not in checked
        or check_key is None
        or key[1].upper() != check_key[1].upper()
    ):
        raise ValidationError(
            "Independent certificate check failed; see " + str(directory)
        )
    pkey, residue = PKEY.search(server), RESIDUE.search(server)
    return {
        "certificate_key": key[1].upper(),
        "proof_key": pkey[1].upper() if pkey else None,
        "residue64": residue[1].upper() if residue else None,
        "probable_prime": candidate + "a probable prime" in server,
        "verification_seconds": st + ct,
        "verification_directory": str(directory),
        "certificate_path": str(prefix) + ".cert",
    }


def atomic_copy(source, destination):
    fd, name = tempfile.mkstemp(
        prefix=destination.name + ".", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as out, Path(source).open("rb") as inp:
            shutil.copyfileobj(inp, out, 1024 * 1024)
        os.replace(name, str(destination))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def routing_module(bundle):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gfn19_routing", bundle / "routing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arithmetic_manifest(bundle, base=28000000):
    routes = routing_module(bundle)
    kind = routes.arithmetic(base)
    data = json.loads((bundle / routes.manifest_name(kind)).read_text())
    if data.get("version") != 2 or data.get("arithmetic") != kind:
        raise ValidationError("Invalid arithmetic manifest")
    return data


def packed_module(bundle):
    # Tests also load this wrapper by filename, outside the bundle's sys.path.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gfn19_packed_parameters", bundle / "packed_kernel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_runtime(
    env, bundle, manifest, base=None, directory=None, packed_info=None
):
    env.pop("GENEFER_PACKED_KERNEL", None)
    env["GENEFER_ARITHMETIC"] = manifest["arithmetic"]
    kernel = str(bundle / manifest["kernel"])
    env["GENEFER_KERNEL"] = kernel
    if manifest.get("packed_squaring") and base is not None:
        selected = packed_module(bundle).select_program(
            base, manifest["gpu"], bundle / "kernels"
        )
        if selected is not None:
            text, information = selected
            require_directory = Path(directory) if directory is not None else None
            if require_directory is None:
                raise ValidationError("Packed kernel needs a working directory")
            path = require_directory / "gfn19-packed.cl"
            path.write_text(text)
            env["GENEFER_PACKED_KERNEL"] = str(path)
            if packed_info is not None:
                packed_info.update(information, source_sha256=sha256(path))
    return env


def verify_tail(checkpoint, proof, base, device, work, conservative):
    # Original upstream arithmetic, with a complete digit-vector comparison.
    checker = conservative.parent / "genefer-tail-check"
    log = work / "tail-check.log"
    started = time.monotonic()
    with log.open("w") as output:
        result = subprocess.run(
            [str(checker), str(base), str(checkpoint), str(proof)],
            cwd=work,
            env=clean_environment(),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    if result.returncode or "TAIL_MATCH=PASS" not in log.read_text():
        raise ValidationError("Final-segment verification failed; see " + str(log))
    return {
        "tail_verified": True,
        "tail_seconds": time.monotonic() - started,
        "tail_checkpoint_sha256": sha256(checkpoint),
    }


def _execute_locked(
    base,
    output,
    device=0,
    verify_existing=False,
    bundle=BUNDLE,
    check_certificate=False,
    check_tail=False,
):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    proof = Path(str(output) + ".proof")
    report_path = Path(str(output) + ".json")
    work = Path(str(output) + ".work")
    if not verify_existing and (proof.exists() or report_path.exists()):
        raise ValidationError(
            "Output already exists; choose another -f or use --verify-existing"
        )
    if verify_existing and not proof.is_file():
        raise ValidationError("No proof exists at " + str(proof))
    conservative = (bundle / "bin" / "genefer-conservative").resolve()
    deployment = json.loads((bundle / "deployment-manifest.json").read_text())
    for name, digest in deployment["files"].items():
        path = bundle / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.resolve().is_relative_to(bundle)
            or sha256(path) != digest
        ):
            raise ValidationError("Bundle file changed: " + name)
    manifest = arithmetic_manifest(bundle, base)
    fast = (bundle / manifest["prover"]).resolve()
    kernel = (bundle / manifest["kernel"]).resolve()

    if not conservative.is_file():
        raise ValidationError("Conservative binary missing; run build.py first")
    work.mkdir(exist_ok=True)
    metadata = work / "candidate.json"
    identity = {
        "base": base,
        "n": 19,
        "device": device,
        "arithmetic": manifest["arithmetic"],
    }
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValidationError("Working directory belongs to another candidate/device")
    write_json(metadata, identity)
    if manifest["gpu"] not in ("3090", "4060"):
        raise ValidationError("Invalid GPU profile")
    backend = {
        "fast_binary": fast.name,
        "fast_sha256": sha256(fast) if fast.is_file() else None,
        "kernel_sha256": sha256(kernel) if kernel.is_file() else None,
        "conservative_sha256": sha256(conservative),
        "runner_sha256": sha256(Path(__file__)),
    }
    if manifest.get("packed_squaring"):
        backend["packed_sources"] = {
            name: sha256(bundle / name) for name in PACKED_ASSETS
        }
    if not verify_existing:
        backend_path = work / "backend.json"
        if backend_path.exists():
            if json.loads(backend_path.read_text()) != backend:
                raise ValidationError(
                    "Checkpoint backend has changed; use a fresh -f prefix"
                )
        elif (work / "fast").exists() or (work / "conservative-fallback").exists():
            raise ValidationError(
                "Legacy checkpoints have no backend identity; use a fresh -f prefix"
            )
        write_json(backend_path, backend)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    started = time.monotonic()
    report = dict(
        identity,
        validated=False,
        certificate_verified=False,
        tail_verified=False,
        checks_requested=dict(
            certificate=check_certificate or verify_existing, tail=check_tail
        ),
        fallback_reason=state.get("fallback_reason"),
        backend_identity=backend,
    )
    if verify_existing:
        prior = json.loads(report_path.read_text()) if report_path.exists() else {}
        original_mode = prior.get("proof_mode", prior.get("mode"))
        if original_mode not in ("fast", "upstream", "conservative-fallback"):
            original_mode = (
                "conservative-fallback"
                if state.get("fast_rejected")
                else ("upstream" if manifest["arithmetic"] == "upstream" else "fast")
            )
        report.update(verify(proof, base, device, work, conservative))
        report["certificate_verified"] = True
        if check_tail:
            report.update(
                verify_tail(
                    work / original_mode / "gproof.tail",
                    proof,
                    base,
                    device,
                    work,
                    conservative,
                )
            )
        report.update(proof_mode=original_mode, mode="verified-existing")
    else:
        source = None
        modes = (
            ("upstream",)
            if manifest["arithmetic"] == "upstream"
            else (
                ("conservative-fallback",)
                if state.get("fast_rejected")
                else ("fast", "conservative-fallback")
            )
        )
        for mode in modes:
            directory = work / mode
            directory.mkdir(exist_ok=True)
            prefix = directory / "gproof"
            env = clean_environment()
            binary = bundle / "bin/genefer-gfn19-upstream"
            packed_info = {}
            if mode in ("fast", "upstream"):
                binary = fast
                configure_runtime(env, bundle, manifest, base, directory, packed_info)
            if check_tail:
                env["GENEFER_CHECK_TAIL"] = "1"
            try:
                text, seconds = command(
                    binary,
                    "-p",
                    base,
                    device,
                    prefix,
                    directory,
                    directory / "proof.log",
                    env,
                )
                generated = Path(str(prefix) + ".proof")
                if "proof file is generated" not in text or not generated.is_file():
                    raise ValidationError(
                        "Prover did not complete successfully; see " + str(directory)
                    )
                proof_header(generated, base)
                candidate = str(base) + "^{2^19} + 1 is "
                if (
                    candidate + "a probable prime" not in text
                    and candidate + "composite" not in text
                ):
                    raise ValidationError(
                        "Prover did not report the requested candidate"
                    )
                key, residue = PKEY.search(text), RESIDUE.search(text)
                report.update(
                    probable_prime=candidate + "a probable prime" in text,
                    proof_key=key[1].upper() if key else None,
                    residue64=residue[1].upper() if residue else None,
                )
                if check_certificate:
                    checked = verify(generated, base, device, work, conservative)
                    for field in ("probable_prime", "proof_key", "residue64"):
                        if report[field] != checked[field]:
                            raise ValidationError(
                                "Native/certificate mismatch: " + field
                            )
                    report.update(checked, certificate_verified=True)
                if check_tail:
                    report.update(
                        verify_tail(
                            Path(str(prefix) + ".tail"),
                            generated,
                            base,
                            device,
                            work,
                            conservative,
                        )
                    )
                report.update(mode=mode, proof_mode=mode, prover_seconds=seconds)
                if mode == "fast" and packed_info:
                    report["packed_squaring"] = packed_info
                source = generated
                break
            except (ValidationError, OSError) as exc:
                if mode != "fast":
                    raise ValidationError("Upstream calculation failed: " + str(exc))
                report["fallback_reason"] = str(exc)
                write_json(
                    state_path, {"fast_rejected": True, "fallback_reason": str(exc)}
                )
                print(
                    "Optimized calculation or requested check failed; retrying with upstream: "
                    + str(exc),
                    flush=True,
                )
        if source is None:
            raise ValidationError("No completed proof was generated")
        atomic_copy(source, proof)
    if report.get("certificate_path"):
        atomic_copy(report["certificate_path"], Path(str(output) + ".cert"))
    report.update(
        completed=True,
        validated=report["certificate_verified"] and report["tail_verified"],
        proof_sha256=sha256(proof),
        total_seconds=time.monotonic() - started,
        working_directory=str(work),
    )
    write_json(report_path, report)
    print(
        "Completed {}: {} ({:.2f}s total)".format(
            base, report["mode"], report["total_seconds"]
        ),
        flush=True,
    )
    print("Proof: " + str(proof), flush=True)
    return report


def execute(
    base,
    output,
    device=0,
    verify_existing=False,
    bundle=BUNDLE,
    check_certificate=False,
    check_tail=False,
):
    output = Path(output).resolve()
    work = Path(str(output) + ".work")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValidationError("Another runner is using " + str(work))
        return _execute_locked(
            base, output, device, verify_existing, bundle, check_certificate, check_tail
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-b", "--base", type=int, required=True)
    p.add_argument("-n", type=int, default=19, choices=[19])
    p.add_argument("-f", "--output", default="gproof", help="Output filename prefix")
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify PREFIX.proof without recomputing it",
    )
    p.add_argument(
        "--check-certificate",
        action="store_true",
        help="Generate and verify a certificate using the original implementation",
    )
    p.add_argument(
        "--check-tail",
        action="store_true",
        help="Recompute and compare the final calculation segment using the original implementation",
    )
    args = p.parse_args(argv)
    b = args.base
    maximum = 2000000000
    if b < 1024 or b > maximum or b % 2 or b & (b - 1) == 0:
        p.error("base must be even, 1024..{} and not a power of two".format(maximum))
    if args.device != 0:
        p.error("Only device 0 is supported")
    try:
        execute(
            b,
            args.output,
            args.device,
            args.verify_existing,
            check_certificate=args.check_certificate,
            check_tail=args.check_tail,
        )
    except KeyboardInterrupt:
        print(
            "Interrupted; checkpoints retained in PREFIX.work for resuming.",
            file=sys.stderr,
        )
        return 130
    except (ValidationError, OSError, ValueError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
