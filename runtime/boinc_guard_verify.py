#!/usr/bin/env python3
"""Private preparation and optional checks for the native BOINC coordinator.

The native parent owns BOINC IPC and publication. This helper never attaches to
BOINC, fetches work, resolves public output names, or publishes result files.
"""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import uuid

sys.dont_write_bytecode = True


class GuardError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GuardError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".new")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checked_module(bundle):
    spec = importlib.util.spec_from_file_location(
        "gfn19_checked_runner", bundle / "run_checked.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(bundle, prefix, base, device, operation, preparing=False):
    require(Path(__file__).resolve().parent == bundle, "helper/bundle mismatch")
    require(
        device == 0, "this experimental bundle supports the single assigned GPU only"
    )
    checked = checked_module(bundle)
    routes = checked.routing_module(bundle)
    arithmetic = os.environ.get("GENEFER_ARITHMETIC")
    require(arithmetic in ("two", "three", "upstream"), "arithmetic selection missing")
    if operation == "proof":
        require(
            arithmetic == "upstream" or arithmetic == routes.arithmetic(base),
            "arithmetic does not match the exact base range",
        )
    manifest_path = bundle / routes.manifest_name(arithmetic)
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest["version"] == 2
        and manifest["arithmetic"] == arithmetic
        and manifest["profile"] == arithmetic,
        "manifest version/profile",
    )
    require(manifest["gpu"] in ("3090", "4060"), "unsupported GPU profile")
    require(
        1024 <= base <= routes.MAX_BASE and base % 2 == 0 and base & (base - 1) != 0,
        "base outside supported range",
    )
    native = {
        "two": "bin/genefer-boinc-two",
        "three": "bin/genefer-boinc-checked",
        "upstream": "bin/genefer-boinc-upstream",
    }[arithmetic]
    required = {
        native,
        "bin/genefer-conservative",
        "bin/genefer-tail-check",
        "boinc_guard_verify.py",
        "run_checked.py",
        "routing.py",
        manifest["kernel"],
    }
    if manifest.get("packed_squaring"):
        require(
            arithmetic == "three" and manifest["backend"] == "cuda",
            "invalid packed backend",
        )
        required.update(checked.PACKED_ASSETS)
    require(set(manifest["files"]) == required, "manifest file set")
    expected_kernel = (
        "kernels/gfn19-three.cl" if arithmetic == "three" else "kernels/gfn19-two.cl"
    )
    require(manifest["kernel"] == expected_kernel, "kernel profile")
    for name, digest in manifest["files"].items():
        path = bundle / name
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "invalid manifest path",
        )
        require(path.resolve().is_relative_to(bundle), "manifest path outside bundle")
        require(sha(path) == digest, "bundle file changed: " + name)
    # The helper is invoked by the native coordinator, never by a shell wrapper.
    parent = Path("/proc/" + str(os.getppid()) + "/exe")
    # BOINC may copy/rename the executable in its project/slot layout.
    require(
        sha(parent) == manifest["files"][native],
        "unexpected parent executable identity",
    )
    require(
        os.environ.get("GENEFER_KERNEL") == str(bundle / manifest["kernel"]),
        "unexpected GPU kernel path",
    )
    expected_prefix = (
        ".gfn_checked_" + ("c_" if operation == "check" else "") + "19_" + str(base)
    )
    require(
        prefix.parent == Path.cwd().resolve() and prefix.name == expected_prefix,
        "unexpected private prefix",
    )
    identity = dict(
        base=base,
        n=19,
        device=device,
        operation=operation,
        arithmetic=arithmetic,
        manifest_sha256=sha(manifest_path),
        files=manifest["files"],
    )
    if manifest.get("packed_squaring") and operation == "proof":
        selected = (
            checked_module(bundle)
            .packed_module(bundle)
            .select_program(base, manifest["gpu"], bundle / "kernels")
        )
        path = prefix.with_name(prefix.name + ".packed.cl")
        if selected is not None:
            text, information = selected
            digest = hashlib.sha256(text.encode()).hexdigest()
            identity["packed_squaring"] = dict(information, source_sha256=digest)
            if path.exists():
                require(sha(path) == digest, "generated packed kernel changed")
            else:
                require(preparing, "generated packed kernel missing")
            if not preparing:
                require(
                    os.environ.get("GENEFER_PACKED_KERNEL") == str(path),
                    "packed kernel path changed",
                )
        else:
            require(not path.exists(), "unexpected packed kernel for this candidate")
    return manifest, identity


def check_owner(prefix, identity, create):
    path = prefix.with_name(prefix.name + ".backend.json")
    if path.exists():
        require(
            json.loads(path.read_text()) == identity, "checkpoint owner/backend changed"
        )
    else:
        require(create, "checkpoint owner missing")
        # The native supervisor opens this log before spawning prepare. It is
        # not a checkpoint and is allowed on the first preparation attempt.
        leftovers = [
            path
            for path in prefix.parent.glob(prefix.name + "*")
            if path.name != prefix.name + ".helper-prepare.log"
        ]
        require(not leftovers, "unowned private checkpoint files")
        atomic_json(path, identity)


def parse_native_result(text, base):
    lines = text.splitlines()
    require(len(lines) == 1, "native output must contain one candidate result")
    line = lines[0]
    label = str(base) + "^{2^19} + 1 is "
    probable = line.startswith(label + "a probable prime")
    require(
        probable or line.startswith(label + "composite"),
        "native result candidate/status",
    )
    require(
        "ckey" not in line and "validation failed" not in line.lower(),
        "unexpected native result content",
    )

    def key(name):
        match = re.search(r"\b" + name + r" = ([0-9a-fA-F]{16})\b", line)
        return match[1].upper() if match else None

    require(key("pkey") is not None, "native proof key missing")
    require((key("res64") is None) == probable, "native residue/status mismatch")
    return dict(
        probable_prime=probable,
        proof_key=key("pkey"),
        residue64=key("res64"),
        old64=key("old64"),
    )


def verify(bundle, prefix, base, device, identity):
    check_owner(prefix, identity, create=False)
    proof = prefix.with_name(prefix.name + ".proof")
    native_path = prefix.with_name(prefix.name + ".native-result.txt")
    original_proof_sha = sha(proof)
    original_native_sha = sha(native_path)
    native = parse_native_result(native_path.read_text(), base)
    spec = importlib.util.spec_from_file_location(
        "immutable_checked_runner", bundle / "run_checked.py"
    )
    checked = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checked)
    work = prefix.with_name(prefix.name + ".verification")
    checked.proof_header(proof, base)
    certificate_requested = os.environ.get("GENEFER_CHECK_CERTIFICATE") == "1"
    tail_requested = os.environ.get("GENEFER_CHECK_TAIL") == "1"
    result, tail = {}, {}
    if certificate_requested:
        work.mkdir(exist_ok=True)
        result = checked.verify(
            proof, base, device, work, bundle / "bin/genefer-conservative"
        )
        for field in ("probable_prime", "proof_key", "residue64"):
            require(
                native[field] == result[field], "native/conservative mismatch: " + field
            )
        server = (Path(result["verification_directory"]) / "server.log").read_text()
        old = re.search(r"\bold64 = ([0-9a-fA-F]{16})\b", server)
        require(
            native["old64"] == (old[1].upper() if old else None),
            "legacy residue mismatch",
        )
        for name in ("server.log", "check.log"):
            log = (Path(result["verification_directory"]) / name).read_text()
            require(
                not re.search(
                    r"validation failed|incorrect|runtime_error|Error:", log, re.I
                ),
                "conservative verifier reported an error",
            )
    if tail_requested:
        checkpoint = prefix.with_name(prefix.name + ".tail")
        checkpoint_sha = sha(checkpoint)
        tail = checked.verify_tail(
            checkpoint,
            proof,
            base,
            device,
            prefix.parent,
            bundle / "bin/genefer-conservative",
        )
        require(
            sha(checkpoint) == checkpoint_sha,
            "final-segment checkpoint changed during verification",
        )
    # Revalidate ownership/identities after the independent GPU subprocesses.
    _, final_identity = load_inputs(bundle, prefix, base, device, "proof")
    require(final_identity == identity, "bundle changed during verification")
    require(
        sha(proof) == original_proof_sha
        and (
            not certificate_requested
            or sha(Path(result["verification_directory"]) / "gproof.proof")
            == original_proof_sha
        ),
        "proof changed during verification",
    )
    require(
        sha(native_path) == original_native_sha,
        "native result changed during verification",
    )
    report = dict(
        completed=True,
        validated=certificate_requested and tail_requested,
        certificate_verified=certificate_requested,
        tail_verified=tail_requested,
        base=base,
        n=19,
        native_result=native,
        native_result_sha256=original_native_sha,
        proof_sha256=original_proof_sha,
        backend_identity=identity,
        verification=result,
        tail_verification=tail,
        scope="Requested optional numerical checks and mandatory input identity checks; native BOINC parent publishes and finishes",
    )
    atomic_json(prefix.with_name(prefix.name + ".verified.json"), report)
    print("BOINC_GUARD_REQUESTED_CHECKS_PASS", flush=True)


def check_certificate(bundle, prefix, base, device, identity):
    """Use only the unchanged conservative implementation for project check jobs."""
    check_owner(prefix, identity, create=False)
    certificate = prefix.with_name(prefix.name + ".cert")
    original_certificate_sha = sha(certificate)
    directory = prefix.with_name(prefix.name + ".check-" + uuid.uuid4().hex)
    directory.mkdir()
    local_prefix = directory / "gproof"
    shutil.copyfile(certificate, directory / "gproof.cert")
    spec = importlib.util.spec_from_file_location(
        "immutable_checked_runner", bundle / "run_checked.py"
    )
    checked = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checked)
    log, elapsed = checked.command(
        bundle / "bin/genefer-conservative",
        "-c",
        base,
        device,
        local_prefix,
        directory,
        directory / "check.log",
        checked.clean_environment(),
    )
    result_path = directory / "results.txt"
    text = result_path.read_text()
    lines = text.splitlines()
    label = str(base) + "^{2^19} + 1 is checked, ckey = "
    require(
        len(lines) == 1 and lines[0].startswith(label),
        "conservative check result missing",
    )
    key = lambda value: re.search(r"\bckey = ([0-9A-Fa-f]{16})\b", value)
    require(
        key(text) is not None
        and key(log) is not None
        and key(text)[1].upper() == key(log)[1].upper(),
        "conservative checker key mismatch",
    )
    require(
        "GFN-19 optimized kernels:" not in log and "Experimental GFN-19" not in log,
        "certificate check did not use conservative arithmetic",
    )
    require(
        not re.search(r"validation failed|incorrect|runtime_error|Error:", log, re.I),
        "conservative certificate checker reported an error",
    )
    _, final_identity = load_inputs(bundle, prefix, base, device, "check")
    require(
        final_identity == identity
        and sha(certificate) == original_certificate_sha
        and sha(directory / "gproof.cert") == original_certificate_sha,
        "input/backend changed during certificate check",
    )
    output = prefix.with_name(prefix.name + ".conservative-result.txt")
    temporary = output.with_name(output.name + ".new")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    atomic_json(
        prefix.with_name(prefix.name + ".checked.json"),
        dict(
            computed=True,
            operation="check",
            base=base,
            n=19,
            validation_status="PROJECT_EXPECTED_KEY_COMPARISON_PENDING",
            certificate_key=key(text)[1].upper(),
            certificate_sha256=original_certificate_sha,
            result_sha256=sha(output),
            checker_seconds=elapsed,
            backend_identity=identity,
            working_directory=str(directory),
        ),
    )
    print("BOINC_GUARD_PRIVATE_CERTIFICATE_CHECK_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "verify", "check"), required=True)
    parser.add_argument("--operation", choices=("proof", "check"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--base", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    bundle, prefix = args.bundle.resolve(), args.prefix.resolve()
    require(
        args.mode == "prepare"
        or (args.mode == "verify" and args.operation == "proof")
        or (args.mode == "check" and args.operation == "check"),
        "mode/operation mismatch",
    )
    manifest, identity = load_inputs(
        bundle,
        prefix,
        args.base,
        args.device,
        args.operation,
        preparing=args.mode == "prepare",
    )
    check_owner(prefix, identity, create=args.mode == "prepare")
    if args.mode == "prepare":
        if identity.get("packed_squaring"):
            text, _ = (
                checked_module(bundle)
                .packed_module(bundle)
                .select_program(args.base, manifest["gpu"], bundle / "kernels")
            )
            path = prefix.with_name(prefix.name + ".packed.cl")
            if not path.exists():
                temporary = path.with_name(path.name + ".new")
                with temporary.open("w") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
        # A previous verification is not authority for a new execution episode.
        for suffix in (".verified.json", ".checked.json", ".conservative-result.txt"):
            old = prefix.with_name(prefix.name + suffix)
            if old.exists():
                old.unlink()
        print("BOINC_GUARD_PRIVATE_INPUTS_PINNED", flush=True)
    elif args.operation == "proof":
        verify(bundle, prefix, args.base, args.device, identity)
    else:
        check_certificate(bundle, prefix, args.base, args.device, identity)


if __name__ == "__main__":
    try:
        main()
    except GuardError as error:
        print("BOINC_GUARD_REJECTED: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
