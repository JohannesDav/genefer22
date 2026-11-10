#!/usr/bin/env python3
"""Two-stage exact CUDA GFN screening, independent of the rental infrastructure.

Loop measurements select candidates for full proofs; they are not headline
benchmarks or a substitute for checking the implementation. Inputs, kernels,
commands, hashes, failures and timings are preserved in a new output directory.
"""

from pathlib import Path
import argparse, importlib.util, json, math, os, re, statistics, subprocess, time, hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_module(path):
    spec = importlib.util.spec_from_file_location("packed_calibration", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def upstream_two_prime_bits(base, exponent):
    for bits, primes in [(31, (2130706433, 2113929217)),
                         (32, (4194304001, 4076863489))]:
        if base * base < (primes[0] * primes[1] // 2) // (1 << exponent):
            return bits
    return None


def plans(m, src, n, b):
    lower = max(14, n - 4)
    upper = min(n, 20)
    preferred = {
        16: (14, 15, 16),
        17: (15, 16, 17),
        18: (16, 17, 18),
        19: (16, 17, 18),
        20: (16, 17, 18),
        21: (17, 18, 19),
        22: (18, 19, 20),
    }[n]
    for ln in preferred:
        if not lower <= ln <= upper:
            continue
        N = 1 << ln
        pack = (1 << n) // N
        accepted = None
        for fields in range(2, 33):
            for ratio in (2, 3, 4):
                try:
                    p = m.parameters(
                        b,
                        fields,
                        length=N,
                        pack=pack,
                        primes=m.field_pool(fields, N)[0],
                        accurate_error=True,
                        digit_ratio=ratio,
                    )
                    if 2 * p["e"] * p["B"] + p["e"] + p["Tmax"] >= 3 * p["B"]:
                        continue
                    accepted = (fields, ratio)
                    break
                except m.UnsupportedPackedParameters:
                    pass
            if accepted:
                break
        if not accepted:
            continue
        fields, ratio = accepted
        for outer in (16, 64, 256):
            chunk = 4 if outer <= 64 else 1
            config = dict(
                length=N,
                pack=pack,
                outer=outer,
                outer_chunk=chunk,
                root_bits=32,
                inner_blocks=1,
                digit_ratio=ratio,
                accurate_error=True,
            )
            try:
                code, meta = m.program(b, fields, src / "kernels/src", **config)
            except m.UnsupportedPackedParameters:
                continue
            yield fields, config, code, meta


def export_profile(checks, winners, memory_gib):
    """Return a build-profile entry from successful probe results."""
    hardware = next((row for row in checks if "device_name" in row), None)
    if hardware is None:
        raise ValueError("Profile export needs a current probe with device metadata")
    memory = hardware["memory_bytes"]
    if (
        not memory_gib * (1 << 30) * 9 // 10
        < memory
        <= memory_gib * (1 << 30) + (128 << 20)
    ):
        raise ValueError("Nominal memory capacity does not match the tested device")
    major, minor = hardware["compute_major"], hardware["compute_minor"]
    architecture = {
        5: "maxwell",
        6: "pascal",
        9: "hopper",
        10: "blackwell",
        12: "blackwell",
    }.get(major, "unknown")
    if major == 7:
        architecture = "volta" if minor < 5 else "turing"
    if major == 8:
        architecture = (
            "ampere_sm80" if minor == 0 else "ada" if minor >= 9 else "ampere"
        )
    projects = {}
    for winner in winners:
        if "configuration" in winner:
            config = winner["configuration"]
            entry = dict(
                implementation="packed",
                length=config["length"],
                outer=config["outer"],
                chunk=config["outer_chunk"],
                root_bits=config["root_bits"],
                inner_blocks=config["inner_blocks"],
                digit_ratio=config["digit_ratio"],
            )
            for flag in ("matrix_mad", "inline_converter"):
                if winner.get("metadata", {}).get(flag):
                    entry[flag] = True
        elif winner["name"].startswith("small"):
            entry = dict(
                implementation="three-prime",
                root_bits=int(winner["extra"]["GENEFER_ROOT_BITS"]),
            )
        else:
            entry = dict(
                implementation={
                    "graphs": "original-cuda",
                    "original": "original-opencl",
                }[winner["name"]]
            )
        projects[str(winner["n"])] = entry
    return dict(
        name=hardware["device_name"],
        architecture=architecture,
        memory_gib=memory_gib,
        projects=projects,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--projects", type=Path, default=HERE / "calibration_bases.json")
    p.add_argument("--probe-source", type=Path, default=HERE / "calibration_probe.cpp")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--probe", type=Path, help="Reuse the portable CUDA probe instead of compiling"
    )
    p.add_argument(
        "--original-probe",
        type=Path,
        help="Reuse original OpenCL arithmetic loop control",
    )
    p.add_argument(
        "--cuda",
        type=Path,
        default=Path("/usr/local/cuda"),
        help="CUDA SDK prefix, headers only",
    )
    p.add_argument("--sizes", default="16,17,18,19,20,21,22")
    p.add_argument(
        "--base", type=int, help="Override benchmark base for every requested size"
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--memory-gib",
        type=int,
        help="nominal GPU capacity; also export a profile.json entry",
    )
    p.add_argument("--squares", type=int, default=8195)
    p.add_argument("--confirm-squares", type=int, default=65539)
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="CPU admission and source generation only",
    )
    a = p.parse_args()
    try:
        sizes = set(map(int, a.sizes.split(",")))
    except ValueError:
        p.error("--sizes must contain comma-separated GFN exponents")
    if not sizes <= set(range(16, 23)):
        p.error("Only GFN sizes 16 through 22 are supported")
    if a.memory_gib is not None and a.memory_gib <= 0:
        p.error("Memory capacity must be positive")
    if min(a.squares, a.confirm_squares) <= 0 or a.device < 0:
        p.error("Invalid counts/device")
    if a.base is not None and (
        not 1026 <= a.base <= 2000000000 or a.base % 2 or not a.base & (a.base - 1)
    ):
        p.error("Base must be even, not a power of two, and between 1026 and 2000000000")
    src = a.source.resolve()
    out = a.output.resolve()
    out.mkdir()
    m = load_module(src / "tools/packed_parameters.py")
    progress = []
    checks = []
    times = []
    winners = []
    admissions = []

    def save(name, value):
        f = out / (name + ".json")
        tmp = f.with_suffix(".partial")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(f)

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GENEFER_", "GFN_CALIB_"))
    }
    env.update(GFN_CALIB_DEVICE_INDEX=str(a.device), GENEFER_PACKED_WRAP_CHECK="1")

    def run(label, args, extra=None):
        print(label, flush=True)
        start = time.monotonic()
        with (out / (label + ".log")).open("w") as f:
            q = subprocess.run(
                list(map(str, args)),
                env=dict(env, **(extra or {})),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        progress.append(
            dict(
                label=label,
                command=list(map(str, args)),
                environment=extra or {},
                exit=q.returncode,
                wall_seconds=time.monotonic() - start,
            )
        )
        save("progress", progress)
        text = (out / (label + ".log")).read_text(errors="replace")
        assert q.returncode == 0, text[-4000:]
        match = re.search(r"RESULT (\{.*\})", text)
        return json.loads(match[1]) if match else {}

    try:
        projects = json.loads(a.projects.read_text())
        save(
            "configuration",
            dict(
                source=str(src),
                device=a.device,
                squares=a.squares,
                confirm_squares=a.confirm_squares,
                projects=projects,
                scope="Loop screening only; original arithmetic control is compiled from this source tree, not an immutable upstream headline benchmark.",
            ),
        )
        if not a.plan_only:
            flags = [
                "g++",
                "-std=c++17",
                "-O3",
                "-DGPU",
                "-I",
                src / "src",
                "-I",
                src / "tests",
                "-I",
                src / "Khronos",
                "-I" + str(a.cuda.resolve() / "include"),
            ]
            original = (
                a.original_probe.resolve() if a.original_probe else out / "original"
            )
            probe = a.probe.resolve() if a.probe else out / "probe"
            if not a.original_probe:
                run(
                    "build-original",
                    [
                        *flags,
                        a.probe_source,
                        src / "src/transform_ocl.cpp",
                        "-lOpenCL",
                        "-lgmp",
                        "-o",
                        original,
                    ],
                )
            if not a.probe:
                run(
                    "build-probe",
                    [
                        *flags,
                        "-DGENEFER_GENERAL_CUDA",
                        "-DGENEFER_GENERAL_PACKED",
                        "-DGENEFER_GENERAL_SMALL3_AVAILABLE",
                        a.probe_source,
                        src / "src/transform_ocl.cpp",
                        src / "src/transform_cuda_small3.cpp",
                        "-lOpenCL",
                        "-lgmp",
                        "-ldl",
                        "-o",
                        probe,
                    ],
                )
        if not a.plan_only:
            save(
                "host-builds",
                {
                    name: dict(
                        path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    for name, path in [("original", original), ("probe", probe)]
                },
            )
            hardware = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit",
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            (out / "hardware.txt").write_text(hardware.stdout + hardware.stderr)
        for project in projects["projects"]:
            n = project["n"]
            b = a.base or project["benchmark_base"]
            if n not in sizes:
                continue
            esize = math.ceil((1 << n) * math.log2(b))
            L = 2 << ((esize.bit_length() - 1) // 2)
            interval = (esize - 1) // L + 1
            candidates = []
            for fields, c, code, meta in plans(m, src, n, b):
                name = f'L{c["length"].bit_length()-1}-O{c["outer"]}-K{fields}-D{c["digit_ratio"]}'
                kernel = out / f"n{n}-{name}.cl"
                kernel.write_text(code)
                sha = hashlib.sha256(code.encode()).hexdigest()
                row = dict(
                    n=n,
                    base=b,
                    name=name,
                    fields=fields,
                    configuration=c,
                    metadata=meta,
                    kernel_sha256=sha,
                    extra=dict(GENEFER_PACKED_KERNEL=str(kernel)),
                )
                candidates.append(row)
                admissions.append(row)
                save("admissions", admissions)
            if a.plan_only:
                continue
            controls = [
                dict(n=n, base=b, name="original", binary=str(original), extra={}),
                dict(n=n, base=b, name="graphs", binary=str(probe), extra={}),
            ]
            if 2046 <= b <= 943718398 and upstream_two_prime_bits(b, n) is None:
                controls.append(
                    dict(
                        n=n,
                        base=b,
                        name="small32",
                        binary=str(probe),
                        extra=dict(
                            GENEFER_SMALL3="1",
                            GENEFER_ROOT_BITS="32",
                            GENEFER_KERNEL=str(src / "kernels/src/three-prime.cl"),
                        ),
                    )
                )
            measured = []
            hashes = {}

            def trial(c, stage, count, repeats):
                binary = c.get("binary", str(probe))
                name = c["name"]
                extra = c["extra"]
                a1 = run(
                    f"{stage}-check-n{n}-{name}",
                    [binary, n, b, 2, "check", "random", interval],
                    extra,
                )
                checks.append(dict(stage=stage, name=name, **a1))
                save("checks", checks)
                vals = []
                for repeat in range(repeats):
                    z = run(
                        f"{stage}-time-n{n}-{name}-{repeat}",
                        [binary, n, b, count, "time", "random", interval],
                        extra,
                    )
                    times.append(dict(stage=stage, name=name, **z))
                    save("timings", times)
                    vals.append(z["us"])
                    if "configuration" in c:
                        conf = c["configuration"]
                        key = (
                            n,
                            conf["length"],
                            c["fields"],
                            conf["digit_ratio"],
                            count,
                        )
                        hashes.setdefault(key, set()).add(z["hash"])
                        assert len(hashes[key]) == 1, (key, hashes[key])
                result = dict(c, us=statistics.median(vals), stage=stage)
                measured.append(result)
                return result

            for c in controls + candidates:
                trial(c, "screen", a.squares, 1)
            packed = [c for c in measured if "configuration" in c]
            if packed:
                best = min(packed, key=lambda c: c["us"])
                for width, chunk, blocks in (
                    (64, best["configuration"]["outer_chunk"], 1),
                    (32, 2, 1),
                    (32, 8, 1),
                    (32, best["configuration"]["outer_chunk"], 2),
                ):
                    config = dict(
                        best["configuration"],
                        root_bits=width,
                        outer_chunk=chunk,
                        inner_blocks=blocks,
                    )
                    if config == best["configuration"]:
                        continue
                    try:
                        code, meta = m.program(
                            b, best["fields"], src / "kernels/src", **config
                        )
                    except m.UnsupportedPackedParameters:
                        continue
                    name = best["name"] + f"-W{width}-C{chunk}-I{blocks}"
                    path = out / f"n{n}-{name}.cl"
                    path.write_text(code)
                    trial(
                        dict(
                            best,
                            name=name,
                            configuration=config,
                            metadata=meta,
                            kernel_sha256=hashlib.sha256(code.encode()).hexdigest(),
                            extra=dict(GENEFER_PACKED_KERNEL=str(path)),
                        ),
                        "tune",
                        a.squares,
                        1,
                    )
            # One source-only integer lowering screen after geometry/root tuning.
            # It retains every existing signed64 bound; the operation is identical.
            packed = [c for c in measured if "configuration" in c]
            if packed:
                best = min(packed, key=lambda c: c["us"])
                code = Path(best["extra"]["GENEFER_PACKED_KERNEL"]).read_text()
                needle_dot = "dot += (long)t[k] * pg_cp.matrix[f][k];"
                assert code.count(needle_dot) == 1
                helper = 'INLINE long pg_mad(long a,int x,int y){long z;asm("mad.wide.s32 %0,%1,%2,%3;":"=l"(z):"r"(x),"r"(y),"l"(a));return z;}\n'
                code = code.replace(
                    "INLINE uint pg_shoup(", helper + "INLINE uint pg_shoup("
                ).replace(needle_dot, "dot = pg_mad(dot,t[k],pg_cp.matrix[f][k]);")
                name = best["name"] + "-mad"
                path = out / f"n{n}-{name}.cl"
                path.write_text(code)
                trial(
                    dict(
                        best,
                        name=name,
                        metadata=dict(best["metadata"], matrix_mad=True),
                        kernel_sha256=hashlib.sha256(code.encode()).hexdigest(),
                        extra=dict(GENEFER_PACKED_KERNEL=str(path)),
                    ),
                    "integer-lowering",
                    a.squares,
                    1,
                )
            shortlisted = sorted(measured, key=lambda c: c["us"])[:2]
            baseline = next(c for c in measured if c["name"] == "original")
            if not any(c["name"] == "original" for c in shortlisted):
                shortlisted.append(baseline)
            confirmed = [trial(c, "confirm", a.confirm_squares, 2) for c in shortlisted]
            best = min(confirmed, key=lambda c: c["us"])
            originaltime = next(c["us"] for c in confirmed if c["name"] == "original")
            if best["us"] > 0.95 * originaltime:
                best = next(c for c in confirmed if c["name"] == "original")
            for operation, pattern in [
                ("normal", "alternating"),
                ("normal", "constant"),
                ("normal", "sparse"),
                ("scalar", "random"),
                ("product", "random"),
                ("checkpoint", "random"),
                ("phase", "random"),
                ("set", "random"),
            ]:
                label = f"deeper-n{n}-{operation}-{pattern}"
                z = run(
                    label,
                    [
                        best.get("binary", str(probe)),
                        n,
                        b,
                        8 if operation in ("normal", "phase") else 4,
                        "check",
                        pattern,
                        interval,
                    ],
                    dict(
                        best["extra"],
                        GFN_CALIB_OPERATION=operation,
                        GFN_CALIB_CONTEXT=str(out / (label + ".context")),
                    ),
                )
                checks.append(
                    dict(name=best["name"], operation=operation, pattern=pattern, **z)
                )
                save("checks", checks)
            winners.append(
                dict(
                    best,
                    original_loop_us=originaltime,
                    reduction_percent=100 * (1 - best["us"] / originaltime),
                    gerbicz_interval=interval,
                    requires_full_proof_comparison=True,
                )
            )
            save("winners", winners)
        if a.memory_gib is not None and not a.plan_only:
            save("profile", export_profile(checks, winners, a.memory_gib))
        save(
            "result",
            dict(
                status="PLAN_ONLY" if a.plan_only else "PASS",
                admissions=len(admissions),
                checks=len(checks),
                winners=winners,
            ),
        )
    except BaseException as e:
        save("result", dict(status="FAILED", error=repr(e)))
        raise


if __name__ == "__main__":
    main()
