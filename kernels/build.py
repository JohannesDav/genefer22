#!/usr/bin/env python3
"""Compile the GFN-19 OpenCL C kernels and export their NVIDIA program binaries."""
import argparse
import ast
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CL_DEVICE_TYPE_GPU = 4
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_NAME = 0x102B
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_PROGRAM_BINARY_SIZES = 0x1165
CL_PROGRAM_BINARIES = 0x1166
CL_PROGRAM_BUILD_LOG = 0x1183

KERNELS = {
    "forward256_ordinary",
    "forward256_0",
    "backward256_0",
    "square2048",
    "fwd2048p",
    "mul2048",
    "normalize1",
    "normalize2",
    "mulscalar",
    "set",
    "copy",
    "copyp",
    "materialize_stairway",
    "boundary_bridge",
    "boundary_bridge_signed",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def configuration(gpu):
    # Share the identifier that the kernel loader checks; no bundled PTX is read.
    text = (ROOT / "src/gfn19/profiles.h").read_text()
    values = re.findall(r'configuration\s*=\s*((?:"[^"]*"\s*)+);', text)
    require(len(values) == 2, "expected two GPU configuration identifiers")
    value = values[0 if gpu == "3090" else 1]
    return "".join(ast.literal_eval(part) for part in re.findall(r'"[^"]*"', value))


def validate_configuration(source, identifier):
    expected = dict(field.split("=", 1) for field in identifier.split(";") if field)
    literals = dict(re.findall(r"(?m)^#define\s+(\w+)\s+(\d+[uUlL]*)\s*$", source))
    for name in (
        "N_SZ",
        "LN_SZ",
        "VSIZE",
        "LVSIZE",
        "CHUNK256",
        "NORM_WG_SZ",
        "P1",
        "Q1",
        "P2",
        "Q2",
        "NORM1",
        "NORM2",
    ):
        require(name in literals, "missing fixed kernel parameter: " + name)
        require(
            int(literals[name].rstrip("uUlL")) == int(expected[name].rstrip("uUlL")),
            "kernel parameter differs from runtime configuration: " + name,
        )


def entry_regions(text):
    entries = {}
    for match in re.finditer(r"(?m)^(?:\.visible )?\.entry (\w+)\(", text):
        name = match[1]
        require(name not in entries, "duplicate PTX entry: " + name)
        opening = text.index("{", match.end())
        end, depth = opening + 1, 1
        while depth and end < len(text):
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        require(depth == 0, "unterminated PTX entry: " + name)
        entries[name] = (match.start(), opening, end)
    return entries


def tune_3090(text):
    """Keep repeatedly used data in L2 and tune bridge local-memory allocation."""
    for name, (_, opening, end) in reversed(list(entry_regions(text).items())):
        if name != "square2048" and not name.startswith("boundary_bridge_signed"):
            continue
        body = text[opening + 1 : end - 1]
        body = "\n\t.reg .b64 %gfnCachePolicy;" + body
        require("\n\tld.param" in body, "missing kernel parameter prologue")
        body = body.replace(
            "\n\tld.param",
            "\n\tcreatepolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 "
            "%gfnCachePolicy, 0f3f800000;\n\tld.param",
            1,
        )
        # Leave non-coherent root-table loads alone. The normal data loads and
        # stores retain their arithmetic and addresses, with an eviction hint.
        pattern = r"\b(ld\.global|st\.global)((?:\.v[24])?\.[busf]\d+)(\s+[^;]+);"
        body, count = re.subn(
            pattern,
            lambda match: match[1] + ".L2::cache_hint" + match[2] + match[3]
            + ", %gfnCachePolicy;",
            body,
        )
        require(count > 0, "missing kernel data memory operations")
        if name.startswith("boundary_bridge_signed"):
            shared = r"(\.shared \.align 16 \.b8 \S+_\$_Z\[)(\d+)(\];)"
            body, count = re.subn(
                shared,
                lambda match: match[1] + str(int(match[2]) + 8192) + match[3],
                body,
            )
            require(count == 1, "unexpected bridge local-memory declaration")
        text = text[: opening + 1] + body + text[end - 1 :]
    return text


def assemble(raw, gpu):
    """Set measured GPU tuning and specialize the bridge's uniform doubling flag."""
    entries = entry_regions(raw)
    require(set(entries) == KERNELS, "unexpected compiler kernel set")
    if gpu == "4060":
        # Measured occupancy settings for the 4060 Ti; the 3090 is uncapped.
        for name, cap in [("square2048", 56), ("boundary_bridge_signed", 64)]:
            start, opening, end = entry_regions(raw)[name]
            require(
                ".maxnreg" not in raw[start:opening], "compiler supplied a register cap"
            )
            raw = raw[:opening] + ".maxnreg " + str(cap) + "\n" + raw[opening:]
    start, _, end = entry_regions(raw)["boundary_bridge_signed"]
    original = raw[start:end]
    pattern = r"ld\.param\.u32\s+(%r\d+),\s*\[boundary_bridge_signed_param_7\];"
    require(
        len(re.findall(pattern, original)) == 1,
        "expected one bridge doubling-flag parameter load",
    )
    copies = []
    for bit in (0, 1):
        specialized = re.sub(
            pattern, lambda m: "mov.u32 " + m[1] + ", " + str(bit) + ";", original
        )
        specialized = specialized.replace(
            "boundary_bridge_signed", "boundary_bridge_signed_dup" + str(bit)
        )
        copies.append(specialized)
    result = raw[:end] + "\n\n" + "\n\n".join(copies) + raw[end:]
    require(
        len(entry_regions(result)) == 17, "assembled module must contain 17 kernels"
    )
    return tune_3090(result) if gpu == "3090" else result


class OpenCL:
    """The small subset of the OpenCL API needed for source compilation."""

    def __init__(self):
        self.lib = ct.CDLL("libOpenCL.so.1")
        pointer, uint, size = ct.c_void_p, ct.c_uint, ct.c_size_t
        signatures = {
            "clGetPlatformIDs": [uint, ct.POINTER(pointer), ct.POINTER(uint)],
            "clGetDeviceIDs": [
                pointer,
                ct.c_ulong,
                uint,
                ct.POINTER(pointer),
                ct.POINTER(uint),
            ],
            "clGetDeviceInfo": [pointer, uint, size, pointer, ct.POINTER(size)],
            "clCreateContext": [
                pointer,
                uint,
                ct.POINTER(pointer),
                pointer,
                pointer,
                ct.POINTER(ct.c_int),
            ],
            "clCreateProgramWithSource": [
                pointer,
                uint,
                ct.POINTER(ct.c_char_p),
                ct.POINTER(size),
                ct.POINTER(ct.c_int),
            ],
            "clBuildProgram": [
                pointer,
                uint,
                ct.POINTER(pointer),
                ct.c_char_p,
                pointer,
                pointer,
            ],
            "clGetProgramBuildInfo": [
                pointer,
                pointer,
                uint,
                size,
                pointer,
                ct.POINTER(size),
            ],
            "clGetProgramInfo": [pointer, uint, size, pointer, ct.POINTER(size)],
            "clReleaseProgram": [pointer],
            "clReleaseContext": [pointer],
        }
        for name, arguments in signatures.items():
            getattr(self.lib, name).argtypes = arguments
            getattr(self.lib, name).restype = ct.c_int
        self.lib.clCreateContext.restype = pointer
        self.lib.clCreateProgramWithSource.restype = pointer

    @staticmethod
    def check(code, operation):
        require(code == 0, operation + " failed (OpenCL status " + str(code) + ")")

    def device_text(self, device, key):
        size = ct.c_size_t()
        self.check(
            self.lib.clGetDeviceInfo(device, key, 0, None, ct.byref(size)),
            "device info size",
        )
        value = ct.create_string_buffer(size.value)
        self.check(
            self.lib.clGetDeviceInfo(device, key, size, value, None), "device info"
        )
        return value.value.decode()

    def select_device(self, gpu):
        count = ct.c_uint()
        self.check(
            self.lib.clGetPlatformIDs(0, None, ct.byref(count)), "platform enumeration"
        )
        platforms = (ct.c_void_p * count.value)()
        self.check(
            self.lib.clGetPlatformIDs(count, platforms, None), "platform enumeration"
        )
        expected = (
            "NVIDIA GeForce RTX 3090" if gpu == "3090" else "NVIDIA GeForce RTX 4060 Ti"
        )
        matches = []
        for platform in platforms:
            count = ct.c_uint()
            status = self.lib.clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_GPU, 0, None, ct.byref(count)
            )
            if status == -1:  # CL_DEVICE_NOT_FOUND: this platform has no GPU.
                continue
            self.check(status, "GPU enumeration")
            devices = (ct.c_void_p * count.value)()
            self.check(
                self.lib.clGetDeviceIDs(
                    platform, CL_DEVICE_TYPE_GPU, count, devices, None
                ),
                "GPU enumeration",
            )
            for handle in devices:
                device = ct.c_void_p(handle)
                if self.device_text(device, CL_DEVICE_NAME) == expected:
                    matches.append(device)
        require(len(matches) == 1, "expected exactly one available " + expected)
        device = matches[0]
        if gpu == "4060":
            memory = ct.c_ulonglong()
            self.check(
                self.lib.clGetDeviceInfo(
                    device,
                    CL_DEVICE_GLOBAL_MEM_SIZE,
                    ct.sizeof(memory),
                    ct.byref(memory),
                    None,
                ),
                "GPU memory size",
            )
            require(
                memory.value >= 15000 * 1024 * 1024,
                "the 4060 Ti profile requires the 16GB card",
            )
        return device

    def compile(self, device, source):
        error = ct.c_int()
        context = self.lib.clCreateContext(
            None, 1, ct.byref(device), None, None, ct.byref(error)
        )
        self.check(error.value, "context creation")
        program = None
        try:
            text = source.encode()
            pointer, length = ct.c_char_p(text), ct.c_size_t(len(text))
            program = self.lib.clCreateProgramWithSource(
                context, 1, ct.byref(pointer), ct.byref(length), ct.byref(error)
            )
            self.check(error.value, "program creation")
            # Use the same default compiler options as the validated exporter.
            status = self.lib.clBuildProgram(
                program, 1, ct.byref(device), b"", None, None
            )
            if status:
                size = ct.c_size_t()
                self.check(
                    self.lib.clGetProgramBuildInfo(
                        program, device, CL_PROGRAM_BUILD_LOG, 0, None, ct.byref(size)
                    ),
                    "compiler log size",
                )
                log = ct.create_string_buffer(size.value)
                self.check(
                    self.lib.clGetProgramBuildInfo(
                        program, device, CL_PROGRAM_BUILD_LOG, size, log, None
                    ),
                    "compiler log",
                )
                raise RuntimeError(
                    "OpenCL compilation failed:\n" + log.value.decode(errors="replace")
                )
            size = ct.c_size_t()
            self.check(
                self.lib.clGetProgramInfo(
                    program,
                    CL_PROGRAM_BINARY_SIZES,
                    ct.sizeof(size),
                    ct.byref(size),
                    None,
                ),
                "program binary size",
            )
            binary = ct.create_string_buffer(size.value)
            pointer = ct.cast(binary, ct.c_void_p)
            self.check(
                self.lib.clGetProgramInfo(
                    program,
                    CL_PROGRAM_BINARIES,
                    ct.sizeof(pointer),
                    ct.byref(pointer),
                    None,
                ),
                "program binary export",
            )
            raw = binary.raw.rstrip(b"\0").decode()
            require(
                ".version " in raw and ".entry " in raw,
                "NVIDIA compiler did not export PTX",
            )
            return raw
        finally:
            if program:
                self.lib.clReleaseProgram(program)
            self.lib.clReleaseContext(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("3090", "4060"), required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new PTX output file"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="require a byte-for-byte match with the bundled PTX",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "output already exists; choose a new file")
    require(
        not output.with_suffix(output.suffix + ".json").exists(),
        "build record already exists",
    )
    source_path = ROOT / "kernels/src" / ("gfn19-" + args.gpu + ".cl")
    source = source_path.read_text()
    source_files = [source_path]
    shared = ROOT / "kernels/src/common.clh"
    if '#include "common.clh"' in source:
        require(
            source.count('#include "common.clh"') == 1,
            "expected one common header include",
        )
        source = source.replace('#include "common.clh"', shared.read_text())
        source_files.append(shared)
    identifier = configuration(args.gpu)
    validate_configuration(source, identifier)
    compiler = OpenCL()
    device = compiler.select_device(args.gpu)
    name = compiler.device_text(device, CL_DEVICE_NAME)
    driver = compiler.device_text(device, CL_DRIVER_VERSION)
    print("Compiling for " + name + " with NVIDIA driver " + driver, flush=True)
    raw = compiler.compile(device, source)
    target = "sm_86" if args.gpu == "3090" else "sm_89"
    require(
        re.search(r"(?m)^\.target " + target + r"[,\s]", raw),
        "compiler target does not match GPU profile",
    )
    result = (
        "// GENEFER_CONFIG " + identifier + "\n" + assemble(raw, args.gpu)
    ).encode()
    matched = None
    if args.verify:
        reference = (
            ROOT
            / "kernels"
            / ("gfn19-" + args.gpu + ".ptx")
        )
        matched = result == reference.read_bytes()
        require(
            matched,
            "generated PTX differs from the bundled reference; verify the documented driver version",
        )
    source_files.extend([Path(__file__).resolve(), ROOT / "src/gfn19/profiles.h"])
    record = dict(
        gpu=args.gpu,
        device=name,
        driver=driver,
        compiler_options="",
        opencl_version=compiler.device_text(device, CL_DEVICE_VERSION),
        compiler_build_id=re.search(r"Compiler Build ID: ([^\n]+)", raw)[1],
        kernels=len(entry_regions(result.decode())),
        ptx_sha256=sha(result),
        source_files={
            str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in source_files
        },
        matches_bundled_ptx=matched,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print("Wrote " + str(output) + " (SHA-256 " + sha(result) + ")")
    if matched:
        print("PASS: byte-for-byte match with the bundled kernels")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print("Kernel build failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
