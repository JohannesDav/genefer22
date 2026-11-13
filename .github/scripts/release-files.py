#!/usr/bin/env python3
"""Validate dependencies and assemble the two-file installation archives."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import zipfile


def inspect_binary(binary, platform):
    report = subprocess.check_output(["objdump", "-p", str(binary)], text=True)
    print(report)
    if platform == "windows":
        if "pei-x86-64" not in report:
            raise ValueError("Expected a Windows x64 executable")
        dependencies = re.findall(r"DLL Name:\s*(\S+)", report)
        allowed = {
            "kernel32.dll", "advapi32.dll", "user32.dll", "gdi32.dll",
            "shell32.dll", "ole32.dll", "oleaut32.dll", "ws2_32.dll",
            "psapi.dll", "msvcrt.dll", "ucrtbase.dll", "ntdll.dll",
            "secur32.dll", "bcrypt.dll", "crypt32.dll", "version.dll",
            "winmm.dll", "shlwapi.dll", "comdlg32.dll", "comctl32.dll",
            "iphlpapi.dll", "userenv.dll", "setupapi.dll", "cfgmgr32.dll",
            "opencl.dll", "nvcuda.dll",
        }
        bad = [name for name in dependencies if name.lower() not in allowed
               and not re.fullmatch(r"api-ms-win-[a-z0-9-]+\.dll", name.lower())]
    else:
        if "elf64-x86-64" not in report:
            raise ValueError("Expected a Linux x64 executable")
        dependencies = re.findall(r"^\s*NEEDED\s+(\S+)", report, re.MULTILINE)
        allowed = {
            "libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2",
            "librt.so.1", "libresolv.so.2", "libutil.so.1",
            "ld-linux-x86-64.so.2", "libOpenCL.so.1",
        }
        bad = [name for name in dependencies if name not in allowed]
        # Also reject GLIBC_PRIVATE and unknown/non-numeric GLIBC requirements.
        versions = set(re.findall(r"\bGLIBC_[A-Za-z0-9_.]+", report))
        if not versions:
            raise ValueError("No GLIBC version requirements found")
        for version in versions:
            suffix = version[len("GLIBC_"):]
            if not re.fullmatch(r"\d+(?:\.\d+)+", suffix) or tuple(map(int, suffix.split("."))) > (2, 31):
                raise ValueError("Unsupported glibc requirement: " + version)
    if not dependencies or bad:
        raise ValueError("Unexpected runtime dependencies: " + repr(bad or dependencies))
    print("PASS: allowed runtime dependencies", dependencies)


def label():
    ref = os.environ["GITHUB_REF"]
    if ref.startswith("refs/tags/"):
        value = ref[len("refs/tags/"):]
        if not re.fullmatch(r"v[A-Za-z0-9][A-Za-z0-9._+-]*", value):
            raise ValueError("Version tags must start with v and use letters, digits, ., _, + or -")
        return value
    return "manual-" + os.environ["GITHUB_SHA"][:12]


def package(platform):
    executable = "geneferg.exe" if platform == "windows" else "geneferg"
    binary = Path("build") / executable
    inspect_binary(binary, platform)
    variant = "-opencl" if os.environ["GENEFER_VARIANT"] == "opencl" else ""
    path = Path("dist") / ("genefer22-johannesdav" + variant + "-" + label() + "-" + platform + "-x64.zip")
    path.parent.mkdir(exist_ok=True)
    template = Path("doc") / ("boinc-app_info-windows.xml" if platform == "windows" else "boinc-app_info.xml")
    # Explicit Unix permissions survive the Actions artifact and ZIP transports.
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name, mode in [(binary, executable, 0o755), (template, "app_info.xml", 0o644)]:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    print("Packaged", path)


def release_notes():
    tag = label()
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["GITHUB_SHA"]
    guide = "opencl" if os.environ["GENEFER_VARIANT"] == "opencl" else "cuda"
    root = "https://github.com/" + repo
    Path("release-notes.md").write_text(
        "Experimental build of this genefer22 fork for NVIDIA GPUs. All changes were written by AI "
        "and haven’t been independently reviewed. Install at your own risk.\n\n"
        "See the [installation guide](" + root + "/blob/main/doc/" + guide +
        ".md#installing-a-prebuilt-release) for how to set it up with BOINC.\n\n"
        "Built from [" + tag + "](" + root + "/tree/" + tag + ") "
        "([`" + sha[:7] + "`](" + root + "/commit/" + sha + ")) by "
        "[GitHub Actions](" + root + "/actions/runs/" + os.environ["GITHUB_RUN_ID"] + ").\n",
        encoding="utf-8",
    )


def checksums():
    paths = sorted(Path("dist").glob("*.zip"))
    variant = "-opencl" if os.environ["GENEFER_VARIANT"] == "opencl" else ""
    expected = {"genefer22-johannesdav" + variant + "-" + label() + "-" + platform + "-x64.zip"
                for platform in ("windows", "linux")}
    if {path.name for path in paths} != expected:
        raise ValueError("Expected exactly the two platform archives")
    Path("dist/SHA256SUMS").write_text("".join(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n" for path in paths
    ), encoding="utf-8")
    release_notes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["windows", "linux", "checksums"])
    command = parser.parse_args().command
    if command == "checksums":
        checksums()
    else:
        package(command)
