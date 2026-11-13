#!/usr/bin/env bash
set -euo pipefail

# GitHub supplies a native Windows path; MSYS tools need its POSIX spelling.
RUNNER_TEMP=$(cygpath -u "$RUNNER_TEMP")
boinc="$RUNNER_TEMP/boinc"
git clone --depth 1 --branch client_release/8.2/8.2.9 https://github.com/BOINC/boinc.git "$boinc"
test "$(git -C "$boinc" rev-parse HEAD)" = "$BOINC_COMMIT"
make -C "$boinc/lib" -f Makefile.mingw -j2 libboinc_api.a libboinc.a libboinc_opencl.a
args=(--boinc "$boinc")
if [[ "$GENEFER_VARIANT" == cuda ]]; then
    archive=cuda_cudart-windows-x86_64-12.8.90-archive.zip
    curl -fL --retry 3 "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/$archive" -o "$RUNNER_TEMP/$archive"
    echo "4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b  $RUNNER_TEMP/$archive" | sha256sum -c -
    unzip -q "$RUNNER_TEMP/$archive" -d "$RUNNER_TEMP"
    args+=(--cuda "$RUNNER_TEMP/cuda_cudart-windows-x86_64-12.8.90-archive")
fi
python3 build.py "${args[@]}"
g++ --version
pacman -Q
