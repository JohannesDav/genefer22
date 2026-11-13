#!/usr/bin/env bash
set -euo pipefail

args=(--boinc /usr)
if [[ "$GENEFER_VARIANT" == cuda ]]; then
    archive=cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz
    curl -fL --retry 3 "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/$archive" -o "$RUNNER_TEMP/$archive"
    echo "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68  $RUNNER_TEMP/$archive" | sha256sum -c -
    tar -xJf "$RUNNER_TEMP/$archive" -C "$RUNNER_TEMP"
    args+=(--cuda "$RUNNER_TEMP/cuda_cudart-linux-x86_64-12.8.90-archive")
fi

# Try Focal's application libraries first. Fall back to BOINC 8.2.9 if
# their headers, static libraries or link dependencies are incompatible.
if ! python3 build.py "${args[@]}"; then
    apt-get update
    apt-get install -y --no-install-recommends autoconf automake libtool pkg-config libcurl4-openssl-dev
    boinc="$RUNNER_TEMP/boinc"
    git clone --depth 1 --branch client_release/8.2/8.2.9 https://github.com/BOINC/boinc.git "$boinc"
    test "$(git -C "$boinc" rev-parse HEAD)" = "$BOINC_COMMIT"
    (
        cd "$boinc"
        ./_autosetup
        ./configure --disable-server --disable-client --disable-manager --disable-apps \
            --enable-libraries --disable-graphics --disable-shared --enable-static --without-x
        make -j2 -C lib libboinc.la
        make -j2 -C api libboinc_api.la libboinc_opencl.la
        # build.py accepts a source prefix, with static archives in lib/ and api/.
        cp lib/.libs/libboinc.a lib/
        cp api/.libs/libboinc_api.a api/.libs/libboinc_opencl.a api/
    )
    python3 build.py "${args[@]:2}" --boinc "$boinc"
fi
g++ --version
dpkg-query -W libc6 libgmp-dev libboinc-app-dev ocl-icd-opencl-dev
