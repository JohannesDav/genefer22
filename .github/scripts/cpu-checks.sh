#!/usr/bin/env bash
set -euo pipefail

backend=cuda
if [[ "$GENEFER_VARIANT" == opencl ]]; then backend=gpu; fi
mkdir -p build/tests
python3 "tools/embed_${backend}_kernels.py" "build/tests/generated/${backend}/embedded_kernels.h"
g++ -std=c++17 -O2 -Isrc "tests/${backend}_parameters_probe.cpp" -lgmp -o build/tests/parameters
python3 "tests/test_${backend}_parameters.py" build/tests/parameters
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated "tests/${backend}_profiles.cpp" \
    "src/${backend}/profiles.cpp" -lgmp -o build/tests/profiles
./build/tests/profiles
g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated "tests/${backend}_handoff.cpp" \
    "src/${backend}/kernel.cpp" -lgmp -o build/tests/handoff
./build/tests/handoff
