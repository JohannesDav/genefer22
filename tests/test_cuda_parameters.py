"""Compare C++ admission/constants with the independent exact Python generator.

Usage: python3 tests/test_cuda_parameters.py /path/to/cuda_parameters_probe
The probe needs only a C++ compiler and the existing GMP dependency, not a GPU.
"""

from pathlib import Path
from fractions import Fraction
import importlib.util
import json
import random
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reference", root / "tools/packed_parameters.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
rng = random.Random(0x47464E)
bases = [
    1026,
    2046,
    4194302,
    4194304,
    8000000,
    28000000,
    30000000,
    100000000,
    520000000,
    620000000,
    943718398,
    1000000000,
    1999999998,
]
cases = []
for n in range(16, 23):
    for ln in range(max(14, n - 4), n + 1):
        for base in bases:
            fields = rng.choice([2, 3, 5, 8, 14, 22, 32])
            ratio = rng.choice([2, 3, 4])
            cases.append((base, 1 << ln, 1 << (n - ln), ratio, fields))
# Several exact accepted and rejected neighbors around useful field counts.
for base in [28000000, 30000000]:
    for fields in range(10, 18):
        for ratio in [2, 3, 4]:
            cases.append((base, 65536, 8, ratio, fields))
lines, expected = [], []
for base, length, pack, ratio, fields in cases:
    primes = reference.field_pool(fields, length)[0]
    lines.append(" ".join(map(str, [base, length, pack, ratio, fields, *primes])))
    try:
        reference.require(
            1026 <= base <= 2000000000 and base % 2 == 0 and base & (base - 1) != 0,
            "packed input range",
        )
        reference.require(base + 2 < min(primes), "encoder range")
        a = reference.parameters(
            base,
            fields,
            pack=pack,
            length=length,
            primes=primes,
            digit_ratio=ratio,
            accurate_error=True,
        )
        reference.require(
            2 * a["e"] * a["B"] + a["e"] + a["Tmax"] < 3 * a["B"], "bridge"
        )
        expected.append(a)
    except reference.UnsupportedPackedParameters:
        expected.append(None)
process = subprocess.run(
    [sys.argv[1]],
    input="\n".join(lines) + "\n",
    capture_output=True,
    text=True,
    check=True,
)
actual = [json.loads(line) for line in process.stdout.splitlines()]
assert len(actual) == len(expected)
accepted = 0
for case, a, b in zip(cases, actual, expected):
    assert a["accepted"] == (b is not None), (case, a, b is not None)
    if b is None:
        continue
    accepted += 1
    for name in ["B", "P", "EP", "Tmax", "groups", "H", "iv", "E"]:
        assert a[name] == b[name], (case, name)
    assert Fraction(a["e"]) == b["e"], case
    assert a["quotient"] == [
        list(reference.split_fraction(x, b["B"], 32)) for x in b["E"]
    ]
    assert a["tail"] == [
        list(reference.split_fraction(x, b["B"] ** 2, 32)) for x in b["H"]
    ]
    assert a["extra_quotient"] == list(reference.split_fraction(b["EP"], b["B"], 32))
    assert a["extra_tail"] == list(reference.split_fraction(b["P"], b["B"] ** 2, 32))
    assert a["bm"] == b["params"][10] and a["epm"] == b["params"][11]
    assert a["matrix"] == b["params"][12], case
print(
    json.dumps(
        dict(
            status="PASS",
            cases=len(cases),
            accepted=accepted,
            rejected=len(cases) - accepted,
            seed=0x47464E,
        )
    )
)
