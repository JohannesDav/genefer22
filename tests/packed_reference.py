#!/usr/bin/env python3
"""CPU reference checks for packed CRT reconstruction and carry invariants.

Uses exact Python integers and a small negacyclic polynomial as an independent
oracle. GPU arithmetic and complete proofs require the separate GPU tests.
"""

from pathlib import Path
import importlib.util
import random
from math import isqrt
from fractions import Fraction as F

r = Path(__file__).resolve().parents[1]
p = r / "runtime/packed_kernel.py"
sp = importlib.util.spec_from_file_location("runtime", p)
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
if not __debug__:
    raise RuntimeError("Run this reference test without Python -O")
rng = random.Random(40893)
checks = 0
squares = 0
for p, g in zip(m.ps, m.generators):
    assert p % 2 and all(p % q for q in range(3, isqrt(p) + 1, 2))
    assert (p - 1) % (2 * m.N) == 0 and pow(g, (p - 1) // 2, p) == p - 1
for bits in [30, 32]:
    for denominator in [2**100, 10**80, 1026**8]:
        for value in [
            denominator // 2 - 1,
            denominator // 2,
            denominator // 2 + 1,
            denominator * 2**100 + denominator // 2 - 1,
        ]:
            q, w = m.split_fraction(value, denominator, bits)
            assert -(2 ** (bits - 1)) <= w < 2 ** (bits - 1)
            assert abs(F(q) + F(w, 2**bits) - F(value, denominator)) <= F(
                1, 2 ** (bits + 1)
            )
            checks += 1
# These representations must be admitted; random cases below may legitimately
# choose a different field count when a bound rejects their parameters.
for base, fields in [
    (28000000, 14),
    (31296290, 14),
    (38000000, 15),
    (114484426, 15),
    (400000000, 16),
    (418735722, 16),
    (528220156, 17),
]:
    m.parameters(base, fields)
eligible = []
for K, maximum, bits in [
    (14, 31296290, 32),
    (15, 114484426, 32),
    (16, 418735722, 32),
    (17, 528220156, 32),
    (17, 528220156, 30),
]:
    bases = [1026, 13427472, 28000000, 30000000, maximum - 2, maximum] + [
        rng.randrange(1026 // 2, maximum // 2) * 2 for _ in range(20)
    ]
    for b in bases:
        if b > maximum:
            continue
        try:
            a = m.parameters(b, K, bits)
        except m.UnsupportedPackedParameters as exc:
            eligible.append(
                dict(K=K, base=b, bits=bits, eligible=False, reason=str(exc))
            )
            continue
        eligible.append(dict(K=K, base=b, bits=bits, eligible=True))
        B = a["B"]
        P = a["P"]
        E = a["E"]
        EP = a["EP"]
        H = a["H"]
        FQ = 2**bits
        ps = m.ps[:K]
        qi, qw, qpi, qpw, ti, tw, tpi, tpw = a["quant"]

        def parts(c):
            t = [m.ctr(c * i % p, p) for i, p in zip(a["iv"], ps)]
            rank = m.near(sum(t), 2**30)
            assert sum(x * h for x, h in zip(t, H)) - rank * P == c
            q = (
                sum(x * w for x, w in zip(t, qi))
                - rank * qpi
                + m.near(sum(x * w for x, w in zip(t, qw)) - rank * qpw, FQ)
            )
            tmod = (
                sum(x * w for x, w in zip(t, ti))
                - rank * tpi
                + m.near(sum(x * w for x, w in zip(t, tw)) - rank * tpw, FQ)
            ) % 2**64
            T = tmod if tmod < 2**63 else tmod - 2**64
            rem = sum(x * e for x, e in zip(t, E)) - rank * EP - q * B
            u = (c - rem) // B - B * T
            assert (
                c == rem + B * u + B * B * T
                and abs(T) <= a["Tmax"]
                and abs(rem) <= a["e"] * B
                and abs(u) <= a["e"] * (B + 1)
            )
            return rem, u, T

        C = 2 * m.N * a["limit"] ** 2
        for c in [0, 1, -1, C, -C, C - 1, 1 - C] + [
            rng.randrange(-C, C + 1) for _ in range(30)
        ]:
            parts(c)
            checks += 1
        length = 8
        limit = a["limit"]
        modulus = B**length + 1
        for pattern in range(3):
            digits = [rng.randrange(-limit, limit + 1) for _ in range(length)]
            if pattern < 2:
                digits = [limit if pattern == 0 else -limit] * length
            for step in range(8):
                dup = step % 2
                coeff = [0] * length
                for i, x in enumerate(digits):
                    for j, y in enumerate(digits):
                        coeff[(i + j) % length] += (
                            (1 if i + j < length else -1) * x * y * (dup + 1)
                        )
                part = [parts(c) for c in coeff]
                out = [
                    part[j][0]
                    + (part[j - 1][1] if j else -part[-1][1])
                    + (part[j - 2][2] if j >= 2 else -part[j - 2 + length][2])
                    for j in range(length)
                ]
                assert (
                    max(map(abs, out)) < limit
                    and sum(x * B**j for j, x in enumerate(out)) % modulus
                    == (dup + 1)
                    * pow(sum(x * B**j for j, x in enumerate(digits)), 2, modulus)
                    % modulus
                )
                digits = out
                squares += 1
for b, K in [(31296292, 14), (114484428, 15), (418735724, 16)]:
    try:
        m.parameters(b, K)
    except m.UnsupportedPackedParameters:
        pass
    else:
        raise AssertionError("Rank upper bound was not enforced")
assert checks > 100 and squares > 100
print(
    f"PASS: 17 prime constants, {checks} CRT/fraction checks, {squares} dependent polynomial squares"
)
