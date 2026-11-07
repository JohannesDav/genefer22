#!/usr/bin/env python3
"""Uniform carry bounds and exact-integer CRT division reference checks."""

from fractions import Fraction as F
import random

if not __debug__:
    raise RuntimeError("Run this reference test without Python -O")
P1, P2, P3 = 1056440321, 1053818881, 1051721729
PAIR = P2 * P3
M = P1 * PAIR
N = 1 << 19
LO, HI = 1026, 528220156
I13 = pow(P3, -1, P1)
I23 = pow(P3, -1, P2)
I12 = pow(P2, -1, P1)
TMAX = P2 // 2 + 1
RMAX = P3 - 1
A = lambda b: 2 * N * (b + 2) ** 2
L = lambda b: (A(b) + b - 2) // (b - 1)
K = lambda b: (A(b) + PAIR - 1) // PAIR + 2
# Uniform inequalities over every integer base in the interval, using endpoint
# bounds only where monotonicity or a positive rational majorant justifies it.
carry_ratio = 2 * N * F(LO + 2, LO) ** 2 * F(LO, LO - 1) + F(1, LO)
residual_ratio = K(HI) + TMAX + F(2 * RMAX, LO) + carry_ratio
assert residual_ratio < 2**29
qbase_bound = (
    2
    * (
        2 * N * HI * F(LO + 2, LO) ** 2
        + N * F((HI + 2) ** 2, PAIR)
        + 3 * F(PAIR, LO)
        + 2
        + TMAX * (F(P3, LO) + F(1, 2))
    )
    + 2**29
)
assert qbase_bound < 2**63
assert A(HI) < (P1 // 2 - 4) * PAIR


def parts(value):
    r1, r2, r3 = value % P1, value % P2, value % P3
    u13 = ((r1 - r3) * I13) % P1
    u23 = ((r2 - r3) * I23) % P2
    u123 = ((u13 - u23) * I12) % P1
    neg3 = 0
    rc = r3
    adjusted = u23 + neg3
    neg2 = int(adjusted >= P2 // 2)
    t = adjusted - neg2 * P2
    k = u123 + neg2
    k -= P1 if k >= P1 // 2 else 0
    assert k * PAIR + t * P3 + rc == value
    return k, t, rc


rng = random.Random(992319)
count = 0
for b in [
    LO,
    2046,
    4094,
    4194304,
    13427472,
    28000000,
    30000000,
    38000000,
    100000000,
    HI,
]:
    qpair = (PAIR + b // 2) // b
    rpair = PAIR - qpair * b
    q3 = (P3 + b // 2) // b
    r3 = P3 - q3 * b
    assert 2 * (K(b) * abs(rpair) + TMAX * abs(r3) + RMAX) + L(b) < b * 2**29
    assert 2 * (K(b) * qpair + TMAX * q3) + 2**29 < 2**63
    for j in range(12000):
        coefficient = rng.randrange(-A(b) // 2, A(b) // 2 + 1)
        if j < 8:
            coefficient = [0, 1, -1, A(b) // 2, -A(b) // 2, PAIR, -PAIR, P3][j]
        coefficient = max(-A(b) // 2, min(A(b) // 2, coefficient))
        k, t, c = parts(coefficient)
        carry = rng.randrange(-L(b), L(b) + 1)
        scale = 1 + j % 2
        residual = scale * (k * rpair + t * r3 + c) + carry
        assert abs(residual) < b * 2**29 and abs(residual) < 2**63
        q = abs(residual) // b
        q = -q if residual < 0 else q
        digit = residual - q * b
        quotient = scale * (k * qpair + t * q3) + q
        assert quotient * b + digit == scale * coefficient + carry
        assert abs(digit) < b and abs(quotient) <= L(b)
        count += 1
record = dict(
    status="PASS",
    cases=count,
    base_interval=[LO, HI],
    uniform_residual_over_base_bound=str(residual_ratio),
    barrett_contract=2**29,
    uniform_quotient_word_bound=str(qbase_bound),
    scope="Exact centered Garner decomposition and one signed64 Barrett division per coefficient; no float, no probabilistic coefficient model",
)
print(record)
