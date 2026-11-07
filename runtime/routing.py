"""Exact GFN-19 routes. Two-prime inequalities match upstream create_ocl."""
N = 1 << 19
SMALL = (2130706433, 2113929217)
WIDE = (4194304001, 4076863489)
PACKED_MIN = 4194304
PACKED_MAX = 528220156
MAX_BASE = 2000000000


def two_prime_width(base):
    for width, primes in [(31, SMALL), (32, WIDE)]:
        if base * base < (primes[0] * primes[1] // 2) // N:
            return width
    return None


def arithmetic(base):
    if two_prime_width(base) is not None:
        return "two"
    return "three" if base <= PACKED_MAX else "upstream"


def manifest_name(kind):
    return {
        "two": "boinc-two-prime-manifest.json",
        "three": "boinc-guard-manifest.json",
        "upstream": "boinc-upstream-manifest.json",
    }[kind]
