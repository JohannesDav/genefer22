"""Generate exact packed CRT parameters; no distribution assumptions."""

from fractions import Fraction
from math import prod
from pathlib import Path

N = 65536
R = 1 << 32
ps = [
    1073479681,
    1071513601,
    1070727169,
    1068236801,
    1065484289,
    1064697857,
    1062862849,
    1062469633,
    1060765697,
    1056440321,
    1056178177,
    1055260673,
    1054212097,
    1053818881,
    1052508161,
    1051721729,
    1049100289,
]
generators = [11, 23, 11, 3, 3, 3, 7, 5, 3, 3, 5, 3, 5, 7, 3, 6, 7]


class UnsupportedPackedParameters(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise UnsupportedPackedParameters(message)


def near(x, b):
    """Round a rational to the nearest integer, with ties toward +infinity."""
    return (2 * x + b) // (2 * b)


def split_fraction(value, denominator, bits):
    whole = near(value, denominator)
    fraction = near((value - whole * denominator) * (1 << bits), denominator)
    if fraction == (1 << (bits - 1)):
        whole += 1
        fraction -= 1 << bits
    require(
        -(1 << (bits - 1)) <= fraction < (1 << (bits - 1)), "fractional weight range"
    )
    return whole, fraction


def ctr(x, p):
    return (x + p // 2) % p - p // 2


def parameters(base, K, bits=32):
    """Construct constants only when every reconstruction and carry bound holds.

    B = base**8 is the packed radix. Coefficients have magnitude at most
    2*N*limit**2, including a doubled square. The returned constants recover
    each coefficient as rem + B*u + B**2*T using integer arithmetic.
    """
    p = ps[:K]
    P = prod(p)
    H = [P // a for a in p]
    iv = [pow(h, -1, a) for h, a in zip(H, p)]
    B = base**8
    E = [h % B for h in H]
    EP = P % B
    rankmax = (K + 1) // 2
    FQ = 1 << bits
    limit_ratio = 3 if K <= 16 else (4 if bits == 32 else 10)
    limit = limit_ratio * B
    error = Fraction(sum(a // 2 for a in p) + rankmax, 2 * FQ)
    e = Fraction(1, 2) + error
    Tmax = 2 * N * limit_ratio**2 + ((e.numerator + e.denominator - 1) // e.denominator)
    # Centered CRT residues t_i satisfy C = sum(t_i*H_i) - rank*P.
    # Rounding sum(t_i)/2**30 gives that exact rank under this bound.
    delta = sum(Fraction((1 << 30) - a, 2 * (1 << 30)) for a in p)
    require(
        Fraction(2 * N * limit**2, P) + delta < Fraction(1, 2), "rank guarantee fails"
    )
    # Quantized quotients give |rem| <= e*B, |u| <= e*(B+1).
    # A shifted carry update is rem[j] + u[j-1] + T[j-2], with negative wrap.
    # Its bound must close on the same packed-digit limit for every square.
    require(2 * e * B + e + Tmax < limit, "digit invariant fails")
    qi_qw = [split_fraction(x, B, bits) for x in E]
    qi, qw = map(list, zip(*qi_qw))
    qpi, qpw = split_fraction(EP, B, bits)
    ti_tw = [split_fraction(x, B * B, bits) for x in H]
    ti, tw = map(list, zip(*ti_tw))
    tpi, tpw = split_fraction(P, B * B, bits)
    require(Tmax < min(p) and base > limit_ratio + 2, "packed carry encoding range")
    # Split fractional dot products to keep every signed accumulator in int64.
    # The integer part of T is accumulated modulo 2**64; the final bounded T
    # is recovered from its low word, after the large CRT terms cancel.
    groups = 1 if bits == 30 else 3
    qbounds = [
        sum(p[j] // 2 * abs(qw[j]) for j in range(K) if j % groups == g)
        + (rankmax * abs(qpw) if g == 0 else 0)
        + (1 << (bits - 1))
        for g in range(groups)
    ]
    tbounds = [
        sum(p[j] // 2 * abs(tw[j]) for j in range(K) if j % groups == g)
        + (rankmax * abs(tpw) if g == 0 else 0)
        + (1 << (bits - 1))
        for g in range(groups)
    ]
    require(max(qbounds + tbounds) < 1 << 63, "quantizer accumulators")
    rows = []
    redbounds = []
    for a in p:
        bm = ctr(B * R, a)
        epm = ctr(EP * R, a)
        mat = [ctr(x * R, a) for x in E]
        # Choose equivalent Montgomery matrix constants near their correlated
        # quotient contribution rather than independently centering each constant.
        mat = [v + near(bm * x - v * B, a * B) * a for v, x in zip(mat, E)]
        epm += near(bm * EP - epm * B, a * B) * a
        require(
            all(-(1 << 31) <= v < 1 << 31 for v in mat + [bm, epm]),
            "packed arithmetic bound",
        )
        partial = sum(x // 2 * abs(v) for x, v in zip(p, mat)) + rankmax * abs(epm)
        qbound = (
            sum(Fraction(x // 2) * Fraction(v, B) for x, v in zip(p, E))
            + rankmax * Fraction(EP, B)
            + e
        )
        total = (
            sum(
                Fraction(p[j] // 2) * abs(Fraction(mat[j]) - Fraction(E[j], B) * bm)
                for j in range(K)
            )
            + rankmax * abs(Fraction(epm) - Fraction(EP, B) * bm)
            + e * abs(bm)
        )
        require(
            partial < 1 << 63
            and qbound * abs(bm) + rankmax * abs(epm) < 1 << 63
            and total < min(1 << 63, 2 * a * R),
            "Montgomery matrix range",
        )
        rows.append((bm, epm, mat))
        redbounds.append(str(total / Fraction(a * R)))
    scale = [[iv[f] * (dup + 1) % p[f] for f in range(K)] for dup in [0, 1]]
    vals = [
        p,
        [pow(a, -1, R) for a in p],
        scale,
        [[x * R // a for x, a in zip(s, p)] for s in scale],
        [h % a for h, a in zip(H, p)],
        [h % a * R // a for h, a in zip(H, p)],
        [pow(B, -1, a) for a in p],
        [pow(B, -1, a) * R // a for a in p],
        [B % a for a in p],
        [B % a * R // a for a in p],
        [v[0] for v in rows],
        [v[1] for v in rows],
        [v[2] for v in rows],
    ]
    quant = [qi, qw, qpi, qpw, [x % 2**64 for x in ti], tw, tpi % 2**64, tpw]
    return dict(
        base=base,
        K=K,
        bits=bits,
        groups=groups,
        B=B,
        P=P,
        H=H,
        E=E,
        EP=EP,
        iv=iv,
        limit=limit,
        e=e,
        Tmax=Tmax,
        params=vals,
        quant=quant,
        ti_exact=ti,
        tpi_exact=tpi,
        bounds=redbounds,
    )


def c_array(values):
    if isinstance(values, list):
        return "{" + ",".join(c_array(x) for x in values) + "}"
    return str(values) + (
        "ul" if values > 2**63 - 1 else "l" if abs(values) > 2**31 - 1 else ""
    )


def uint_array(values):
    return (
        "{"
        + ",".join(
            uint_array(x) if isinstance(x, list) else str(x) + "u" for x in values
        )
        + "}"
    )


def program(base, fields=15, kernel_directory=None, root_bits=32, outer_chunk=8):
    """Generate the measured 64-by-1024 squaring profile for one base."""
    bits = 32
    require(outer_chunk in (4, 8, 16), "unsupported outer transform chunk")
    require(
        1026 <= base <= 528220156 and base % 2 == 0 and base & (base - 1) != 0,
        "unsupported GFN-19 base",
    )
    require(
        fields in range(7, 18),
        "unsupported packed representation",
    )
    a = parameters(base, fields, bits)
    directory = (
        Path(kernel_directory)
        if kernel_directory
        else Path(__file__).resolve().parent / "kernels"
    )
    source = (directory / "gfn19-packed.cl").read_text()
    require(root_bits in (32, 64), "unsupported root word width")
    constants = []
    for p, g in zip(ps[:fields], generators[:fields]):
        i = pow(g, (p - 1) // 4, p)
        si = pow(g, (p - 1) // 8, p)
        constants.append(
            dict(
                p=p,
                q=pow(p, -1, R),
                im=i * R % p,
                si=si * R % p,
                isi=i * si * R % p,
                norm=2 * R * R * pow(N, -1, p) % p,
            )
        )
    for name, typ, keys in [
        ("g_pq", "uint2_32", ["p", "q"]),
        ("g_b0", "uint4_32", ["isi", "si", "im", None]),
        ("g_f0_ordinary", "uint4_32", [None, "im", "si", "isi"]),
    ]:
        declaration = (
            "__constant "
            + typ
            + " "
            + name
            + "[PG_K]={"
            + ",".join(
                "("
                + typ
                + ")("
                + ",".join(str(row[k]) + "u" if k else "0u" for k in keys)
                + ")"
                for row in constants
            )
            + "};"
        )
        source = source.replace("/* @" + name.upper() + "@ */", declaration)
    source = source.replace(
        "/* @G_PACKED_NORM@ */",
        "__constant uint g_packed_norm[PG_K]="
        + uint_array([c["norm"] for c in constants])
        + ";",
    )
    header = (
        "\n".join(
            "#define PG_" + k + " " + str(v)
            for k, v in [
                ("OUTER_THREADS", 16 * outer_chunk),
                ("INNER_THREADS", 64),
                ("CHUNK", outer_chunk),
                ("ROOT_WORDS", N // 2),
                ("ROOT_BITS", root_bits),
                ("K", fields),
                ("BASE", base),
                ("BITS", bits),
                ("GROUPS", a["groups"]),
                ("DIGIT_RATIO", a["limit"] // a["B"]),
                ("CODEC_LIMBS", (fields * 30 + 31) // 32 + 1),
            ]
        )
        + "\n"
    )
    for i in range(fields):
        header += (
            "#define PG_PRIME"
            + str(i)
            + " "
            + str(ps[i])
            + "\n#define PG_GENERATOR"
            + str(i)
            + " "
            + str(generators[i])
            + "\n"
        )
    structures = """typedef struct {
    uint p[PG_K], pinv[PG_K], scale[2][PG_K], scaleq[2][PG_K],
         h[PG_K], hq[PG_K], binv[PG_K], binvq[PG_K], b[PG_K], bq[PG_K];
    int bm[PG_K], epm[PG_K], matrix[PG_K][PG_K];
} PgParam;
typedef struct {
    int qi[PG_K], qw[PG_K], qpi, qpw;
    ulong ti[PG_K];
    int tw[PG_K];
    ulong tpi;
    int tpw;
} PgQuant;
"""
    structures += (
        "__constant PgParam pg_cp="
        + c_array(a["params"])
        + ";\n__constant PgQuant pg_cq="
        + c_array(a["quant"])
        + ";\n"
    )
    limbs = (fields * 30 + 31) // 32 + 1
    structures += (
        "__constant uint pg_codec_h[PG_K][PG_CODEC_LIMBS]="
        + uint_array([[(h >> (32 * j)) % 2**32 for j in range(limbs)] for h in a["H"]])
        + ";\n"
    )
    structures += (
        "__constant uint pg_codec_p[PG_CODEC_LIMBS]="
        + uint_array([(a["P"] >> (32 * j)) % 2**32 for j in range(limbs)])
        + ";\n"
    )
    structures += (
        "__constant uint pg_codec_bq[PG_K]="
        + uint_array([(base << 32) // p for p in ps[:fields]])
        + ";\n"
    )
    source = (
        header
        + source
        + structures
        + (directory / "gfn19-packed-convert.clh").read_text()
        + (directory / "gfn19-packed-codec.clh").read_text()
    )
    require("@G_" not in source, "unresolved packed source parameter")
    return source, dict(
        fields=fields,
        bits=bits,
        transform_length=N,
        digits_per_coefficient=8,
        outer_chunk=outer_chunk,
        root_bits=root_bits,
        compact_roots=True,
        maximum_digit=str(a["limit"]),
        rank_bound="deterministic",
    )


def select_program(base, gpu, kernel_directory):
    """Select a bounded packed representation; retain three-prime execution when
    no packed parameter set meets every reconstruction and accumulator bound.
    """
    require(gpu in ("3090", "4060"), "unsupported GPU profile")
    for fields in range(7, 18):
        try:
            return program(
                base,
                fields,
                kernel_directory=kernel_directory,
                root_bits=64 if gpu == "4060" else 32,
                outer_chunk=8,
            )
        except UnsupportedPackedParameters:
            continue
    return None
