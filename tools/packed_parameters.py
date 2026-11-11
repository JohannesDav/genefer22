"""Generate exact packed CRT parameters; no distribution assumptions."""

from fractions import Fraction
from math import prod
from pathlib import Path
from functools import lru_cache

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


def parameters(
    base,
    K,
    bits=32,
    pack=8,
    accurate_error=False,
    length=N,
    primes=None,
    digit_ratio=None,
    prime_bits=30,
    quantizer_groups=None,
):
    """Construct constants only when every reconstruction and carry bound holds.

    B = base**pack is the packed radix. Coefficients have magnitude at most
    2*N*limit**2, including a doubled square. The returned constants recover
    each coefficient as rem + B*u + B**2*T using integer arithmetic.
    """
    N = length
    require(prime_bits == 30, "unsupported prime width")
    p = list(primes[:K]) if primes is not None else ps[:K]
    P = prod(p)
    H = [P // a for a in p]
    iv = [pow(h, -1, a) for h, a in zip(H, p)]
    B = base**pack
    E = [h % B for h in H]
    EP = P % B
    rankmax = (K + 1) // 2
    FQ = 1 << bits
    limit_ratio = (
        digit_ratio
        if digit_ratio is not None
        else (3 if K <= 16 else (4 if bits == 32 else 10))
    )
    require(limit_ratio in (2, 3, 4, 10), "unsupported digit invariant")
    limit = limit_ratio * B
    error = Fraction(sum(a // 2 for a in p) + rankmax, 2 * FQ)
    if accurate_error:
        # Bound the actual fixed rounding constants, still for every possible
        # centered residue. This is never larger than the uniform bound above.
        def quantizer_error(values, extra, denominator):
            def epsilon(value):
                whole, fraction = split_fraction(value, denominator, bits)
                return abs(
                    Fraction(whole)
                    + Fraction(fraction, FQ)
                    - Fraction(value, denominator)
                )

            return sum(
                (prime // 2) * epsilon(value) for prime, value in zip(p, values)
            ) + rankmax * epsilon(extra)

        exact_error = max(quantizer_error(E, EP, B), quantizer_error(H, P, B * B))
        require(exact_error <= error, "invalid exact quantizer error bound")
        error = exact_error
    e = Fraction(1, 2) + error
    Tmax = 2 * N * limit_ratio**2 + ((e.numerator + e.denominator - 1) // e.denominator)
    # Centered CRT residues t_i satisfy C = sum(t_i*H_i) - rank*P.
    # Rounding sum(t_i)/2**30 gives that exact rank under this bound.
    delta = sum(Fraction((1 << prime_bits) - a, 2 * (1 << prime_bits)) for a in p)
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
    groups = (
        (1 if bits == 30 else max(3, (K + 7) // 8))
        if prime_bits == 30
        else max(3, (K + 3) // 4)
    )
    if quantizer_groups is not None:
        groups = quantizer_groups
    require(1 <= groups <= 16, "unsupported quantizer grouping")
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


@lru_cache(maxsize=None)
def field_pool(count, length, prime_bits=30):
    if length == N and prime_bits == 30:
        if count > len(ps):
            extend_fields(count)
        return tuple(ps[:count]), tuple(generators[:count])
    # The transform retains quadratic leaves: roots of order length suffice.
    # X^N+1 splits as product(X^2-omega^(2j+1)), omega primitive of order N.
    step = length
    candidate = (1 << prime_bits) - step + 1
    values, gs = [], []
    while len(values) < count:
        require(candidate > step, "insufficient field primes")
        if is_prime32(candidate):
            g = next(
                g
                for g in range(2, 1000)
                if pow(g, (candidate - 1) // 2, candidate) == candidate - 1
            )
            values.append(candidate)
            gs.append(g)
        candidate -= step
    return tuple(values), tuple(gs)


def outer_source(outer):
    """Generate the same radix-four network for a different outer factor."""
    require(outer in (16, 64, 256, 1024), "unsupported outer transform")
    B = outer // 4
    log = outer.bit_length() - 1
    stages = []
    m = B // 4
    while m > 1:
        stages.append(m)
        m //= 4
    common = [
        f"const int lm=LN_SZ-LVSIZE-{log};",
        f"const unsigned s={B};",
        f"__local VTYPE Z[{outer}*PG_CHUNK];",
        f"DECLARE_VAR({B},PG_CHUNK);",
    ]
    forward = [
        "__kernel __attribute__((reqd_work_group_size(PG_OUTER_THREADS,1,1)))",
        "void packed_forward(__global VTYPE *restrict const zg,__global const root_word_t *restrict const wg){",
        *common,
        "DECLARE_VAR_FORWARD(); VTYPE v[4];",
        f"_loadg(4,v,zi,{B}<<lm);",
        "forward4_ordinary(pq,g_f0_ordinary[lid],v);",
        f"_storel(4,&Z[i],{B}*PG_CHUNK,v);",
    ]
    for m in stages:
        forward += [
            f"const sz_t k{m}=((4*threadIdx)&~({4*m}-1))+(threadIdx%{m});",
            f"forward_4(pq,{m}*PG_CHUNK,&Zi[PG_CHUNK*k{m}],w,sj/{m});",
        ]
    forward += [
        "forward_4o(pq,(sz_t)1<<lm,zo,PG_CHUNK,&Zi[PG_CHUNK*4*threadIdx],w,sj);",
        "}",
        "/* @G_PACKED_NORM@ */",
    ]
    backward = [
        "__kernel __attribute__((reqd_work_group_size(PG_OUTER_THREADS,1,1)))",
        "void packed_backward(__global VTYPE *restrict const zg,__global const root_word_t *restrict const wg){",
        *common,
        "DECLARE_VAR_BACKWARD();",
        "backward_4i(pq,PG_CHUNK,&Zi[PG_CHUNK*4*threadIdx],(sz_t)1<<lm,zi,wi,sji);",
    ]
    for m in reversed(stages):
        backward += [
            f"const sz_t k{m}=((4*threadIdx)&~({4*m}-1))+(threadIdx%{m});",
            f"backward_4(pq,{m}*PG_CHUNK,&Zi[PG_CHUNK*k{m}],wi,sji/{m});",
        ]
    backward += [
        "barrier(CLK_LOCAL_MEM_FENCE); VTYPE v[4];",
        f"_loadl(4,v,&Z[i],{B}*PG_CHUNK);",
        "_backward4_0(pq,g_b0[lid],v);",
        "for(int k=0;k<4;++k)v[k]=lscalar4(fold24(v[k],pq.s0),g_packed_norm[lid],pq);",
        f"_storeg(4,zo,{B}<<lm,v);",
        "}",
    ]
    return "\n".join(forward + backward) + "\n"


def inner_source(inner):
    require(
        inner in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
        "unsupported inner transform",
    )
    L = inner // 4
    threads = L // 4
    stages = []
    m = threads
    while m >= 1:
        stages.append(m)
        m //= 4
    # log2(inner) even => a four-point leaf; odd => eight-point leaf.
    leaf = 4 if (inner.bit_length() - 1) % 2 == 0 else 8
    lines = [
        "__kernel __attribute__((reqd_work_group_size(PG_INNER_THREADS,1,1)))",
        "void packed_square1024(__global VTYPE *restrict const zg, __global const root_word_t *restrict const wg) {",
        f"__local VTYPE Z[{L} * PG_INNER_BLOCKS];",
        "DECLARE_VAR_REG();",
        f"const sz_t local_id=id%({threads}*PG_INNER_BLOCKS), group_id=id/({threads}*PG_INNER_BLOCKS);",
        f"const sz_t start=(local_id & ~({threads}-1))*4, t=local_id%{threads};",
        f"const sz_t k=group_id*{L}*PG_INNER_BLOCKS+start+t;",
        "const sz_t sj=N_SZ/4/VSIZE+id; DECLARE_IVAR(N_SZ/4/VSIZE,id);",
        "__global VTYPE *restrict const zk=&z[k]; __local VTYPE *const Zleaf=&Z[start];",
        "__local VTYPE *const p0=&Zleaf[t];",
    ]
    for j, m in enumerate(stages[1:], 1):
        lines.append(f"__local VTYPE *const p{j}=&Zleaf[((4*t)&~({4*m}-1))+(t%{m})];")
    lines.append(f"forward_4i(pq,{stages[0]},p0,{stages[0]},zk,w,sj/{stages[0]});")
    for j, m in enumerate(stages[1:], 1):
        lines.append(f"forward_4(pq,{m},p{j},w,sj/{m});")
    lines.append(f"square_{leaf}(pq,&Zleaf[4*t],w,wi,sj,sji);")
    for j in range(len(stages) - 1, 0, -1):
        m = stages[j]
        lines.append(f"backward_4(pq,{m},p{j},wi,sji/{m});")
    lines.append(f"backward_4o(pq,{stages[0]},zk,{stages[0]},p0,wi,sji/{stages[0]});")
    return "\n".join(lines + ["}"]) + "\n"


def program(
    base,
    fields=15,
    kernel_directory=None,
    root_bits=32,
    outer_chunk=8,
    pack=8,
    accurate_error=False,
    inner_blocks=1,
    length=N,
    outer=64,
    digit_ratio=None,
    prime_bits=30,
    quantizer_groups=None,
):
    """Generate a bounded squaring profile with a resource-checked transform split."""
    N = length
    require(prime_bits == 30, "unsupported prime width")
    require(N in tuple(1 << j for j in range(14, 23)), "unsupported transform length")
    bits = 32
    require(outer in (16, 64, 256, 1024), "unsupported outer factor")
    inner = N // outer
    require(
        outer * outer_chunk * 16 <= 49152 and (outer // 4) * outer_chunk <= 1024,
        "outer resource limit",
    )
    require(
        inner * 4 * inner_blocks <= 49152 and (inner // 16) * inner_blocks <= 1024,
        "inner resource limit",
    )
    require(outer_chunk in (1, 2, 4, 8, 16, 32), "unsupported outer transform chunk")
    require(
        inner_blocks in (1, 2, 4, 8, 16, 32), "unsupported inner transform grouping"
    )
    require(
        1026 <= base <= 2000000000 and base % 2 == 0 and base & (base - 1) != 0,
        "unsupported candidate base",
    )
    require(
        fields in range(2, 33),
        "unsupported packed representation",
    )
    require(pack in (1, 2, 4, 8, 16, 32, 64), "unsupported packed radix width")
    pool, roots = field_pool(fields, N, prime_bits)
    require(base + 2 < min(pool), "packed encoder base range")
    a = parameters(
        base,
        fields,
        bits,
        pack=pack,
        accurate_error=accurate_error,
        length=N,
        primes=pool,
        digit_ratio=digit_ratio,
        prime_bits=prime_bits,
        quantizer_groups=quantizer_groups,
    )
    require(
        2 * a["e"] * a["B"] + a["e"] + a["Tmax"] < 3 * a["B"],
        "ordinary digit bridge bound",
    )
    directory = (
        Path(kernel_directory)
        if kernel_directory
        else Path(__file__).resolve().parents[1] / "kernels/src"
    )
    source = (directory / "packed.cl").read_text()
    begin = source.index(
        "__kernel __attribute__((reqd_work_group_size(PG_OUTER_THREADS"
    )
    source = source[:begin] + outer_source(outer) + inner_source(inner)
    require(root_bits in (32, 64), "unsupported root word width")
    constants = []
    for p, g in zip(pool, roots):
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
    # Division visits one 32-bit word per radix digit even when a small base
    # needs fewer CRT words. Pad both the temporary and its constants equally.
    limbs = max(pack, (fields * prime_bits + 31) // 32 + 1)
    header = (
        "\n".join(
            "#define PG_" + k + " " + str(v)
            for k, v in [
                ("OUTER_THREADS", (outer // 4) * outer_chunk),
                ("INNER_THREADS", (inner // 16) * inner_blocks),
                ("INNER_BLOCKS", inner_blocks),
                ("CHUNK", outer_chunk),
                ("ROOT_WORDS", N // 2),
                ("ROOT_BITS", root_bits),
                ("PRIME_BITS", prime_bits),
                ("K", fields),
                ("N", N),
                ("LN", N.bit_length() - 1),
                ("INNER_N", inner),
                ("PACK", pack),
                ("BASE", base),
                ("BITS", bits),
                ("GROUPS", a["groups"]),
                ("DIGIT_RATIO", a["limit"] // a["B"]),
                ("CODEC_LIMBS", limbs),
            ]
        )
        + "\n"
    )
    for i in range(fields):
        header += (
            "#define PG_PRIME"
            + str(i)
            + " "
            + str(pool[i])
            + "\n#define PG_GENERATOR"
            + str(i)
            + " "
            + str(roots[i])
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
        + uint_array([(base << 32) // p for p in pool])
        + ";\n"
    )
    source = (
        header
        + source
        + structures
        + (directory / "packed-convert.clh").read_text()
        + (directory / "packed-codec.clh").read_text()
    )
    require("@G_" not in source, "unresolved packed source parameter")
    return source, dict(
        fields=fields,
        bits=bits,
        transform_length=N,
        digits_per_coefficient=pack,
        outer_chunk=outer_chunk,
        inner_blocks=inner_blocks,
        outer_factor=outer,
        digit_ratio=a["limit"] // a["B"],
        root_bits=root_bits,
        compact_roots=True,
        prime_bits=prime_bits,
        quantizer_groups=a["groups"],
        maximum_digit=str(a["limit"]),
        rank_bound="deterministic",
    )


def is_prime32(value):
    if value < 2:
        return False
    for p in (2, 3, 5, 7, 11):
        if value % p == 0:
            return value == p
    d = value - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11):
        x = pow(a, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def extend_fields(count=32):
    candidate = min(ps) - 2 * N
    while len(ps) < count:
        if is_prime32(candidate):
            g = 2
            while pow(g, (candidate - 1) // 2, candidate) == 1:
                g += 1
            ps.append(candidate)
            generators.append(g)
        candidate -= 2 * N
