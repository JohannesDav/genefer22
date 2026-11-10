#include "kernel.h"
#include "parameters.h"
#include "checkpoint.h"
#include <cuda/embedded_kernels.h>
#include <sstream>

namespace cuda_gfn {
namespace {
unsigned logarithm(unsigned n) {
    unsigned log = 0;
    while (n > 1) {
        n >>= 1;
        ++log;
    }
    return log;
}
void replace(std::string &text, const std::string &needle, const std::string &value) {
    const size_t at = text.find(needle);
    require(at != std::string::npos, "Missing embedded kernel marker");
    text.replace(at, needle.size(), value);
}
std::string literal(const integer &a) {
    return a.get_str() + (a > (integer(1) << 63) - 1              ? "ul"
                          : magnitude(a) > (integer(1) << 31) - 1 ? "l"
                                                                  : "");
}
template <class T> std::string array(const std::vector<T> &a);
std::string array(const integer &a) { return literal(a); }
template <class T> std::string array(const std::vector<T> &a) {
    std::string out = "{";
    for (size_t j = 0; j < a.size(); ++j) {
        if (j)
            out += ",";
        out += array(a[j]);
    }
    return out + "}";
}
template <class T> std::string unsigned_array(const std::vector<T> &a);
std::string unsigned_array(const integer &a) { return a.get_str() + "u"; }
template <class T> std::string unsigned_array(const std::vector<T> &a) {
    std::string out = "{";
    for (size_t j = 0; j < a.size(); ++j) {
        if (j)
            out += ",";
        out += unsigned_array(a[j]);
    }
    return out + "}";
}
integer powmod(const integer &a, const integer &exponent, const integer &p) {
    integer out;
    mpz_powm(out.get_mpz_t(), a.get_mpz_t(), exponent.get_mpz_t(), p.get_mpz_t());
    return out;
}
struct prime_pool {
    const uint32_t *primes, *roots;
    unsigned count;
};
prime_pool pool(unsigned log) {
    using namespace embedded;
    switch (log) {
#define POOL(N)                                                                                              \
    case N:                                                                                                  \
        return {primes_##N, roots_##N, prime_count_##N};
        POOL(14)
        POOL(15)
        POOL(16) POOL(17) POOL(18) POOL(19) POOL(20) POOL(21) POOL(22)
#undef POOL
            default : throw std::invalid_argument("Unsupported packed length");
    }
}
const char *outer_network(unsigned n) {
    switch (n) {
#define OUTER(N)                                                                                             \
    case N:                                                                                                  \
        return embedded::outer_##N;
        OUTER(16)
        OUTER(64)
        OUTER(256) OUTER(1024)
#undef OUTER
            default : throw std::invalid_argument("Unsupported outer factor");
    }
}
const char *inner_network(unsigned n) {
    switch (n) {
#define INNER(N)                                                                                             \
    case N:                                                                                                  \
        return embedded::inner_##N;
        INNER(16)
        INNER(32)
        INNER(64) INNER(128) INNER(256) INNER(512) INNER(1024) INNER(2048) INNER(4096) INNER(8192)
#undef INNER
            default : throw std::invalid_argument("Unsupported inner factor");
    }
}
std::vector<integer> column(const std::vector<weight> &a, bool fractions) {
    std::vector<integer> out;
    for (const auto &w : a)
        out.push_back(fractions ? w.fraction : w.whole);
    return out;
}
std::string source(uint32_t base, const kernel_plan &plan, unsigned fields, const prime_pool &p,
                   const packed_parameters &a) {
    const integer unit = integer(1) << 32;
    const unsigned limbs = std::max(plan.pack, (fields * 30 + 31) / 32 + 1);
    std::ostringstream header;
    auto define = [&](const char *name, unsigned value) {
        header << "#define PG_" << name << " " << value << "\n";
    };
    define("OUTER_THREADS", plan.outer / 4 * plan.chunk);
    define("INNER_THREADS", plan.length / plan.outer / 16 * plan.inner_blocks);
    define("INNER_BLOCKS", plan.inner_blocks);
    define("CHUNK", plan.chunk);
    define("ROOT_WORDS", plan.length / 2);
    define("ROOT_BITS", plan.root_bits);
    define("PRIME_BITS", 30);
    define("K", fields);
    define("N", plan.length);
    define("LN", logarithm(plan.length));
    define("INNER_N", plan.length / plan.outer);
    define("PACK", plan.pack);
    define("BASE", base);
    define("BITS", 32);
    define("GROUPS", a.groups);
    define("DIGIT_RATIO", plan.ratio);
    define("CODEC_LIMBS", limbs);
    for (unsigned j = 0; j < fields; ++j)
        header << "#define PG_PRIME" << j << " " << p.primes[j] << "\n#define PG_GENERATOR" << j << " "
               << p.roots[j] << "\n";
    std::string body = std::string(embedded::packed_base) + outer_network(plan.outer) +
                       inner_network(plan.length / plan.outer);
    std::vector<integer> pinv, im, si, isi, norm;
    for (unsigned j = 0; j < fields; ++j) {
        const integer prime = p.primes[j], g = p.roots[j];
        const integer i = powmod(g, (prime - 1) / 4, prime), s = powmod(g, (prime - 1) / 8, prime);
        pinv.push_back(inverse(prime, unit));
        im.push_back(mod(i * unit, prime));
        si.push_back(mod(s * unit, prime));
        isi.push_back(mod(i * s * unit, prime));
        norm.push_back(mod(2 * unit * unit * inverse(integer(plan.length), prime), prime));
    }
    const std::vector<integer> zero(fields, integer(0));
    auto declaration = [&](const char *name, const char *type,
                           const std::vector<std::vector<integer>> &columns) {
        std::string out = std::string("__constant ") + type + " " + name + "[PG_K]={";
        for (unsigned j = 0; j < fields; ++j) {
            if (j)
                out += ",";
            out += std::string("(") + type + ")(";
            for (size_t k = 0; k < columns.size(); ++k) {
                if (k)
                    out += ",";
                out += columns[k][j].get_str() + "u";
            }
            out += ")";
        }
        return out + "};";
    };
    replace(body, "/* @G_PQ@ */", declaration("g_pq", "uint2_32", {a.primes, pinv}));
    replace(body, "/* @G_B0@ */", declaration("g_b0", "uint4_32", {isi, si, im, zero}));
    replace(body, "/* @G_F0_ORDINARY@ */", declaration("g_f0_ordinary", "uint4_32", {zero, im, si, isi}));
    replace(body, "/* @G_PACKED_NORM@ */",
            "__constant uint g_packed_norm[PG_K]=" + unsigned_array(norm) + ";");
    std::string structures = R"(typedef struct {
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
)";
    std::vector<std::vector<integer>> scale(2), scaleq(2);
    std::vector<integer> h, hq, binv, binvq, b, bq;
    for (unsigned j = 0; j < fields; ++j) {
        const integer &prime = a.primes[j];
        for (unsigned d = 0; d < 2; ++d) {
            const integer v = mod(a.inverses[j] * (d + 1), prime);
            scale[d].push_back(v);
            scaleq[d].push_back(v * unit / prime);
        }
        h.push_back(mod(a.cofactors[j], prime));
        hq.push_back(h.back() * unit / prime);
        binv.push_back(inverse(a.radix, prime));
        binvq.push_back(binv.back() * unit / prime);
        b.push_back(mod(a.radix, prime));
        bq.push_back(b.back() * unit / prime);
    }
    std::string params = "{" + array(a.primes) + "," + array(pinv) + "," + array(scale) + "," +
                         array(scaleq) + "," + array(h) + "," + array(hq) + "," + array(binv) + "," +
                         array(binvq) + "," + array(b) + "," + array(bq) + "," + array(a.radix_montgomery) +
                         "," + array(a.extra_montgomery) + "," + array(a.matrix) + "}";
    auto ti = column(a.tail, false);
    const integer u64 = integer(1) << 64;
    for (auto &v : ti)
        v = mod(v, u64);
    const std::string quant = "{" + array(column(a.quotient, false)) + "," + array(column(a.quotient, true)) +
                              "," + literal(a.extra_quotient.whole) + "," +
                              literal(a.extra_quotient.fraction) + "," + array(ti) + "," +
                              array(column(a.tail, true)) + "," + literal(mod(a.extra_tail.whole, u64)) +
                              "," + literal(a.extra_tail.fraction) + "}";
    structures += "__constant PgParam pg_cp=" + params + ";\n__constant PgQuant pg_cq=" + quant + ";\n";
    auto words = [&](const integer &value) {
        std::vector<integer> out;
        for (unsigned j = 0; j < limbs; ++j)
            out.push_back(mod(integer(value >> (32 * j)), unit));
        return out;
    };
    std::vector<std::vector<integer>> cofactors;
    std::vector<integer> division;
    for (unsigned j = 0; j < fields; ++j) {
        cofactors.push_back(words(a.cofactors[j]));
        division.push_back(integer(base) * unit / a.primes[j]);
    }
    structures += "__constant uint pg_codec_h[PG_K][PG_CODEC_LIMBS]=" + unsigned_array(cofactors) + ";\n";
    structures += "__constant uint pg_codec_p[PG_CODEC_LIMBS]=" + unsigned_array(words(a.modulus)) + ";\n";
    structures += "__constant uint pg_codec_bq[PG_K]=" + unsigned_array(division) + ";\n";
    std::string converter = embedded::converter;
    if (plan.matrix_mad) {
        replace(converter, "INLINE uint pg_shoup(",
                "INLINE long pg_mad(long a,int x,int y){long z;asm(\"mad.wide.s32 "
                "%0,%1,%2,%3;\":\"=l\"(z):\"r\"(x),\"r\"(y),\"l\"(a));return z;}\nINLINE uint pg_shoup(");
        replace(converter, "dot += (long)t[k] * pg_cp.matrix[f][k];",
                "dot = pg_mad(dot,t[k],pg_cp.matrix[f][k]);");
    }
    if (plan.inline_converter)
        replace(converter, "INLINE void pg_convert(",
                "INLINE __attribute__((always_inline)) void pg_convert(");
    std::string out = header.str() + body + structures + converter + embedded::codec;
    require(out.find("@G_") == std::string::npos, "Unresolved kernel parameter");
    return out;
}
} // namespace

generated_kernel generate_kernel(uint32_t base, unsigned exponent, const kernel_plan &plan) {
    require(exponent >= 16 && exponent <= 22, "Unsupported CUDA GFN size");
    require(plan.length >= 16384 && plan.length <= (1u << 22) && !(plan.length & (plan.length - 1)),
            "Unsupported packed length");
    require(uint64_t(plan.length) * plan.pack == (uint64_t(1) << exponent), "Packed exponent mismatch");
    require(packed_handoff_fits(base, exponent, plan.pack), "Packed digits exceed ordinary two-prime bound");
    require(plan.root_bits == 32 || plan.root_bits == 64, "Unsupported root word width");
    require(plan.outer && plan.length % plan.outer == 0, "Invalid transform split");
    const unsigned inner = plan.length / plan.outer;
    require(plan.chunk && plan.chunk <= 32 && !(plan.chunk & (plan.chunk - 1)), "Unsupported outer grouping");
    require(plan.inner_blocks && plan.inner_blocks <= 32 && !(plan.inner_blocks & (plan.inner_blocks - 1)),
            "Unsupported inner grouping");
    require(uint64_t(plan.outer) * plan.chunk * 16 <= 49152 && plan.outer / 4 * plan.chunk <= 1024,
            "Outer resource limit");
    require(uint64_t(inner) * 4 * plan.inner_blocks <= 49152 && inner / 16 * plan.inner_blocks <= 1024,
            "Inner resource limit");
    (void)outer_network(plan.outer);
    (void)inner_network(inner);
    const prime_pool p = pool(logarithm(plan.length));
    const unsigned first = plan.fields ? plan.fields : 2, last = plan.fields ? plan.fields : p.count;
    require(first >= 2 && last <= p.count, "Unavailable prime count");
    for (unsigned fields = first; fields <= last; ++fields) {
        std::vector<uint32_t> primes(p.primes, p.primes + fields);
        packed_parameters a;
        try {
            a = admit(base, plan.length, plan.pack, plan.ratio, primes);
        } catch (const std::invalid_argument &) {
            if (plan.fields)
                throw;
            else
                continue;
        }
        return {source(base, plan, fields, p, a), fields};
    }
    throw std::invalid_argument("No admitted packed prime count");
}
} // namespace cuda_gfn
