#pragma once

// Exact packed CRT admission. All divisions that model Python // round down,
// including negative inputs. No floating-point estimates enter admission.
#include <algorithm>
#include <cstdint>
#include <gmpxx.h>
#include <stdexcept>
#include <vector>

namespace cuda_gfn {
using integer = mpz_class;
using rational = mpq_class;

inline void require(bool ok, const char *message) {
    if (!ok)
        throw std::invalid_argument(message);
}
inline integer floor_div(const integer &a, const integer &b) {
    require(b > 0, "Nonpositive exact divisor");
    integer out;
    mpz_fdiv_q(out.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
    return out;
}
inline integer mod(const integer &a, const integer &p) {
    integer out;
    mpz_mod(out.get_mpz_t(), a.get_mpz_t(), p.get_mpz_t());
    return out;
}
inline integer power(const integer &a, unsigned exponent) {
    integer out;
    mpz_pow_ui(out.get_mpz_t(), a.get_mpz_t(), exponent);
    return out;
}
inline integer inverse(const integer &a, const integer &p) {
    integer out;
    require(mpz_invert(out.get_mpz_t(), a.get_mpz_t(), p.get_mpz_t()) != 0, "Noninvertible packed constant");
    return out;
}
inline integer nearest(const integer &a, const integer &b) { return floor_div(2 * a + b, 2 * b); }
inline rational fraction(const integer &a, const integer &b) {
    rational out(a, b);
    out.canonicalize();
    return out;
}
inline integer magnitude(const integer &a) { return a >= 0 ? a : integer(-a); }
inline rational magnitude(const rational &a) { return a >= 0 ? a : rational(-a); }
inline integer centered(const integer &a, const integer &p) {
    return mod(a + floor_div(p, 2), p) - floor_div(p, 2);
}
struct weight {
    integer whole, fraction;
};
inline weight split(const integer &a, const integer &b) {
    const integer unit = integer(1) << 32, half = unit / 2;
    weight out;
    out.whole = nearest(a, b);
    out.fraction = nearest((a - out.whole * b) * unit, b);
    if (out.fraction == half) {
        ++out.whole;
        out.fraction -= unit;
    }
    require(out.fraction >= -half && out.fraction < half, "Fractional weight range");
    return out;
}
struct packed_parameters {
    unsigned fields = 0, groups = 0, ratio = 0;
    integer radix, modulus, extra_remainder, maximum_carry;
    rational error;
    std::vector<integer> primes, cofactors, inverses, remainders;
    std::vector<weight> quotient, tail;
    weight extra_quotient, extra_tail;
    std::vector<integer> radix_montgomery, extra_montgomery;
    std::vector<std::vector<integer>> matrix;
};

inline packed_parameters admit(uint32_t base, unsigned length, unsigned pack, unsigned ratio,
                               const std::vector<uint32_t> &primes) {
    require(base >= 1026 && base <= 2000000000 && !(base & 1) && (base & (base - 1)),
            "Unsupported packed base");
    require(length >= 16384 && length <= (1u << 22) && !(length & (length - 1)),
            "Unsupported transform length");
    require(pack && pack <= 64 && !(pack & (pack - 1)), "Unsupported packed radix");
    require(ratio >= 2 && ratio <= 4, "Unsupported digit invariant");
    require(primes.size() >= 2 && primes.size() <= 32, "Unsupported field count");
    const integer unit = integer(1) << 32, signed_limit = integer(1) << 63;
    packed_parameters out;
    out.fields = unsigned(primes.size());
    out.ratio = ratio;
    out.groups = std::max(3u, (out.fields + 7) / 8);
    out.radix = power(integer(base), pack);
    out.modulus = 1;
    for (uint32_t prime : primes) {
        require(prime > base + 2 && prime < (1u << 30), "Packed encoder prime range");
        out.primes.emplace_back(prime);
        out.modulus *= prime;
    }
    const unsigned rank = (out.fields + 1) / 2;
    const integer radix_squared = out.radix * out.radix;
    for (const integer &prime : out.primes) {
        const integer cofactor = out.modulus / prime;
        out.cofactors.push_back(cofactor);
        out.inverses.push_back(inverse(cofactor, prime));
        out.remainders.push_back(mod(cofactor, out.radix));
    }
    out.extra_remainder = mod(out.modulus, out.radix);
    auto quantizer_error = [&](const std::vector<integer> &values, const integer &extra,
                               const integer &denominator) {
        auto epsilon = [&](const integer &value) -> rational {
            const weight w = split(value, denominator);
            return magnitude(rational(w.whole) + fraction(w.fraction, unit) - fraction(value, denominator));
        };
        rational total = rank * epsilon(extra);
        for (unsigned j = 0; j < out.fields; ++j)
            total += rational(out.primes[j] / 2) * epsilon(values[j]);
        return total;
    };
    const rational error_q = quantizer_error(out.remainders, out.extra_remainder, out.radix);
    const rational error_t = quantizer_error(out.cofactors, out.modulus, radix_squared);
    integer rounding_terms = rank;
    for (const integer &prime : out.primes)
        rounding_terms += prime / 2;
    require(std::max(error_q, error_t) <= fraction(rounding_terms, 2 * unit),
            "Invalid exact quantizer error bound");
    out.error = fraction(1, 2) + std::max(error_q, error_t);
    out.maximum_carry =
        2 * integer(length) * ratio * ratio - floor_div(-out.error.get_num(), out.error.get_den());
    rational delta = 0;
    for (const integer &prime : out.primes)
        delta += fraction((integer(1) << 30) - prime, integer(1) << 31);
    require(fraction(2 * integer(length) * ratio * ratio * radix_squared, out.modulus) + delta <
                fraction(1, 2),
            "Rank guarantee fails");
    const rational bridge = 2 * out.error * out.radix + out.error + rational(out.maximum_carry);
    require(bridge < rational(ratio * out.radix), "Digit invariant fails");
    require(bridge < rational(3 * out.radix), "Ordinary digit bridge bound");
    require(out.maximum_carry < *std::min_element(out.primes.begin(), out.primes.end()) && base > ratio + 2,
            "Packed carry encoding range");
    for (unsigned j = 0; j < out.fields; ++j) {
        out.quotient.push_back(split(out.remainders[j], out.radix));
        out.tail.push_back(split(out.cofactors[j], radix_squared));
    }
    out.extra_quotient = split(out.extra_remainder, out.radix);
    out.extra_tail = split(out.modulus, radix_squared);
    for (unsigned group = 0; group < out.groups; ++group) {
        integer qbound = unit / 2, tbound = unit / 2;
        if (!group) {
            qbound += rank * magnitude(out.extra_quotient.fraction);
            tbound += rank * magnitude(out.extra_tail.fraction);
        }
        for (unsigned j = group; j < out.fields; j += out.groups) {
            qbound += (out.primes[j] / 2) * magnitude(out.quotient[j].fraction);
            tbound += (out.primes[j] / 2) * magnitude(out.tail[j].fraction);
        }
        require(qbound < signed_limit && tbound < signed_limit, "Quantizer accumulators");
    }
    for (const integer &prime : out.primes) {
        const integer bm = centered(out.radix * unit, prime);
        integer epm = centered(out.extra_remainder * unit, prime);
        epm += nearest(bm * out.extra_remainder - epm * out.radix, prime * out.radix) * prime;
        std::vector<integer> row;
        integer partial = rank * magnitude(epm);
        rational qbound = rank * fraction(out.extra_remainder, out.radix) + out.error;
        rational total = rank * magnitude(rational(epm) - fraction(out.extra_remainder, out.radix) * bm) +
                         out.error * magnitude(bm);
        for (unsigned j = 0; j < out.fields; ++j) {
            integer v = centered(out.remainders[j] * unit, prime);
            v += nearest(bm * out.remainders[j] - v * out.radix, prime * out.radix) * prime;
            require(v >= -(integer(1) << 31) && v < (integer(1) << 31), "Packed matrix constant range");
            row.push_back(v);
            partial += (out.primes[j] / 2) * magnitude(v);
            qbound += rational(out.primes[j] / 2) * fraction(out.remainders[j], out.radix);
            total += rational(out.primes[j] / 2) *
                     magnitude(rational(v) - fraction(out.remainders[j], out.radix) * bm);
        }
        require(bm >= -(integer(1) << 31) && bm < (integer(1) << 31) && epm >= -(integer(1) << 31) &&
                    epm < (integer(1) << 31),
                "Packed Montgomery constant range");
        require(partial < signed_limit &&
                    qbound * magnitude(bm) + rank * magnitude(epm) < rational(signed_limit) &&
                    total < rational(signed_limit) && total < rational(2 * prime * unit),
                "Montgomery matrix range");
        out.radix_montgomery.push_back(bm);
        out.extra_montgomery.push_back(epm);
        out.matrix.push_back(std::move(row));
    }
    return out;
}
} // namespace cuda_gfn
