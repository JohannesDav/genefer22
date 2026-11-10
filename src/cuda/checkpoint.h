#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

namespace cuda_gfn {
struct checkpoint_representation {
    unsigned fields;
    std::array<uint32_t, 3> primes;
};
inline unsigned checkpoint_exponent(uint32_t size) {
    if (!size || (size & (size - 1)))
        throw std::invalid_argument("Checkpoint size");
    unsigned out = 0;
    while (size > 1) {
        size >>= 1;
        ++out;
    }
    return out;
}
inline checkpoint_representation upstream_checkpoint(uint32_t base, unsigned exponent) {
    if (exponent < 12 || exponent > 23)
        throw std::invalid_argument("Checkpoint exponent");
    constexpr std::array<uint32_t, 3> narrow = {{2130706433u, 2113929217u, 2013265921u}};
    constexpr std::array<uint32_t, 3> wide = {{4194304001u, 4076863489u, 3942645761u}};
    const uint64_t square = uint64_t(base) * base;
    if (square < uint64_t(narrow[0]) * narrow[1] / 2 / (uint64_t(1) << exponent))
        return {2, narrow};
    if (square < uint64_t(wide[0]) * wide[1] / 2 / (uint64_t(1) << exponent))
        return {2, wide};
    return {3, base <= 1000000000 ? narrow : wide};
}
inline uint32_t digit_residue(int64_t digit, uint32_t prime) {
    // Admitted ordinary digits have magnitude below every original prime.
    if (digit <= -int64_t(prime) || digit >= int64_t(prime))
        throw std::invalid_argument("Checkpoint digit range");
    return uint32_t(digit < 0 ? digit + prime : digit);
}
inline int64_t checkpoint_digit(uint32_t first, uint32_t prime) {
    if (first >= prime)
        throw std::invalid_argument("Checkpoint residue range");
    return first > prime / 2 ? int64_t(first) - prime : int64_t(first);
}
} // namespace cuda_gfn
