#pragma once
#include <cstdint>

// GFN-19, three small-prime channels, normalized digit-domain storage.
// This bounds coefficient reconstruction and carry arithmetic. It does not
// assert that a GPU compiler or the complete implementation is bug-free.
namespace gfn19_guaranteed {
using wide = unsigned __int128;
constexpr uint32_t size = 1u << 19;
constexpr uint32_t prime1 = 1056440321u;
constexpr uint32_t prime2 = 1053818881u;
constexpr uint32_t prime3 = 1051721729u;
constexpr wide modulus = wide(prime1) * prime2 * prime3;
constexpr uint32_t minimum_base = 1024;
// |digit| <= b+2, strictly below prime1/2 for get_int's signed decoding.
constexpr uint32_t maximum_base = ((prime1 / 2 - 3) & ~uint32_t(1));
constexpr wide ceil_div(wide x, wide y) { return x / y + (x % y != 0); }
constexpr wide digit_bound(uint32_t base) { return wide(base) + 2; }
constexpr wide coefficient_bound(uint32_t base) {
    return 2 * wide(size) * digit_bound(base) * digit_bound(base);
}
constexpr wide carry_bound(uint32_t base) {
    return ceil_div(coefficient_bound(base), base - 1);
}
constexpr bool supported(uint32_t base) {
    if (base < minimum_base || base > maximum_base || base % 2 || !(base & (base-1)))
        return false;
    const wide digits = digit_bound(base), coefficients = coefficient_bound(base);
    const wide carry = carry_bound(base), cube = wide(base) * base * base;
    const wide scalar_product = digits * 2147483647u;
    const wide scalar_carry = ceil_div(scalar_product, base - 1);
    return digits < prime1 / 2 && digits < prime3
        && coefficients + carry < modulus / 2
        && coefficients + carry < (wide(1) << 93)
        && carry + digits < (wide(1) << 63)
        && scalar_product + scalar_carry < (wide(1) << 63)
        && ((scalar_product + scalar_carry) >> 29) < (wide(1) << 29) * base
        && ((coefficients + carry) >> 29) < (wide(1) << 29) * base
        && ((carry + digits) >> 29) < (wide(1) << 29) * base
        && wide(base-1) + ceil_div(carry, cube) <= digits
        && wide(base-1) + ceil_div(scalar_carry, cube) <= digits;
}
static_assert(4 * uint64_t(prime1) < (uint64_t(1) << 32), "lazy arithmetic range");
static_assert(4 * uint64_t(prime2) < (uint64_t(1) << 32), "lazy arithmetic range");
static_assert(4 * uint64_t(prime3) < (uint64_t(1) << 32), "lazy arithmetic range");
static_assert(supported(1026) && supported(28000000) && supported(maximum_base), "bounds");
static_assert(!supported(maximum_base+2), "signed digit decoding boundary");
}
