#pragma once
#include <gmp.h>
// Independent divide-and-conquer base conversion; no Genefer transform code.
inline void radix_to_mpz(mpz_t out, const int32_t *d, size_t n, unsigned b)
{
    if (n <= 32)
    {
        mpz_set_ui(out, 0);
        for (size_t i = n; i-- > 0;)
        {
            mpz_mul_ui(out, out, b);
            if (d[i] >= 0)
                mpz_add_ui(out, out, d[i]);
            else
                mpz_sub_ui(out, out, -int64_t(d[i]));
        }
        return;
    }
    mpz_t hi, p;
    mpz_inits(hi, p, nullptr);
    size_t half = n / 2;
    radix_to_mpz(out, d, half, b);
    radix_to_mpz(hi, d + half, n - half, b);
    mpz_ui_pow_ui(p, b, half);
    mpz_mul(hi, hi, p);
    mpz_add(out, out, hi);
    mpz_clears(hi, p, nullptr);
}
