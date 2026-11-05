#include <chrono>
#include <iostream>
#include <memory>
#include "cuda/transform.h"
#include "reference.h"
class ArithmeticTest : public transformGPUs<2, false>
{
  public:
    using transformGPUs::getZi;
    using transformGPUs::setZi;
    using transformGPUs::transformGPUs;
};
int main(int argc, char **argv)
{
    unsigned b = argc > 1 ? std::stoul(argv[1]) : 28000000,
             count = argc > 2 ? std::stoul(argv[2]) : 2000;
    ArithmeticTest t(b, 19, false, 0, 3, 0, 0, true);
    unsigned offset = std::getenv("DUP_OFFSET") ? std::stoul(std::getenv("DUP_OFFSET")) : 0;
    auto dup = [&](unsigned i) {
        return std::getenv("DUP_ALL8") ? bool((((i / 8 + offset) & 255) >> (i % 8)) & 1)
                                       : bool(i % 3 == 0);
    };
    std::vector<int32_t> z(1 << 19);
    uint64_t rng = std::getenv("GATE_SEED") ? std::stoull(std::getenv("GATE_SEED")) : 123456789;
    for (auto &v : z)
    {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        v = int32_t(rng % b) - int32_t(b / 2);
    }
    if (const char *name = std::getenv("GATE_PATTERN"))
    {
        const std::string pattern = name;
        const int amplitude =
            std::getenv("GATE_AMPLITUDE") ? std::stoi(std::getenv("GATE_AMPLITUDE")) : 1;
        std::fill(z.begin(), z.end(), 0);
        if (pattern == "constant")
            std::fill(z.begin(), z.end(), amplitude);
        else if (pattern == "negative")
            std::fill(z.begin(), z.end(), -amplitude);
        else if (pattern == "alternating")
            for (size_t i = 0; i < z.size(); ++i)
                z[i] = (i & 1) ? -amplitude : amplitude;
        else if (pattern == "sparse")
            for (size_t i :
                 {size_t(0), size_t(1), z.size() / 2 - 1, z.size() / 2, z.size() - 2, z.size() - 1})
                z[i] = (i & 1) ? -int(b / 2 - 1) : int(b / 2 - 1);
        else if (pattern != "zero")
            throw std::runtime_error("Unknown GATE_PATTERN");
    }

    mpz_t ref, modulus, initial;
    mpz_inits(ref, modulus, initial, nullptr);
    if (true)
    {
        radix_to_mpz(ref, z.data(), z.size(), b);
        mpz_ui_pow_ui(modulus, b, z.size());
        mpz_add_ui(modulus, modulus, 1);
        mpz_mod(ref, ref, modulus);
        mpz_set(initial, ref);
    }
    t.setZi(z.data());
    if (std::getenv("MULTIPLY"))
    {
        t.copy(1, 0);
        t.initMultiplicand(1);
    }
    const unsigned warm = std::getenv("GATE_WARMUP") ? std::stoul(std::getenv("GATE_WARMUP")) : 0;
    for (unsigned i = 0; i < warm; ++i)
        t.squareDup(dup(i));
    t.setZi(z.data());
    auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < count; ++i)
    {
        if (std::getenv("SCALAR"))
            t.squareMul(std::atoi(std::getenv("SCALAR")));
        else
            t.squareDup(dup(i));
        if (std::getenv("ROUNDTRIP") && i % 3 == 0)
        {
            std::vector<int32_t> temp(z.size());
            t.getZi(temp.data());
            t.setZi(temp.data());
        }
        if (std::getenv("COPY_RESTORE") && i % 4 == 0)
        {
            t.copy(2, 0);
            t.set(1);
            t.copy(0, 2);
        }
    }
    if (std::getenv("MULTIPLY"))
        t.mul();
    t.getZi(z.data());
    auto stop = std::chrono::steady_clock::now();
    if (true)
    {
        for (unsigned i = 0; i < count; ++i)
        {
            mpz_mul(ref, ref, ref);
            if (std::getenv("SCALAR"))
                mpz_mul_ui(ref, ref, std::atoi(std::getenv("SCALAR")));
            else if (dup(i))
                mpz_mul_2exp(ref, ref, 1);
            mpz_mod(ref, ref, modulus);
        }
        if (std::getenv("MULTIPLY"))
        {
            mpz_mul(ref, ref, initial);
            mpz_mod(ref, ref, modulus);
        }
        mpz_t actual;
        mpz_init(actual);
        radix_to_mpz(actual, z.data(), z.size(), b);
        mpz_mod(actual, actual, modulus);
        bool ok = mpz_cmp(actual, ref) == 0;
        mpz_clear(actual);
        std::cout << "GMP_REFERENCE=" << (ok ? "PASS" : "FAIL") << std::endl;
        if (!ok)
            return 2;
    }
    mpz_clears(ref, modulus, initial, nullptr);
    gint g(1 << 19, b);
    std::copy(z.begin(), z.end(), g.data());
    g.unbalance();
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned i = 0; i < z.size(); ++i)
    {
        hash ^= uint32_t(g.data()[i]);
        hash *= 1099511628211ULL;
    }
    std::cout << "RESULT base=" << b << " count=" << count
              << " us/square=" << std::chrono::duration<double>(stop - start).count() * 1e6 / count
              << " hash=" << std::hex << hash << std::dec << std::endl;
}
