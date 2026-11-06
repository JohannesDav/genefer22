// Recompute the final proof segment with the original three-prime transform.
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "transformGPU.h"

static void require_canonical(const gint &value)
{
    bool minus_one = value.data()[0] == -1;
    for (size_t i = 1; i < value.getSize() && minus_one; ++i)
        minus_one = value.data()[i] == 0;
    if (minus_one) return;
    for (size_t i = 0; i < value.getSize(); ++i)
        if (value.data()[i] < 0 || uint32_t(value.data()[i]) >= value.getBase())
            throw std::runtime_error("Noncanonical integer file");
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 4)
            throw std::runtime_error("Usage: genefer-tail-check BASE CHECKPOINT1 PROOF");
        const auto requested = std::stoull(argv[1]);
        if (requested < 1024 || requested > 528220156 || requested % 2 ||
            !(requested & (requested - 1)))
            throw std::runtime_error("Unsupported GFN-19 base");
        const uint32_t base = uint32_t(requested);
        constexpr size_t size = 1u << 19;
        gint start(size, base), target(size, base);
        int version = 0, depth = 0;
        if (std::filesystem::file_size(argv[2]) != 8 + 4 * size + 4)
            throw std::runtime_error("Checkpoint length");
        {
            file input(argv[2], "rb", true);
            start.read(input);
            require_canonical(start);
            if (!input.check_crc32()) throw std::runtime_error("Checkpoint CRC");
        }
        {
            file input(argv[3], "rb", true);
            if (!input.read(reinterpret_cast<char *>(&version), sizeof(version)) ||
                !input.read(reinterpret_cast<char *>(&depth), sizeof(depth)) ||
                version != 1 || depth < 1 || depth > 20)
                throw std::runtime_error("Proof format");
            if (std::filesystem::file_size(argv[3]) != 8 + (depth + 1) * (8 + 4 * size) + 4)
                throw std::runtime_error("Proof length");
            target.read(input);
            require_canonical(target);
            gint temporary(size, base);
            for (int i = 0; i < depth; ++i)
            {
                temporary.read(input);
                require_canonical(temporary);
            }
            if (!input.check_crc32()) throw std::runtime_error("Proof CRC");
        }
        mpz_t exponent;
        mpz_init(exponent);
        mpz_ui_pow_ui(exponent, base, size);
        const size_t bits = mpz_sizeinbase(exponent, 2);
        const size_t steps = ((bits - 1) >> depth) + 1;
        transformGPUs<3, false> arithmetic(base, 19, false, 0, 3, 0, 0, false);
        arithmetic.setInt(start);
        for (size_t i = steps; i-- > 0;)
            arithmetic.squareDup(mpz_tstbit(exponent, i) != 0);
        mpz_clear(exponent);
        gint computed(size, base);
        arithmetic.getInt(computed);
        computed.unbalance();
        target.unbalance();
        const bool equal = std::equal(computed.data(), computed.data() + size, target.data());
        std::cout << "TAIL_MATCH=" << (equal ? "PASS" : "FAIL")
                  << " steps=" << steps << " exact_digit_comparison=1\n";
        return equal ? 0 : 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 3;
    }
}
