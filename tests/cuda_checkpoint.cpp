// Verify checkpoint exchange with the original OpenCL implementation.
#include "ocl.h"
#include "transform.h"
#include <array>
#include <cuda/embedded_kernels.h>
#include <filesystem>
#include <iostream>
#include <memory>

transform *create_general_small3(uint32_t, uint32_t, bool, size_t, size_t, cl_platform_id, cl_device_id, bool,
                                 unsigned, const std::string &);

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Usage: cuda_checkpoint EMPTY_OUTPUT_DIRECTORY");
        const std::filesystem::path directory(argv[1]);
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("Output directory must not exist");
        constexpr unsigned n = 16;
        constexpr size_t size = size_t(1) << n;
        // Exercise upstream's signed two-prime, unsigned two-prime and
        // three-prime checkpoint formats using both root-table widths.
        for (const uint32_t base : {2046u, 10000000u, 620000000u}) {
            for (const unsigned roots : {32u, 64u}) {
                std::unique_ptr<transform> original(
                    transform::create_original_gpu(base, n, false, 0, 3, nullptr, nullptr, false));
                std::unique_ptr<transform> optimized(create_general_small3(
                    base, n, false, 0, 3, nullptr, nullptr, false, roots, cuda_gfn::embedded::three_prime));
                if (!optimized)
                    throw std::runtime_error("Small-prime case unexpectedly rejected");
                gint digits(size, base), expected(size, base), actual(size, base);
                for (size_t i = 0; i < size; ++i)
                    digits.data()[i] = (i % 3 == 0) ? -int(base / 2) : int((i * 31337u) % (base / 2));
                digits.balance();
                const auto path = directory / (std::to_string(base) + "-" + std::to_string(roots) + ".ctx");
                for (const bool from_original : {true, false}) {
                    transform *from = from_original ? original.get() : optimized.get();
                    transform *to = from_original ? optimized.get() : original.get();
                    from->setInt(digits);
                    from->copy(1, 0);
                    from->squareDup(true);
                    from->copy(2, 0);
                    from->squareDup(false);
                    {
                        file context(path.string(), "wb", true);
                        from->saveContext(context, 3);
                        context.write_crc32();
                    }
                    {
                        file context(path.string(), "rb", true);
                        if (!to->readContext(context, 3) || !context.check_crc32())
                            throw std::runtime_error("Cross-backend checkpoint was rejected");
                    }
                    for (const size_t reg : {size_t(0), size_t(1), size_t(2)}) {
                        if (reg) {
                            from->copy(0, reg);
                            to->copy(0, reg);
                        }
                        from->getInt(expected);
                        to->getInt(actual);
                        expected.unbalance();
                        actual.unbalance();
                        if (!std::equal(expected.data(), expected.data() + size, actual.data()))
                            throw std::runtime_error("Cross-backend register mismatch");
                        from->squareDup(true);
                        to->squareDup(true);
                        from->getInt(expected);
                        to->getInt(actual);
                        expected.unbalance();
                        actual.unbalance();
                        if (!std::equal(expected.data(), expected.data() + size, actual.data()))
                            throw std::runtime_error("Resumed arithmetic mismatch");
                    }
                }
            }
        }
        std::cout << "PASS: cross-backend checkpoint formats and resumed arithmetic\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
