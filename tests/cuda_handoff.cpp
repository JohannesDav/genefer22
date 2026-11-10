#include "cuda/checkpoint.h"
#include "cuda/kernel.h"
#include <cuda/embedded_kernels.h>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    using namespace cuda_gfn;
    for (unsigned n : {16u, 18u}) {
        const std::array<uint32_t, 2> limits = n == 16
            ? std::array<uint32_t, 2>{5862084, 11421892}
            : std::array<uint32_t, 2>{2931042, 5710946};
        for (uint32_t b : limits) {
            for (unsigned pack : {1u, 2u, 4u}) {
                kernel_plan plan;
                plan.pack = pack;
                plan.length = (1u << n) / pack;
                for (int offset : {-2, 0, 2}) {
                    const uint32_t base = b + offset;
                    const bool expected = offset != 0 || pack >= 4;
                    if (packed_handoff_fits(base, n, pack) != expected)
                        throw std::runtime_error("Unexpected boundary admission");
                    try {
                        const auto kernel = generate_kernel(base, n, plan);
                        if (!expected || kernel.source.empty())
                            throw std::runtime_error("Unsafe layout admitted");
                    } catch (const std::invalid_argument &error) {
                        if (expected || std::string(error.what()) !=
                                            "Packed digits exceed ordinary two-prime bound")
                            throw;
                    }
                }
            }
        }
    }
    const uint32_t bases[] = {620000000, 520000000, 100000000, 30000000,
                              8000000, 2800000, 660000};
    for (const auto &profile : embedded::device_profiles)
        if (profile.kind == implementation::packed &&
            !packed_handoff_fits(bases[profile.exponent - 16], profile.exponent, profile.plan.pack))
            throw std::runtime_error("Benchmark profile admission changed");
    std::cout << "PASS: 36 packed handoff boundary cases and all measured frontier profiles\n";
}
