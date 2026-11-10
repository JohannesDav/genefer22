#pragma once
#include <cstdint>
#include <string>

namespace cuda_gfn {
struct kernel_plan {
    unsigned length = 65536, pack = 8, outer = 64, chunk = 8, inner_blocks = 1;
    unsigned root_bits = 32, ratio = 3, fields = 0;
    bool matrix_mad = false, inline_converter = false;
};
struct generated_kernel {
    std::string source;
    unsigned fields = 0;
};
generated_kernel generate_kernel(uint32_t base, unsigned exponent, const kernel_plan &plan);
} // namespace cuda_gfn
