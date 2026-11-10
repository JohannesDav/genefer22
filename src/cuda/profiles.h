#pragma once
#include "kernel.h"
#include <string>
#include <vector>

namespace cuda_gfn {
enum class architecture { unknown, maxwell, pascal, volta, turing, ampere, ampere_sm80, ada, hopper, blackwell };
enum class implementation { packed, three_prime, original_cuda, original_opencl };
struct profile_entry {
    const char *name;
    unsigned memory_gib, exponent;
    architecture arch;
    implementation kind;
    kernel_plan plan;
    int shared_memory_preference;
};
struct device_properties {
    std::string name;
    architecture arch = architecture::unknown;
    uint64_t memory_bytes = 0, local_memory_bytes = 0;
    size_t max_threads = 0;
};
architecture architecture_for(unsigned major, unsigned minor);
std::vector<const profile_entry *> profiles_for(const device_properties &device, unsigned exponent,
                                                uint32_t base = 0);
bool resources_fit(const device_properties &device, const kernel_plan &plan, unsigned fields);
} // namespace cuda_gfn
