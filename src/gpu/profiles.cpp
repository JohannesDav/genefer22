#include "profiles.h"
#include "checkpoint.h"
#include <gpu/embedded_kernels.h>

namespace gpu_gfn {
namespace {
std::string model_name(const std::string &name) {
    constexpr const char *prefix = "NVIDIA ";
    return name.compare(0, 7, prefix) == 0 ? name.substr(7) : name;
}
} // namespace
architecture architecture_for(unsigned major, unsigned minor) {
    if (major == 5)
        return architecture::maxwell;
    if (major == 6)
        return architecture::pascal;
    if (major == 7)
        return minor < 5 ? architecture::volta : architecture::turing;
    if (major == 8) {
        if (minor == 0) return architecture::ampere_sm80;
        return minor >= 9 ? architecture::ada : architecture::ampere;
    }
    if (major == 9)
        return architecture::hopper;
    if (major == 10 || major == 12)
        return architecture::blackwell;
    return architecture::unknown;
}
std::vector<const profile_entry *> profiles_for(const device_properties &device, unsigned exponent,
                                                uint32_t base) {
    std::vector<const profile_entry *> out;
    const bool two_primes = base && exponent >= 16 && exponent <= 22 &&
                            upstream_checkpoint(base, exponent).fields == 2;
    // Frontier calibration can prefer the three-small-prime transform. At
    // smaller bases retain upstream's two-prime arithmetic and try the packed
    // defaults instead; their field count has its own exact admission bound.
    auto eligible = [&](const profile_entry &p) {
        return !two_primes || p.kind != implementation::three_prime;
    };
    const uint64_t gib = uint64_t(1) << 30;
    for (const auto &p : embedded::device_profiles)
        if (eligible(p) && p.exponent == exponent && model_name(device.name) == model_name(p.name) &&
            device.memory_bytes > uint64_t(p.memory_gib) * gib * 9 / 10 &&
            device.memory_bytes <= (uint64_t(p.memory_gib) * 1024 + 128) * (gib / 1024)) {
            out.push_back(&p);
            break;
        }
    // An uncalibrated memory capacity can reuse the same model's settings.
    // Keep exact capacity matches first; use the nearest nominal capacity as
    // the substitute, preferring the smaller one when distances are equal.
    if (out.empty() && device.memory_bytes) {
        const profile_entry *nearest = nullptr;
        uint64_t nearest_distance = 0;
        for (const auto &p : embedded::device_profiles) {
            if (!eligible(p) || p.exponent != exponent || p.arch != device.arch ||
                model_name(device.name) != model_name(p.name))
                continue;
            const uint64_t memory = uint64_t(p.memory_gib) * gib;
            const uint64_t distance = memory > device.memory_bytes
                ? memory - device.memory_bytes : device.memory_bytes - memory;
            if (!nearest || distance < nearest_distance ||
                (distance == nearest_distance && p.memory_gib < nearest->memory_gib)) {
                nearest = &p;
                nearest_distance = distance;
            }
        }
        if (nearest)
            out.push_back(nearest);
    }
    for (const auto &p : embedded::architecture_profiles)
        if (eligible(p) && p.exponent == exponent && p.arch == device.arch) {
            out.push_back(&p);
            break;
        }
    for (const auto &p : embedded::project_profiles)
        if (eligible(p) && p.exponent == exponent) {
            out.push_back(&p);
            break;
        }
    return out;
}
bool resources_fit(const device_properties &device, const kernel_plan &plan, unsigned fields) {
    if (!plan.outer || plan.length % plan.outer)
        return false;
    const uint64_t inner = plan.length / plan.outer;
    const uint64_t outer_memory = uint64_t(plan.outer) * plan.chunk * 16;
    const uint64_t inner_memory = inner * 4 * plan.inner_blocks;
    const uint64_t outer_threads = uint64_t(plan.outer) / 4 * plan.chunk;
    const uint64_t inner_threads = inner / 16 * plan.inner_blocks;
    const uint64_t packed_memory = uint64_t(fields) * plan.length * (8 + plan.root_bits / 16) +
                                   uint64_t(plan.pack) * plan.length * 4 + uint64_t(plan.length) * 4 + 4;
    return outer_memory <= device.local_memory_bytes && inner_memory <= device.local_memory_bytes &&
           outer_threads <= device.max_threads && inner_threads <= device.max_threads &&
           packed_memory < device.memory_bytes;
}
} // namespace gpu_gfn
