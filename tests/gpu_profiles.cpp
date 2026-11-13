#include "gpu/profiles.h"
#include "gpu/checkpoint.h"
#include <gpu/embedded_kernels.h>
#include <iostream>
#include <stdexcept>

static void expect(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() {
    using namespace gpu_gfn;
    try {
        const uint64_t gib = uint64_t(1) << 30;
        device_properties known;
        known.name = "NVIDIA GeForce RTX 4060 Ti";
        known.memory_bytes = 16 * gib - 512 * 1024 * 1024;
        known.arch = architecture::ada;
        auto choices = profiles_for(known, 19);
        expect(choices.size() == 3, "Tested device must precede architecture and GFN defaults");
        expect(choices.front()->memory_gib == 16, "Wrong tested memory variant");
        known.name = "GeForce RTX 4060 Ti";
        choices = profiles_for(known, 19);
        expect(choices.front()->memory_gib == 16, "Optional NVIDIA name prefix must not change the profile");
        known.memory_bytes = 8 * gib;
        choices = profiles_for(known, 19);
        expect(choices.front()->memory_gib == 8, "8GB device must use its own tested settings");
        unsigned three_prime_profiles = 0;
        for (const auto &entry : embedded::device_profiles) {
            if (entry.kind != implementation::three_prime) continue;
            known.name = entry.name;
            known.memory_bytes = uint64_t(entry.memory_gib) * gib;
            known.arch = entry.arch;
            choices = profiles_for(known, entry.exponent);
            expect(!choices.empty() && choices.front() == &entry,
                   "Model routing must retain the calibrated three-prime profile");
            choices = profiles_for(known, entry.exponent, 2046);
            expect(!choices.empty(), "Small bases must retain an arithmetic profile");
            for (const auto *profile : choices)
                expect(profile->kind != implementation::three_prime,
                       "A three-prime profile must not override upstream's two-prime range");
            ++three_prime_profiles;
        }
        expect(three_prime_profiles > 0, "Three-prime routing cases must be exercised");
        known.name = "NVIDIA GeForce RTX 5080";
        known.memory_bytes = 15 * gib + gib / 2;
        known.arch = architecture::blackwell;
        choices = profiles_for(known, 21);
        expect(choices.front()->memory_gib == 16, "Reserved VRAM must retain the nominal 16GB profile");
        known.name = "NVIDIA GeForce RTX 4060 Ti";
        known.arch = architecture::ada;
        known.memory_bytes = 24 * gib;
        choices = profiles_for(known, 19);
        expect(choices.size() == 3 && choices.front()->memory_gib == 16,
               "Untested capacity must use the nearest capacity of the same model");
        known.memory_bytes = 12 * gib;
        choices = profiles_for(known, 19);
        expect(choices.front()->memory_gib == 8,
               "Equal memory distances must prefer the smaller calibrated capacity");
        known.memory_bytes = 4 * gib;
        choices = profiles_for(known, 19);
        expect(choices.front()->memory_gib == 8,
               "A smaller untested capacity must retain a same-model candidate");
        known.arch = architecture::nvidia_pascal;
        choices = profiles_for(known, 19);
        expect(choices.size() == 2 && choices.front()->memory_gib == 0,
               "A same-name substitute must not cross architectures");
        known.arch = architecture::ada;
        known.memory_bytes = 0;
        choices = profiles_for(known, 19);
        expect(choices.size() == 2 && choices.front()->memory_gib == 0,
               "Missing memory information must not select a substitute capacity");
        known.name = "Future GPU";
        known.arch = architecture::unknown;
        choices = profiles_for(known, 19);
        expect(choices.size() == 1, "Unknown architecture must use the GFN default");
        known.arch = architecture::hopper;
        choices = profiles_for(known, 22);
        expect(choices.size() == 2 && choices.front()->arch == architecture::hopper,
               "Untested Hopper models must inherit the measured architecture default");
        expect(profiles_for(known, 23).empty(), "Unsupported GFN must have no optimized profile");
        expect(architecture_for(7, 0) == architecture::volta, "Volta classification");
        expect(architecture_for(7, 5) == architecture::turing, "Turing classification");
        expect(architecture_for(8, 6) == architecture::ampere, "Ampere classification");
        expect(architecture_for(8, 0) == architecture::ampere_sm80, "A100 architecture classification");
        expect(architecture_for(8, 9) == architecture::ada, "Ada classification");
        expect(architecture_for(9, 0) == architecture::hopper, "Hopper classification");
        expect(architecture_for(99, 0) == architecture::unknown, "Future architecture classification");
        const uint32_t frontiers[] = {620000000, 520000000, 100000000, 30000000,
                                      8000000, 2800000, 660000};
        for (const auto &profile : embedded::device_profiles) {
            device_properties device;
            device.name = profile.name;
            device.memory_bytes = uint64_t(profile.memory_gib) * gib;
            device.arch = profile.arch;
            const auto previous = profiles_for(device, profile.exponent);
            const auto frontier = profiles_for(device, profile.exponent,
                                                frontiers[profile.exponent - 16]);
            expect(previous.front() == frontier.front(),
                   "Small-base routing must not change a measured frontier profile");
            for (unsigned n = 16; n <= 22; ++n) {
                const auto calibrated = profiles_for(device, n);
                expect(!calibrated.empty() &&
                           std::string(calibrated.front()->name) == profile.name &&
                           calibrated.front()->memory_gib == profile.memory_gib,
                       "Every calibrated model must cover GFN-16 through GFN-22");
                for (uint32_t base : {1026u, 2046u, 100000u, 1000000u, 10000000u}) {
                    if (upstream_checkpoint(base, n).fields != 2) continue;
                    for (const auto *choice : profiles_for(device, n, base))
                        expect(choice->kind != implementation::three_prime,
                               "Two-prime admission must apply to every model and GFN size");
                    device.memory_bytes = 256 * gib;
                    for (const auto *choice : profiles_for(device, n, base))
                        expect(choice->kind != implementation::three_prime,
                               "Same-model substitutes must respect two-prime admission");
                    device.memory_bytes = uint64_t(profile.memory_gib) * gib;
                }
            }
        }
        std::cout << "PASS: tested model, memory variant and unfamiliar-device routing\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
