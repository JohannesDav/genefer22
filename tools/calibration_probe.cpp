#include "ocl.h"
#include "reference.h"
#include "transform.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
int main(int argc, char **argv) {
    try {
        if (argc < 6)
            throw std::runtime_error("n base count check|time pattern [gerbicz interval]");
        unsigned n = std::stoul(argv[1]), b = std::stoul(argv[2]), count = std::stoul(argv[3]);
        bool check = std::string(argv[4]) == "check";
        std::string pattern = argv[5];
        unsigned interval = argc > 6 ? std::stoul(argv[6]) : 0;
        if (count == 0 || n < 16 || n > 22 || b < 1026 || b > 2000000000 || b % 2 || !(b & (b - 1)))
            throw std::runtime_error("Unsupported calibration input");
        const size_t size = size_t(1) << n;
        gint digits(size, b);
        uint64_t state = 123456789;
        for (size_t i = 0; i < size; ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            digits.data()[i] = int64_t(state % b) - b / 2;
            if (pattern == "alternating")
                digits.data()[i] = (i & 1) ? -int(b / 2) : int(b / 2);
            else if (pattern == "constant")
                digits.data()[i] = b / 2;
            else if (pattern == "sparse")
                digits.data()[i] = (i == 0 || i == size / 2 || i == size - 1) ? int(b / 2) : 0;
        }
        digits.balance();
        mpz_t expected, modulus, actual;
        mpz_inits(expected, modulus, actual, nullptr);
        if (check) {
            radix_to_mpz(expected, digits.data(), size, b);
            mpz_ui_pow_ui(modulus, b, size);
            mpz_add_ui(modulus, modulus, 1);
            mpz_mod(expected, expected, modulus);
        }
        size_t device =
            std::getenv("GFN_CALIB_DEVICE_INDEX") ? std::stoul(std::getenv("GFN_CALIB_DEVICE_INDEX")) : 0;
        std::unique_ptr<transform> t(transform::create_gpu(b, n, false, device, 3, nullptr, nullptr,
                                                           std::getenv("GFN_CALIB_VERBOSE") != nullptr));
        platform devices;
        cl_device_id gpu = devices.getDevice(device);
        char device_name[1024]{};
        cl_ulong memory = 0;
        cl_uint major = 0, minor = 0;
        clGetDeviceInfo(gpu, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        clGetDeviceInfo(gpu, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memory), &memory, nullptr);
        clGetDeviceInfo(gpu, 0x4000, sizeof(major), &major, nullptr);
        clGetDeviceInfo(gpu, 0x4001, sizeof(minor), &minor, nullptr);

        auto doubled = [](unsigned i) {
            uint32_t z = (i + 1) * 2654435761u;
            z ^= z >> 13;
            z *= 2246822519u;
            return bool((z >> 17) & 1);
        };
        std::string operation =
            std::getenv("GFN_CALIB_OPERATION") ? std::getenv("GFN_CALIB_OPERATION") : "normal";
        auto initialize = [&] {
            if (operation == "set")
                t->set(b + 2);
            else
                t->setInt(digits);
        };
        if (check && operation == "set")
            mpz_set_ui(expected, b + 2);
        initialize();
        if (check) {
            gint recovered(size, b);
            t->getInt(recovered);
            radix_to_mpz(actual, recovered.data(), size, b);
            mpz_mod(actual, actual, modulus);
            if (mpz_cmp(expected, actual))
                throw std::runtime_error("Initial digit round-trip mismatch");
        }
        for (unsigned i = 0; i < (check ? 0 : 2048); ++i)
            t->squareDup(bool(((i / 8) >> (i % 8)) & 1));
        initialize();
        t->copy(1, 0);
        t->synchronizeForCalibration();
        auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < count; ++i) {
            bool dup = doubled(i);
            if (operation == "phase" && i == 2)
                t->setProofGeneration(true);
            if (operation == "phase" && i == 4)
                t->setProofGeneration(false);
            if (operation == "scalar")
                t->squareMul(2147483647);
            else
                t->squareDup(dup);
            if (interval && i % interval == 0) {
                t->copy(2, 0);
                t->mul(1);
                t->copy(1, 0);
                t->copy(0, 2);
            }
        }
        if (operation == "product") {
            t->initMultiplicand(0);
            t->mul();
        }
        if (operation == "checkpoint") {
            const char *path = std::getenv("GFN_CALIB_CONTEXT");
            if (!path)
                throw std::runtime_error("Checkpoint test needs a filename");
            t->copy(2, 0);
            {
                file saved(path, "wb", true);
                t->saveContext(saved, 3);
                saved.write_crc32();
            }
            t->set(2);
            {
                file saved(path, "rb", true);
                if (!t->readContext(saved, 3) || !saved.check_crc32())
                    throw std::runtime_error("Checkpoint round-trip failed");
            }
        }
        t->synchronizeForCalibration();
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        t->getInt(digits);
        // Backends can return different redundant digit representations of
        // the same residue. Hash the canonical base-b digits instead.
        digits.unbalance();
        uint64_t hash = 0;
        for (size_t i = 0; i < size; ++i)
            hash = hash * 0x9e3779b97f4a7c15ull + uint32_t(digits.data()[i]);
        if (check) {
            for (unsigned i = 0; i < count; ++i) {
                mpz_mul(expected, expected, expected);
                if (operation == "scalar")
                    mpz_mul_ui(expected, expected, 2147483647);
                else if (doubled(i))
                    mpz_mul_2exp(expected, expected, 1);
                mpz_mod(expected, expected, modulus);
            }
            if (operation == "product") {
                mpz_mul(expected, expected, expected);
                mpz_mod(expected, expected, modulus);
            }
            radix_to_mpz(actual, digits.data(), size, b);
            mpz_mod(actual, actual, modulus);
            if (mpz_cmp(expected, actual))
                throw std::runtime_error("GMP mismatch");
        }
        std::cout << "RESULT {\"device_name\":" << std::quoted(device_name) << ",\"memory_bytes\":" << memory
                  << ",\"compute_major\":" << major << ",\"compute_minor\":" << minor << ",\"n\":" << n
                  << ",\"base\":" << b << ",\"count\":" << count
                  << ",\"ordinary_fields\":" << t->getOrdinaryPrimeCount()
                  << ",\"check\":" << (check ? "true" : "false") << ",\"us\":" << seconds * 1e6 / count
                  << ",\"hash\":\"" << std::hex << hash << "\"}" << std::endl;
        mpz_clears(expected, modulus, actual, nullptr);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
