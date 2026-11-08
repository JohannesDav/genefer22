#pragma once
#if defined(GENEFER_EXACT_TWO)
#include "../transformGPU.h"
#else
#include "transform.h"
#endif
#include <array>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <regex>
#include <vector>

// The packed and ordinary transforms own independent OpenCL contexts. The
// adapter exchanges ordinary digits through host memory at representation changes.
class packed_device {
    static constexpr unsigned n = 65536;
    unsigned outer_threads_ = 256, inner_threads_ = 64;
    unsigned fields_ = 0, root_bits_ = 32, root_words_ = 32768, current_ = 0, queued_ = 0;
    cl_context context_{};
    cl_command_queue queue_{};
    cl_program program_{};
    std::array<cl_kernel, 3> transforms_{};
    std::array<cl_kernel, 2> converters_{};
    cl_kernel encode_{}, decode_{}, repair_{};
    cl_mem state_[2]{}, roots_{}, digits_{}, carries_{}, error_{};

    static void check(cl_int code, const char *where) {
        if (code != CL_SUCCESS)
            throw std::runtime_error(std::string("Packed OpenCL ") + where + ": " +
                                     std::to_string(code));
    }
    static uint32_t power(uint32_t x, uint64_t k, uint32_t p) {
        uint64_t out = 1;
        for (; k; k >>= 1, x = uint64_t(x) * x % p)
            if (k & 1) out = out * x % p;
        return uint32_t(out);
    }
    static unsigned reverse(unsigned j, unsigned size) {
        unsigned value = 0;
        for (; size > 1; size >>= 1, j >>= 1) value = 2 * value + (j & 1);
        return value;
    }
    static unsigned literal(const std::string &source, const std::string &name) {
        std::smatch m;
        if (!std::regex_search(source, m, std::regex("#define\\s+" + name + "\\s+(\\d+)")))
            throw std::runtime_error("Packed kernel has no " + name);
        return std::stoul(m[1]);
    }
    cl_mem buffer(size_t bytes) {
        cl_int code;
        cl_mem result = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &code);
        check(code, "allocate buffer");
        return result;
    }
    cl_kernel kernel(const char *name) {
        cl_int code;
        cl_kernel result = clCreateKernel(program_, name, &code);
        check(code, name);
        return result;
    }
    void write(cl_mem destination, const void *source, size_t bytes) {
        check(clEnqueueWriteBuffer(queue_, destination, CL_TRUE, 0, bytes, source, 0,
                                   nullptr, nullptr), "write buffer");
    }
    void launch(cl_kernel fn, std::initializer_list<cl_mem> arguments,
                size_t global, size_t local) {
        if (!local || global % local) throw std::runtime_error("Invalid packed OpenCL geometry");
        cl_uint index = 0;
        for (cl_mem value : arguments)
            check(clSetKernelArg(fn, index++, sizeof(value), &value), "set kernel argument");
        check(clEnqueueNDRangeKernel(queue_, fn, 1, nullptr, &global, &local, 0,
                                     nullptr, nullptr), "enqueue kernel");
    }
    void checkFlags() {
        uint32_t flags = 0;
        check(clEnqueueReadBuffer(queue_, error_, CL_TRUE, 0, sizeof(flags), &flags, 0,
                                  nullptr, nullptr), "read conversion flags");
        if (flags) throw std::runtime_error("Packed digit conversion invariant failed: " +
                                           std::to_string(flags));
    }
    void release() noexcept {
        if (queue_) clFinish(queue_);
        for (auto fn : transforms_) if (fn) clReleaseKernel(fn);
        for (auto fn : converters_) if (fn) clReleaseKernel(fn);
        for (auto fn : {encode_, decode_, repair_}) if (fn) clReleaseKernel(fn);
        for (auto mem : {state_[0], state_[1], roots_, digits_, carries_, error_})
            if (mem) clReleaseMemObject(mem);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

  public:
    packed_device(unsigned base, cl_device_id device, const std::string &path, bool verbose) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot read packed kernel source");
        const std::string source(std::istreambuf_iterator<char>(file), {});
        fields_ = literal(source, "PG_K");
        root_bits_ = literal(source, "PG_ROOT_BITS");
        root_words_ = literal(source, "PG_ROOT_WORDS");
        if (source.find("#define PG_OUTER_THREADS") != std::string::npos)
            outer_threads_ = literal(source, "PG_OUTER_THREADS");
        if (source.find("#define PG_INNER_THREADS") != std::string::npos)
            inner_threads_ = literal(source, "PG_INNER_THREADS");
        if (base != literal(source, "PG_BASE") || fields_ < 7 || fields_ > 17 ||
            (root_bits_ != 32 && root_bits_ != 64) || root_words_ != n / 2)
            throw std::runtime_error("Packed kernel does not match the candidate base");
        for (unsigned threads : {outer_threads_, inner_threads_})
            if (!threads || threads > 1024 || (threads & (threads - 1)) || 4096 % threads)
                throw std::runtime_error("Invalid packed transform thread count");
        try {
            cl_int code;
            context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &code);
            check(code, "create context");
            queue_ = clCreateCommandQueue(context_, device, 0, &code);
            check(code, "create in-order queue");
            const char *text = source.data();
            const size_t length = source.size();
            program_ = clCreateProgramWithSource(context_, 1, &text, &length, &code);
            check(code, "create program");
            code = clBuildProgram(program_, 1, &device, "", nullptr, nullptr);
            if (code != CL_SUCCESS) {
                size_t length = 0;
                clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
                std::vector<char> log(length + 1);
                clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
                throw std::runtime_error(std::string("Packed OpenCL compilation failed: ") + log.data());
            }
            const char *transform_names[] = {"packed_forward", "packed_square1024", "packed_backward"};
            for (unsigned i = 0; i < 3; ++i) transforms_[i] = kernel(transform_names[i]);
            converters_[0] = kernel("packed_convert_dup0");
            converters_[1] = kernel("packed_convert_dup1");
            encode_ = kernel("packed_encode");
            decode_ = kernel("packed_decode");
            repair_ = kernel("packed_repair");
            for (auto &value : state_) value = buffer(size_t(fields_) * n * 4);
            roots_ = buffer(size_t(fields_) * root_words_ * (root_bits_ / 8));
            digits_ = buffer(8 * n * 4);
            carries_ = buffer(n * 4);
            error_ = buffer(4);
            std::vector<uint32_t> roots32(root_bits_ == 32 ? size_t(fields_) * root_words_ : 0);
            std::vector<uint64_t> roots64(root_bits_ == 64 ? size_t(fields_) * root_words_ : 0);
            for (unsigned field = 0; field < fields_; ++field) {
                uint32_t p = literal(source, "PG_PRIME" + std::to_string(field)),
                         generator = literal(source, "PG_GENERATOR" + std::to_string(field));
                if (uint64_t(p) * 4 >= (uint64_t(1) << 32) || (p - 1) % (2 * n) ||
                    power(generator, (p - 1) / 2, p) == 1)
                    throw std::runtime_error("Invalid packed transform field");
                for (unsigned s = 1; s < n / 2; s *= 2) {
                    uint32_t root = power(generator, (p - 1) / (4 * s), p);
                    for (unsigned j = 0; j < s; ++j) {
                        uint32_t w = power(root, reverse(j, 2 * s) + 1, p),
                                 q = (uint64_t(w) << 32) / p;
                        if (root_bits_ == 32) {
                            roots32[size_t(field) * root_words_ + s + j] = q;
                        } else {
                            uint64_t word = uint64_t(w) | (uint64_t(q) << 32);
                            roots64[size_t(field) * root_words_ + s + j] = word;
                        }
                    }
                }
            }
            if (root_bits_ == 32) write(roots_, roots32.data(), roots32.size() * 4);
            else write(roots_, roots64.data(), roots64.size() * 8);
            if (verbose) pio::print("GFN-19 exact packed-digit OpenCL square (" +
                                    std::to_string(fields_) + " prime channels)\n");
        } catch (...) { release(); throw; }
    }
    packed_device(const packed_device &) = delete;
    packed_device &operator=(const packed_device &) = delete;
    ~packed_device() { release(); }
    void synchronize() {
        check(clFinish(queue_), "finish queue");
        queued_ = 0;
    }
    void setDigits(const int32_t *input) {
        synchronize();
        write(digits_, input, 8 * n * 4);
        const uint32_t zero = 0;
        write(error_, &zero, sizeof(zero));
        current_ = 0;
        launch(encode_, {digits_, state_[0], error_}, fields_ * n, 128);
        checkFlags();
    }
    void getDigits(int32_t *output) {
        synchronize();
        const uint32_t zero = 0;
        write(error_, &zero, sizeof(zero));
        launch(decode_, {state_[current_], digits_, carries_, error_}, n, 128);
        launch(repair_, {digits_, carries_, error_}, n, 128);
        checkFlags();
        check(clEnqueueReadBuffer(queue_, digits_, CL_TRUE, 0, 8 * n * 4, output, 0,
                                  nullptr, nullptr), "read digits");
    }
    void square(bool doubled) {
        const cl_mem input = state_[current_], output = state_[current_ ^ 1];
        const size_t global = size_t(fields_) * n / 16;
        launch(transforms_[0], {input, roots_}, global, outer_threads_);
        launch(transforms_[1], {input, roots_}, global, inner_threads_);
        launch(transforms_[2], {input, roots_}, global, outer_threads_);
        launch(converters_[doubled ? 1 : 0], {input, output}, ((n + 29) / 30 + 3) / 4 * 128, 128);
        current_ ^= 1;
        // Submit periodically without waiting, so host enqueue and GPU execution overlap.
        if (++queued_ == 8) {
            check(clFlush(queue_), "flush queue");
            queued_ = 0;
        }
    }
    size_t memory() const {
        return size_t(fields_) * n * 8 + size_t(fields_) * root_words_ * (root_bits_ / 8) +
               (9 * n + 1) * 4;
    }
};
