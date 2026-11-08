#pragma once
#if defined(GENEFER_EXACT_TWO)
#include "../transformGPU.h"
#else
#include "transform.h"
#endif
#include <array>
#include <cuda.h>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <vector>

class packed_device {
    static constexpr unsigned n = 65536;
    unsigned outer_threads_ = 256, inner_threads_ = 64;
    unsigned base_, fields_, root_bits_ = 32, root_words_ = 32768, current_ = 0, pending_ = 0,
                             mask_ = 0, start_ = 0;
    CUdevice device_{};
    CUcontext context_{};
    CUstream stream_{};
    CUmodule module_{};
    CUdeviceptr state_[2]{}, roots_{}, digits_{}, carries_{}, error_{};
    std::array<CUfunction, 3> transforms_{};
    std::array<CUfunction, 2> converters_{};
    CUfunction encode_{}, decode_{}, repair_{};
    std::map<unsigned, CUgraphExec> graphs_;

    static uint32_t power(uint32_t x, uint64_t k, uint32_t p) {
        uint64_t out = 1;
        for (; k; k >>= 1, x = uint64_t(x) * x % p)
            if (k & 1)
                out = out * x % p;
        return uint32_t(out);
    }
    static unsigned reverse(unsigned j, unsigned size) {
        unsigned value = 0;
        for (; size > 1; size >>= 1, j >>= 1)
            value = 2 * value + (j & 1);
        return value;
    }
    static unsigned literal(const std::string &source, const std::string &name) {
        std::smatch m;
        if (!std::regex_search(source, m, std::regex("#define\\s+" + name + "\\s+(\\d+)")))
            throw std::runtime_error("Packed kernel has no " + name);
        return std::stoul(m[1]);
    }
    static void checkCL(cl_int code, const char *where = "OpenCL") {
        if (code != CL_SUCCESS)
            throw std::runtime_error(std::string(where) + ": " + std::to_string(code));
    }
    static std::string compile(cl_device_id device, const std::string &source) {
        cl_int error;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        checkCL(error, "packed compiler context");
        cl_program program = nullptr;
        try {
            const char *data = source.data();
            const size_t source_length = source.size();
            program = clCreateProgramWithSource(context, 1, &data, &source_length, &error);
            checkCL(error, "packed program");
            error = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
            if (error != CL_SUCCESS) {
                size_t length = 0;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
                std::vector<char> log(length + 1);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log.data(),
                                      nullptr);
                throw std::runtime_error(std::string("Packed kernel build: ") + log.data());
            }
            size_t length = 0;
            checkCL(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(length), &length,
                                     nullptr));
            std::string binary(length, '\0');
            char *target = binary.data();
            checkCL(
                clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(target), &target, nullptr));
            while (!binary.empty() && binary.back() == 0)
                binary.pop_back();
            clReleaseProgram(program);
            clReleaseContext(context);
            program = nullptr;
            context = nullptr;
            size_t at = 0;
            while ((at = binary.find("%envreg3", at)) != std::string::npos)
                binary.replace(at, 8, "0");
            if (binary.find("%envreg") != std::string::npos ||
                binary.find(".entry ") == std::string::npos)
                throw std::runtime_error("Unsupported packed PTX");
            return binary;
        } catch (...) {
            if (program)
                clReleaseProgram(program);
            if (context)
                clReleaseContext(context);
            throw;
        }
    }
    void launch(CUfunction function, unsigned blocks, unsigned threads, void **args) {
        three_cuda_check(
            cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream_, args, nullptr));
    }
    void addNode(CUgraph graph, CUgraphNode &previous, CUfunction fn, unsigned blocks,
                 unsigned threads, void **args) {
        CUDA_KERNEL_NODE_PARAMS p{};
        p.func = fn;
        p.gridDimX = blocks;
        p.gridDimY = p.gridDimZ = 1;
        p.blockDimX = threads;
        p.blockDimY = p.blockDimZ = 1;
        p.kernelParams = args;
        CUgraphNode node;
        three_cuda_check(cuGraphAddKernelNode(&node, graph, previous ? &previous : nullptr,
                                              previous ? 1 : 0, &p));
        previous = node;
    }
    void flush() {
        if (!pending_)
            return;
        if (pending_ == 8) {
            const unsigned key = (start_ << 8) | mask_;
            auto &exec = graphs_[key];
            if (!exec) {
                CUgraph graph;
                three_cuda_check(cuGraphCreate(&graph, 0));
                CUgraphNode previous{};
                for (unsigned i = 0; i < 8; ++i) {
                    unsigned index = start_ ^ (i & 1), bit = (mask_ >> i) & 1;
                    CUdeviceptr input = state_[index], output = state_[index ^ 1], roots = roots_;
                    void *transformArgs[] = {&input, &roots};
                    void *convertArgs[] = {&input, &output};
                    addNode(graph, previous, transforms_[0], fields_ * n / 16 / outer_threads_,
                            outer_threads_, transformArgs);
                    addNode(graph, previous, transforms_[1], fields_ * n / 16 / inner_threads_,
                            inner_threads_, transformArgs);
                    addNode(graph, previous, transforms_[2], fields_ * n / 16 / outer_threads_,
                            outer_threads_, transformArgs);
                    addNode(graph, previous, converters_[bit], ((n + 29) / 30 + 3) / 4, 128,
                            convertArgs);
                }
                three_cuda_check(cuGraphInstantiate(&exec, graph, 0));
                three_cuda_check(cuGraphDestroy(graph));
            }
            three_cuda_check(cuGraphLaunch(exec, stream_));
        } else
            for (unsigned i = 0; i < pending_; ++i) {
                unsigned index = start_ ^ (i & 1), bit = (mask_ >> i) & 1;
                CUdeviceptr input = state_[index], output = state_[index ^ 1], roots = roots_;
                void *args[] = {&input, &roots};
                void *convert[] = {&input, &output};
                launch(transforms_[0], fields_ * n / 16 / outer_threads_, outer_threads_, args);
                launch(transforms_[1], fields_ * n / 16 / inner_threads_, inner_threads_, args);
                launch(transforms_[2], fields_ * n / 16 / outer_threads_, outer_threads_, args);
                launch(converters_[bit], ((n + 29) / 30 + 3) / 4, 128, convert);
            }
        pending_ = 0;
        mask_ = 0;
    }
    void checkFlags() {
        three_cuda_check(cuStreamSynchronize(stream_));
        uint32_t bits = 0;
        three_cuda_check(cuMemcpyDtoH(&bits, error_, sizeof(bits)));
        if (bits)
            throw std::runtime_error("Packed digit conversion invariant failed: " +
                                     std::to_string(bits));
    }

  public:
    packed_device(unsigned base, cl_device_id cldevice, const std::string &path, bool verbose)
        : base_(base) {
        std::ifstream file(path);
        if (!file)
            throw std::runtime_error("Cannot read packed kernel source");
        std::string source(std::istreambuf_iterator<char>(file), {});
        fields_ = literal(source, "PG_K");
        if (source.find("#define PG_OUTER_THREADS") != std::string::npos)
            outer_threads_ = literal(source, "PG_OUTER_THREADS");
        if (source.find("#define PG_INNER_THREADS") != std::string::npos)
            inner_threads_ = literal(source, "PG_INNER_THREADS");
        for (unsigned threads : {outer_threads_, inner_threads_})
            if (!threads || threads > 1024 || (threads & (threads - 1)) || 4096 % threads)
                throw std::runtime_error("Invalid packed transform thread count");

        root_bits_ = literal(source, "PG_ROOT_BITS");
        root_words_ = literal(source, "PG_ROOT_WORDS");
        if (root_words_ != n / 2)
            throw std::runtime_error("Invalid root table size");
        if (root_bits_ != 32 && root_bits_ != 64)
            throw std::runtime_error("Invalid packed root width");
        if (base != literal(source, "PG_BASE") ||
            (fields_ < 7 || fields_ > 17))
            throw std::runtime_error("Packed kernel does not match the candidate base");
        try {
            three_cuda_check(cuInit(0));
            std::array<unsigned char, 16> uuid{};
            checkCL(clGetDeviceInfo(cldevice, 0x106A, uuid.size(), uuid.data(), nullptr),
                    "packed device UUID");
            int count = 0, matches = 0;
            three_cuda_check(cuDeviceGetCount(&count));
            for (int index = 0; index < count; ++index) {
                CUdevice candidate;
                CUuuid other{};
                three_cuda_check(cuDeviceGet(&candidate, index));
                three_cuda_check(cuDeviceGetUuid(&other, candidate));
                if (std::memcmp(uuid.data(), other.bytes, uuid.size()) == 0) {
                    device_ = candidate;
                    ++matches;
                }
            }
            if (matches != 1)
                throw std::runtime_error("Cannot match packed OpenCL and CUDA GPU");
            three_cuda_check(cuDevicePrimaryCtxRetain(&context_, device_));
            three_cuda_check(cuCtxSetCurrent(context_));
            std::string binary = compile(cldevice, source);
            three_cuda_check(cuCtxSetCurrent(context_));
            three_cuda_check(cuModuleLoadData(&module_, binary.c_str()));
            three_cuda_check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
            for (unsigned i = 0; i < 3; ++i)
                three_cuda_check(cuModuleGetFunction(&transforms_[i], module_,
                                                     i == 0   ? "packed_forward"
                                                     : i == 1 ? "packed_square1024"
                                                              : "packed_backward"));
            for (unsigned i = 0; i < 2; ++i)
                three_cuda_check(cuModuleGetFunction(
                    &converters_[i], module_, i ? "packed_convert_dup1" : "packed_convert_dup0"));
            three_cuda_check(cuModuleGetFunction(&encode_, module_, "packed_encode"));
            three_cuda_check(cuModuleGetFunction(&decode_, module_, "packed_decode"));
            three_cuda_check(cuModuleGetFunction(&repair_, module_, "packed_repair"));
            for (auto fn : transforms_)
                three_cuda_check(
                    cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 50));
            for (auto &value : state_)
                three_cuda_check(cuMemAlloc(&value, size_t(fields_) * n * 4));
            three_cuda_check(cuMemAlloc(&roots_, size_t(fields_) * root_words_ * (root_bits_ / 8)));
            three_cuda_check(cuMemAlloc(&digits_, 8 * n * 4));
            three_cuda_check(cuMemAlloc(&carries_, n * 4));
            three_cuda_check(cuMemAlloc(&error_, 4));
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
            if (root_bits_ == 32)
                three_cuda_check(cuMemcpyHtoD(roots_, roots32.data(), roots32.size() * 4));
            else
                three_cuda_check(cuMemcpyHtoD(roots_, roots64.data(), roots64.size() * 8));
            three_cuda_check(cuCtxSynchronize());
            if (verbose)
                pio::print("GFN-19 exact packed-digit CUDA square (" + std::to_string(fields_) +
                           " prime channels)\n");

        } catch (...) {
            release();
            throw;
        }
    }
    ~packed_device() { release(); }

  private:
    void release() noexcept {
        if (context_)
            cuCtxSetCurrent(context_);
        try {
            flush();
            if (stream_)
                cuStreamSynchronize(stream_);
        } catch (...) {
        }
        for (auto &entry : graphs_)
            if (entry.second)
                cuGraphExecDestroy(entry.second);
        for (auto value : state_)
            if (value)
                cuMemFree(value);
        for (auto value : {roots_, digits_, carries_, error_})
            if (value)
                cuMemFree(value);
        if (module_)
            cuModuleUnload(module_);
        if (stream_)
            cuStreamDestroy(stream_);
        if (context_)
            cuDevicePrimaryCtxRelease(device_);
    }

  public:
    void synchronize() {
        flush();
        three_cuda_check(cuStreamSynchronize(stream_));
    }
    void setDigits(const int32_t *input) {
        synchronize();
        three_cuda_check(cuMemcpyHtoD(digits_, input, 8 * n * 4));
        three_cuda_check(cuMemsetD32(error_, 0, 1));
        three_cuda_check(cuCtxSynchronize());
        current_ = 0;
        CUdeviceptr out = state_[0];
        void *args[] = {&digits_, &out, &error_};
        launch(encode_, fields_ * n / 128, 128, args);
        checkFlags();
    }
    void getDigits(int32_t *output) {
        synchronize();
        three_cuda_check(cuMemsetD32(error_, 0, 1));
        three_cuda_check(cuCtxSynchronize());
        CUdeviceptr input = state_[current_];
        void *args[] = {&input, &digits_, &carries_, &error_};
        launch(decode_, n / 128, 128, args);
        void *repairArgs[] = {&digits_, &carries_, &error_};
        launch(repair_, n / 128, 128, repairArgs);
        checkFlags();
        three_cuda_check(cuMemcpyDtoH(output, digits_, 8 * n * 4));
    }
    void square(bool doubled) {
        if (!pending_)
            start_ = current_;
        mask_ |= (unsigned(doubled) << pending_);
        ++pending_;
        current_ ^= 1;
        if (pending_ == 8)
            flush();
    }
    size_t memory() const {
        return size_t(fields_) * n * 8 + size_t(fields_) * root_words_ * (root_bits_ / 8) +
               (9 * n + 1) * 4;
    }
};
