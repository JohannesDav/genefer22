// OpenCL execution of the fixed GFN-19 kernels.
#pragma once

#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>

class device : oclObject
{
    struct Kernel
    {
        cl_kernel normal = nullptr;
        std::array<cl_kernel, 2> doubled{};
        cl_int bit = 0;

        ~Kernel()
        {
            for (auto handle : doubled)
                if (handle)
                    clReleaseKernel(handle);
            if (normal)
                clReleaseKernel(normal);
        }
    };

    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    size_t maxWorkGroupSize_ = 0;

    static Kernel *kernel(cl_kernel handle)
    {
        return reinterpret_cast<Kernel *>(handle);
    }

  public:
    device(const platform &parent, size_t index, bool verbose)
    {
        if (index >= parent.getDeviceCount())
            throw std::runtime_error("OpenCL device index out of range");
        device_ = parent.getDevice(index);
        char name[256]{};
        oclFatal(clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr));
#if defined(GENEFER_GUARD_3090)
        const std::string expected = "NVIDIA GeForce RTX 3090";
#else
        const std::string expected = "NVIDIA GeForce RTX 4060 Ti";
        cl_ulong memory = 0;
        oclFatal(clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memory), &memory,
                                 nullptr));
        if (memory < cl_ulong(15000) * 1024 * 1024)
            throw std::runtime_error("The 4060 Ti profile requires the 16GB card");
#endif
        if (name != expected)
            throw std::runtime_error("GPU does not match the selected GFN-19 profile");
        oclFatal(clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(maxWorkGroupSize_), &maxWorkGroupSize_, nullptr));
        cl_int error = CL_SUCCESS;
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
        oclFatal(error, "create GFN-19 context");
        queue_ = clCreateCommandQueue(context_, device_, 0, &error);
        if (error != CL_SUCCESS)
        {
            clReleaseContext(context_);
            context_ = nullptr;
            oclFatal(error, "create GFN-19 queue");
        }
        if (verbose)
            pio::print(std::string("GFN-19 optimized kernels: ") + name + "\n");
    }

    virtual ~device()
    {
        if (queue_)
            clFinish(queue_);
        if (program_)
            clReleaseProgram(program_);
        if (queue_)
            clReleaseCommandQueue(queue_);
        if (context_)
            clReleaseContext(context_);
    }

    size_t getMaxWorkGroupSize() const
    {
        return maxWorkGroupSize_;
    }

    void loadProgram(const std::string &configuration)
    {
        const char *path = std::getenv("GENEFER_KERNEL");
        if (!path)
            throw std::runtime_error("GENEFER_KERNEL must specify the matching GPU kernels");
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("Cannot open GFN-19 kernels");
        const std::string text(std::istreambuf_iterator<char>(input), {});
        const std::string marker = "// GENEFER_CONFIG ";
        const size_t end = text.find('\n');
        if (text.compare(0, marker.size(), marker) != 0 || end == std::string::npos ||
            text.substr(marker.size(), end - marker.size()) != configuration)
            throw std::runtime_error("Kernel configuration does not match this transform");

        const size_t bytes = text.size();
        const auto *binary = reinterpret_cast<const unsigned char *>(text.data());
        cl_int error = CL_SUCCESS, status = CL_SUCCESS;
        program_ = clCreateProgramWithBinary(context_, 1, &device_, &bytes, &binary, &status,
                                              &error);
        oclFatal(error, "load GFN-19 kernels");
        oclFatal(status, "GFN-19 binary status");
        error = clBuildProgram(program_, 1, &device_, "", nullptr, nullptr);
        if (error != CL_SUCCESS)
        {
            size_t size = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::vector<char> log(size + 1);
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, size, log.data(),
                                  nullptr);
            pio::print(std::string(log.data()) + "\n");
            oclFatal(error, "build GFN-19 kernels");
        }
    }

    void clearProgram()
    {
        oclFatal(clFinish(queue_));
        if (program_)
        {
            oclFatal(clReleaseProgram(program_));
            program_ = nullptr;
        }
    }

  protected:
    cl_mem _createBuffer(cl_mem_flags flags, size_t bytes, bool clear = true)
    {
        cl_int error = CL_SUCCESS;
        cl_mem memory = clCreateBuffer(context_, flags, bytes, nullptr, &error);
        oclFatal(error);
        if (clear)
        {
            std::vector<unsigned char> zeros(bytes);
            error = clEnqueueWriteBuffer(queue_, memory, CL_TRUE, 0, bytes, zeros.data(), 0,
                                         nullptr, nullptr);
            if (error != CL_SUCCESS)
            {
                clReleaseMemObject(memory);
                oclFatal(error);
            }
        }
        return memory;
    }

    void _releaseBuffer(cl_mem &memory)
    {
        if (memory)
        {
            oclFatal(clFinish(queue_));
            oclFatal(clReleaseMemObject(memory));
            memory = nullptr;
        }
    }

    void _readBuffer(cl_mem &memory, void *data, size_t bytes, size_t offset = 0)
    {
        oclFatal(clEnqueueReadBuffer(queue_, memory, CL_TRUE, offset, bytes, data, 0, nullptr,
                                     nullptr));
    }

    void _writeBuffer(cl_mem &memory, const void *data, size_t bytes, size_t offset = 0)
    {
        oclFatal(clEnqueueWriteBuffer(queue_, memory, CL_TRUE, offset, bytes, data, 0, nullptr,
                                      nullptr));
    }

    cl_kernel _createKernel(const char *name)
    {
        auto handle = std::make_unique<Kernel>();
        cl_int error = CL_SUCCESS;
        handle->normal = clCreateKernel(program_, name, &error);
        oclFatal(error, name);
        if (std::string(name) == "boundary_bridge_signed")
        {
            for (unsigned bit = 0; bit < 2; ++bit)
            {
                const auto entry = std::string(name) + "_dup" + std::to_string(bit);
                handle->doubled[bit] = clCreateKernel(program_, entry.c_str(), &error);
                oclFatal(error, entry.c_str());
            }
        }
        return reinterpret_cast<cl_kernel>(handle.release());
    }

    void _releaseKernel(cl_kernel &handle)
    {
        delete kernel(handle);
        handle = nullptr;
    }

    static void _setKernelArg(cl_kernel handle, cl_uint index, size_t bytes, const void *value)
    {
        auto *k = kernel(handle);
        oclFatal(clSetKernelArg(k->normal, index, bytes, value));
        for (auto duplicate : k->doubled)
            if (duplicate)
                oclFatal(clSetKernelArg(duplicate, index, bytes, value));
        if (k->doubled[0] && index == 7)
        {
            if (bytes != sizeof(k->bit))
                throw std::runtime_error("Invalid doubling flag size");
            std::memcpy(&k->bit, value, sizeof(k->bit));
            if (k->bit != 0 && k->bit != 1)
                throw std::runtime_error("Specialized bridge doubling must be boolean");
        }
    }

    void _executeKernel(cl_kernel handle, size_t global, size_t local = 0)
    {
        auto *k = kernel(handle);
        const auto selected = k->doubled[0] ? k->doubled[k->bit] : k->normal;
        oclFatal(clEnqueueNDRangeKernel(queue_, selected, 1, nullptr, &global,
                                        local ? &local : nullptr, 0, nullptr, nullptr));
    }
};
