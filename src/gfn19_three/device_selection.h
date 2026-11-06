#pragma once
#include <string>
#include <stdexcept>

inline cl_device_id three_prime_device(const platform &parent, size_t index)
{
    if (index >= parent.getDeviceCount()) throw std::runtime_error("OpenCL device index out of range");
    cl_device_id id=parent.getDevice(index);
    char name[256]{};
    if (clGetDeviceInfo(id,CL_DEVICE_NAME,sizeof(name),name,nullptr)!=CL_SUCCESS) throw std::runtime_error("Cannot read GPU name");
#if defined(GENEFER_GUARD_3090)
    const std::string expected="NVIDIA GeForce RTX 3090";
#else
    const std::string expected="NVIDIA GeForce RTX 4060 Ti";
    cl_ulong memory=0;
    if (clGetDeviceInfo(id,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(memory),&memory,nullptr)!=CL_SUCCESS) throw std::runtime_error("Cannot read GPU memory size");
    if (memory<cl_ulong(15000)*1024*1024) throw std::runtime_error("This profile requires the 16GB 4060 Ti");
#endif
    if (name!=expected) throw std::runtime_error("GPU does not match the selected GFN-19 build");
    return id;
}
