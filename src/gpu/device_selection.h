#pragma once
#include <stdexcept>

inline cl_device_id nvidia_device(const platform &parent,size_t index) {
    if(index>=parent.getDeviceCount()) throw std::runtime_error("OpenCL device index out of range");
    cl_device_id gpu=parent.getDevice(index);
    cl_uint vendor=0;
    if(clGetDeviceInfo(gpu,CL_DEVICE_VENDOR_ID,sizeof(vendor),&vendor,nullptr)!=CL_SUCCESS || vendor!=0x10de)
        throw std::runtime_error("OpenCL arithmetic requires an NVIDIA GPU");
    return gpu;
}
