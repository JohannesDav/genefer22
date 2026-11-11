// Exercise the combined executable's original OpenCL path on any non-NVIDIA
// OpenCL device. A CPU ICD is sufficient for this compatibility test.
#include "../src/ocl.h"
#include "../src/transform.h"
#include <iostream>
#include <memory>
#include <vector>

int main() {
    cl_uint count = 0;
    if (clGetPlatformIDs(0, nullptr, &count) != CL_SUCCESS || !count)
        return 77;
    std::vector<cl_platform_id> platforms(count);
    if (clGetPlatformIDs(count, platforms.data(), nullptr) != CL_SUCCESS)
        return 1;
    for (cl_platform_id platform : platforms) {
        cl_uint n = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &n) != CL_SUCCESS)
            continue;
        std::vector<cl_device_id> devices(n);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, n, devices.data(), nullptr) != CL_SUCCESS)
            return 1;
        for (cl_device_id device : devices) {
            cl_uint vendor = 0;
            if (clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, nullptr) != CL_SUCCESS)
                return 1;
            if (vendor == 0x10de)
                continue;
            // Supply explicit device IDs through the same route BOINC uses.
            // No BOINC client or project connection is involved.
            std::unique_ptr<transform> value(
                transform::create_gpu(1026, 16, true, 0, 3, platform, device, false));
            value->set(3);
            value->squareDup(false);
            value->squareDup(true); // 2 * 9^2 = 162
            gint result(1u << 16, 1026);
            value->getInt(result);
            if (result.data()[0] != 162)
                return 1;
            for (size_t j = 1; j < result.getSize(); ++j)
                if (result.data()[j])
                    return 1;
            std::cout << "PASS: original OpenCL fallback on a non-NVIDIA device\n";
            return 0;
        }
    }
    return 77;
}
