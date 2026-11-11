/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#if defined(GENEFER_GENERAL_PACKED)
#include "gpu/packed.h"
#else
#include "transformGPU.h"
#endif

#if defined(GENEFER_GENERAL_SMALL3_AVAILABLE)
// Defined in a separate translation unit: its prime constants and C++ types
// must not collide with the original upstream arithmetic.
transform * create_general_small3(uint32_t, uint32_t, bool, size_t, size_t,
                                  cl_platform_id, cl_device_id, bool,
                                  unsigned = 0, const std::string & = {});
#endif

#if defined(GENEFER_AUTOMATIC_OPENCL)
#include "gpu/profiles.h"
#include <gpu/embedded_kernels.h>
transform *create_original_opencl(uint32_t,uint32_t,bool,size_t,size_t,
                                  cl_platform_id,cl_device_id,bool);

namespace {
gpu_gfn::device_properties properties(cl_device_id gpu) {
    gpu_gfn::device_properties out;
    char name[1024]{};
    cl_ulong memory=0,local=0;
    cl_uint major=0,minor=0;
    size_t threads=0;
    auto query=[&](cl_device_info key,size_t bytes,void *value) {
        const cl_int code=clGetDeviceInfo(gpu,key,bytes,value,nullptr);
        if(code!=CL_SUCCESS) throw std::runtime_error("Cannot query OpenCL device properties: "+std::to_string(code));
    };
    query(CL_DEVICE_NAME,sizeof(name),name);
    query(CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(memory),&memory);
    query(CL_DEVICE_LOCAL_MEM_SIZE,sizeof(local),&local);
    query(CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(threads),&threads);
    // NVIDIA's compute-capability query is optional; project defaults remain
    // available when the driver does not expose this extension.
    if(clGetDeviceInfo(gpu,0x4000,sizeof(major),&major,nullptr)==CL_SUCCESS
       && clGetDeviceInfo(gpu,0x4001,sizeof(minor),&minor,nullptr)==CL_SUCCESS)
        out.arch=gpu_gfn::architecture_for(major,minor);
    out.name=name; out.memory_bytes=memory; out.local_memory_bytes=local;
    out.max_threads=threads;
    return out;
}
transform *make_optimized(uint32_t b,uint32_t n,bool boinc,size_t device,size_t registers,
                     cl_platform_id platform_id,cl_device_id gpu,bool verbose,
                     const std::string &source={}) {
    if(uint64_t(b)*b < uint64_t(P1S)*P2S/2/(size_t(1)<<n))
        return new packed_transform<2,false>(b,n,boinc,device,registers,platform_id,gpu,verbose,source);
    if(uint64_t(b)*b < uint64_t(P1U)*P2U/2/(size_t(1)<<n))
        return new packed_transform<2,true>(b,n,boinc,device,registers,platform_id,gpu,verbose,source);
    if(b<=1000000000)
        return new packed_transform<3,false>(b,n,boinc,device,registers,platform_id,gpu,verbose,source);
    return new packed_transform<3,true>(b,n,boinc,device,registers,platform_id,gpu,verbose,source);
}
transform *make_automatic(uint32_t b,uint32_t n,bool boinc,size_t device,size_t registers,
                          cl_platform_id platform_id,cl_device_id boinc_gpu,bool verbose) {
    auto stock=[&] { return create_original_opencl(b,n,boinc,device,registers,platform_id,boinc_gpu,verbose); };
    if(n<16 || n>22 || std::getenv("GENEFER_FORCE_OPENCL")) return stock();
    cl_device_id gpu=boinc_gpu;
    if(!boinc || !gpu) {
        platform devices;
        if(device>=devices.getDeviceCount()) return stock();
        gpu=devices.getDevice(device);
    }
    cl_uint vendor=0;
    if(clGetDeviceInfo(gpu,CL_DEVICE_VENDOR_ID,sizeof(vendor),&vendor,nullptr)!=CL_SUCCESS
       || vendor!=0x10de) return stock();
    try {
        const auto info=properties(gpu);
        cl_uint warp=0;
        if(info.max_threads<1024 || info.local_memory_bytes<49152
           || clGetDeviceInfo(gpu,0x4003,sizeof(warp),&warp,nullptr)!=CL_SUCCESS || warp!=32)
            return stock();
        for(const auto *profile:gpu_gfn::profiles_for(info,n,b)) {
            if(profile->kind==gpu_gfn::implementation::original_opencl) {
                if(verbose) pio::print(std::string("GPU profile: ")+profile->name+" (original OpenCL)\n");
                return stock();
            }
            if(profile->kind==gpu_gfn::implementation::original_optimized) {
                if(verbose) pio::print(std::string("GPU profile: ")+profile->name+" (original OpenCL arithmetic)\n");
                return make_optimized(b,n,boinc,device,registers,platform_id,boinc_gpu,verbose);
            }
            if(profile->kind==gpu_gfn::implementation::three_prime) {
                if(info.max_threads<1024 || info.local_memory_bytes<49152) continue;
                if(auto *value=create_general_small3(b,n,boinc,device,registers,platform_id,boinc_gpu,
                                                     verbose,profile->plan.root_bits,gpu_gfn::embedded::three_prime)) {
                    if(verbose) pio::print(std::string("OpenCL profile: ")+profile->name+" (three primes)\n");
                    return value;
                }
                continue;
            }
            gpu_gfn::generated_kernel kernel;
            try { kernel=gpu_gfn::generate_kernel(b,n,profile->plan); }
            catch(const std::invalid_argument &) { continue; }
            if(!gpu_gfn::resources_fit(info,profile->plan,kernel.fields)) continue;
            if(verbose) pio::print(std::string("OpenCL profile: ")+profile->name+" (packed)\n");
            return make_optimized(b,n,boinc,device,registers,platform_id,boinc_gpu,verbose,
                             kernel.source);
        }
        if(verbose) pio::print("OpenCL: original arithmetic\n");
        return make_optimized(b,n,boinc,device,registers,platform_id,boinc_gpu,verbose);
    } catch(const std::exception &error) {
        if(verbose) pio::print(std::string("OpenCL initialization unavailable; using original OpenCL: ")+error.what()+"\n");
        return stock();
    }
}
} // namespace

transform *transform::create_original_gpu(uint32_t b,uint32_t n,bool boinc,size_t device,
                                          size_t registers,cl_platform_id platform_id,
                                          cl_device_id gpu,bool verbose) {
    return create_original_opencl(b,n,boinc,device,registers,platform_id,gpu,verbose);
}
#endif

// inline uint64_t bt12(const double pp_2N)
// {
// 	uint64_t b = static_cast<uint64_t>(sqrt(pp_2N));
// 	if (b % 2 == 1) --b;
// 	if ((b * double(b) >= pp_2N) || ((b + 2) * double(b + 2) < pp_2N)) std::cout << "ERROR! ";
// 	return b;
// }

transform * transform::create_ocl(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								  const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
{
#if defined(GENEFER_AUTOMATIC_OPENCL)
    return make_automatic(b,n,isBoinc,device,num_regs,boinc_platform_id,boinc_device_id,verbose);
#endif
	// NTT limits
	// for (int m = 11; m <= 23; ++m)
	// {
	// 	std::cout << m << ": ";
	// 	std::cout << bt12(P1S * double(P2S) / 2 / (1 << m));
	// 	std::cout << ", " << bt12(P1U * double(P2U) / 2 / (1 << m));
	// 	std::cout << ", " << std::min(bt12(P1S * double(P2S) * P3S / 2 / (1 << m)), uint64_t(P3S / 2));
	// 	std::cout << ", " << std::min(bt12(P1U * double(P2U) * P3U / 2 / (1 << m)), uint64_t(P3U / 2));
	// 	std::cout << "." << std::endl;
	// }

#if defined(GENEFER_GENERAL_SMALL3_AVAILABLE)
    if (std::getenv("GENEFER_SMALL3"))
        if (auto *value=create_general_small3(b,n,isBoinc,device,num_regs,
                                               boinc_platform_id,boinc_device_id,verbose))
            return value;
#endif
#if defined(GENEFER_GENERAL_PACKED)
    if (b * uint64_t(b) < (P1S * uint64_t(P2S) / 2) / (size_t(1) << n))
        return new packed_transform<2, false>(b,n,isBoinc,device,num_regs,boinc_platform_id,boinc_device_id,verbose);
    if (b * uint64_t(b) < (P1U * uint64_t(P2U) / 2) / (size_t(1) << n))
        return new packed_transform<2, true>(b,n,isBoinc,device,num_regs,boinc_platform_id,boinc_device_id,verbose);
    if (b <= 1000000000)
        return new packed_transform<3, false>(b,n,isBoinc,device,num_regs,boinc_platform_id,boinc_device_id,verbose);
    return new packed_transform<3, true>(b,n,isBoinc,device,num_regs,boinc_platform_id,boinc_device_id,verbose);

#else
	transform * pTransform = nullptr;
	if (b * static_cast<uint64_t>(b) < (P1S * static_cast<uint64_t>(P2S) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPUs<2, false>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else if (b * static_cast<uint64_t>(b) < (P1U * static_cast<uint64_t>(P2U) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPUs<2, true>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else if (b <= 1000000000)
	{
		pTransform = new transformGPUs<3, false>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else
	{
		pTransform = new transformGPUs<3, true>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	return pTransform;
#endif
}
