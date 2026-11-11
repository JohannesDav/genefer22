// Preserve the original OpenCL implementation as the portable fallback.
// Keep its template types separate from OpenCL implementations with other
// execution devices or prime constants.
#undef GENEFER_GENERAL_OPENCL
#undef GENEFER_GENERAL_PACKED
#undef GENEFER_GENERAL_SMALL3_AVAILABLE
#include <cstdint>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include "ocl.h"
#include "transform.h"
#include "ocl/kernel.h"

namespace original_opencl {
#include "transformGPU.h"

transform *create(uint32_t base,uint32_t exponent,bool boinc,size_t device,size_t registers,
                  cl_platform_id platform,cl_device_id gpu,bool verbose) {
    const uint64_t square=uint64_t(base)*base;
    if (square < uint64_t(P1S)*P2S/2/(size_t(1)<<exponent))
        return new transformGPUs<2,false>(base,exponent,boinc,device,registers,platform,gpu,verbose);
    if (square < uint64_t(P1U)*P2U/2/(size_t(1)<<exponent))
        return new transformGPUs<2,true>(base,exponent,boinc,device,registers,platform,gpu,verbose);
    if (base<=1000000000)
        return new transformGPUs<3,false>(base,exponent,boinc,device,registers,platform,gpu,verbose);
    return new transformGPUs<3,true>(base,exponent,boinc,device,registers,platform,gpu,verbose);
}
} // namespace original_opencl

transform *create_original_opencl(uint32_t base,uint32_t exponent,bool boinc,size_t device,
                                  size_t registers,cl_platform_id platform,cl_device_id gpu,bool verbose) {
    return original_opencl::create(base,exponent,boinc,device,registers,platform,gpu,verbose);
}
