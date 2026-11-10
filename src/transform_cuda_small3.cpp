// Exact small-prime transform selected by NVIDIA GPU profiles.
// Keep its types and constants separate from the upstream transform.
#include "cuda/three_prime.h"

transform * create_general_small3(uint32_t b, uint32_t n, bool boinc, size_t device,
                                  size_t registers, cl_platform_id platform,
                                  cl_device_id gpu, bool verbose, unsigned root_bits,
                                  const std::string &source)
{
    if (n < 16 || n > 22 || !small_prime_bounds::supported(b)) return nullptr;
    return new cuda_small_prime::transformGPUs<3,false>(b,n,boinc,device,registers,
                                                     platform,gpu,verbose,root_bits,source);
}
