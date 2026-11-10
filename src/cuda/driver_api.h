#pragma once
// Resolve the CUDA driver lazily: non-NVIDIA devices do not need libcuda.
#include <cuda.h>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace cuda_gfn {
#define GENEFER_SYMBOL_INNER(x) #x
#define GENEFER_SYMBOL(x) GENEFER_SYMBOL_INNER(x)
using type_cuCtxSetCurrent = decltype(&::cuCtxSetCurrent);
inline constexpr const char symbol_cuCtxSetCurrent[] = GENEFER_SYMBOL(cuCtxSetCurrent);
using type_cuCtxSynchronize = decltype(&::cuCtxSynchronize);
inline constexpr const char symbol_cuCtxSynchronize[] = GENEFER_SYMBOL(cuCtxSynchronize);
using type_cuDeviceGet = decltype(&::cuDeviceGet);
inline constexpr const char symbol_cuDeviceGet[] = GENEFER_SYMBOL(cuDeviceGet);
using type_cuDeviceGetAttribute = decltype(&::cuDeviceGetAttribute);
inline constexpr const char symbol_cuDeviceGetAttribute[] = GENEFER_SYMBOL(cuDeviceGetAttribute);
using type_cuDeviceGetCount = decltype(&::cuDeviceGetCount);
inline constexpr const char symbol_cuDeviceGetCount[] = GENEFER_SYMBOL(cuDeviceGetCount);
using type_cuDeviceGetUuid = decltype(&::cuDeviceGetUuid);
inline constexpr const char symbol_cuDeviceGetUuid[] = GENEFER_SYMBOL(cuDeviceGetUuid);
using type_cuDevicePrimaryCtxRelease = decltype(&::cuDevicePrimaryCtxRelease);
inline constexpr const char symbol_cuDevicePrimaryCtxRelease[] = GENEFER_SYMBOL(cuDevicePrimaryCtxRelease);
using type_cuDevicePrimaryCtxRetain = decltype(&::cuDevicePrimaryCtxRetain);
inline constexpr const char symbol_cuDevicePrimaryCtxRetain[] = GENEFER_SYMBOL(cuDevicePrimaryCtxRetain);
using type_cuDeviceTotalMem = decltype(&::cuDeviceTotalMem);
inline constexpr const char symbol_cuDeviceTotalMem[] = GENEFER_SYMBOL(cuDeviceTotalMem);
using type_cuDriverGetVersion = decltype(&::cuDriverGetVersion);
inline constexpr const char symbol_cuDriverGetVersion[] = GENEFER_SYMBOL(cuDriverGetVersion);
using type_cuFuncGetAttribute = decltype(&::cuFuncGetAttribute);
inline constexpr const char symbol_cuFuncGetAttribute[] = GENEFER_SYMBOL(cuFuncGetAttribute);
using type_cuFuncSetAttribute = decltype(&::cuFuncSetAttribute);
inline constexpr const char symbol_cuFuncSetAttribute[] = GENEFER_SYMBOL(cuFuncSetAttribute);
using type_cuGetErrorString = decltype(&::cuGetErrorString);
inline constexpr const char symbol_cuGetErrorString[] = GENEFER_SYMBOL(cuGetErrorString);
using type_cuGraphAddKernelNode = decltype(&::cuGraphAddKernelNode);
inline constexpr const char symbol_cuGraphAddKernelNode[] = GENEFER_SYMBOL(cuGraphAddKernelNode);
using type_cuGraphCreate = decltype(&::cuGraphCreate);
inline constexpr const char symbol_cuGraphCreate[] = GENEFER_SYMBOL(cuGraphCreate);
using type_cuGraphDestroy = decltype(&::cuGraphDestroy);
inline constexpr const char symbol_cuGraphDestroy[] = GENEFER_SYMBOL(cuGraphDestroy);
using type_cuGraphExecDestroy = decltype(&::cuGraphExecDestroy);
inline constexpr const char symbol_cuGraphExecDestroy[] = GENEFER_SYMBOL(cuGraphExecDestroy);
using type_cuGraphGetNodes = decltype(&::cuGraphGetNodes);
inline constexpr const char symbol_cuGraphGetNodes[] = GENEFER_SYMBOL(cuGraphGetNodes);
using type_cuGraphGetRootNodes = decltype(&::cuGraphGetRootNodes);
inline constexpr const char symbol_cuGraphGetRootNodes[] = GENEFER_SYMBOL(cuGraphGetRootNodes);
using type_cuGraphInstantiate = decltype(&::cuGraphInstantiate);
inline constexpr const char symbol_cuGraphInstantiate[] = GENEFER_SYMBOL(cuGraphInstantiate);
using type_cuGraphKernelNodeGetParams = decltype(&::cuGraphKernelNodeGetParams);
inline constexpr const char symbol_cuGraphKernelNodeGetParams[] = GENEFER_SYMBOL(cuGraphKernelNodeGetParams);
using type_cuGraphLaunch = decltype(&::cuGraphLaunch);
inline constexpr const char symbol_cuGraphLaunch[] = GENEFER_SYMBOL(cuGraphLaunch);
using type_cuGraphNodeGetDependentNodes = decltype(&::cuGraphNodeGetDependentNodes);
inline constexpr const char symbol_cuGraphNodeGetDependentNodes[] =
    GENEFER_SYMBOL(cuGraphNodeGetDependentNodes);
using type_cuGraphNodeGetType = decltype(&::cuGraphNodeGetType);
inline constexpr const char symbol_cuGraphNodeGetType[] = GENEFER_SYMBOL(cuGraphNodeGetType);
using type_cuInit = decltype(&::cuInit);
inline constexpr const char symbol_cuInit[] = GENEFER_SYMBOL(cuInit);
using type_cuLaunchKernel = decltype(&::cuLaunchKernel);
inline constexpr const char symbol_cuLaunchKernel[] = GENEFER_SYMBOL(cuLaunchKernel);
using type_cuMemAlloc = decltype(&::cuMemAlloc);
inline constexpr const char symbol_cuMemAlloc[] = GENEFER_SYMBOL(cuMemAlloc);
using type_cuMemFree = decltype(&::cuMemFree);
inline constexpr const char symbol_cuMemFree[] = GENEFER_SYMBOL(cuMemFree);
using type_cuMemcpyDtoH = decltype(&::cuMemcpyDtoH);
inline constexpr const char symbol_cuMemcpyDtoH[] = GENEFER_SYMBOL(cuMemcpyDtoH);
using type_cuMemcpyHtoD = decltype(&::cuMemcpyHtoD);
inline constexpr const char symbol_cuMemcpyHtoD[] = GENEFER_SYMBOL(cuMemcpyHtoD);
using type_cuMemsetD32 = decltype(&::cuMemsetD32);
inline constexpr const char symbol_cuMemsetD32[] = GENEFER_SYMBOL(cuMemsetD32);
using type_cuMemsetD8 = decltype(&::cuMemsetD8);
inline constexpr const char symbol_cuMemsetD8[] = GENEFER_SYMBOL(cuMemsetD8);
using type_cuModuleGetFunction = decltype(&::cuModuleGetFunction);
inline constexpr const char symbol_cuModuleGetFunction[] = GENEFER_SYMBOL(cuModuleGetFunction);
using type_cuModuleLoadData = decltype(&::cuModuleLoadData);
inline constexpr const char symbol_cuModuleLoadData[] = GENEFER_SYMBOL(cuModuleLoadData);
using type_cuModuleUnload = decltype(&::cuModuleUnload);
inline constexpr const char symbol_cuModuleUnload[] = GENEFER_SYMBOL(cuModuleUnload);
using type_cuStreamBeginCapture = decltype(&::cuStreamBeginCapture);
inline constexpr const char symbol_cuStreamBeginCapture[] = GENEFER_SYMBOL(cuStreamBeginCapture);
using type_cuStreamCreate = decltype(&::cuStreamCreate);
inline constexpr const char symbol_cuStreamCreate[] = GENEFER_SYMBOL(cuStreamCreate);
using type_cuStreamDestroy = decltype(&::cuStreamDestroy);
inline constexpr const char symbol_cuStreamDestroy[] = GENEFER_SYMBOL(cuStreamDestroy);
using type_cuStreamEndCapture = decltype(&::cuStreamEndCapture);
inline constexpr const char symbol_cuStreamEndCapture[] = GENEFER_SYMBOL(cuStreamEndCapture);
using type_cuStreamSynchronize = decltype(&::cuStreamSynchronize);
inline constexpr const char symbol_cuStreamSynchronize[] = GENEFER_SYMBOL(cuStreamSynchronize);
#undef GENEFER_SYMBOL
#undef GENEFER_SYMBOL_INNER
class driver_api {
    void *library_ = nullptr;
    void *resolve(const char *name) const {
#if defined(_WIN32)
        void *value = reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(library_), name));
#else
        void *value = dlsym(library_, name);
#endif
        if (!value)
            throw std::runtime_error(std::string("Missing CUDA driver entry: ") + name);
        return value;
    }

  public:
    type_cuCtxSetCurrent f_cuCtxSetCurrent = nullptr;
    type_cuCtxSynchronize f_cuCtxSynchronize = nullptr;
    type_cuDeviceGet f_cuDeviceGet = nullptr;
    type_cuDeviceGetAttribute f_cuDeviceGetAttribute = nullptr;
    type_cuDeviceGetCount f_cuDeviceGetCount = nullptr;
    type_cuDeviceGetUuid f_cuDeviceGetUuid = nullptr;
    type_cuDevicePrimaryCtxRelease f_cuDevicePrimaryCtxRelease = nullptr;
    type_cuDevicePrimaryCtxRetain f_cuDevicePrimaryCtxRetain = nullptr;
    type_cuDeviceTotalMem f_cuDeviceTotalMem = nullptr;
    type_cuDriverGetVersion f_cuDriverGetVersion = nullptr;
    type_cuFuncGetAttribute f_cuFuncGetAttribute = nullptr;
    type_cuFuncSetAttribute f_cuFuncSetAttribute = nullptr;
    type_cuGetErrorString f_cuGetErrorString = nullptr;
    type_cuGraphAddKernelNode f_cuGraphAddKernelNode = nullptr;
    type_cuGraphCreate f_cuGraphCreate = nullptr;
    type_cuGraphDestroy f_cuGraphDestroy = nullptr;
    type_cuGraphExecDestroy f_cuGraphExecDestroy = nullptr;
    type_cuGraphGetNodes f_cuGraphGetNodes = nullptr;
    type_cuGraphGetRootNodes f_cuGraphGetRootNodes = nullptr;
    type_cuGraphInstantiate f_cuGraphInstantiate = nullptr;
    type_cuGraphKernelNodeGetParams f_cuGraphKernelNodeGetParams = nullptr;
    type_cuGraphLaunch f_cuGraphLaunch = nullptr;
    type_cuGraphNodeGetDependentNodes f_cuGraphNodeGetDependentNodes = nullptr;
    type_cuGraphNodeGetType f_cuGraphNodeGetType = nullptr;
    type_cuInit f_cuInit = nullptr;
    type_cuLaunchKernel f_cuLaunchKernel = nullptr;
    type_cuMemAlloc f_cuMemAlloc = nullptr;
    type_cuMemFree f_cuMemFree = nullptr;
    type_cuMemcpyDtoH f_cuMemcpyDtoH = nullptr;
    type_cuMemcpyHtoD f_cuMemcpyHtoD = nullptr;
    type_cuMemsetD32 f_cuMemsetD32 = nullptr;
    type_cuMemsetD8 f_cuMemsetD8 = nullptr;
    type_cuModuleGetFunction f_cuModuleGetFunction = nullptr;
    type_cuModuleLoadData f_cuModuleLoadData = nullptr;
    type_cuModuleUnload f_cuModuleUnload = nullptr;
    type_cuStreamBeginCapture f_cuStreamBeginCapture = nullptr;
    type_cuStreamCreate f_cuStreamCreate = nullptr;
    type_cuStreamDestroy f_cuStreamDestroy = nullptr;
    type_cuStreamEndCapture f_cuStreamEndCapture = nullptr;
    type_cuStreamSynchronize f_cuStreamSynchronize = nullptr;
    driver_api() {
#if defined(_WIN32)
        library_ = reinterpret_cast<void *>(LoadLibraryA("nvcuda.dll"));
#else
        library_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library_)
            throw std::runtime_error("NVIDIA CUDA driver unavailable");
        try {
            f_cuCtxSetCurrent = reinterpret_cast<type_cuCtxSetCurrent>(resolve(symbol_cuCtxSetCurrent));
            f_cuCtxSynchronize = reinterpret_cast<type_cuCtxSynchronize>(resolve(symbol_cuCtxSynchronize));
            f_cuDeviceGet = reinterpret_cast<type_cuDeviceGet>(resolve(symbol_cuDeviceGet));
            f_cuDeviceGetAttribute =
                reinterpret_cast<type_cuDeviceGetAttribute>(resolve(symbol_cuDeviceGetAttribute));
            f_cuDeviceGetCount = reinterpret_cast<type_cuDeviceGetCount>(resolve(symbol_cuDeviceGetCount));
            f_cuDeviceGetUuid = reinterpret_cast<type_cuDeviceGetUuid>(resolve(symbol_cuDeviceGetUuid));
            f_cuDevicePrimaryCtxRelease =
                reinterpret_cast<type_cuDevicePrimaryCtxRelease>(resolve(symbol_cuDevicePrimaryCtxRelease));
            f_cuDevicePrimaryCtxRetain =
                reinterpret_cast<type_cuDevicePrimaryCtxRetain>(resolve(symbol_cuDevicePrimaryCtxRetain));
            f_cuDeviceTotalMem = reinterpret_cast<type_cuDeviceTotalMem>(resolve(symbol_cuDeviceTotalMem));
            f_cuDriverGetVersion =
                reinterpret_cast<type_cuDriverGetVersion>(resolve(symbol_cuDriverGetVersion));
            f_cuFuncGetAttribute =
                reinterpret_cast<type_cuFuncGetAttribute>(resolve(symbol_cuFuncGetAttribute));
            f_cuFuncSetAttribute =
                reinterpret_cast<type_cuFuncSetAttribute>(resolve(symbol_cuFuncSetAttribute));
            f_cuGetErrorString = reinterpret_cast<type_cuGetErrorString>(resolve(symbol_cuGetErrorString));
            f_cuGraphAddKernelNode =
                reinterpret_cast<type_cuGraphAddKernelNode>(resolve(symbol_cuGraphAddKernelNode));
            f_cuGraphCreate = reinterpret_cast<type_cuGraphCreate>(resolve(symbol_cuGraphCreate));
            f_cuGraphDestroy = reinterpret_cast<type_cuGraphDestroy>(resolve(symbol_cuGraphDestroy));
            f_cuGraphExecDestroy =
                reinterpret_cast<type_cuGraphExecDestroy>(resolve(symbol_cuGraphExecDestroy));
            f_cuGraphGetNodes = reinterpret_cast<type_cuGraphGetNodes>(resolve(symbol_cuGraphGetNodes));
            f_cuGraphGetRootNodes =
                reinterpret_cast<type_cuGraphGetRootNodes>(resolve(symbol_cuGraphGetRootNodes));
            f_cuGraphInstantiate =
                reinterpret_cast<type_cuGraphInstantiate>(resolve(symbol_cuGraphInstantiate));
            f_cuGraphKernelNodeGetParams =
                reinterpret_cast<type_cuGraphKernelNodeGetParams>(resolve(symbol_cuGraphKernelNodeGetParams));
            f_cuGraphLaunch = reinterpret_cast<type_cuGraphLaunch>(resolve(symbol_cuGraphLaunch));
            f_cuGraphNodeGetDependentNodes = reinterpret_cast<type_cuGraphNodeGetDependentNodes>(
                resolve(symbol_cuGraphNodeGetDependentNodes));
            f_cuGraphNodeGetType =
                reinterpret_cast<type_cuGraphNodeGetType>(resolve(symbol_cuGraphNodeGetType));
            f_cuInit = reinterpret_cast<type_cuInit>(resolve(symbol_cuInit));
            f_cuLaunchKernel = reinterpret_cast<type_cuLaunchKernel>(resolve(symbol_cuLaunchKernel));
            f_cuMemAlloc = reinterpret_cast<type_cuMemAlloc>(resolve(symbol_cuMemAlloc));
            f_cuMemFree = reinterpret_cast<type_cuMemFree>(resolve(symbol_cuMemFree));
            f_cuMemcpyDtoH = reinterpret_cast<type_cuMemcpyDtoH>(resolve(symbol_cuMemcpyDtoH));
            f_cuMemcpyHtoD = reinterpret_cast<type_cuMemcpyHtoD>(resolve(symbol_cuMemcpyHtoD));
            f_cuMemsetD32 = reinterpret_cast<type_cuMemsetD32>(resolve(symbol_cuMemsetD32));
            f_cuMemsetD8 = reinterpret_cast<type_cuMemsetD8>(resolve(symbol_cuMemsetD8));
            f_cuModuleGetFunction =
                reinterpret_cast<type_cuModuleGetFunction>(resolve(symbol_cuModuleGetFunction));
            f_cuModuleLoadData = reinterpret_cast<type_cuModuleLoadData>(resolve(symbol_cuModuleLoadData));
            f_cuModuleUnload = reinterpret_cast<type_cuModuleUnload>(resolve(symbol_cuModuleUnload));
            f_cuStreamBeginCapture =
                reinterpret_cast<type_cuStreamBeginCapture>(resolve(symbol_cuStreamBeginCapture));
            f_cuStreamCreate = reinterpret_cast<type_cuStreamCreate>(resolve(symbol_cuStreamCreate));
            f_cuStreamDestroy = reinterpret_cast<type_cuStreamDestroy>(resolve(symbol_cuStreamDestroy));
            f_cuStreamEndCapture =
                reinterpret_cast<type_cuStreamEndCapture>(resolve(symbol_cuStreamEndCapture));
            f_cuStreamSynchronize =
                reinterpret_cast<type_cuStreamSynchronize>(resolve(symbol_cuStreamSynchronize));
        } catch (...) {
#if defined(_WIN32)
            FreeLibrary(reinterpret_cast<HMODULE>(library_));
#else
            dlclose(library_);
#endif
            throw;
        }
    }
    // Keep the driver loaded until process exit, including static-object teardown.
};
inline driver_api &driver() {
    static driver_api api;
    return api;
}
} // namespace cuda_gfn
#ifdef cuCtxSetCurrent
#undef cuCtxSetCurrent
#endif
#define cuCtxSetCurrent(...) (::cuda_gfn::driver().f_cuCtxSetCurrent(__VA_ARGS__))
#ifdef cuCtxSynchronize
#undef cuCtxSynchronize
#endif
#define cuCtxSynchronize(...) (::cuda_gfn::driver().f_cuCtxSynchronize(__VA_ARGS__))
#ifdef cuDeviceGet
#undef cuDeviceGet
#endif
#define cuDeviceGet(...) (::cuda_gfn::driver().f_cuDeviceGet(__VA_ARGS__))
#ifdef cuDeviceGetAttribute
#undef cuDeviceGetAttribute
#endif
#define cuDeviceGetAttribute(...) (::cuda_gfn::driver().f_cuDeviceGetAttribute(__VA_ARGS__))
#ifdef cuDeviceGetCount
#undef cuDeviceGetCount
#endif
#define cuDeviceGetCount(...) (::cuda_gfn::driver().f_cuDeviceGetCount(__VA_ARGS__))
#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif
#define cuDeviceGetUuid(...) (::cuda_gfn::driver().f_cuDeviceGetUuid(__VA_ARGS__))
#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
#define cuDevicePrimaryCtxRelease(...) (::cuda_gfn::driver().f_cuDevicePrimaryCtxRelease(__VA_ARGS__))
#ifdef cuDevicePrimaryCtxRetain
#undef cuDevicePrimaryCtxRetain
#endif
#define cuDevicePrimaryCtxRetain(...) (::cuda_gfn::driver().f_cuDevicePrimaryCtxRetain(__VA_ARGS__))
#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
#define cuDeviceTotalMem(...) (::cuda_gfn::driver().f_cuDeviceTotalMem(__VA_ARGS__))
#ifdef cuDriverGetVersion
#undef cuDriverGetVersion
#endif
#define cuDriverGetVersion(...) (::cuda_gfn::driver().f_cuDriverGetVersion(__VA_ARGS__))
#ifdef cuFuncGetAttribute
#undef cuFuncGetAttribute
#endif
#define cuFuncGetAttribute(...) (::cuda_gfn::driver().f_cuFuncGetAttribute(__VA_ARGS__))
#ifdef cuFuncSetAttribute
#undef cuFuncSetAttribute
#endif
#define cuFuncSetAttribute(...) (::cuda_gfn::driver().f_cuFuncSetAttribute(__VA_ARGS__))
#ifdef cuGetErrorString
#undef cuGetErrorString
#endif
#define cuGetErrorString(...) (::cuda_gfn::driver().f_cuGetErrorString(__VA_ARGS__))
#ifdef cuGraphAddKernelNode
#undef cuGraphAddKernelNode
#endif
#define cuGraphAddKernelNode(...) (::cuda_gfn::driver().f_cuGraphAddKernelNode(__VA_ARGS__))
#ifdef cuGraphCreate
#undef cuGraphCreate
#endif
#define cuGraphCreate(...) (::cuda_gfn::driver().f_cuGraphCreate(__VA_ARGS__))
#ifdef cuGraphDestroy
#undef cuGraphDestroy
#endif
#define cuGraphDestroy(...) (::cuda_gfn::driver().f_cuGraphDestroy(__VA_ARGS__))
#ifdef cuGraphExecDestroy
#undef cuGraphExecDestroy
#endif
#define cuGraphExecDestroy(...) (::cuda_gfn::driver().f_cuGraphExecDestroy(__VA_ARGS__))
#ifdef cuGraphGetNodes
#undef cuGraphGetNodes
#endif
#define cuGraphGetNodes(...) (::cuda_gfn::driver().f_cuGraphGetNodes(__VA_ARGS__))
#ifdef cuGraphGetRootNodes
#undef cuGraphGetRootNodes
#endif
#define cuGraphGetRootNodes(...) (::cuda_gfn::driver().f_cuGraphGetRootNodes(__VA_ARGS__))
#ifdef cuGraphInstantiate
#undef cuGraphInstantiate
#endif
#define cuGraphInstantiate(...) (::cuda_gfn::driver().f_cuGraphInstantiate(__VA_ARGS__))
#ifdef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeGetParams
#endif
#define cuGraphKernelNodeGetParams(...) (::cuda_gfn::driver().f_cuGraphKernelNodeGetParams(__VA_ARGS__))
#ifdef cuGraphLaunch
#undef cuGraphLaunch
#endif
#define cuGraphLaunch(...) (::cuda_gfn::driver().f_cuGraphLaunch(__VA_ARGS__))
#ifdef cuGraphNodeGetDependentNodes
#undef cuGraphNodeGetDependentNodes
#endif
#define cuGraphNodeGetDependentNodes(...) (::cuda_gfn::driver().f_cuGraphNodeGetDependentNodes(__VA_ARGS__))
#ifdef cuGraphNodeGetType
#undef cuGraphNodeGetType
#endif
#define cuGraphNodeGetType(...) (::cuda_gfn::driver().f_cuGraphNodeGetType(__VA_ARGS__))
#ifdef cuInit
#undef cuInit
#endif
#define cuInit(...) (::cuda_gfn::driver().f_cuInit(__VA_ARGS__))
#ifdef cuLaunchKernel
#undef cuLaunchKernel
#endif
#define cuLaunchKernel(...) (::cuda_gfn::driver().f_cuLaunchKernel(__VA_ARGS__))
#ifdef cuMemAlloc
#undef cuMemAlloc
#endif
#define cuMemAlloc(...) (::cuda_gfn::driver().f_cuMemAlloc(__VA_ARGS__))
#ifdef cuMemFree
#undef cuMemFree
#endif
#define cuMemFree(...) (::cuda_gfn::driver().f_cuMemFree(__VA_ARGS__))
#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
#define cuMemcpyDtoH(...) (::cuda_gfn::driver().f_cuMemcpyDtoH(__VA_ARGS__))
#ifdef cuMemcpyHtoD
#undef cuMemcpyHtoD
#endif
#define cuMemcpyHtoD(...) (::cuda_gfn::driver().f_cuMemcpyHtoD(__VA_ARGS__))
#ifdef cuMemsetD32
#undef cuMemsetD32
#endif
#define cuMemsetD32(...) (::cuda_gfn::driver().f_cuMemsetD32(__VA_ARGS__))
#ifdef cuMemsetD8
#undef cuMemsetD8
#endif
#define cuMemsetD8(...) (::cuda_gfn::driver().f_cuMemsetD8(__VA_ARGS__))
#ifdef cuModuleGetFunction
#undef cuModuleGetFunction
#endif
#define cuModuleGetFunction(...) (::cuda_gfn::driver().f_cuModuleGetFunction(__VA_ARGS__))
#ifdef cuModuleLoadData
#undef cuModuleLoadData
#endif
#define cuModuleLoadData(...) (::cuda_gfn::driver().f_cuModuleLoadData(__VA_ARGS__))
#ifdef cuModuleUnload
#undef cuModuleUnload
#endif
#define cuModuleUnload(...) (::cuda_gfn::driver().f_cuModuleUnload(__VA_ARGS__))
#ifdef cuStreamBeginCapture
#undef cuStreamBeginCapture
#endif
#define cuStreamBeginCapture(...) (::cuda_gfn::driver().f_cuStreamBeginCapture(__VA_ARGS__))
#ifdef cuStreamCreate
#undef cuStreamCreate
#endif
#define cuStreamCreate(...) (::cuda_gfn::driver().f_cuStreamCreate(__VA_ARGS__))
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
#define cuStreamDestroy(...) (::cuda_gfn::driver().f_cuStreamDestroy(__VA_ARGS__))
#ifdef cuStreamEndCapture
#undef cuStreamEndCapture
#endif
#define cuStreamEndCapture(...) (::cuda_gfn::driver().f_cuStreamEndCapture(__VA_ARGS__))
#ifdef cuStreamSynchronize
#undef cuStreamSynchronize
#endif
#define cuStreamSynchronize(...) (::cuda_gfn::driver().f_cuStreamSynchronize(__VA_ARGS__))
