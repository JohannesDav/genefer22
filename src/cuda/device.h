// CUDA driver interface for the fixed GFN-19 PTX kernels.
#pragma once
#include <cuda.h>
#include <array>
#include <cstdlib>
#include <stdexcept>

inline void cudaCheck(CUresult result)
{
    if (result == CUDA_SUCCESS)
        return;
    const char *msg = nullptr;
    cuGetErrorString(result, &msg);
    throw std::runtime_error(std::string("CUDA: ") + (msg ? msg : "unknown error"));
}
class device
{
    struct Kernel
    {
        CUfunction function;
        std::array<CUfunction, 2> dupFunctions{};
        std::string name;
        std::array<std::array<unsigned char, 16>, 8> args{};
        std::array<void *, 8> params{};
        Kernel(CUfunction f, const char *n) : function(f), name(n)
        {
            for (size_t i = 0; i < args.size(); ++i)
                params[i] = args[i].data();
        }
    };
    CUcontext _context{};
    CUdevice _device{};
    CUmodule _module{};
    CUstream _stream{};
    CUaccessPolicyWindow _dataWindow{};
    bool _dataPersist = false;
    std::array<CUgraphExec, 2> _squareGraphs{};
    static Kernel *kernel(cl_kernel k)
    {
        return reinterpret_cast<Kernel *>(k);
    }
    // Host-side deferred single-candidate squarings. Kernel arithmetic is unchanged.
    std::array<CUgraph, 2> _squareTemplates{};
    std::array<std::array<CUgraphNode, 2>, 2> _squareTemplateNodes{};
    std::map<unsigned, CUgraphExec> _squareGroups;
    unsigned _squareSpan = 1, _pendingSquares = 0, _pendingMask = 0;
    bool _capturingSquare = false;

    void retainSquareTemplate(unsigned bit, CUgraph graph)
    {
        size_t n = 0;
        cudaCheck(cuGraphGetNodes(graph, nullptr, &n));
        if (n != 2)
            throw std::runtime_error("Grouped square requires two kernel nodes");
        std::array<CUgraphNode, 2> nodes{};
        cudaCheck(cuGraphGetNodes(graph, nodes.data(), &n));
        size_t roots = 0;
        cudaCheck(cuGraphGetRootNodes(graph, nullptr, &roots));
        if (roots != 1)
            throw std::runtime_error("Grouped square requires one graph root");
        CUgraphNode first{};
        cudaCheck(cuGraphGetRootNodes(graph, &first, &roots));
        _squareTemplateNodes[bit] = {first, nodes[0] == first ? nodes[1] : nodes[0]};
        for (auto node : _squareTemplateNodes[bit])
        {
            CUgraphNodeType type;
            cudaCheck(cuGraphNodeGetType(node, &type));
            if (type != CU_GRAPH_NODE_TYPE_KERNEL)
                throw std::runtime_error("Grouped square has non-kernel node");
        }
        _squareTemplates[bit] = graph;
    }
    void flushSquares()
    {
        if (!_pendingSquares)
            return;
        if (_pendingSquares < _squareSpan)
        {
            for (unsigned i = 0; i < _pendingSquares; ++i)
                cudaCheck(cuGraphLaunch(_squareGraphs[(_pendingMask >> i) & 1], _stream));
        }
        else
        {
            auto &exec = _squareGroups[_pendingMask];
            if (!exec)
            {
                CUgraph graph;
                cudaCheck(cuGraphCreate(&graph, 0));
                CUgraphNode previous{};
                for (unsigned i = 0; i < _pendingSquares; ++i)
                {
                    unsigned bit = (_pendingMask >> i) & 1;
                    for (auto src : _squareTemplateNodes[bit])
                    {
                        CUDA_KERNEL_NODE_PARAMS params{};
                        cudaCheck(cuGraphKernelNodeGetParams(src, &params));
                        CUgraphNode dst;
                        cudaCheck(cuGraphAddKernelNode(&dst, graph, previous ? &previous : nullptr,
                                                       previous ? 1 : 0, &params));
                        if (_dataPersist)
                        {
                            CUkernelNodeAttrValue attr{};
                            attr.accessPolicyWindow = _dataWindow;
                            cudaCheck(cuGraphKernelNodeSetAttribute(
                                dst, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
                            CUkernelNodeAttrValue got{};
                            cudaCheck(cuGraphKernelNodeGetAttribute(
                                dst, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &got));
                            if (got.accessPolicyWindow.base_ptr != _dataWindow.base_ptr ||
                                got.accessPolicyWindow.num_bytes != _dataWindow.num_bytes ||
                                got.accessPolicyWindow.hitProp != CU_ACCESS_PROPERTY_PERSISTING ||
                                got.accessPolicyWindow.hitRatio != 1.0f)
                                throw std::runtime_error(
                                    "Grouped graph persistence readback mismatch");
                        }
                        previous = dst;
                    }
                }
                cudaCheck(cuGraphInstantiate(&exec, graph, 0));
                cudaCheck(cuGraphDestroy(graph));
            }
            cudaCheck(cuGraphLaunch(exec, _stream));
        }
        _pendingSquares = 0;
        _pendingMask = 0;
    }

  public:
    device(const platform &, size_t d, bool verbose)
    {
        cudaCheck(cuInit(0));
        CUdevice dev;
        cudaCheck(cuDeviceGet(&dev, static_cast<int>(d)));
        _device = dev;
        char card[256]{};
        cudaCheck(cuDeviceGetName(card, sizeof(card), dev));
#if defined(GENEFER_GUARD_3090)
        const std::string expected = "NVIDIA GeForce RTX 3090";
#else
        const std::string expected = "NVIDIA GeForce RTX 4060 Ti";
        size_t memory = 0;
        cudaCheck(cuDeviceTotalMem(&memory, dev));
        if (memory < size_t(15000) * 1024 * 1024)
            throw std::runtime_error("The 4060 Ti profile requires the 16GB card");
#endif
        if (card != expected)
            throw std::runtime_error("GPU does not match the selected GFN-19 profile");

        // Primary context allows tools and the runtime to share the same context.
        cudaCheck(cuDevicePrimaryCtxRetain(&_context, dev));
        cudaCheck(cuCtxSetCurrent(_context));
        cudaCheck(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
        if (const char *span = std::getenv("CUDA_GRAPH_SPAN"))
        {
            _squareSpan = std::stoul(span);
            if (_squareSpan != 1 && _squareSpan != 4 && _squareSpan != 8)
                throw std::runtime_error("CUDA_GRAPH_SPAN must be 1,4,8");
        }
        size_t initialPersist = 0;
        cudaCheck(cuCtxGetLimit(&initialPersist, CU_LIMIT_PERSISTING_L2_CACHE_SIZE));
        cudaCheck(cuCtxResetPersistingL2Cache());
        cudaCheck(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 0));
        std::cout << "PERSIST_INITIAL bytes=" << initialPersist << " reset=1" << std::endl;

        if (verbose)
        {
            char name[256];
            cudaCheck(cuDeviceGetName(name, sizeof(name), dev));
            pio::print(std::string("CUDA PTX backend: ") + name + "\n");
        }
    }
    virtual ~device()
    {
        flushSquares();
        cuStreamSynchronize(_stream);
        if (_dataPersist)
        {
            CUstreamAttrValue attr{};
            cudaCheck(
                cuStreamSetAttribute(_stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
            cudaCheck(cuCtxResetPersistingL2Cache());
            cudaCheck(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 0));
        }
        cuStreamDestroy(_stream);
        cuDevicePrimaryCtxRelease(_device);
    }
    size_t getMaxWorkGroupSize() const
    {
        return 1024;
    }
    size_t getLocalMemSize() const
    {
        return 49152;
    }
    size_t getTimerResolution() const
    {
        return 1;
    }
    bool isIntel() const
    {
        return false;
    }
    void loadProgram(const std::string &source)
    {
        const char *path = std::getenv("CUDA_PTX");
        if (!path)
            throw std::runtime_error("CUDA_PTX must specify matching exported kernel PTX");
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("Cannot open CUDA PTX");
        std::string ptx(std::istreambuf_iterator<char>(input), {});
        // NVIDIA OpenCL encodes get_global_offset(0) in envreg3. CUDA does not
        // initialize OpenCL's environment registers. Every Genefer launch has
        // offset zero, so specialize this register instead of reading garbage.
        size_t pos = 0;
        const std::string offset = "%envreg3";
        while ((pos = ptx.find(offset, pos)) != std::string::npos)
            ptx.replace(pos, offset.size(), "0");
        if (ptx.find("%envreg") != std::string::npos)
            throw std::runtime_error("Unsupported OpenCL environment register in PTX");
        const std::string marker = "// GENEFER_CONFIG ";
        const size_t identity = ptx.find(marker);
        const size_t end =
            identity == std::string::npos ? std::string::npos : ptx.find('\n', identity);
        if (identity == std::string::npos || end == std::string::npos ||
            ptx.substr(identity + marker.size(), end - identity - marker.size()) != source)
            throw std::runtime_error(
                "PTX configuration does not match this transform; rebuild kernels");
        cudaCheck(cuModuleLoadData(&_module, ptx.c_str()));
    }
    template <class F> void runSquareGraph(bool dup, F launch)
    {
        if (!std::getenv("CUDA_GRAPH"))
        {
            flushSquares();
            launch();
            return;
        }
        auto &exec = _squareGraphs[dup ? 1 : 0];
        if (!exec)
        {
            CUgraph graph;
            cudaCheck(cuStreamBeginCapture(_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            _capturingSquare = true;
            launch();
            _capturingSquare = false;
            cudaCheck(cuStreamEndCapture(_stream, &graph));
            if (_dataPersist)
            {
                size_t count = 0;
                cudaCheck(cuGraphGetNodes(graph, nullptr, &count));
                std::vector<CUgraphNode> nodes(count);
                cudaCheck(cuGraphGetNodes(graph, nodes.data(), &count));
                size_t changed = 0;
                for (auto node : nodes)
                {
                    CUgraphNodeType type;
                    cudaCheck(cuGraphNodeGetType(node, &type));
                    if (type != CU_GRAPH_NODE_TYPE_KERNEL)
                        continue;
                    CUkernelNodeAttrValue attr{};
                    attr.accessPolicyWindow = _dataWindow;
                    cudaCheck(cuGraphKernelNodeSetAttribute(
                        node, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
                    CUkernelNodeAttrValue check{};
                    cudaCheck(cuGraphKernelNodeGetAttribute(
                        node, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &check));
                    if (check.accessPolicyWindow.base_ptr != _dataWindow.base_ptr ||
                        check.accessPolicyWindow.num_bytes != _dataWindow.num_bytes ||
                        check.accessPolicyWindow.hitProp != CU_ACCESS_PROPERTY_PERSISTING ||
                        check.accessPolicyWindow.hitRatio != 1.0f)
                        throw std::runtime_error("Graph persistence attribute readback mismatch");
                    ++changed;
                }
                std::cout << "PERSIST_GRAPH kernels=" << changed
                          << " bytes=" << _dataWindow.num_bytes << " verified=1" << std::endl;
            }
            cudaCheck(cuGraphInstantiate(&exec, graph, 0));
            if (_squareSpan > 1)
                retainSquareTemplate(dup ? 1 : 0, graph);
            else
                cudaCheck(cuGraphDestroy(graph));
        }
        if (_squareSpan == 1)
            cudaCheck(cuGraphLaunch(exec, _stream));
        else
        {
            _pendingMask |= unsigned(dup) << _pendingSquares;
            if (++_pendingSquares == _squareSpan)
                flushSquares();
        }
    }
    void configureDataWindow(cl_mem memory, size_t bytes)
    {
        flushSquares();
        if (!std::getenv("CUDA_DATA_PERSIST"))
            return;
        const size_t request = size_t(std::stoul(std::getenv("CUDA_DATA_PERSIST"))) * 1024 * 1024;
        int maxPersist = 0, maxWindow = 0, l2 = 0;
        cudaCheck(cuDeviceGetAttribute(&maxPersist,
                                       CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, _device));
        cudaCheck(cuDeviceGetAttribute(&maxWindow,
                                       CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, _device));
        cudaCheck(cuDeviceGetAttribute(&l2, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, _device));
        if (request > size_t(maxPersist) || bytes > size_t(maxWindow))
            throw std::runtime_error("Requested persistence exceeds device limit");
        cudaCheck(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, request));
        size_t actual = 0;
        cudaCheck(cuCtxGetLimit(&actual, CU_LIMIT_PERSISTING_L2_CACHE_SIZE));
        _dataWindow.base_ptr = reinterpret_cast<void *>(memory);
        _dataWindow.num_bytes = bytes;
        _dataWindow.hitRatio = 1.0f;
        _dataWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
        _dataWindow.missProp = CU_ACCESS_PROPERTY_NORMAL;
        CUstreamAttrValue attr{};
        attr.accessPolicyWindow = _dataWindow;
        cudaCheck(cuStreamSetAttribute(_stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
        _dataPersist = true;
        std::cout << "PERSIST_CONFIG requested=" << request << " actual=" << actual
                  << " window=" << bytes << " l2=" << l2 << " max_persist=" << maxPersist
                  << std::endl;
    }
    void clearProgram()
    {
        flushSquares();
        cudaCheck(cuStreamSynchronize(_stream));
        for (auto &p : _squareGroups)
            cudaCheck(cuGraphExecDestroy(p.second));
        _squareGroups.clear();
        for (auto &g : _squareTemplates)
            if (g)
            {
                cudaCheck(cuGraphDestroy(g));
                g = nullptr;
            }
        for (auto &g : _squareGraphs)
            if (g)
            {
                cudaCheck(cuGraphExecDestroy(g));
                g = nullptr;
            }
        cudaCheck(cuStreamSynchronize(_stream));
        cudaCheck(cuModuleUnload(_module));
        _module = nullptr;
    }

  protected:
    cl_mem _createBuffer(cl_mem_flags, size_t bytes, bool clear = true) const
    {
        CUdeviceptr ptr;
        cudaCheck(cuMemAlloc(&ptr, bytes));
        if (clear)
        {
            cudaCheck(cuMemsetD8(ptr, 0, bytes));
            cudaCheck(cuCtxSynchronize());
        }
        return reinterpret_cast<cl_mem>(ptr);
    }
    void _releaseBuffer(cl_mem &mem)
    {
        flushSquares();
        cudaCheck(cuStreamSynchronize(_stream));
        if (mem)
            cudaCheck(cuMemFree(reinterpret_cast<CUdeviceptr>(mem)));
        mem = nullptr;
    }
    void _readBuffer(cl_mem &mem, void *ptr, size_t bytes, size_t offset = 0)
    {
        flushSquares();
        cudaCheck(cuStreamSynchronize(_stream));
        cudaCheck(cuMemcpyDtoH(ptr, reinterpret_cast<CUdeviceptr>(mem) + offset, bytes));
    }
    void _writeBuffer(cl_mem &mem, const void *ptr, size_t bytes, size_t offset = 0)
    {
        flushSquares();
        cudaCheck(cuStreamSynchronize(_stream));
        cudaCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mem) + offset, ptr, bytes));
        // Pageable H2D may return after staging but before DMA completes. Our
        // nonblocking stream does not implicitly wait for the default stream.
        cudaCheck(cuCtxSynchronize());
    }
    cl_kernel _createKernel(const char *name)
    {
        CUfunction f;
        cudaCheck(cuModuleGetFunction(&f, _module, name));
        if (const char *carveout = std::getenv("CUDA_CACHE"))
            cudaCheck(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                         std::atoi(carveout)));
        if (std::string(name) == "square2048" || std::string(name) == "boundary_bridge_signed")
        {
            const bool inner = std::string(name) == "square2048";
            const char *setting = std::getenv(inner ? "CUDA_INNER_CACHE" : "CUDA_BRIDGE_CACHE");
            if (setting)
                cudaCheck(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                             std::stoi(setting)));
        }
        auto *k = new Kernel(f, name);
        if (std::string(name) == "boundary_bridge_signed" && std::getenv("CUDA_DUP_SPECIAL"))
        {
            int pref = 50;
            if (const char *v = std::getenv("CUDA_CACHE"))
                pref = std::stoi(v);
            if (const char *v = std::getenv("CUDA_BRIDGE_CACHE"))
                pref = std::stoi(v);
            for (unsigned bit = 0; bit < 2; ++bit)
            {
                std::string entry = "boundary_bridge_signed_dup" + std::to_string(bit);
                auto &fn = k->dupFunctions[bit];
                cudaCheck(cuModuleGetFunction(&fn, _module, entry.c_str()));
                cudaCheck(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                             pref));
            }
        }
        return reinterpret_cast<cl_kernel>(k);
    }
    void _releaseKernel(cl_kernel &k)
    {
        flushSquares();
        delete kernel(k);
        k = nullptr;
    }
    static void _setKernelArg(cl_kernel k, cl_uint index, size_t bytes, const void *value)
    {
        if (index >= 8 || bytes > 16)
            throw std::runtime_error("CUDA argument bounds");
        std::memcpy(kernel(k)->args[index].data(), value, bytes);
    }
    void _executeKernel(cl_kernel k, size_t global, size_t local = 0)
    {
        if (!_capturingSquare)
            flushSquares();
        if (!local)
            local = std::min(size_t(128), global);
        if (global % local)
            throw std::runtime_error("CUDA launch size must divide global size");
        auto *p = kernel(k);
        CUfunction selected = p->function;
        if (p->dupFunctions[0])
        {
            int32_t bit = 0;
            std::memcpy(&bit, p->args[7].data(), sizeof(bit));
            if (bit != 0 && bit != 1)
                throw std::runtime_error("Specialized bridge duplication must be boolean");
            selected = p->dupFunctions[bit];
        }
        cudaCheck(cuLaunchKernel(selected, global / local, 1, 1, local, 1, 1, 0, _stream,
                                 p->params.data(), nullptr));
    }
};
