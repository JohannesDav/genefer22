// CUDA driver execution of NVIDIA OpenCL C kernels.
#pragma once
#include "device_selection.h"
#include "driver_api.h"
#include <array>
#include <map>
#include <cstring>
#include <vector>
#include <regex>

inline void cuda_check(CUresult result)
{
    if (result==CUDA_SUCCESS) return;
    const char *message=nullptr;
    cuGetErrorString(result,&message);
    throw std::runtime_error(std::string("CUDA: ")+(message?message:"unknown error"));
}

class cuda_device : protected oclObject
{
    struct kernel
    {
        CUfunction function{};
        std::array<std::array<unsigned char,16>,8> arguments{};
        std::array<void *,8> pointers{};
        explicit kernel(CUfunction f):function(f)
        {
            for(size_t i=0;i<pointers.size();++i) pointers[i]=arguments[i].data();
        }
    };
    cl_device_id opencl_device_{};
    CUdevice device_{};
    CUcontext context_{};
    CUstream stream_{};
    CUmodule module_{};
    std::array<CUgraph,2> templates_{};
    std::array<CUgraphExec,2> squares_{};
    std::array<std::vector<CUgraphNode>,2> nodes_;
    std::map<unsigned,CUgraphExec> batches_;
    unsigned pending_=0, mask_=0;
    bool capturing_=false;

    static kernel *handle(cl_kernel value) { return reinterpret_cast<kernel *>(value); }
    static CUresult dependents(CUgraphNode node,CUgraphNode *next,size_t *count)
    {
#if CUDA_VERSION >= 13000
        return cuGraphNodeGetDependentNodes(node,next,nullptr,count);
#else
        return cuGraphNodeGetDependentNodes(node,next,count);
#endif
    }
    void remember(unsigned bit,CUgraph graph)
    {
        size_t count=0,roots=0;
        cuda_check(cuGraphGetNodes(graph,nullptr,&count));
        cuda_check(cuGraphGetRootNodes(graph,nullptr,&roots));
        if(count==0||count>64||roots!=1) throw std::runtime_error("Unexpected CUDA square graph");
        CUgraphNode node{};
        cuda_check(cuGraphGetRootNodes(graph,&node,&roots));
        for(size_t i=0;i<count;++i)
        {
            CUgraphNodeType type;
            cuda_check(cuGraphNodeGetType(node,&type));
            if(type!=CU_GRAPH_NODE_TYPE_KERNEL) throw std::runtime_error("Non-kernel square graph node");
            nodes_[bit].push_back(node);
            size_t successors=0;
            cuda_check(dependents(node,nullptr,&successors));
            if(successors!=(i+1<count?1u:0u)) throw std::runtime_error("Non-serial square graph");
            if(successors) { CUgraphNode next{};cuda_check(dependents(node,&next,&successors));node=next; }
        }
        templates_[bit]=graph;
    }
    void flush()
    {
        if(!pending_) return;
        if(pending_<8)
        {
            for(unsigned i=0;i<pending_;++i) cuda_check(cuGraphLaunch(squares_[(mask_>>i)&1],stream_));
        }
        else
        {
            auto &batch=batches_[mask_];
            if(!batch)
            {
                CUgraph graph{};
                cuda_check(cuGraphCreate(&graph,0));
                CUgraphNode previous{};
                for(unsigned i=0;i<8;++i) for(auto original:nodes_[(mask_>>i)&1])
                {
                    CUDA_KERNEL_NODE_PARAMS parameters{};
                    cuda_check(cuGraphKernelNodeGetParams(original,&parameters));
                    CUgraphNode next{};
                    cuda_check(cuGraphAddKernelNode(&next,graph,previous?&previous:nullptr,previous?1:0,&parameters));
                    previous=next;
                }
                cuda_check(cuGraphInstantiate(&batch,graph,0));
                cuda_check(cuGraphDestroy(graph));
            }
            cuda_check(cuGraphLaunch(batch,stream_));
        }
        pending_=0;mask_=0;
    }
    // Use the selected GPU's NVIDIA compiler, then restore our CUDA context.
    // No OpenCL buffers or queues are shared with CUDA.
    std::string compile(const std::string &source)
    {
        cl_context context=nullptr;
        cl_program program=nullptr;
        std::string binary;
        try
        {
            cl_int error;
            context=clCreateContext(nullptr,1,&opencl_device_,nullptr,nullptr,&error);
            oclFatal(error,"create compiler context");
            const char *text=source.data();const size_t length=source.size();
            program=clCreateProgramWithSource(context,1,&text,&length,&error);
            oclFatal(error,"create GPU program");
            error=clBuildProgram(program,1,&opencl_device_,"",nullptr,nullptr);
            if(error!=CL_SUCCESS)
            {
                size_t bytes=0;
                clGetProgramBuildInfo(program,opencl_device_,CL_PROGRAM_BUILD_LOG,0,nullptr,&bytes);
                std::vector<char> log(bytes+1);
                clGetProgramBuildInfo(program,opencl_device_,CL_PROGRAM_BUILD_LOG,bytes,log.data(),nullptr);
                throw std::runtime_error(std::string("NVIDIA kernel compilation failed: ")+log.data());
            }
            size_t bytes=0;
            oclFatal(clGetProgramInfo(program,CL_PROGRAM_BINARY_SIZES,sizeof(bytes),&bytes,nullptr));
            std::vector<char> data(bytes);char *destination=data.data();
            oclFatal(clGetProgramInfo(program,CL_PROGRAM_BINARIES,sizeof(destination),&destination,nullptr));
            while(!data.empty()&&data.back()==0) data.pop_back();
            binary.assign(data.begin(),data.end());
        }
        catch(...)
        {
            if(program) clReleaseProgram(program);
            if(context) clReleaseContext(context);
            cuCtxSetCurrent(context_);
            throw;
        }
        clReleaseProgram(program);clReleaseContext(context);
        cuda_check(cuCtxSetCurrent(context_));
        if(binary.find(".version ")==std::string::npos||binary.find(".entry ")==std::string::npos)
            throw std::runtime_error("NVIDIA OpenCL compiler did not export PTX");
        // Every launch has global offset zero. CUDA does not initialize the
        // environment register that OpenCL uses for get_global_offset(0).
        size_t pos=0;
        while((pos=binary.find("%envreg3",pos))!=std::string::npos) {
            // Do not mistake another environment register, such as envreg30,
            // for the global-offset register. Unsupported registers stay below.
            if(pos+8<binary.size() && binary[pos+8]>='0' && binary[pos+8]<='9') {
                pos+=8;
                continue;
            }
            binary.replace(pos,8,"0");
            ++pos;
        }
        if(binary.find("%envreg")!=std::string::npos) throw std::runtime_error("Unsupported OpenCL PTX environment register");
        return binary;
    }
public:
    cuda_device(const platform &parent,size_t index,bool verbose)
    {
        try {
        opencl_device_=nvidia_device(parent,index);
        cuda_check(cuInit(0));
        // cl_khr_device_uuid identifies the same physical GPU across APIs.
        std::array<unsigned char,16> uuid{};
        oclFatal(clGetDeviceInfo(opencl_device_,0x106A,uuid.size(),uuid.data(),nullptr),"query OpenCL device UUID");
        int count=0,matches=0;
        cuda_check(cuDeviceGetCount(&count));
        for(int i=0;i<count;++i)
        {
            CUdevice candidate;CUuuid other{};
            cuda_check(cuDeviceGet(&candidate,i));cuda_check(cuDeviceGetUuid(&other,candidate));
            if(std::memcmp(uuid.data(),other.bytes,uuid.size())==0) { device_=candidate;++matches; }
        }
        if(matches!=1) throw std::runtime_error("Cannot match the assigned OpenCL GPU to CUDA");
        cuda_check(cuDevicePrimaryCtxRetain(&context_,device_));
        cuda_check(cuCtxSetCurrent(context_));
        cuda_check(cuStreamCreate(&stream_,CU_STREAM_NON_BLOCKING));
        if(verbose) pio::print("CUDA kernel execution\n");
        } catch (...) {
            if(stream_) cuStreamDestroy(stream_);
            if(context_) cuDevicePrimaryCtxRelease(device_);
            throw;
        }
    }
    virtual ~cuda_device()
    {
        try { flush(); } catch(...) {}
        if(stream_) cuStreamSynchronize(stream_);
        for(auto &entry:batches_) if(entry.second) cuGraphExecDestroy(entry.second);
        for(auto graph:squares_) if(graph) cuGraphExecDestroy(graph);
        for(auto graph:templates_) if(graph) cuGraphDestroy(graph);
        if(module_) cuModuleUnload(module_);
        if(stream_) cuStreamDestroy(stream_);
        if(context_) cuDevicePrimaryCtxRelease(device_);
    }
    void synchronizeForCalibration() { flush(); cuda_check(cuStreamSynchronize(stream_)); }
    size_t getMaxWorkGroupSize() const { return 1024; }
    size_t getLocalMemSize() const { return 49152; }
    size_t getTimerResolution() const { return 1; }
    bool isIntel() const { return false; }
    void setProfiling(bool enabled) { if(enabled) throw std::runtime_error("CUDA profiling is not supported"); }
    void resetProfiles() {}
    cl_ulong getProfileTime() const { return 0; }
    void displayProfiles(size_t) const { throw std::runtime_error("CUDA profiling is not supported"); }
    void loadProgram(const std::string &source)
    {
        std::string binary=compile(source);
        cuda_check(cuModuleLoadData(&module_,binary.c_str()));
    }
    void clearProgram() { flush();cuda_check(cuStreamSynchronize(stream_)); }
    template<class F> void runSquareGraph(bool doubled,F launch)
    {
        const unsigned bit=doubled?1:0;
        if(!squares_[bit])
        {
            CUgraph graph{};
            cuda_check(cuStreamBeginCapture(stream_,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            capturing_=true;launch();capturing_=false;
            cuda_check(cuStreamEndCapture(stream_,&graph));
            cuda_check(cuGraphInstantiate(&squares_[bit],graph,0));
            remember(bit,graph);
        }
        mask_|=bit<<pending_;
        if(++pending_==8) flush();
    }
protected:
    cl_mem _createBuffer(cl_mem_flags,size_t bytes,bool clear=true)
    {
        CUdeviceptr data;cuda_check(cuMemAlloc(&data,bytes));
        if(clear) {cuda_check(cuMemsetD8(data,0,bytes));cuda_check(cuCtxSynchronize());}
        return reinterpret_cast<cl_mem>(data);
    }
    void _releaseBuffer(cl_mem &data)
    {
        flush();cuda_check(cuStreamSynchronize(stream_));
        if(data) cuda_check(cuMemFree(reinterpret_cast<CUdeviceptr>(data)));
        data=nullptr;
    }
    void _readBuffer(cl_mem &data,void *destination,size_t bytes,size_t offset=0)
    {
        flush();cuda_check(cuStreamSynchronize(stream_));
        cuda_check(cuMemcpyDtoH(destination,reinterpret_cast<CUdeviceptr>(data)+offset,bytes));
    }
    void _writeBuffer(cl_mem &data,const void *source,size_t bytes,size_t offset=0)
    {
        flush();cuda_check(cuStreamSynchronize(stream_));
        cuda_check(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(data)+offset,source,bytes));
        cuda_check(cuCtxSynchronize());
    }
    cl_kernel _createKernel(const char *name)
    {
        CUfunction function;cuda_check(cuModuleGetFunction(&function,module_,name));
        return reinterpret_cast<cl_kernel>(new kernel(function));
    }
    void _releaseKernel(cl_kernel &value) {flush();delete handle(value);value=nullptr;}
    static void _setKernelArg(cl_kernel value,cl_uint index,size_t bytes,const void *argument)
    {
        if(index>=8||bytes>16) throw std::runtime_error("CUDA kernel argument bounds");
        std::memcpy(handle(value)->arguments[index].data(),argument,bytes);
    }
    void _executeKernel(cl_kernel value,size_t global,size_t local=0)
    {
        if(!capturing_) flush();
        if(!local) local=std::min(size_t(128),global);
        if(global%local) throw std::runtime_error("Invalid CUDA launch geometry");
        auto *k=handle(value);
        cuda_check(cuLaunchKernel(k->function,global/local,1,1,local,1,1,0,stream_,k->pointers.data(),nullptr));
    }
};
