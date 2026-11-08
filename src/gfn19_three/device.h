// OpenCL execution for the three-prime transform.
#pragma once
#include <cstring>
#include "device_selection.h"
class three_device : oclObject
{
private:
	enum class EVendor { Unknown, NVIDIA, AMD, INTEL };

	const cl_platform_id _platform;
	const cl_device_id _device;
#if defined(ocl_debug)
	const size_t _d;
#endif
	bool _profile = false;
#if defined(__APPLE__)
	bool _isSync = true;
#else
	bool _isSync = false;
#endif
	size_t _syncCount = 0;
	cl_ulong _localMemSize = 0;
	size_t _maxWorkGroupSize = 0;
	cl_ulong _timerResolution = 0;
	EVendor _vendor = EVendor::Unknown;
	cl_context _context = nullptr;
	cl_command_queue _queueF = nullptr;
	cl_command_queue _queueP = nullptr;
	cl_command_queue _queue = nullptr;
	cl_program _program = nullptr;

	struct profile
	{
		std::string name;
		size_t count;
		cl_ulong time;

		profile() {}
		profile(const std::string & name) : name(name), count(0), time(0) {}
	};
	std::map<cl_kernel, profile> _profileMap;

public:
	three_device(const platform & parent, const size_t d, const bool verbose) : _platform((three_prime_device(parent,d),parent.getPlatform(d))), _device(parent.getDevice(d))
#if defined(ocl_debug)
		, _d(d)
#endif
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl device " << d << "." << std::endl;
		pio::display(ss.str());
#endif

		char deviceName[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_NAME, 1024, deviceName, nullptr));
		char deviceVendor[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_VENDOR, 1024, deviceVendor, nullptr));
		char deviceVersion[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_VERSION, 1024, deviceVersion, nullptr));
		char driverVersion[1024]; oclFatal(clGetDeviceInfo(_device, CL_DRIVER_VERSION, 1024, driverVersion, nullptr));
		_vendor = getVendor(deviceVendor);

		cl_uint computeUnits; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr));
		cl_uint maxClockFrequency; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, nullptr));
		cl_ulong memSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr));
		cl_ulong memCacheSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(memCacheSize), &memCacheSize, nullptr));
		cl_uint memCacheLineSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(memCacheLineSize), &memCacheLineSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(_localMemSize), &_localMemSize, nullptr));
		cl_ulong memConstSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(memConstSize), &memConstSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_maxWorkGroupSize), &_maxWorkGroupSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(_timerResolution), &_timerResolution, nullptr));

		if (verbose)
		{
			std::ostringstream ssd;
			ssd << "Running on device '" << deviceName << "', vendor '" << deviceVendor
				<< "', version '" << deviceVersion << "', driver '" << driverVersion << "'";
#if defined(ocl_debug)
			ssd << std::endl << computeUnits << " compUnits @ " << maxClockFrequency << "MHz, mem=" << (memSize >> 20) << "MB, cache="
			 	<< (memCacheSize >> 10) << "kB, cacheLine=" << memCacheLineSize << "B, localMem=" << (_localMemSize >> 10)
			 	<< "kB, constMem=" << (memConstSize >> 10) << "kB, maxWorkGroup=" << _maxWorkGroupSize << ".";
#endif
			pio::print(ssd.str());
		}

		const cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)_platform, 0 };
		cl_int err_cc;
		_context = clCreateContext(contextProperties, 1, &_device, nullptr, nullptr, &err_cc);
		oclFatal(err_cc);
		cl_int err_ccq;
		_queueF = clCreateCommandQueue(_context, _device, 0, &err_ccq);
		_queueP = clCreateCommandQueue(_context, _device, CL_QUEUE_PROFILING_ENABLE, &err_ccq);
		_queue = _queueF;	// default queue is fast
		oclFatal(err_ccq);

		if (_vendor != EVendor::NVIDIA) _isSync = true;
	}

public:
	virtual ~three_device()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Delete ocl device " << _d << "." << std::endl;
		pio::display(ss.str());
#endif
		oclFatal(clReleaseCommandQueue(_queueP));
		oclFatal(clReleaseCommandQueue(_queueF));
		oclFatal(clReleaseContext(_context));
	}

public:
	size_t getMaxWorkGroupSize() const { return _maxWorkGroupSize; }
	size_t getLocalMemSize() const { return _localMemSize; }
	size_t getTimerResolution() const { return _timerResolution; }
	bool isIntel() const { return (_vendor == EVendor::INTEL); }

private:
	static EVendor getVendor(const std::string & vendorString)
	{
		std::string lVendorString; lVendorString.resize(vendorString.size());
		std::transform(vendorString.begin(), vendorString.end(), lVendorString.begin(), [](char c){ return std::tolower(c); });

		if (strstr(lVendorString.c_str(), "nvidia") != nullptr) return EVendor::NVIDIA;
		if (strstr(lVendorString.c_str(), "amd") != nullptr) return EVendor::AMD;
		if (strstr(lVendorString.c_str(), "advanced micro devices") != nullptr) return EVendor::AMD;
		if (strstr(lVendorString.c_str(), "intel") != nullptr) return EVendor::INTEL;
		// must be tested after 'Intel' because 'ati' is in 'Intel(R) Corporation' string
		if (strstr(lVendorString.c_str(), "ati") != nullptr) return EVendor::AMD;
		return EVendor::Unknown;
	}

public:

public:
	void resetProfiles()
	{
		for (auto it : _profileMap)
		{
			profile & prof = _profileMap[it.first];	// it.first is not a reference!
			prof.count = 0;
			prof.time = 0;
		}
	}

public:
	cl_ulong getProfileTime() const
	{
		cl_ulong time = 0;
		for (auto it : _profileMap) time += it.second.time;
		return time;
	}

public:
	void displayProfiles(const size_t count) const
	{
		cl_ulong ptime = 0;
		for (auto it : _profileMap) ptime += it.second.time;
		ptime /= count;

		std::ostringstream ss;
		for (auto it : _profileMap)
		{
			const profile & prof = it.second;
			if (prof.count != 0)
			{
				const size_t ncount = prof.count / count;
				const cl_ulong ntime = prof.time / count;
				ss << "- " << prof.name << ": " << ncount << ", " << std::setprecision(3)
					<< ntime * 100.0 / ptime << " %, " << ntime << " (" << (ntime / ncount) << ")" << std::endl;
			}
		}
		pio::display(ss.str());
	}

public:
	void setProfiling(const bool enable)
	{
		_profile = enable;
		_queue = enable ? _queueP : _queueF;
		resetProfiles();
	}

public:
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::ostringstream & src) const
	{
		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2022, Yves Gallot" << std::endl << std::endl;
		hFile << "genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "#pragma once" << std::endl << std::endl;
		hFile << "#include <cstdint>" << std::endl << std::endl;

		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;

			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

public:
	void loadProgram(const std::string & programSrc)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Load ocl program." << std::endl;
		pio::display(ss.str());
#endif
		const char * src[1]; src[0] = programSrc.c_str();
		cl_int err_cpws;
		_program = clCreateProgramWithSource(_context, 1, src, nullptr, &err_cpws);
		oclFatal(err_cpws);

		char pgmOptions[1024];
		strcpy(pgmOptions, "");
#if defined(ocl_debug)
		if (_vendor == EVendor::NVIDIA) strcat(pgmOptions, " -cl-nv-verbose");
		if (_vendor == EVendor::AMD) strcat(pgmOptions, " -save-temps=.");
#endif
		const cl_int err = clBuildProgram(_program, 1, &_device, pgmOptions, nullptr, nullptr);

#if !defined(ocl_debug)
		if (err != CL_SUCCESS)
#endif
		{
			size_t logSize; clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
			if (logSize > 2)
			{
				std::vector<char> buildLog(logSize + 1);
				clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
				buildLog[logSize] = '\0';
#if defined(ocl_debug)
				std::ofstream fileOut("pgm.log"); 
				fileOut << buildLog.data() << std::endl;
				fileOut.close();
#else
				std::ostringstream ss; ss << buildLog.data() << std::endl;
				pio::print(ss.str());
#endif
			}
		}

		oclFatal(err);

#if defined(ocl_debug)
		size_t binSize; clGetProgramInfo(_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, nullptr);
		std::vector<char> binary(binSize);
		clGetProgramInfo(_program, CL_PROGRAM_BINARIES, sizeof(char *), &binary, nullptr);
		std::ofstream fileOut((_vendor == EVendor::NVIDIA) ? "pgm.ptx" : "pgm.bin", std::ios::binary);
		fileOut.write(binary.data(), std::streamsize(binSize));
		fileOut.close();
#endif
	}

public:
	void clearProgram()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Clear ocl program." << std::endl;
		pio::display(ss.str());
#endif
		oclFatal(clReleaseProgram(_program));
		_program = nullptr;
		_profileMap.clear();
	}

private:
	void _sync()
	{
		_syncCount = 0;
		oclFatal(clFinish(_queue));
	}

protected:
	cl_mem _createBuffer(const cl_mem_flags flags, const size_t size, const bool clear = true) const
	{
		cl_int err;
		cl_mem mem = clCreateBuffer(_context, flags, size, nullptr, &err);
		oclFatal(err);
		if (clear)
		{
			std::vector<uint8_t> ptr(size);
			for (size_t i = 0; i < size; ++i) ptr[i] = 0x00;	// debug 0xff;
			oclFatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, 0, size, ptr.data(), 0, nullptr, nullptr));
		}
		return mem;
	}

protected:
	static void _releaseBuffer(cl_mem & mem)
	{
		if (mem != nullptr)
		{
			oclFatal(clReleaseMemObject(mem));
			mem = nullptr;
		}
	}

protected:
	void _readBuffer(cl_mem & mem, void * const ptr, const size_t size, const size_t offset = 0)
	{
		// Poison readback with invalid residue words without per-byte random-number calls.
		char * const cptr = static_cast<char *>(ptr);
		std::memset(cptr, 0xff, size);
		_sync();
		oclFatal(clEnqueueReadBuffer(_queue, mem, CL_TRUE, offset, size, ptr, 0, nullptr, nullptr));
	}

protected:
	void _writeBuffer(cl_mem & mem, const void * const ptr, const size_t size, const size_t offset = 0)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, offset, size, ptr, 0, nullptr, nullptr));
	}

protected:
	cl_kernel _createKernel(const char * const kernelName)
	{
		cl_int err;
		cl_kernel kernel = clCreateKernel(_program, kernelName, &err);
		oclFatal(err, kernelName);
		_profileMap[kernel] = profile(kernelName);
		return kernel;
	}

protected:
	static void _releaseKernel(cl_kernel & kernel)
	{
		if (kernel != nullptr)
		{
			oclFatal(clReleaseKernel(kernel));
			kernel = nullptr;
		}
	}

protected:
	static void _setKernelArg(cl_kernel kernel, const cl_uint arg_index, const size_t arg_size, const void * const arg_value)
	{
#if !defined(ocl_fast_exec) || defined(ocl_debug)
		cl_int err =
#endif
		clSetKernelArg(kernel, arg_index, arg_size, arg_value);
#if !defined(ocl_fast_exec) || defined(ocl_debug)
		oclFatal(err);
#endif
	}

protected:
	void _executeKernel(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
		if (!_profile)
		{
#if !defined(ocl_fast_exec) || defined(ocl_debug)
			cl_int err =
#endif
			clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, nullptr);
#if !defined(ocl_fast_exec) || defined(ocl_debug)
			oclFatal(err);
#endif
			if (_isSync)
			{
				++_syncCount;
				if (_syncCount == 16 * 1024) _sync();
			}
		}
		else
		{
			_sync();
			cl_event evt;
			oclFatal(clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, &evt));
			cl_ulong dt = 0;
			if (clWaitForEvents(1, &evt) == CL_SUCCESS)
			{
				cl_ulong start, end;
				cl_int err_s = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
				cl_int err_e = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
				if ((err_s == CL_SUCCESS) && (err_e == CL_SUCCESS)) dt = end - start;
			}
			clReleaseEvent(evt);

			profile & prof = _profileMap[kernel];
			prof.count++;
			prof.time += dt;
		}
	}
};
