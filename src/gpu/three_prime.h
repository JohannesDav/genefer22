/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <type_traits>
#include <iterator>

#include "../ocl.h"
#include "../transform.h"
#include "bounds.h"
#if defined(GENEFER_AUTOMATIC_OPENCL)
#include "checkpoint.h"
#endif

namespace gpu_small_prime {



// #define USE_WI	1
// #define	TUNE	1


inline size_t configured_root_word_bytes()
{
    if (const char *value=std::getenv("GENEFER_ROOT_BITS")) {
        if (std::string(value)=="32") return 4;
        if (std::string(value)=="64") return 8;
        throw std::runtime_error("Root table format must be 32 or 64 bits");
    }
    return 4;
}

#define VSIZE	4

// TUNE must be set if CHECK is set
// #define CHECK_ALL_FUNCTIONS		1
// #define CHECK_RADIX4_FUNCTIONS	1
// #define CHECK_FUNC_1		1	// GFN-12&13&14: square4, square32, square64, forward4_0, forward64, forward256_0, forward256
// #define CHECK_FUNC_2		1	// GFN-11&12&13: square8, square128, forward4, forward64_0, forward1024_0
// #define CHECK_FUNC_3		1	// GFN-12&13&14: square256, square512, forward1024
// #define CHECK_FUNC_4		1	// GFN-12&13&14: square1024, square2048, square4096

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

#define	P1S			(127 * (uint32(1) << 24) + 1)
#define	Q1S			2164260865u		// p * q = 1 (mod 2^32)
#define	R1S			33554430u		// 2^32 mod p
#define	RSQ1S		402124772u		// (2^32)^2 mod p
#define	H1S			100663290u		// Montgomery form of the primitive root 3
#define	IM1S		2063729671u		// MF of I = 3^{(p - 1)/4}
#define	MFIM1S		1930170389u		// MF of MF of I to convert input into MF
#define	SQRTI1S		1626730317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1S	856006302u		// MF of i * sqrt(i)

#define	P2S			(63 * (uint32(1) << 25) + 1)
#define	Q2S			2181038081u
#define	R2S			67108862u
#define	RSQ2S		2111798781u
#define	H2S			335544310u		// MF of the primitive root 5
#define	IM2S		530075385u
#define	MFIM2S		1036950657u
#define	SQRTI2S		338852760u
#define	ISQRTI2S	1090446030u

#define	P3S			(15 * (uint32(1) << 27) + 1)
#define	Q3S			2281701377u
#define	R3S			268435454u
#define	RSQ3S		1172168163u
#define	H3S			268435390u		// MF of the primitive root 31
#define	IM3S		473486609u
#define	MFIM3S		734725699u
#define	SQRTI3S		1032137103u
#define	ISQRTI3S	1964242958u

#define	INVP2_P1S	2130706177u		// MF of 1 / P2 (mod P1)
#define	INVP3_P1S	608773230u		// MF of 1 / P3 (mod P1)
#define	INVP3_P2S	1409286102u		// MF of 1 / P3 (mod P2)

#define	P1P2P3LS	1962934273u					// (P1 * P2 * P3) mod 2^32
#define	P1P2P3HS	2111326211158966273ul		// (P1 * P2 * P3) >> 32

#define	P1P2P3_2LS	3128950784u					// (P1 * P2 * P3 / 2) mod 2^32
#define	P1P2P3_2HS	1055663105579483136ul		// (P1 * P2 * P3 / 2) >> 32

#define P1U			(125 * (uint32(1) << 25) + 1)
#define	Q1U			100663297u		// p * q = 1 (mod 2^32)
#define	R1U			100663295u		// 2^32 mod p
#define	RSQ1U		232465106u		// (2^32)^2 mod p
#define	H1U			301989885u		// Montgomery form of the primitive root 3
#define	IM1U		1486287593u		// MF of I = 3^{(p - 1)/4}
#define	MFIM1U		3645424034u		// MF of MF of I to convert input into MF
#define	SQRTI1U		3580437317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1U	2017881188u		// MF of i * sqrt(i)

#define P2U			(243 * (uint32(1) << 24) + 1)
#define	Q2U			218103809u
#define	R2U			218103807u
#define	RSQ2U		3444438393u
#define	H2U			1526726649u		// MF of the primitive root 7
#define	IM2U		99906823u
#define	MFIM2U		1773796560u
#define	SQRTI2U		2024944857u
#define	ISQRTI2U	2119710515u

#define P3U			(235 * (uint32(1) << 24) + 1)
#define	Q3U			352321537u
#define	R3U			352321535u
#define	RSQ3U		3810498414u
#define	H3U			1056964605u		// MF of the primitive root 3
#define	IM3U		2213106415u
#define	MFIM3U		2454519270u
#define	SQRTI3U		3448990025u
#define	ISQRTI3U	3659377330u

#define	INVP2_P1U	1797558821u		// MF of 1 / P2 (mod P1)
#define	INVP3_P1U	3075822917u		// MF of 1 / P3 (mod P1)
#define	INVP3_P2U	4076863457u		// MF of 1 / P3 (mod P2)

#define	P1P2P3LU	3623878657u					// (P1 * P2 * P3) mod 2^32
#define	P1P2P3HU	15696902887611105282ul		// (P1 * P2 * P3) >> 32

#define	P1P2P3_2LU	1811939328u					// (P1 * P2 * P3 / 2) mod 2^32
#define	P1P2P3_2HU	7848451443805552641ul		// (P1 * P2 * P3 / 2) >> 32

#include "primes.h"

inline int64_t decode_normalized_digit(uint32_t first, uint32_t second)
{
    // The host admission requires |digit| < the smallest of all three primes.
    return int64_t(first) - (first == second ? int64_t(0) : int64_t(P1S));
}


class ZP
{
protected:
	uint32 _n;

public:
	ZP() {}
	explicit ZP(const uint32 n) : _n(n) {}

	uint32 get() const { return _n; }
};

template<uint32 P, uint32 Q, uint32 R, uint32 H>
class ZPT : public ZP
{
private:
	static uint32 _add(const uint32 a, const uint32 b) { return a + b - ((a >= P - b) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { return a - b + ((a < b) ? P : 0); }

	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

public:
	ZPT() {}
	explicit ZPT(const uint32 n) : ZP(n) {}

	int32 get_int() const { return (_n >= P / 2) ? int32(_n - P) : int32(_n); }
	ZPT & set_int(const int32 i) { const uint32 a = uint32(i < 0 ? -int64(i) : i); const uint32 r = a >= P ? a - P : a; _n = (i < 0 && r != 0) ? P - r : r; return *this; }

	ZPT mul(const ZPT & rhs) const { return ZPT(_mul(_n, rhs._n)); }

	ZPT pow(const size_t e) const
	{
		if (e == 0) return ZPT(R);	// MF of one is R
		ZPT r = ZPT(R), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.mul(y); }
		r = r.mul(y);
		return r;
	}

	static const ZPT primroot_n(const uint32 n) { return ZPT(H).pow((P - 1) / n); }
	static ZPT norm(const uint32 n) { return ZPT(P - (P - 1) / n); }
};

#if VSIZE == 4
#define LVSIZE	2
#elif VSIZE == 2
#define LVSIZE	1
#else
#define LVSIZE	0
#endif

// Warning: DECLARE_VAR_xx in kernel.cl must be modified if BLKxx = 1 or != 1.

#define BLK32		32		// local size =   4KB, workgroup size =  256 / VSIZE
#define BLK64		16		// local size =   4KB, workgroup size =  256 / VSIZE
#define BLK128		8		// local size =   4KB, workgroup size =  256 / VSIZE
#define BLK256		4		// local size =   4KB, workgroup size =  256 / VSIZE
#define BLK512		2		// local size =   4KB, workgroup size =  256 / VSIZE
#define BLK1024		1		// local size =   4KB, workgroup size =  256 / VSIZE
#define BLK2048		1		// local size =   8KB, workgroup size =  512 / VSIZE
#define BLK4096		1		// local size =  16KB, workgroup size = 1024 / VSIZE

#define CHUNK64		4		// local size =  VSIZE * 1KB, workgroup size = 64
#define CHUNK256	4		// local size =  VSIZE * 4KB, workgroup size = 256
#define CHUNK1024	1		// local size =  VSIZE * 4KB, workgroup size = 256

#define CREATE_TRANSFORM_KERNEL(name) _##name = createTransformKernel(#name);
#define CREATE_TRANSFORM_KERNELP(name) _##name = createTransformKernel(#name, false);
#define CREATE_MUL_KERNEL(name) _##name = createMulKernel(#name);
#define CREATE_NORMALIZE_KERNEL(name, b, b_inv, b_s) _##name = createNormalizeKernel(#name, b, b_inv, b_s);
#define CREATE_SETCOPY_KERNEL(name) _##name = createSetCopyKernel(#name);
#define CREATE_COPYP_KERNEL(name) _##name = createCopypKernel(#name);

#define DEFINE_FORWARD(u) void forward##u(const int lm) { ek_fb(_forward##u, lm - LVSIZE, u / 4 * CHUNK##u, 4 * VSIZE); }
#define DEFINE_BACKWARD(u) void backward##u(const int lm) { ek_fb(_backward##u, lm - LVSIZE, u / 4 * CHUNK##u, 4 * VSIZE); }
#define DEFINE_FORWARDn(u, v) void forward##u##_##v() { ek(_forward##u##_##v, u / 4 * CHUNK##u, 4 * VSIZE); }
#define DEFINE_BACKWARDn(u, v) void backward##u##_##v() { ek(_backward##u##_##v, u / 4 * CHUNK##u, 4 * VSIZE); }

#define DEFINE_SQUARE(u) void square##u() { ek(_square##u, (u * BLK##u) / (4 * VSIZE), 4 * VSIZE); }
#define DEFINE_FWDP(u) void fwd##u##p() { ek(_fwd##u##p, (u * BLK##u) / (4 * VSIZE), 4 * VSIZE); }
#define DEFINE_MUL(u) void mul##u() { ek(_mul##u, (u * BLK##u) / (4 * VSIZE), 4 * VSIZE); }

#define DEFINE_FORWARDP(u) \
	void forward##u##p(const int lm) { setTransformArgs(_forward##u, false); forward##u(lm); setTransformArgs(_forward##u);	}
#define DEFINE_FORWARDPn(u, v) \
	void forward##u##p_##v() { setTransformArgs(_forward##u##_##v, false); forward##u##_##v(); setTransformArgs(_forward##u##_##v);	}

template<size_t RNS_SIZE, bool is32>
class engines : public optimized_device
{
    const size_t root_bytes_;
    size_t root_word_bytes() const { return root_bytes_; }
	using ZP1 = ZPT<is32 ? P1U : P1S, is32 ? Q1U : Q1S, is32 ? R1U : R1S, is32 ? H1U : H1S>;
	using ZP2 = ZPT<is32 ? P2U : P2S, is32 ? Q2U : Q2S, is32 ? R2U : R2S, is32 ? H2U : H2S>;
	using ZP3 = ZPT<is32 ? P3U : P3S, is32 ? Q3U : Q3S, is32 ? R3U : R3S, is32 ? H3U : H3S>;

private:
	const size_t _n;
	const int _ln;
	const bool _isBoinc;
	const size_t _num_regs;
	const int _lnormWGsize;
	cl_mem _z = nullptr, _zp = nullptr, _w = nullptr, _c = nullptr;
#if defined(TUNE)
	cl_kernel _forward4 = nullptr, _backward4 = nullptr, _forward4_0 = nullptr, _backward4_0 = nullptr;
	cl_kernel _square2x2 = nullptr, _square4 = nullptr, _square8 = nullptr;
	cl_kernel _fwd4p = nullptr, _fwd8p = nullptr;
	cl_kernel _mul2x2 = nullptr, _mul4 = nullptr, _mul8 = nullptr;
#endif
	cl_kernel _forward64_0 = nullptr, _forward64_9 = nullptr, _forward64_11 = nullptr;
	cl_kernel _backward64_0 = nullptr, _backward64_9 = nullptr, _backward64_11 = nullptr;
	cl_kernel _forward256_0 = nullptr, _backward256_0 = nullptr, _forward1024_0 = nullptr, _backward1024_0 = nullptr;
#if defined(TUNE)
	cl_kernel _forward64 = nullptr, _forward256 = nullptr, _forward1024 = nullptr;
	cl_kernel _backward64 = nullptr, _backward256 = nullptr, _backward1024 = nullptr;
#endif
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr;
	cl_kernel _square512 = nullptr, _square1024 = nullptr, _square2048 = nullptr, _square4096 = nullptr;
	cl_kernel _fwd32p = nullptr, _fwd64p = nullptr, _fwd128p = nullptr, _fwd256p = nullptr;
	cl_kernel _fwd512p = nullptr, _fwd1024p = nullptr, _fwd2048p = nullptr, _fwd4096p = nullptr;
	cl_kernel _mul32 = nullptr, _mul64 = nullptr, _mul128 = nullptr, _mul256 = nullptr;
	cl_kernel _mul512 = nullptr, _mul1024 = nullptr, _mul2048 = nullptr, _mul4096 = nullptr;
	cl_kernel _normalize1dup0 = nullptr, _normalize1dup1 = nullptr, _normalize2 = nullptr, _mulscalar = nullptr;
	cl_kernel _set = nullptr, _copy = nullptr, _copyp = nullptr;
#if defined(TUNE)
	splitter * _pSplit = nullptr;
	size_t _splitIndex = 0;
#endif
	// bool _first = true;

	static constexpr int ilog2_32(const uint32_t n) { return 31 - __builtin_clz(n); }

public:
	engines(const platform & platform, const size_t d, const int ln, const bool isBoinc, const size_t num_regs, const bool verbose, unsigned root_bits = 0)
		: optimized_device(platform, d, verbose), root_bytes_(root_bits ? root_bits/8 : configured_root_word_bytes()), _n(size_t(1) << ln), _ln(ln), _isBoinc(isBoinc), _num_regs(num_regs),
		_lnormWGsize(std::min(std::max(5, ln / 2 - 3), ilog2_32(uint32_t(getMaxWorkGroupSize())))) {}
	virtual ~engines() {
        try { releaseKernels(); } catch (...) {}
        try { releaseMemory(); } catch (...) {}
        try { clearProgram(); } catch (...) {}
    }

///////////////////////////////

public:
	size_t getNormWGsize() const { return size_t(1 << _lnormWGsize); }

	void allocMemory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t n = _n;
		if (n != 0)
		{
			_z = _createBuffer(CL_MEM_READ_WRITE, RNS_SIZE * n * _num_regs * sizeof(ZP));
			_zp = _createBuffer(CL_MEM_READ_WRITE, RNS_SIZE * n * sizeof(ZP));
			_w = _createBuffer(CL_MEM_READ_ONLY, RNS_SIZE * n * root_word_bytes());
			_c = _createBuffer(CL_MEM_READ_WRITE, n / 4 * sizeof(int64));
		}
	}

	void releaseMemory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Free gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		if (_n != 0)
		{
			_releaseBuffer(_z); _releaseBuffer(_zp);
			_releaseBuffer(_w); _releaseBuffer(_c);
		}
	}

///////////////////////////////

private:
	cl_kernel createTransformKernel(const char * const kernelName, const bool isMultiplier = true)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_w);
		return kernel;
	}

	cl_kernel createNormalizeKernel(const char * const kernelName, const uint32 b, const uint32 b_inv, const int32 b_s)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_c);
		_setKernelArg(kernel, 2, sizeof(uint32), &b);
		_setKernelArg(kernel, 3, sizeof(uint32), &b_inv);
		_setKernelArg(kernel, 4, sizeof(int32), &b_s);
		return kernel;
	}

	cl_kernel createMulKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_zp);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_w);
		return kernel;
	}

	cl_kernel createSetCopyKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		return kernel;
	}

	cl_kernel createCopypKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_z);
		return kernel;
	}

public:
	void createKernels(const uint32_t b)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl kernels." << std::endl;
		pio::display(ss.str());
#endif

#if defined(TUNE)
		CREATE_TRANSFORM_KERNEL(forward4);
		CREATE_TRANSFORM_KERNEL(backward4);
		CREATE_TRANSFORM_KERNEL(forward4_0);
		CREATE_TRANSFORM_KERNEL(backward4_0);

		CREATE_TRANSFORM_KERNEL(square2x2);
		CREATE_TRANSFORM_KERNEL(square4);
		CREATE_TRANSFORM_KERNEL(square8);

		CREATE_TRANSFORM_KERNELP(fwd4p);
		CREATE_TRANSFORM_KERNELP(fwd8p);

		CREATE_MUL_KERNEL(mul2x2);
		CREATE_MUL_KERNEL(mul4);
		CREATE_MUL_KERNEL(mul8);
#endif

#if !defined(CHECK_RADIX4_FUNCTIONS)
		CREATE_TRANSFORM_KERNEL(forward64_0);
		CREATE_TRANSFORM_KERNEL(forward64_9);
		CREATE_TRANSFORM_KERNEL(forward64_11);
		CREATE_TRANSFORM_KERNEL(backward64_0);
		CREATE_TRANSFORM_KERNEL(backward64_9);
		CREATE_TRANSFORM_KERNEL(backward64_11);
		CREATE_TRANSFORM_KERNEL(forward256_0);
		CREATE_TRANSFORM_KERNEL(backward256_0);
		CREATE_TRANSFORM_KERNEL(forward1024_0);
		CREATE_TRANSFORM_KERNEL(backward1024_0);
#if defined(TUNE)
		CREATE_TRANSFORM_KERNEL(forward64);
		CREATE_TRANSFORM_KERNEL(backward64);
		CREATE_TRANSFORM_KERNEL(forward256);
		CREATE_TRANSFORM_KERNEL(backward256);
		CREATE_TRANSFORM_KERNEL(forward1024);
		CREATE_TRANSFORM_KERNEL(backward1024);
#endif
		CREATE_TRANSFORM_KERNEL(square32);
		CREATE_TRANSFORM_KERNEL(square64);
		CREATE_TRANSFORM_KERNEL(square128);
		CREATE_TRANSFORM_KERNEL(square256);
		CREATE_TRANSFORM_KERNEL(square512);
		CREATE_TRANSFORM_KERNEL(square1024);
		CREATE_TRANSFORM_KERNEL(square2048);
		CREATE_TRANSFORM_KERNEL(square4096);

		CREATE_TRANSFORM_KERNELP(fwd32p);
		CREATE_TRANSFORM_KERNELP(fwd64p);
		CREATE_TRANSFORM_KERNELP(fwd128p);
		CREATE_TRANSFORM_KERNELP(fwd256p);
		CREATE_TRANSFORM_KERNELP(fwd512p);
		CREATE_TRANSFORM_KERNELP(fwd1024p);
		CREATE_TRANSFORM_KERNELP(fwd2048p);
		CREATE_TRANSFORM_KERNELP(fwd4096p);

		CREATE_MUL_KERNEL(mul32);
		CREATE_MUL_KERNEL(mul64);
		CREATE_MUL_KERNEL(mul128);
		CREATE_MUL_KERNEL(mul256);
		CREATE_MUL_KERNEL(mul512);
		CREATE_MUL_KERNEL(mul1024);
		CREATE_MUL_KERNEL(mul2048);
		CREATE_MUL_KERNEL(mul4096);
#endif
		const uint32 b_ui = static_cast<uint32>(b);
		const int32 b_s = static_cast<int32>(31 - __builtin_clz(b) - 1);
		const uint32 b_inv = static_cast<uint32>((static_cast<uint64_t>(1) << (b_s + 32)) / b);
		CREATE_NORMALIZE_KERNEL(normalize1dup0, b_ui, b_inv, b_s);
        CREATE_NORMALIZE_KERNEL(normalize1dup1, b_ui, b_inv, b_s);
		CREATE_NORMALIZE_KERNEL(normalize2, b_ui, b_inv, b_s);
		CREATE_NORMALIZE_KERNEL(mulscalar, b_ui, b_inv, b_s);

		CREATE_SETCOPY_KERNEL(set);
		CREATE_SETCOPY_KERNEL(copy);
		CREATE_COPYP_KERNEL(copyp);

#if defined(TUNE)
		_pSplit = new splitter(size_t(_ln), CHUNK256, CHUNK1024, sizeof(ZP), VSIZE, 12, getLocalMemSize(), getMaxWorkGroupSize());
#endif
	}

	void releaseKernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
#if defined(TUNE)
		delete _pSplit;

		_releaseKernel(_forward4); _releaseKernel(_backward4); _releaseKernel(_forward4_0); _releaseKernel(_backward4_0);
		_releaseKernel(_square2x2); _releaseKernel(_square4); _releaseKernel(_square8);
		_releaseKernel(_fwd4p); _releaseKernel(_fwd8p);
		_releaseKernel(_mul2x2); _releaseKernel(_mul4); _releaseKernel(_mul8);
#endif
		_releaseKernel(_forward64_0); _releaseKernel(_forward64_9); _releaseKernel(_forward64_11);
		_releaseKernel(_backward64_0); _releaseKernel(_backward64_9); _releaseKernel(_backward64_11);
		_releaseKernel(_forward256_0); _releaseKernel(_backward256_0); _releaseKernel(_forward1024_0); _releaseKernel(_backward1024_0);
#if defined(TUNE)
		_releaseKernel(_forward64); _releaseKernel(_forward256); _releaseKernel(_forward1024);
		_releaseKernel(_backward64); _releaseKernel(_backward256); _releaseKernel(_backward1024);
#endif
		_releaseKernel(_square32); _releaseKernel(_square64); _releaseKernel(_square128); _releaseKernel(_square256);
		_releaseKernel(_square512); _releaseKernel(_square1024); _releaseKernel(_square2048); _releaseKernel(_square4096);
		_releaseKernel(_fwd32p); _releaseKernel(_fwd64p); _releaseKernel(_fwd128p); _releaseKernel(_fwd256p);
		_releaseKernel(_fwd512p); _releaseKernel(_fwd1024p); _releaseKernel(_fwd2048p); _releaseKernel(_fwd4096p);
		_releaseKernel(_mul32); _releaseKernel(_mul64); _releaseKernel(_mul128); _releaseKernel(_mul256);
		_releaseKernel(_mul512); _releaseKernel(_mul1024); _releaseKernel(_mul2048); _releaseKernel(_mul4096);

		_releaseKernel(_normalize1dup0); _releaseKernel(_normalize1dup1); _releaseKernel(_normalize2); _releaseKernel(_mulscalar);

		_releaseKernel(_set); _releaseKernel(_copy); _releaseKernel(_copyp);
	}

///////////////////////////////

	void readMemory_z(ZP * const zPtr, const size_t count = 1) { _readBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP)); }
	void writeMemory_z(const ZP * const zPtr, const size_t count = 1) { _writeBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP)); }
	void writeMemory_w(const ZP * const wPtr, const size_t offset) {
 const uint64 p=(offset==0?P1S:(offset==1?P2S:P3S)),q=(offset==0?Q1S:(offset==1?Q2S:Q3S));
 const uint64 ri=p-((p*q)>>32);
 auto write=[&](auto word) {
     using Word=decltype(word);
     std::vector<Word> roots(_n);
     for(size_t i=0;i<_n;++i) {
         uint64 w=(uint64(wPtr[i].get())*ri)%p;
         if constexpr(sizeof(Word)==8) roots[i]=Word(w|(((w<<32)/p)<<32));
         else roots[i]=Word((w<<32)/p);
     }
     _writeBuffer(_w,roots.data(),_n*sizeof(Word),offset*_n*sizeof(Word));
 };
 if(root_word_bytes()==8) write(uint64_t(0));
 else write(uint32_t(0));
}

///////////////////////////////

private:
	void ek(cl_kernel & kernel, const size_t localWorkSize, const size_t step)
	{
		const size_t n_s = _n / step;
		_executeKernel(kernel, RNS_SIZE * n_s, localWorkSize);
	}

	void ek_fb(cl_kernel & kernel, const int lm, const size_t localWorkSize, const size_t step)
	{
		const size_t n_s = _n / step;
		const int32 ilm = static_cast<int32>(lm);
		const uint32 is = static_cast<uint32>(n_s >> lm);
		_setKernelArg(kernel, 2, sizeof(int32), &ilm);
		_setKernelArg(kernel, 3, sizeof(uint32), &is);
		_executeKernel(kernel, RNS_SIZE * n_s, localWorkSize);
	}

#if defined(TUNE)
	void forward4(const int lm) { ek_fb(_forward4, lm - LVSIZE, 0, 4 * VSIZE); }
	void backward4(const int lm) { ek_fb(_backward4, lm - LVSIZE, 0, 4 * VSIZE); }
	void forward4_0() { ek(_forward4_0, 0, 4 * VSIZE); }
	void backward4_0() { ek(_backward4_0, 0, 4 * VSIZE); }

	void square4() { ek(_square4, 0, 4 * VSIZE); }
	void fwd4p() { ek(_fwd4p, 0, 4 * VSIZE); }
	void mul4() { ek(_mul4, 0, 4 * VSIZE); }

#if VSIZE == 1
	void square8() { forward4(1); ek(_square2x2, 0, 4); backward4(1); }
	void fwd8p() { forward4p(1); }
	void mul8() { forward4(1); ek(_mul2x2, 0, 4); backward4(1); }
#else
	void square8() { ek(_square8, 0, 4 * VSIZE); }
	void fwd8p() { ek(_fwd8p, 0, 4 * VSIZE); }
	void mul8() { ek(_mul8, 0, 4 * VSIZE); }
#endif
#endif

	DEFINE_FORWARDn(64, 0);
	DEFINE_BACKWARDn(64, 0);
	DEFINE_FORWARDn(64, 9);
	DEFINE_BACKWARDn(64, 9);
	DEFINE_FORWARDn(64, 11);
	DEFINE_BACKWARDn(64, 11);
	DEFINE_FORWARDn(256, 0);
	DEFINE_BACKWARDn(256, 0);
	DEFINE_FORWARDn(1024, 0);
	DEFINE_BACKWARDn(1024, 0);
#if defined(TUNE)
	DEFINE_FORWARD(64);
	DEFINE_BACKWARD(64);
	DEFINE_FORWARD(256);
	DEFINE_BACKWARD(256);
	DEFINE_FORWARD(1024);
	DEFINE_BACKWARD(1024);
#endif
	DEFINE_SQUARE(32);
	DEFINE_SQUARE(64);
	DEFINE_SQUARE(128);
	DEFINE_SQUARE(256);
	DEFINE_SQUARE(512);
	DEFINE_SQUARE(1024);
	DEFINE_SQUARE(2048);
	DEFINE_SQUARE(4096);

	DEFINE_FWDP(32);
	DEFINE_FWDP(64);
	DEFINE_FWDP(128);
	DEFINE_FWDP(256);
	DEFINE_FWDP(512);
	DEFINE_FWDP(1024);
	DEFINE_FWDP(2048);
	DEFINE_FWDP(4096);

	DEFINE_MUL(32);
	DEFINE_MUL(64);
	DEFINE_MUL(128);
	DEFINE_MUL(256);
	DEFINE_MUL(512);
	DEFINE_MUL(1024);
	DEFINE_MUL(2048);
	DEFINE_MUL(4096);

	void setTransformArgs(cl_kernel & kernel, const bool isMultiplier = true)
	{
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
	}

#if defined(TUNE)
	DEFINE_FORWARDP(4);
	DEFINE_FORWARDP0(4);
#endif
	DEFINE_FORWARDPn(64, 0);
	DEFINE_FORWARDPn(64, 9);
	DEFINE_FORWARDPn(64, 11);
	DEFINE_FORWARDPn(256, 0);
	DEFINE_FORWARDPn(1024, 0);
#if defined(TUNE)
	DEFINE_FORWARDP(64);
	DEFINE_FORWARDP(256);
	DEFINE_FORWARDP(1024);
#endif

#if defined(TUNE)
	void _mul(const size_t sIndex, const bool isSquare)
	{
#if defined(CHECK_FUNC_1)
		if (_ln == 12) { forward4_0(); forward256(12 - 10); if (isSquare) square4(); else mul4(); backward256(12 - 10); backward4_0(); return; }
		if (_ln == 13) { forward4_0(); forward64(13 - 8); if (isSquare) { square32(); } else mul32(); backward64(13 - 8); backward4_0(); return; }
		if (_ln == 14) { forward256_0(); if (isSquare) square64(); else mul64(); backward256_0(); return; }
#endif
#if defined(CHECK_FUNC_2)
		if (_ln == 11) { forward4_0(); forward4(11 - 4); if (isSquare) square128(); else mul128(); backward4(11 - 4); backward4_0(); return; }
		if (_ln == 12) { forward64_0(); if (isSquare) square64(); else mul64(); backward64_0(); return; }
		if (_ln == 13) { forward1024_0(); if (isSquare) square8(); else mul8(); backward1024_0(); return; }
#endif
#if defined(CHECK_FUNC_3)
		if (_ln == 12) { forward4_0(); forward4(12 - 4); if (isSquare) square256(); else mul256(); backward4(12 - 4); backward4_0(); return; }
		if (_ln == 13) { forward4_0(); forward4(13 - 4); if (isSquare) square512(); else mul512(); backward4(13 - 4); backward4_0(); return; }
		if (_ln == 14) { forward4_0(); forward1024(14 - 12); if (isSquare) square4(); else mul4(); backward1024(14 - 12); backward4_0(); return; }
#endif
#if defined(CHECK_FUNC_4)
		if (_ln == 12) { forward4_0(); if (isSquare) square1024(); else mul1024(); backward4_0(); return; }
		if (_ln == 13) { forward4_0(); if (isSquare) square2048(); else mul2048(); backward4_0(); return; }
		if (_ln == 14) { forward4_0(); if (isSquare) square4096(); else mul4096(); backward4_0(); return; }
#endif

		int lm = _ln;

#if defined(CHECK_RADIX4_FUNCTIONS)
		(void)sIndex;
		lm -= 2; forward4_0();
		while (lm > 3) { lm -= 2; forward4(lm); }
		if (isSquare) { if (lm == 3) square8(); else square4(); } else { if (lm == 3) mul8(); else mul4(); }
		while (lm < _ln - 2) { backward4(lm); lm += 2; }
		backward4_0(); lm += 2;
#else
		const splitter * const pSplit = _pSplit;
		const size_t s = pSplit->getPartSize(sIndex);

		const uint32_t k0 = pSplit->getPart(sIndex, 0);
		lm -= int(k0);
		if (k0 == 10) forward1024_0();
		else if (k0 == 8) forward256_0();
		else forward64_0();	// k0 = 6

		for (size_t i = 2; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			lm -= int(k);
			if (k == 10) forward1024(lm);
			else if (k == 8) forward256(lm);
			else forward64(lm); // k = 6
		}

		// lm = pSplit->getPart(sIndex, s - 1);
		if (isSquare)
		{
			if (lm == 12) square4096();
			else if (lm == 11) square2048();
			else if (lm == 10) square1024();
			else if (lm == 9) square512();
			else if (lm == 8) square256();
			else if (lm == 7) square128();
			else if (lm == 6) square64();
			else if (lm == 5) square32();
		}
		else
		{
			if (lm == 12) mul4096();
			else if (lm == 11) mul2048();
			else if (lm == 10) mul1024();
			else if (lm == 9) mul512();
			else if (lm == 8) mul256();
			else if (lm == 7) mul128();
			else if (lm == 6) mul64();
			else if (lm == 5) mul32();
		}

		for (size_t i = s - 1; i >= 2; --i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10) backward1024(lm);
			else if (k == 8) backward256(lm);
			else backward64(lm);	// k = 6
			lm += int(k);
		}
		if (k0 == 10) backward1024_0();
		else if (k0 == 8) backward256_0();
		else backward64_0();	// k = 6
		lm += int(k0);

#endif
	}
#endif	// TUNE

public:
	void square()
	{
		// if (_first) { info(); _first = false; }

#if defined(TUNE)
		const size_t splitIndex =
#if defined(CHECK_ALL_FUNCTIONS)
		size_t(rand()) % _pSplit->getSize();
#else
		_splitIndex;
#endif
		_mul(splitIndex, true);
#else
		const int ln = _ln;
		if (ln == 11) { forward64_0(); square32(); backward64_0(); }
		else if (ln == 12) { forward64_0(); square64(); backward64_0(); }
		else if (ln == 13) { forward64_0(); square128(); backward64_0(); }
		else if (ln == 14) { forward64_0(); square256(); backward64_0(); }
		else if (ln == 15) { forward64_0(); square512(); backward64_0(); }
		else if (ln == 16) { forward64_0(); square1024(); backward64_0(); }
		else if (ln == 17) { forward64_0(); square2048(); backward64_0(); }
		else if (ln == 18) { forward256_0(); square1024(); backward256_0(); }
		else if (ln == 19) { forward256_0(); square2048(); backward256_0(); }
		else if (ln == 20) { forward256_0(); square4096(); backward256_0(); }
		else if (ln == 21) { forward64_0(); forward64_9(); square512(); backward64_9(); backward64_0(); }
		else if (ln == 22) { forward1024_0(); square4096(); backward1024_0(); }
		else { forward64_0(); forward64_11(); square2048(); backward64_11(); backward64_0(); }
#endif
	}

	void mul()
	{
#if defined(TUNE)
		_mul(_splitIndex, false);
#else
		const int ln = _ln;
		if (ln == 11) { forward64_0(); mul32(); backward64_0(); }
		else if (ln == 12) { forward64_0(); mul64(); backward64_0(); }
		else if (ln == 13) { forward64_0(); mul128(); backward64_0(); }
		else if (ln == 14) { forward64_0(); mul256(); backward64_0(); }
		else if (ln == 15) { forward64_0(); mul512(); backward64_0(); }
		else if (ln == 16) { forward64_0(); mul1024(); backward64_0(); }
		else if (ln == 17) { forward64_0(); mul2048(); backward64_0(); }
		else if (ln == 18) { forward256_0(); mul1024(); backward256_0(); }
		else if (ln == 19) { forward256_0(); mul2048(); backward256_0(); }
		else if (ln == 20) { forward256_0(); mul4096(); backward256_0(); }
		else if (ln == 21) { forward64_0(); forward64_9(); mul512(); backward64_9(); backward64_0(); }
		else if (ln == 22) { forward1024_0(); mul4096(); backward1024_0(); }
		else { forward64_0(); forward64_11(); mul2048(); backward64_11(); backward64_0(); }
#endif
	}

	void initMultiplicand(const size_t src)
	{
		const uint32 isrc = static_cast<uint32>(src * RNS_SIZE * _n / 4);
		_setKernelArg(_copyp, 2, sizeof(uint32), &isrc);
		_executeKernel(_copyp, RNS_SIZE * _n / 4);

#if defined(CHECK_FUNC_1)
		if (_ln == 12) { forward4p_0(); forward256p(12 - 10); fwd4p(); return; }
		if (_ln == 13) { forward4p_0(); forward64p(13 - 8); fwd32p(); return; }
		if (_ln == 14) { forward256p_0(); fwd64p(); return; }
#endif
#if defined(CHECK_FUNC_2)
		if (_ln == 11) { forward4p_0(); forward4p(11 - 4); fwd128p(); return; }
		if (_ln == 12) { forward64p_0(); fwd64p(); return; }
		if (_ln == 13) { forward1024p_0(); fwd8p(); return; }
#endif
#if defined(CHECK_FUNC_3)
		if (_ln == 12) { forward4p_0(); forward4p(12 - 4); fwd256p(); return; }
		if (_ln == 13) { forward4p_0(); forward4p(13 - 4); fwd512p(); return; }
		if (_ln == 14) { forward4p_0(); forward1024p(14 - 12); fwd4p(); return; }
#endif
#if defined(CHECK_FUNC_4)
		if (_ln == 12) { forward4p_0(); fwd1024p(); return; }
		if (_ln == 13) { forward4p_0(); fwd2048p(); return; }
		if (_ln == 14) { forward4p_0(); fwd4096p(); return; }
#endif

#if defined(CHECK_ALL_FUNCTIONS)
		_splitIndex = size_t(rand()) % _pSplit->getSize();
#endif

		int lm = _ln;

#if defined(CHECK_RADIX4_FUNCTIONS)
		lm -= 2; forward4p_0();
		while (lm > 3) { lm -= 2; forward4p(lm); }
		if (lm == 3) fwd8p(); else fwd4p();
		return;
#elif defined(TUNE)
		const splitter * const pSplit = _pSplit;
		const size_t sIndex = _splitIndex;
		const size_t s = pSplit->getPartSize(sIndex);

		const uint32_t k0 = pSplit->getPart(sIndex, 0);
		lm -= int(k0);
		if (k0 == 10) forward1024p_0();
		else if (k0 == 8) forward256p_0();
		else forward64p_0();	// k0 = 6

		for (size_t i = 2; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			lm -= int(k);
			if (k == 10) forward1024p(lm);
			else if (k == 8) forward256p(lm);
			else forward64p(lm);	// k = 6
		}

		// lm = pSplit->getPart(sIndex, s - 1);
		if (lm == 12) fwd4096p();
		else if (lm == 11) fwd2048p();
		else if (lm == 10) fwd1024p();
		else if (lm == 9) fwd512p();
		else if (lm == 8) fwd256p();
		else if (lm == 7) fwd128p();
		else if (lm == 6) fwd64p();
		else if (lm == 5) fwd32p();
#else
		if (lm == 11) { forward64p_0(); fwd32p(); }
		else if (lm == 12) { forward64p_0(); fwd64p(); }
		else if (lm == 13) { forward64p_0(); fwd128p(); }
		else if (lm == 14) { forward64p_0(); fwd256p(); }
		else if (lm == 15) { forward64p_0(); fwd512p(); }
		else if (lm == 16) { forward64p_0(); fwd1024p(); }
		else if (lm == 17) { forward64p_0(); fwd2048p(); }
		else if (lm == 18) { forward256p_0(); fwd1024p(); }
		else if (lm == 19) { forward256p_0(); fwd2048p(); }
		else if (lm == 20) { forward256p_0(); fwd4096p(); }
		else if (lm == 21) { forward64p_0(); forward64p_9(); fwd512p(); }
		else if (lm == 22) { forward1024p_0(); fwd4096p(); }
		else { forward64p_0(); forward64p_11(); fwd2048p(); }
#endif
	}

	void set(const uint32_t a)
	{
		const uint32 ia = static_cast<uint32>(a);
		_setKernelArg(_set, 1, sizeof(uint32), &ia);
		_executeKernel(_set, RNS_SIZE * _n / 4);
	}

	void copy(const size_t dst, const size_t src)
	{
		const size_t size = _n / 4;
		const uint32 idst = static_cast<uint32>(dst * RNS_SIZE * size), isrc = static_cast<uint32>(src * RNS_SIZE * size);
		_setKernelArg(_copy, 1, sizeof(uint32), &idst);
		_setKernelArg(_copy, 2, sizeof(uint32), &isrc);
		_executeKernel(_copy, RNS_SIZE * size);
	}

public:
	void baseMod(const bool dup)
	{
		const int32 idup = dup ? 1 : 0;
		const size_t size = _n / 4;

		cl_kernel kernel = dup ? _normalize1dup1 : _normalize1dup0;
        _setKernelArg(kernel, 5, sizeof(int32), &idup);
        _executeKernel(kernel, size, 1u << _lnormWGsize);
		_executeKernel(_normalize2, size >> _lnormWGsize);
	}

	void baseModMul(const int a)
	{
		baseMod(false);

		const int32 ia = static_cast<int32>(a);
		const size_t size = _n / 4;
		_setKernelArg(_mulscalar, 5, sizeof(int32), &ia);
		_executeKernel(_mulscalar, size, 1u << _lnormWGsize);
		_executeKernel(_normalize2, size >> _lnormWGsize);
	}

#if defined(TUNE)
private:
	void squareTune(const size_t count, const size_t sIndex, const ZP * const Z)
	{
		for (size_t j = 0; j != count; ++j)
		{
			writeMemory_z(Z);
			_mul(sIndex, true);
		}
	}

public:
	void tune()
	{
		const size_t n = _n;

		ZP * const Z = new ZP[RNS_SIZE * n];
		for (size_t i = 0; i != n; ++i)
		{
			Z[0 * n + i] = ZP1().set_int(static_cast<int32>(((is32 ? P1U : P1S) - 1) * cos(i + 0.25)));
			if (RNS_SIZE >= 2) Z[1 * n + i] = ZP2().set_int(static_cast<int32>(((is32 ? P2U : P2S) - 1) * cos(i + 0.33)));
			if (RNS_SIZE >= 3) Z[2 * n + i] = ZP3().set_int(static_cast<int32>(((is32 ? P3U : P3S) - 1) * cos(i + 0.47)));
		}

		setProfiling(true);

		const splitter * const pSplit = _pSplit;
		const size_t ns = pSplit->getSize();
		if (ns > 1)
		{
			cl_ulong minT = cl_ulong(-1);
			for (size_t i = 0; i < ns; ++i)
			{
				resetProfiles();
				squareTune(16, i, Z);
				const cl_ulong t = getProfileTime();

#if defined(ocl_debug)
				std::ostringstream ss; ss << "[" << i << "]";
				for (size_t j = 0, nps = pSplit->getPartSize(i); j < nps; ++j) ss << " " << pSplit->getPart(i, j);
				ss << ": " << t << std::endl;
				pio::display(ss.str());
#endif
				if (t < minT)
				{
					minT = t;
					_splitIndex = i;
				}
			}
		}
#if defined(ocl_debug)
		{
			std::ostringstream ss;
			for (size_t j = 0, nps = pSplit->getPartSize(_splitIndex); j < nps; ++j) ss << " " << pSplit->getPart(_splitIndex, j);
			ss << std::endl;
			pio::display(ss.str());
		}
#endif
		delete[] Z;

		setProfiling(false);
	}

public:
	void info()
	{
		std::ostringstream ss; ss << "split:";
		for (size_t sIndex = 0, ns = _pSplit->getSize(); sIndex < ns; ++sIndex)
		{
			if (sIndex != 0) ss << ",";

			int lm = _ln;
			const size_t s = _pSplit->getPartSize(sIndex);
			for (size_t i = 1; i < s; ++i)
			{
				const uint32_t k = _pSplit->getPart(sIndex, i - 1);
				lm -= int(k);
				ss << " " << k;
				if (i != 1) ss << "(" << lm << ")"; else ss << "_0";
			}
			ss << " s" << lm;

			if (sIndex == _splitIndex) ss << " *";
		}

		ss << "." << std::endl;
		pio::display(ss.str());
	}
#endif	// TUNE
};


template<size_t RNS_SIZE, bool is32>
class transformGPUs : public transform
{
    const size_t root_bytes_;
    size_t root_word_bytes() const { return root_bytes_; }
    static_assert(RNS_SIZE == 3 && !is32, "This implementation requires three small primes");
    static_assert(P1S == small_prime_bounds::prime1 && P2S == small_prime_bounds::prime2 && P3S == small_prime_bounds::prime3, "Arithmetic bounds must match kernel primes");
	using ZP1 = ZPT<is32 ? P1U : P1S, is32 ? Q1U : Q1S, is32 ? R1U : R1S, is32 ? H1U : H1S>;
	using ZP2 = ZPT<is32 ? P2U : P2S, is32 ? Q2U : Q2S, is32 ? R2U : R2S, is32 ? H2U : H2S>;
	using ZP3 = ZPT<is32 ? P3U : P3S, is32 ? Q3U : Q3S, is32 ? R3U : R3S, is32 ? H3U : H3S>;

private:

	const size_t _mem_size, _cache_size;
	const size_t _num_regs;
	std::unique_ptr<ZP[]> _z;
	std::unique_ptr<engines<RNS_SIZE, is32>> _pEngine;

public:
	transformGPUs(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
				 const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose,
                     unsigned root_bits = 0, const std::string &embedded_kernel = {})
		: transform(size_t(1) << n, n, b, (RNS_SIZE == 2) ? EKind::NTT2 : EKind::NTT3),
        root_bytes_(root_bits ? root_bits/8 : configured_root_word_bytes()),
        _mem_size(RNS_SIZE * (size_t(1) << n) * ((num_regs + 1) * sizeof(ZP) + root_word_bytes()) + (size_t(1) << n) / 4 * sizeof(int64)),
        _cache_size(RNS_SIZE * (size_t(1) << n) * root_word_bytes()),
		_num_regs(num_regs), _z(new ZP[RNS_SIZE * (size_t(1) << n) * num_regs])
	{
		if (
            (n < 16 || n > 22) ||
            !small_prime_bounds::supported(b))
            throw std::runtime_error("Outside the bounded small-prime arithmetic range");

		const size_t size = getSize();

		const bool is_boinc_platform = isBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pEngine.reset(new engines<RNS_SIZE, is32>(eng_platform, is_boinc_platform ? 0 : device, static_cast<int>(n), isBoinc, num_regs, verbose, unsigned(root_bytes_*8)));

		std::ostringstream src; src << "#define GFN_ROOT_BITS " << (root_word_bytes()*8) << "\n"; src << "#define GFN_PAIRED_DIGITS " << (uint64_t(b)+2 >= P1S/2) << "\n"; src << "#define WARP_TAIL_STAGES 1\n"; src << "#define QUARTIC_LEAF 1\n"; src << "#define LAZY 1\n"; src << "#define SHOUP 3\n";


		src << "#define N_SZ\t" << (1u << n) << "u" << std::endl;
		src << "#define LN_SZ\t" << n << std::endl;
		src << "#define RNS_SZ\t" << RNS_SIZE << std::endl;
		src << "#define VSIZE\t" << VSIZE << std::endl;
		src << "#define LVSIZE\t" << LVSIZE << std::endl;

		if (is32) src << "#define IS32\t" << 1 << std::endl;

		src << "#define P1\t" << (is32 ? P1U : P1S) << "u" << std::endl;
		src << "#define Q1\t" << (is32 ? Q1U : Q1S) << "u" << std::endl;
		src << "#define RSQ1\t" << (is32 ? RSQ1U : RSQ1S) << "u" << std::endl;
		src << "#define IM1\t" << (is32 ? IM1U : IM1S) << "u" << std::endl;
		src << "#define MFIM1\t" << (is32 ? MFIM1U : MFIM1S) << "u" << std::endl;
		src << "#define SQRTI1\t" << (is32 ? SQRTI1U : SQRTI1S) << "u" << std::endl;
		src << "#define ISQRTI1\t" << (is32 ? ISQRTI1U : ISQRTI1S) << "u" << std::endl;

		src << "#define P2\t" << (is32 ? P2U : P2S) << "u" << std::endl;
		src << "#define Q2\t" << (is32 ? Q2U : Q2S) << "u" << std::endl;
		src << "#define RSQ2\t" << (is32 ? RSQ2U : RSQ2S) << "u" << std::endl;
		src << "#define IM2\t" << (is32 ? IM2U : IM2S) << "u" << std::endl;
		src << "#define MFIM2\t" << (is32 ? MFIM2U : MFIM2S) << "u" << std::endl;
		src << "#define SQRTI2\t" << (is32 ? SQRTI2U : SQRTI2S) << "u" << std::endl;
		src << "#define ISQRTI2\t" << (is32 ? ISQRTI2U : ISQRTI2S) << "u" << std::endl;

		src << "#define P3\t" << (is32 ? P3U : P3S) << "u" << std::endl;
		src << "#define Q3\t" << (is32 ? Q3U : Q3S) << "u" << std::endl;
		src << "#define RSQ3\t" << (is32 ? RSQ3U : RSQ3S) << "u" << std::endl;
		src << "#define IM3\t" << (is32 ? IM3U : IM3S) << "u" << std::endl;
		src << "#define MFIM3\t" << (is32 ? MFIM3U : MFIM3S) << "u" << std::endl;
		src << "#define SQRTI3\t" << (is32 ? SQRTI3U : SQRTI3S) << "u" << std::endl;
		src << "#define ISQRTI3\t" << (is32 ? ISQRTI3U : ISQRTI3S) << "u" << std::endl;

		src << "#define INVP2_P1\t" << (is32 ? INVP2_P1U : INVP2_P1S) << "u" << std::endl;
		src << "#define INVP3_P1\t" << (is32 ? INVP3_P1U : INVP3_P1S) << "u" << std::endl;
		src << "#define INVP3_P2\t" << (is32 ? INVP3_P2U : INVP3_P2S) << "u" << std::endl;

		src << "#define P1P2P3L\t" << (is32 ? P1P2P3LU : P1P2P3LS) << "u" << std::endl;
		src << "#define P1P2P3H\t" << (is32 ? P1P2P3HU : P1P2P3HS) << "ul" << std::endl;
		src << "#define P1P2P3_2L\t" << (is32 ? P1P2P3_2LU : P1P2P3_2LS) << "u" << std::endl;
		src << "#define P1P2P3_2H\t" << (is32 ? P1P2P3_2HU : P1P2P3_2HS) << "ul" << std::endl;

		src << "#define NORM1\t" << ZP1::norm(uint32(size / 2)).get() << "u" << std::endl;
		src << "#define NORM2\t" << ZP2::norm(uint32(size / 2)).get() << "u" << std::endl;
		src << "#define NORM3\t" << ZP3::norm(uint32(size / 2)).get() << "u" << std::endl;

		src << "#define W_SHFT\t" << size << "u" << std::endl;
		src << "#define WI_SHFT\t" << size / 2 << "u" << std::endl;
#if defined(USE_WI)
		src << "#define USE_WI\t" << 1 << std::endl;
#endif
		src << "#define BLK32\t" << BLK32 << std::endl;
		src << "#define BLK64\t" << BLK64 << std::endl;
		src << "#define BLK128\t" << BLK128 << std::endl;
		src << "#define BLK256\t" << BLK256 << std::endl;
		src << "#define BLK512\t" << BLK512 << std::endl;
		src << "#define BLK1024\t" << BLK1024 << std::endl;

		src << "#define CHUNK64\t" << CHUNK64 << std::endl;
		src << "#define CHUNK256\t" << CHUNK256 << std::endl;
		src << "#define CHUNK1024\t" << CHUNK1024 << std::endl;

#if defined(CHECK_RADIX4_FUNCTIONS)
		src << "#define SHORT_FUNC\t" << 1 << std::endl;
#endif
#if defined(TUNE)
		src << "#define ALL_FUNC\t" << 1 << std::endl;
#endif

		src << "#define NORM_WG_SZ\t" << _pEngine->getNormWGsize() << std::endl;

		src << "#define MAX_WG_SZ\t" << _pEngine->getMaxWorkGroupSize() << std::endl << std::endl;

        src << "#define GFN_CRT_SHOUP 1\n";
        // Shoup reciprocals belong to the selected prime set, not the transform size.
        auto inverseMod = [](uint64_t a, uint32_t p) {
            uint64_t value=1;
            for (uint32_t exponent=p-2; exponent; exponent>>=1, a=a*a%p)
                if (exponent&1) value=value*a%p;
            return uint32_t(value);
        };
        const uint32_t moduli[3]={P1S,P2S,P1S};
        const uint32_t divisors[3]={P3S,P3S,P2S};
        const char *names[3]={"CRT_INV13","CRT_INV23","CRT_INV12"};
        for (unsigned i=0;i<3;++i) {
            const uint32_t value=inverseMod(divisors[i],moduli[i]);
            src << "#define " << names[i] << " " << value << "u\n";
            src << "#define " << names[i] << "_Q " << ((uint64_t(value)<<32)/moduli[i]) << "u\n";
        }

        const int shift=31-__builtin_clz(b)-1;
        const uint32_t inverse=uint32_t((uint64_t(1)<<(shift+32))/b);
        src << "#define GFN_FIXED_B " << b << "u\n#define GFN_FIXED_INV " << inverse << "u\n#define GFN_FIXED_SHIFT " << shift << "\n";
        // Reconstruct C = k*(p2*p3) + t*p3 + r3, and divide using bounded
        // signed64 residuals. Check the entire supported coefficient interval.
        const int64_t pair = int64_t(P2S) * P3S;
        const int64_t qpair = (pair + b/2) / b, rpair = pair - qpair*b;
        const int64_t qthird = (int64_t(P3S) + b/2) / b, rthird = int64_t(P3S) - qthird*b;
        using wide = small_prime_bounds::wide;
        const wide coefficient = small_prime_bounds::coefficient_bound(b);
        const wide kmax = small_prime_bounds::ceil_div(coefficient, wide(pair)) + 2;
        const wide tmax = uint64_t(P2S)/2 + 1, rmax = uint64_t(P3S)-1;
        const wide carry = small_prime_bounds::carry_bound(b);
        auto magnitude = [](int64_t x) { return uint64_t(x < 0 ? -x : x); };
        if (2*(kmax*magnitude(rpair) + tmax*magnitude(rthird) + rmax) + carry >= wide(b)*(wide(1)<<29)
            || 2*(kmax*uint64_t(qpair) + tmax*uint64_t(qthird)) + (wide(1)<<29) >= (wide(1)<<63)
            || coefficient >= wide(P1S/2-4)*wide(pair))
            throw std::runtime_error("Three-prime CRT division range check failed");
        src << "#define CRT_QPAIR " << qpair << "l\n#define CRT_RPAIR " << rpair << "l\n";
        src << "#define CRT_QTHIRD " << qthird << "l\n#define CRT_RTHIRD " << rthird << "l\n";
        if (!embedded_kernel.empty()) src << embedded_kernel;
        else {
            const char *path = std::getenv("GENEFER_KERNEL");
            if (!path) throw std::runtime_error("Three-prime kernel source unavailable");
            std::ifstream kernel(path, std::ios::binary);
            if (!kernel) throw std::runtime_error("Cannot open the three-prime kernel source");
            src << std::string(std::istreambuf_iterator<char>(kernel), {});
        }
        _pEngine->loadProgram(src.str());
		_pEngine->allocMemory();
		_pEngine->createKernels(b);

		std::unique_ptr<ZP1[]> wr1(new ZP1[size]{});
		std::unique_ptr<ZP2[]> wr2(new ZP2[size]{});
        wr1[0]=wr1[size/2]=ZP1(0);
        wr2[0]=wr2[size/2]=ZP2(0);
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const ZP1 r_s1 = ZP1::primroot_n(4 * s);
			const ZP2 r_s2 = ZP2::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				const size_t sj = s + j, sji = s + (s - j - 1);
				const size_t e = bitRev(j, 2 * s) + 1;
				wr1[size / 2 + sji] = wr1[sj] = r_s1.pow(e);
				wr2[size / 2 + sji] = wr2[sj] = r_s2.pow(e);
			}
		}
		_pEngine->writeMemory_w(wr1.get(), 0);
		_pEngine->writeMemory_w(wr2.get(), 1);



		if (RNS_SIZE == 3)
		{
			std::unique_ptr<ZP3[]> wr3(new ZP3[size]{});
            wr3[0]=wr3[size/2]=ZP3(0);
			for (size_t s = 1; s < size / 2; s *= 2)
			{
				const ZP3 r_s3 = ZP3::primroot_n(4 * s);
				for (size_t j = 0; j < s; ++j)
				{
					wr3[size / 2 + s + (s - j - 1)] = wr3[s + j] = r_s3.pow(bitRev(j, 2 * s) + 1);
				}
			}
			_pEngine->writeMemory_w(wr3.get(), 2);

		}

#if defined(TUNE)
		_pEngine->tune();
#endif
	}

	virtual ~transformGPUs() = default;

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return _cache_size; }
    void synchronizeForCalibration() const override { _pEngine->synchronizeForCalibration(); }

protected:
	void getZi(int32_t * const zi) const override
	{
		_pEngine->readMemory_z(_z.get());

		const size_t size = getSize();

		const ZP * const z = _z.get();
		for (size_t i = 0; i < size; ++i)
		{
            const int64_t digit = decode_normalized_digit(z[i].get(),z[size+i].get());
            if (std::abs(int64_t(digit)) > int64_t(small_prime_bounds::digit_bound(getB()))
                || z[i].get() != ZP1().set_int(static_cast<int32_t>(digit)).get()
                || z[size+i].get() != ZP2().set_int(static_cast<int32_t>(digit)).get()
                || z[2*size+i].get() != ZP3().set_int(static_cast<int32_t>(digit)).get())
                throw std::runtime_error("Output residues violate the normalized digit invariant");
            zi[i] = static_cast<int32_t>(digit);
		}
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size = getSize();

		ZP * const z = _z.get();
		for (size_t i = 0; i < size; ++i)
		{
			if (std::abs(int64_t(zi[i])) > int64_t(small_prime_bounds::digit_bound(getB())))
                throw std::runtime_error("Input digit exceeds the arithmetic bound");
            z[0 * size + i] = ZP1().set_int(zi[i]);
			if (RNS_SIZE >= 2) z[1 * size + i] = ZP2().set_int(zi[i]);
			if (RNS_SIZE >= 3) z[2 * size + i] = ZP3().set_int(zi[i]);
		}
		_pEngine->writeMemory_z(z);
	}

public:
	bool readContext(file & cFile, const size_t nregs) override
	{
#if defined(GENEFER_AUTOMATIC_OPENCL)
        const auto representation=gpu_gfn::upstream_checkpoint(getB(),gpu_gfn::checkpoint_exponent(uint32_t(getSize())));
        int kind=0;
        const int expected_kind=static_cast<int>(representation.fields==2 ? EKind::NTT2 : EKind::NTT3);
        if(!cFile.read(reinterpret_cast<char *>(&kind),sizeof(kind)) || kind!=expected_kind) return false;
        const size_t size=getSize(),registers=nregs ? nregs : _num_regs;
        if(registers>_num_regs) return false;
        std::vector<uint32_t> buffer(representation.fields*size);
        for(size_t reg=0;reg<registers;++reg) {
            if(!cFile.read(reinterpret_cast<char *>(buffer.data()),buffer.size()*sizeof(uint32_t))) return false;
            ZP *z=_z.get()+reg*RNS_SIZE*size;
            for(size_t i=0;i<size;++i) {
                int64_t digit;
                try {
                    digit=gpu_gfn::checkpoint_digit(buffer[i],representation.primes[0]);
                    if(std::abs(digit)>int64_t(small_prime_bounds::digit_bound(getB()))) return false;
                    for(unsigned f=0;f<representation.fields;++f)
                        if(buffer[f*size+i]!=gpu_gfn::digit_residue(digit,representation.primes[f])) return false;
                } catch(const std::invalid_argument &) { return false; }
                z[i]=ZP1().set_int(int32_t(digit));
                z[size+i]=ZP2().set_int(int32_t(digit));
                z[2*size+i]=ZP3().set_int(int32_t(digit));
            }
        }
        _pEngine->writeMemory_z(_z.get(),registers);
        return true;
#else
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(getKind())) return false;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

        if (num_regs > _num_regs) return false;
		if (!cFile.read(reinterpret_cast<char *>(_z.get()), RNS_SIZE * size * num_regs * sizeof(ZP))) return false;
        // Checkpoint register encoding must match all three small primes.
        // Do not accept another implementation's NTT3 file based only on EKind.
        for (size_t reg = 0; reg < num_regs; ++reg)
        {
            const ZP *z = _z.get() + reg * RNS_SIZE * size;
            for (size_t i = 0; i < size; ++i)
            {
                const int64_t digit = decode_normalized_digit(z[i].get(),z[size+i].get());
                if (std::abs(int64_t(digit)) > int64_t(small_prime_bounds::digit_bound(getB()))
                    || z[i].get() != ZP1().set_int(static_cast<int32_t>(digit)).get()
                    || z[size+i].get() != ZP2().set_int(static_cast<int32_t>(digit)).get()
                    || z[2*size+i].get() != ZP3().set_int(static_cast<int32_t>(digit)).get()) return false;
            }
        }
		_pEngine->writeMemory_z(_z.get(), num_regs);

		return true;
#endif
	}

	void saveContext(file & cFile, const size_t nregs) const override
	{
#if defined(GENEFER_AUTOMATIC_OPENCL)
        const auto representation=gpu_gfn::upstream_checkpoint(getB(),gpu_gfn::checkpoint_exponent(uint32_t(getSize())));
        const int kind=static_cast<int>(representation.fields==2 ? EKind::NTT2 : EKind::NTT3);
        if(!cFile.write(reinterpret_cast<const char *>(&kind),sizeof(kind))) return;
        const size_t size=getSize(),registers=nregs ? nregs : _num_regs;
        if(registers>_num_regs) throw std::runtime_error("Checkpoint register count");
        _pEngine->readMemory_z(_z.get(),registers);
        std::vector<uint32_t> buffer(representation.fields*size);
        for(size_t reg=0;reg<registers;++reg) {
            const ZP *z=_z.get()+reg*RNS_SIZE*size;
            for(size_t i=0;i<size;++i) {
                const int64_t digit=decode_normalized_digit(z[i].get(),z[size+i].get());
                if(std::abs(digit)>int64_t(small_prime_bounds::digit_bound(getB()))
                   || z[i].get()!=ZP1().set_int(int32_t(digit)).get()
                   || z[size+i].get()!=ZP2().set_int(int32_t(digit)).get()
                   || z[2*size+i].get()!=ZP3().set_int(int32_t(digit)).get())
                    throw std::runtime_error("Checkpoint digit invariant");
                for(unsigned f=0;f<representation.fields;++f)
                    buffer[f*size+i]=gpu_gfn::digit_residue(digit,representation.primes[f]);
            }
            if(!cFile.write(reinterpret_cast<const char *>(buffer.data()),buffer.size()*sizeof(uint32_t))) return;
        }
#else
		const int kind = static_cast<int>(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

		_pEngine->readMemory_z(_z.get(), num_regs);
		if (!cFile.write(reinterpret_cast<const char *>(_z.get()), RNS_SIZE * size * num_regs * sizeof(ZP))) return;
#endif
	}

	void set(const uint32_t a) override
	{
		if (a > small_prime_bounds::digit_bound(getB()))
            throw std::runtime_error("Initial value exceeds the arithmetic bound");
        _pEngine->set(a);
	}

	void squareDup(const bool dup) override
    {
        _pEngine->runSquare(dup, [&] { _pEngine->square(); _pEngine->baseMod(dup); });
    }

	void squareMul(const int32_t a) override
	{
		if (a < 1) throw std::runtime_error("Scalar must be positive");
        if (a <= 2) squareDup(a == 2);
		else
		{
			_pEngine->square();
			_pEngine->baseModMul(a);
		}
	}

	void initMultiplicand(const size_t src) override
	{
		_pEngine->initMultiplicand(src);
	}

	void mul() override
	{
		_pEngine->mul();
		_pEngine->baseMod(false);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		_pEngine->copy(dst, src);
	}

	void info() const override
	{
#if defined(TUNE)
		_pEngine->info();
#endif
	}
};

} // namespace gpu_small_prime
