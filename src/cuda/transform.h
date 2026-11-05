/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or
modify it. Please give feedback to the authors if improvement is realized. It is distributed in the
hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>
#include <fstream>

#include "../ocl.h"
#include "../transform.h"

#include "profiles.h"

#define VSIZE 4

typedef cl_uint uint32;
typedef cl_int int32;
typedef cl_ulong uint64;
typedef cl_long int64;

#define P1S 1056440321u
#define Q1S 3238526977u
#define R1S 69206012u
#define RSQ1S 596300470u
#define H1S 207618036u
#define IM1S 160598537u
#define MFIM1S 350572486u
#define SQRTI1S 574280268u
#define ISQRTI1S 986507723u

#define P2S 1053818881u
#define Q2S 3241148417u
#define R2S 79691772u
#define RSQ2S 159648582u
#define H2S 557842404u
#define IM2S 882957072u
#define MFIM2S 676445185u
#define SQRTI2S 516220135u
#define ISQRTI2S 182280430u

#define P3S 1051721729u
#define Q3S 3243245569u
#define R3S 88080380u
#define RSQ3S 748646691u
#define H3S 528482280u
#define IM3S 467413740u
#define MFIM3S 410649836u
#define SQRTI3S 414819672u
#define ISQRTI3S 201694522u

#define INVP2_P1S 422574490u
#define INVP3_P1S 821674895u
#define INVP3_P2S 1053816833u

#define P1P2P3LS 3161980929u
#define P1P2P3HS 272616369202144000ul

#define P1P2P3_2LS 1580990464u
#define P1P2P3_2HS 136308184601072000ul

#define P1U (125 * (uint32(1) << 25) + 1)
#define Q1U 100663297u       // p * q = 1 (mod 2^32)
#define R1U 100663295u       // 2^32 mod p
#define RSQ1U 232465106u     // (2^32)^2 mod p
#define H1U 301989885u       // Montgomery form of the primitive root 3
#define IM1U 1486287593u     // MF of I = 3^{(p - 1)/4}
#define MFIM1U 3645424034u   // MF of MF of I to convert input into MF
#define SQRTI1U 3580437317u  // MF of 3^{(p - 1)/8}
#define ISQRTI1U 2017881188u // MF of i * sqrt(i)

#define P2U (243 * (uint32(1) << 24) + 1)
#define Q2U 218103809u
#define R2U 218103807u
#define RSQ2U 3444438393u
#define H2U 1526726649u // MF of the primitive root 7
#define IM2U 99906823u
#define MFIM2U 1773796560u
#define SQRTI2U 2024944857u
#define ISQRTI2U 2119710515u

#define P3U (235 * (uint32(1) << 24) + 1)
#define Q3U 352321537u
#define R3U 352321535u
#define RSQ3U 3810498414u
#define H3U 1056964605u // MF of the primitive root 3
#define IM3U 2213106415u
#define MFIM3U 2454519270u
#define SQRTI3U 3448990025u
#define ISQRTI3U 3659377330u

#define INVP2_P1U 1797558821u // MF of 1 / P2 (mod P1)
#define INVP3_P1U 3075822917u // MF of 1 / P3 (mod P1)
#define INVP3_P2U 4076863457u // MF of 1 / P3 (mod P2)

#define P1P2P3LU 3623878657u            // (P1 * P2 * P3) mod 2^32
#define P1P2P3HU 15696902887611105282ul // (P1 * P2 * P3) >> 32

#define P1P2P3_2LU 1811939328u           // (P1 * P2 * P3 / 2) mod 2^32
#define P1P2P3_2HU 7848451443805552641ul // (P1 * P2 * P3 / 2) >> 32

class ZP
{
  protected:
    uint32 _n;

  public:
    ZP()
    {
    }
    explicit ZP(const uint32 n) : _n(n)
    {
    }

    uint32 get() const
    {
        return _n;
    }
};

template <uint32 P, uint32 Q, uint32 R, uint32 H> class ZPT : public ZP
{
  private:
    static uint32 _add(const uint32 a, const uint32 b)
    {
        return a + b - ((a >= P - b) ? P : 0);
    }
    static uint32 _sub(const uint32 a, const uint32 b)
    {
        return a - b + ((a < b) ? P : 0);
    }

    static uint32 _mul(const uint32 lhs, const uint32 rhs)
    {
        const uint64 t = lhs * uint64(rhs);
        const uint32 lo = uint32(t), hi = uint32(t >> 32);
        const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
        return _sub(hi, mp);
    }

  public:
    ZPT()
    {
    }
    explicit ZPT(const uint32 n) : ZP(n)
    {
    }

    int32 get_int() const
    {
        return (_n >= P / 2) ? int32(_n - P) : int32(_n);
    }
    ZPT &set_int(const int32 i)
    {
        _n = (i < 0) ? (uint32(i) + P) : uint32(i);
        return *this;
    }

    ZPT mul(const ZPT &rhs) const
    {
        return ZPT(_mul(_n, rhs._n));
    }

    ZPT pow(const size_t e) const
    {
        if (e == 0)
            return ZPT(R); // MF of one is R
        ZPT r = ZPT(R), y = *this;
        for (size_t i = e; i != 1; i /= 2)
        {
            if (i % 2 != 0)
                r = r.mul(y);
            y = y.mul(y);
        }
        r = r.mul(y);
        return r;
    }

    static const ZPT primroot_n(const uint32 n)
    {
        return ZPT(H).pow((P - 1) / n);
    }
    static ZPT norm(const uint32 n)
    {
        return ZPT(P - (P - 1) / n);
    }
};

#if VSIZE == 4
#define LVSIZE 2
#elif VSIZE == 2
#define LVSIZE 1
#else
#define LVSIZE 0
#endif

// Launch geometry is fixed by the matching PTX kernels.

#define BLK32 32  // local size =   4KB, workgroup size =  256 / VSIZE
#define BLK64 16  // local size =   4KB, workgroup size =  256 / VSIZE
#define BLK128 8  // local size =   4KB, workgroup size =  256 / VSIZE
#define BLK256 4  // local size =   4KB, workgroup size =  256 / VSIZE
#define BLK512 2  // local size =   4KB, workgroup size =  256 / VSIZE
#define BLK1024 1 // local size =   4KB, workgroup size =  256 / VSIZE
#define BLK2048 1 // local size =   8KB, workgroup size =  512 / VSIZE
#define BLK4096 1 // local size =  16KB, workgroup size = 1024 / VSIZE

#define CHUNK64 4 // local size =  VSIZE * 1KB, workgroup size = 64
#define STAIR_GROUPS (512 / CHUNK256)
#define STAIR_META_OFFSET (STAIR_GROUPS * 2048 + (STAIR_GROUPS + 4) * 4)
#define STAIR_SHADOW_OFFSET (STAIR_META_OFFSET + 16)
#define CHUNK256 4  // local size =  VSIZE * 4KB, workgroup size = 256
#define CHUNK1024 1 // local size =  VSIZE * 4KB, workgroup size = 256

#define CREATE_TRANSFORM_KERNEL(name) _##name = createTransformKernel(#name);
#define CREATE_TRANSFORM_KERNELP(name) _##name = createTransformKernel(#name, false);
#define CREATE_MUL_KERNEL(name) _##name = createMulKernel(#name);
#define CREATE_NORMALIZE_KERNEL(name, b, b_inv, b_s)                                               \
    _##name = createNormalizeKernel(#name, b, b_inv, b_s);
#define CREATE_SETCOPY_KERNEL(name) _##name = createSetCopyKernel(#name);
#define CREATE_COPYP_KERNEL(name) _##name = createCopypKernel(#name);

#define DEFINE_FORWARD(u)                                                                          \
    void forward##u(const int lm)                                                                  \
    {                                                                                              \
        ek_fb(_forward##u, lm - LVSIZE, u / 4 * CHUNK##u, 4 * VSIZE);                              \
    }
#define DEFINE_BACKWARD(u)                                                                         \
    void backward##u(const int lm)                                                                 \
    {                                                                                              \
        ek_fb(_backward##u, lm - LVSIZE, u / 4 * CHUNK##u, 4 * VSIZE);                             \
    }
#define DEFINE_FORWARDn(u, v)                                                                      \
    void forward##u##_##v()                                                                        \
    {                                                                                              \
        ek(_forward##u##_##v, u / 4 * CHUNK##u, 4 * VSIZE);                                        \
    }
#define DEFINE_BACKWARDn(u, v)                                                                     \
    void backward##u##_##v()                                                                       \
    {                                                                                              \
        ek(_backward##u##_##v, u / 4 * CHUNK##u, 4 * VSIZE);                                       \
    }

#define DEFINE_SQUARE(u)                                                                           \
    void square##u()                                                                               \
    {                                                                                              \
        ek(_square##u, (u * BLK##u) / (4 * VSIZE), 4 * VSIZE);                                     \
    }
#define DEFINE_FWDP(u)                                                                             \
    void fwd##u##p()                                                                               \
    {                                                                                              \
        ek(_fwd##u##p, (u * BLK##u) / (4 * VSIZE), 4 * VSIZE);                                     \
    }
#define DEFINE_MUL(u)                                                                              \
    void mul##u()                                                                                  \
    {                                                                                              \
        ek(_mul##u, (u * BLK##u) / (4 * VSIZE), 4 * VSIZE);                                        \
    }

#define DEFINE_FORWARDP(u)                                                                         \
    void forward##u##p(const int lm)                                                               \
    {                                                                                              \
        setTransformArgs(_forward##u, false);                                                      \
        forward##u(lm);                                                                            \
        setTransformArgs(_forward##u);                                                             \
    }
#define DEFINE_FORWARDPn(u, v)                                                                     \
    void forward##u##p_##v()                                                                       \
    {                                                                                              \
        setTransformArgs(_forward##u##_##v, false);                                                \
        forward##u##_##v();                                                                        \
        setTransformArgs(_forward##u##_##v);                                                       \
    }

template <size_t RNS_SIZE, bool is32> class engines : public device
{
    using ZP1 = ZPT<is32 ? P1U : P1S, is32 ? Q1U : Q1S, is32 ? R1U : R1S, is32 ? H1U : H1S>;
    using ZP2 = ZPT<is32 ? P2U : P2S, is32 ? Q2U : Q2S, is32 ? R2U : R2S, is32 ? H2U : H2S>;
    using ZP3 = ZPT<is32 ? P3U : P3S, is32 ? Q3U : Q3S, is32 ? R3U : R3S, is32 ? H3U : H3S>;

  private:
    const size_t _n;
    const size_t _num_regs;
    const int _lnormWGsize;
    cl_mem _bridgeAux = nullptr;
    cl_kernel _bridge = nullptr, _stairMaterialize = nullptr;
    cl_mem _parity = nullptr;
    bool _squareParity = false;
    bool _ordinaryProduct = false;
    cl_mem _z = nullptr, _zp = nullptr, _w = nullptr, _c = nullptr;
    cl_kernel _forward256_ordinary = nullptr;
    cl_kernel _forward256_0 = nullptr, _backward256_0 = nullptr;
    cl_kernel _square2048 = nullptr, _fwd2048p = nullptr, _mul2048 = nullptr;
    cl_kernel _normalize1 = nullptr, _normalize2 = nullptr, _mulscalar = nullptr;
    cl_kernel _set = nullptr, _copy = nullptr, _copyp = nullptr;

    static constexpr int ilog2_32(const uint32_t n)
    {
        return 31 - __builtin_clz(n);
    }

  public:
    engines(const platform &platform, const size_t d, const int ln, const bool isBoinc,
            const size_t num_regs, const bool verbose)
        : device(platform, d, verbose), _n(size_t(1) << ln), _num_regs(num_regs),
          _lnormWGsize(std::min(std::max(5, ln / 2 - 3), ilog2_32(uint32_t(getMaxWorkGroupSize()))))
    {
    }
    virtual ~engines()
    {
    }

    ///////////////////////////////

  public:
    size_t getNormWGsize() const
    {
        return size_t(1 << _lnormWGsize);
    }

    void allocMemory()
    {
        const size_t n = _n;
        if (n != 0)
        {
            _bridgeAux = _createBuffer(CL_MEM_READ_WRITE, STAIR_SHADOW_OFFSET + 32);
            _parity = _createBuffer(CL_MEM_READ_WRITE, n / 2);
            _z = _createBuffer(CL_MEM_READ_WRITE, RNS_SIZE * n * _num_regs * sizeof(ZP));
            configureDataWindow(_z, RNS_SIZE * n * sizeof(ZP));

            _zp = _createBuffer(CL_MEM_READ_WRITE, RNS_SIZE * n * sizeof(ZP));
            _w = _createBuffer(CL_MEM_READ_ONLY, RNS_SIZE * n * sizeof(ZP));
            _c = _createBuffer(CL_MEM_READ_WRITE, n / 4 * sizeof(int64));
        }
    }

    void releaseMemory()
    {
        if (_n != 0)
        {
            _releaseBuffer(_bridgeAux);
            _releaseBuffer(_parity);
            _releaseBuffer(_z);
            _releaseBuffer(_zp);
            _releaseBuffer(_w);
            _releaseBuffer(_c);
        }
    }

    ///////////////////////////////

  private:
    cl_kernel createTransformKernel(const char *const kernelName, const bool isMultiplier = true)
    {
        cl_kernel kernel = _createKernel(kernelName);
        _setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
        _setKernelArg(kernel, 1, sizeof(cl_mem), &_w);
        if (std::string(kernelName) == "square2048")
            _setKernelArg(kernel, 2, sizeof(cl_mem), &_bridgeAux);
        if ((std::string(kernelName) == "forward256_0" ||
             std::string(kernelName) == "forward256_ordinary"))
            _setKernelArg(kernel, 2, sizeof(cl_mem), &_parity);
        return kernel;
    }

    cl_kernel createNormalizeKernel(const char *const kernelName, const uint32 b,
                                    const uint32 b_inv, const int32 b_s)
    {
        cl_kernel kernel = _createKernel(kernelName);
        _setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
        _setKernelArg(kernel, 1, sizeof(cl_mem), &_c);
        _setKernelArg(kernel, 2, sizeof(uint32), &b);
        _setKernelArg(kernel, 3, sizeof(uint32), &b_inv);
        _setKernelArg(kernel, 4, sizeof(int32), &b_s);
        if (std::string(kernelName) == "normalize1")
            _setKernelArg(kernel, 6, sizeof(cl_mem), &_parity);
        return kernel;
    }

    cl_kernel createMulKernel(const char *const kernelName)
    {
        cl_kernel kernel = _createKernel(kernelName);
        _setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
        _setKernelArg(kernel, 1, sizeof(cl_mem), &_zp);
        _setKernelArg(kernel, 2, sizeof(cl_mem), &_w);
        return kernel;
    }

    cl_kernel createSetCopyKernel(const char *const kernelName)
    {
        cl_kernel kernel = _createKernel(kernelName);
        _setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
        return kernel;
    }

    cl_kernel createCopypKernel(const char *const kernelName)
    {
        cl_kernel kernel = _createKernel(kernelName);
        _setKernelArg(kernel, 0, sizeof(cl_mem), &_zp);
        _setKernelArg(kernel, 1, sizeof(cl_mem), &_z);
        return kernel;
    }

  public:
    void createKernels(const uint32_t b)
    {
        CREATE_TRANSFORM_KERNEL(forward256_ordinary);
        CREATE_TRANSFORM_KERNEL(forward256_0);
        CREATE_TRANSFORM_KERNEL(backward256_0);
        CREATE_TRANSFORM_KERNEL(square2048);
        CREATE_TRANSFORM_KERNELP(fwd2048p);
        CREATE_MUL_KERNEL(mul2048);
        const uint32 b_ui = static_cast<uint32>(b);
        const int32 b_s = static_cast<int32>(31 - __builtin_clz(b) - 1);
        const uint32 b_inv = static_cast<uint32>((static_cast<uint64_t>(1) << (b_s + 32)) / b);
        CREATE_NORMALIZE_KERNEL(normalize1, b_ui, b_inv, b_s);
        CREATE_NORMALIZE_KERNEL(normalize2, b_ui, b_inv, b_s);
        CREATE_NORMALIZE_KERNEL(mulscalar, b_ui, b_inv, b_s);
        _stairMaterialize = _createKernel("materialize_stairway");
        _setKernelArg(_stairMaterialize, 0, sizeof(cl_mem), &_z);
        const bool signedBridge = (b >= 4194304u && b <= 30000000u);
        _bridge = _createKernel(signedBridge ? "boundary_bridge_signed" : "boundary_bridge");
        int64 sqp = P2S / b_ui, srp = P2S % b_ui;
        if (srp > int64(b_ui / 2))
        {
            srp -= b_ui;
            ++sqp;
        }
        const int64 smod = int64(uint64(P1S) * P2S);
        int64 sqm = smod / b_ui, srm = smod % b_ui;
        if (srm > int64(b_ui / 2))
        {
            srm -= b_ui;
            ++sqm;
        }
        const int64 scfg[4] = {sqp, srp, sqm, srm};
        if (signedBridge)
        {
            const int64 R = int64(1) << 32;
            const int64 numer = srp * R + b_ui / 2;
            const int64 rw = numer >= 0 ? numer / b_ui : -((-numer + b_ui - 1) / b_ui);
            const uint64 bb = uint64(b_ui) * b_ui;
            const uint32 pcfg[8] = {uint32(sqp),
                                    uint32(srp),
                                    uint32(rw),
                                    uint32((R + b_ui / 2) / b_ui),
                                    uint32((uint64(P2S) * uint64(R) + bb / 2) / bb),
                                    uint32(b_s),
                                    uint32(((uint64(1) << (32 + b_s)) + b_ui - 1) / b_ui),
                                    0};
            _writeBuffer(_bridgeAux, pcfg, sizeof(pcfg), STAIR_SHADOW_OFFSET);
        }
        else
            _writeBuffer(_bridgeAux, scfg, sizeof(scfg), STAIR_SHADOW_OFFSET);

        _setKernelArg(_bridge, 0, sizeof(cl_mem), &_z);
        _setKernelArg(_bridge, 1, sizeof(cl_mem), &_w);
        _setKernelArg(_bridge, 2, sizeof(cl_mem), &_parity);
        _setKernelArg(_bridge, 3, sizeof(cl_mem), &_bridgeAux);
        _setKernelArg(_bridge, 4, sizeof(uint32), &b_ui);
        const uint32 bridgeInv =
            signedBridge ? uint32(((uint64_t(1) << (b_s + 32)) + b / 2) / b) : b_inv;
        _setKernelArg(_bridge, 5, sizeof(uint32), &bridgeInv);
        _setKernelArg(_bridge, 6, sizeof(int32), &b_s);

        int64 qp = P2S / b, rp = P2S % b;
        if (rp > int64(b / 2))
        {
            rp -= b;
            ++qp;
        }
        const int64 modulus = int64(uint64(P1S) * P2S);
        int64 qm = modulus / b, rm = modulus % b;
        if (rm > int64(b / 2))
        {
            rm -= b;
            ++qm;
        }
        const int64 config[4] = {qp, rp, qm, rm};
        _writeBuffer(_c, config, sizeof(config), (_n / 4 - 4) * sizeof(int64));

        CREATE_SETCOPY_KERNEL(set);
        CREATE_SETCOPY_KERNEL(copy);
        CREATE_COPYP_KERNEL(copyp);
    }

    void releaseKernels()
    {
        _releaseKernel(_forward256_ordinary);
        _releaseKernel(_forward256_0);
        _releaseKernel(_backward256_0);
        _releaseKernel(_square2048);
        _releaseKernel(_fwd2048p);
        _releaseKernel(_mul2048);

        _releaseKernel(_stairMaterialize);
        _releaseKernel(_bridge);
        _releaseKernel(_normalize1);
        _releaseKernel(_normalize2);
        _releaseKernel(_mulscalar);

        _releaseKernel(_set);
        _releaseKernel(_copy);
        _releaseKernel(_copyp);
    }

    ///////////////////////////////

    void checkRegister(size_t r)
    {
        if (r >= _num_regs)
            throw std::runtime_error("Bridge register out of range");
    }
    void materializeZero()
    {
        const int32 before = 0, after = 1;
        _setKernelArg(_stairMaterialize, 1, sizeof(int32), &before);
        _executeKernel(_stairMaterialize, RNS_SIZE * _n / 4);
        backward256_0();
        _setKernelArg(_stairMaterialize, 1, sizeof(int32), &after);
        _executeKernel(_stairMaterialize, RNS_SIZE * _n / 4);
    }
    void restoreZero()
    {
        std::vector<uint32> epoch(STAIR_GROUPS + 8);
        _writeBuffer(_bridgeAux, epoch.data(), epoch.size() * sizeof(uint32), STAIR_GROUPS * 2048);
        forward256_ordinary();
    }
    void readMemory_z(ZP *const zPtr, const size_t count = 1)
    {
        if (!count || count > _num_regs)
            throw std::runtime_error("Invalid bridge register read count");
        materializeZero();
        _readBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP));
        restoreZero();
    }
    void writeMemory_z(const ZP *const zPtr, const size_t count = 1)
    {
        if (!count || count > _num_regs)
            throw std::runtime_error("Invalid bridge register write count");
        _writeBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP));
        restoreZero();
    }
    void writeMemory_w(const ZP *const wPtr, const size_t offset)
    {
        const uint64 p = (offset == 0 ? P1S : P2S), q = (offset == 0 ? Q1S : Q2S);
        const uint64 ri = p - ((p * q) >> 32);
        std::vector<uint32> roots(_n);
#if defined(GENEFER_GUARD_4060)
        for (size_t i = 0; i < _n / 2; ++i)
        {
            uint64 w = i ? (uint64(wPtr[i].get()) * ri) % p : 0;
            roots[i] = uint32((w << 32) / p);
        }
        for (size_t i = 0; i < _n / 4; ++i)
        {
            const uint32 q = roots[i], normal = uint32((uint64(q) * p >> 32) + (q != 0));
            roots[_n / 2 + 2 * i] = q;
            roots[_n / 2 + 2 * i + 1] = normal;
        }
#else
        for (size_t i = 0; i < _n; ++i)
        {
            uint64 w = (uint64(wPtr[i].get()) * ri) % p;
            roots[i] = uint32((w << 32) / p);
        }
#endif
        _writeBuffer(_w, roots.data(), _n * sizeof(uint32), offset * _n * sizeof(uint32));
    }

    ///////////////////////////////

  private:
    void ek(cl_kernel &kernel, const size_t localWorkSize, const size_t step)
    {
        const size_t n_s = _n / step;
        _executeKernel(kernel, RNS_SIZE * n_s, localWorkSize);
    }

    void ek_fb(cl_kernel &kernel, const int lm, const size_t localWorkSize, const size_t step)
    {
        const size_t n_s = _n / step;
        const int32 ilm = static_cast<int32>(lm);
        const uint32 is = static_cast<uint32>(n_s >> lm);
        _setKernelArg(kernel, 2, sizeof(int32), &ilm);
        _setKernelArg(kernel, 3, sizeof(uint32), &is);
        _executeKernel(kernel, RNS_SIZE * n_s, localWorkSize);
    }

    DEFINE_FORWARDn(256, ordinary);
    DEFINE_FORWARDn(256, 0);
    DEFINE_BACKWARDn(256, 0);
    DEFINE_SQUARE(2048);
    DEFINE_FWDP(2048);
    DEFINE_MUL(2048);

    void setTransformArgs(cl_kernel &kernel, const bool isMultiplier = true)
    {
        _setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
    }

    DEFINE_FORWARDPn(256, 0);

  public:
    void square()
    {
        _squareParity = true;
        square2048();
    }

    void mul()
    {
        materializeZero();
        _ordinaryProduct = true;
        _squareParity = false;
        forward256_0();
        mul2048();
        backward256_0();
    }

    void initMultiplicand(const size_t src)
    {
        checkRegister(src);
        if (src == 0)
            materializeZero();

        const uint32 isrc = static_cast<uint32>(src * RNS_SIZE * _n / 4);
        _setKernelArg(_copyp, 2, sizeof(uint32), &isrc);
        _executeKernel(_copyp, RNS_SIZE * _n / 4);

        forward256p_0();
        fwd2048p();
        if (src == 0)
            restoreZero();
    }

    void set(const uint32_t a)
    {

        const uint32 ia = static_cast<uint32>(a);
        _setKernelArg(_set, 1, sizeof(uint32), &ia);
        _executeKernel(_set, RNS_SIZE * _n / 4);
        restoreZero();
    }

    void copy(const size_t dst, const size_t src)
    {
        checkRegister(src);
        checkRegister(dst);
        if (src == dst)
            return;
        if (src == 0)
            materializeZero();

        const size_t size = _n / 4;
        const uint32 idst = static_cast<uint32>(dst * RNS_SIZE * size),
                     isrc = static_cast<uint32>(src * RNS_SIZE * size);
        _setKernelArg(_copy, 1, sizeof(uint32), &idst);
        _setKernelArg(_copy, 2, sizeof(uint32), &isrc);
        _executeKernel(_copy, RNS_SIZE * size);
        if (src == 0 || dst == 0)
            restoreZero();
    }

  public:
    void baseMod(const bool dup)
    {
        const int32 idup = dup ? 1 : 0;
        if (!_ordinaryProduct)
        {
            _setKernelArg(_bridge, 7, sizeof(int32), &idup);
            _executeKernel(_bridge, ((512 / CHUNK256) + 1) * (64 * CHUNK256) * 1,
                           64 * CHUNK256 * 1);
            return;
        }

        const size_t size = _n / 4;

        _setKernelArg(_normalize1, 5, sizeof(int32), &idup);
        const int32 useParity = _squareParity ? 1 : 0;
        _setKernelArg(_normalize1, 7, sizeof(int32), &useParity);
        _executeKernel(_normalize1, size, 1u << _lnormWGsize);
        _executeKernel(_normalize2, size >> _lnormWGsize);
        _ordinaryProduct = false;
        restoreZero();
    }

    void baseModMul(const int a)
    {

        baseMod(false);
        materializeZero();

        const int32 ia = static_cast<int32>(a);
        const size_t size = _n / 4;
        _setKernelArg(_mulscalar, 5, sizeof(int32), &ia);
        _executeKernel(_mulscalar, size, 1u << _lnormWGsize);
        _executeKernel(_normalize2, size >> _lnormWGsize);
        restoreZero();
    }
};

template <size_t RNS_SIZE, bool is32> class transformGPUs : public transform
{
    using ZP1 = ZPT<is32 ? P1U : P1S, is32 ? Q1U : Q1S, is32 ? R1U : R1S, is32 ? H1U : H1S>;
    using ZP2 = ZPT<is32 ? P2U : P2S, is32 ? Q2U : Q2S, is32 ? R2U : R2S, is32 ? H2U : H2S>;
    using ZP3 = ZPT<is32 ? P3U : P3S, is32 ? Q3U : Q3S, is32 ? R3U : R3S, is32 ? H3U : H3S>;

  private:
    const size_t _mem_size, _cache_size;
    const size_t _num_regs;
    ZP *const _z;
    engines<RNS_SIZE, is32> *_pEngine = nullptr;

  public:
    transformGPUs(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device,
                  const size_t num_regs, const cl_platform_id boinc_platform_id,
                  const cl_device_id boinc_device_id, const bool verbose)
        : transform(size_t(1) << n, n, b, (RNS_SIZE == 2) ? EKind::NTT2 : EKind::NTT3),
          _mem_size(RNS_SIZE * (size_t(1) << n) * (2 * num_regs + 3) / 2 * sizeof(ZP) +
                    (size_t(1) << n) / 4 * sizeof(int64)),
          _cache_size(RNS_SIZE * (size_t(1) << n) * 3 / 2 * sizeof(ZP)), _num_regs(num_regs),
          _z(new ZP[RNS_SIZE * (size_t(1) << n) * num_regs])
    {
        // std::cout << "NTT" << RNS_SIZE << (is32 ? "u" : "i") << std::endl;

        static_assert(RNS_SIZE == 2 && !is32, "GFN-19 requires the signed two-prime transform");
        if (n != 19)
            throw std::runtime_error("This CUDA transform supports GFN-19 only");

        const size_t size = getSize();

        const bool is_boinc_platform =
            isBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
        const platform eng_platform =
            is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

        _pEngine = new engines<RNS_SIZE, is32>(eng_platform, is_boinc_platform ? 0 : device,
                                               static_cast<int>(n), isBoinc, num_regs, verbose);

        _pEngine->loadProgram(gfn19::configuration);
        _pEngine->allocMemory();
        _pEngine->createKernels(b);

        ZP1 *const wr1 = new ZP1[size];
        ZP2 *const wr2 = new ZP2[size];
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
        _pEngine->writeMemory_w(wr1, 0);
        _pEngine->writeMemory_w(wr2, 1);
        delete[] wr1;
        delete[] wr2;

        if (RNS_SIZE == 3)
        {
            ZP3 *const wr3 = new ZP3[size];
            for (size_t s = 1; s < size / 2; s *= 2)
            {
                const ZP3 r_s3 = ZP3::primroot_n(4 * s);
                for (size_t j = 0; j < s; ++j)
                {
                    wr3[size / 2 + s + (s - j - 1)] = wr3[s + j] = r_s3.pow(bitRev(j, 2 * s) + 1);
                }
            }
            _pEngine->writeMemory_w(wr3, 2);
            delete[] wr3;
        }
    }

    virtual ~transformGPUs()
    {
        _pEngine->releaseKernels();
        _pEngine->releaseMemory();
        _pEngine->clearProgram();
        delete _pEngine;

        delete[] _z;
    }

    size_t getMemSize() const override
    {
        return _mem_size;
    }
    size_t getCacheSize() const override
    {
        return _cache_size;
    }

  protected:
    void getZi(int32_t *const zi) const override
    {
        _pEngine->readMemory_z(_z);

        const size_t size = getSize();

        const ZP *const z = _z;
        for (size_t i = 0; i < size; ++i)
        {
            zi[i] = ZP1(z[i].get()).get_int();
        }
    }

    void setZi(const int32_t *const zi) override
    {
        const size_t size = getSize();

        ZP *const z = _z;
        for (size_t i = 0; i < size; ++i)
        {
            z[0 * size + i] = ZP1().set_int(zi[i]);
            if (RNS_SIZE >= 2)
                z[1 * size + i] = ZP2().set_int(zi[i]);
            if (RNS_SIZE >= 3)
                z[2 * size + i] = ZP3().set_int(zi[i]);
        }
        _pEngine->writeMemory_z(z);
    }

  public:
    bool readContext(file &cFile, const size_t nregs) override
    {
        int kind = 0;
        if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind)))
            return false;
        if (kind != static_cast<int>(getKind()))
            return false;

        const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

        if (!cFile.read(reinterpret_cast<char *>(_z), RNS_SIZE * size * num_regs * sizeof(ZP)))
            return false;
        _pEngine->writeMemory_z(_z, num_regs);

        return true;
    }

    void saveContext(file &cFile, const size_t nregs) const override
    {
        const int kind = static_cast<int>(getKind());
        if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind)))
            return;

        const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

        _pEngine->readMemory_z(_z, num_regs);
        if (!cFile.write(reinterpret_cast<const char *>(_z),
                         RNS_SIZE * size * num_regs * sizeof(ZP)))
            return;
    }

    void set(const uint32_t a) override
    {
        _pEngine->set(a);
    }

    void squareDup(const bool dup) override
    {
        _pEngine->runSquareGraph(dup, [&]() {
            _pEngine->square();
            _pEngine->baseMod(dup);
        });
    }

    void squareMul(const int32_t a) override
    {
        if (a <= 2)
            squareDup(a == 2);
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
    }
};
