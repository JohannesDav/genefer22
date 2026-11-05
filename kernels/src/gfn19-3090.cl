/*
Copyright 2022, Yves Gallot
Derived from genefer22. Distributed under the repository's MIT license.
*/

#include "common.clh"

// Decode the inner roots on demand; cache decoded roots in the carry bridge.

INLINE void quartic_square8(const uint2_32 pq, uint4_32 z[2], const ulong w1, const ulong win1)
{
    const uint_32 q = (uint_32)w1, gamma = mul_hi(q, pq.s0) + (q != 0);
    FWD2v4(z[0], z[1], w1);
    z[0] = quartic_dot(pq, z[0], gamma, q);
    z[1] = quartic_dot(pq, z[1], pq.s0 - gamma, ~q);
    BCK2v4(z[0], z[1], win1);
}

INLINE void fwd8x2_write(const uint2_32 pq, const sz_t mg, __global uint4_32 *restrict const z,
                         __local const uint4_32 *restrict const Z,
                         __global const uint_32 *restrict const w, const sz_t sj)
{
    DECLARE_W12_24(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint4_32 zl[4];
    _loadl4(4, zl, Z, 1);
    _fwd8v4(pq, &zl[0], w1[0], &w2[0]);
    _fwd8v4(pq, &zl[2], w1[1], &w2[2]);
    _storeg4(4, z, mg, zl);
}

INLINE void mul_8x2(const uint2_32 pq, __local uint4_32 *restrict const Z, const sz_t mg,
                    const __global uint4_32 *restrict const zp,
                    __global const uint_32 *restrict const w,
                    __global const uint_32 *restrict const wi, const sz_t sj, const sz_t sji)
{
    DECLARE_W12_24(sj);
    DECLARE_WIN12_24(sji);
    uint4_32 zpl[4];
    _loadg4(4, zpl, zp, mg);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint4_32 zl[4];
    _loadl4(4, zl, Z, 1);
    _mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);
    _mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);
    _storel4(4, Z, 1, zl);
}

// --- v1, v2, v4 -- no barrier

// --- v2, v4 -- no barrier

INLINE void fwd8_write(const uint2_32 pq, const sz_t mg, __global VTYPE *restrict const z,
                       __local const VTYPE *restrict const Z,
                       __global const uint_32 *restrict const w, const sz_t sj)
{
    fwd8x2_write(pq, mg, z, Z, w, sj);
}

INLINE void mul_8(const uint2_32 pq, __local VTYPE *restrict const Z, const sz_t mg,
                  const __global VTYPE *restrict const zp, __global const uint_32 *restrict const w,
                  __global const uint_32 *restrict const wi, const sz_t sj, const sz_t sji)
{
    mul_8x2(pq, Z, mg, zp, w, wi, sj, sji);
}

// --- transform/macro ---

#define DECLARE_VAR_REGv4()                                                                        \
    const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 4),                             \
               mid = gid & ~((N_SZ / 16) - 1), id = gid % (N_SZ / 16);                             \
    const uint2_32 pq = g_pq[lid];                                                                 \
    __global uint4_32 *restrict const z = &zg[4 * mid];                                            \
    __global const uint_32 *restrict const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv4() __global const uint4_32 *restrict const zp = &zpg[4 * mid];

#define DECLARE_VAR_REG DECLARE_VAR_REGv4
#define DECLARE_VARP_REG DECLARE_VARP_REGv4

// --- transform without local mem ---

// --- transform ---

#define DECLARE_VAR(B_N, CHUNK_N)                                                                  \
    DECLARE_VAR_REG();                                                                             \
    const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N);                   \
    const sz_t i = local_id, chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N,                     \
               blockIdx = group_id * CHUNK_N + chunk_idx;                                          \
    __local VTYPE *const Zi = &Z[chunk_idx];                                                       \
    const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx;                  \
    const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm;                               \
    const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm,                               \
               ko = blockIdx - blockIdx_mm + idx_mm * 4;                                           \
    const sz_t sj = s + idx_m;                                                                     \
    DECLARE_IVAR(s, idx_m);

#define DECLARE_VAR_FORWARD()                                                                      \
    __global VTYPE *restrict const zi = &z[ki];                                                    \
    __global VTYPE *restrict const zo = &z[ko];

#define DECLARE_VAR_BACKWARD()                                                                     \
    __global VTYPE *restrict const zi = &z[ko];                                                    \
    __global VTYPE *restrict const zo = &z[ki];

#define BACKWARD_I(B_N, CHUNK_N)                                                                   \
    DECLARE_VAR(B_N, CHUNK_N);                                                                     \
    DECLARE_VAR_BACKWARD();                                                                        \
    backward_4i(pq, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sji / 1);

// -----------------

// -----------------

#define B_256 (256 / 4)

#define ATTR_256() __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))

#define FORWARD_256()                                                                              \
    const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);                         \
    forward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);                                 \
    const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);                            \
    forward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);                                    \
    forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);

__kernel ATTR_256() void forward256_0(__global VTYPE *restrict const zg,
                                      __global const uint_32 *restrict const wg,
                                      __global uchar *par)
{
    const int lm = LN_SZ - LVSIZE - 8;
    const unsigned int s = 256 / 4;
    __local VTYPE Z[4 * B_256 * CHUNK256];
    DECLARE_VAR(B_256, CHUNK256);
    DECLARE_VAR_FORWARD();
    forward_4i_parity(pq, g_f0[lid], B_256 * CHUNK256, &Z[i], B_256 << lm, zi, zg, par);
    FORWARD_256();
}

__kernel ATTR_256() void forward256_ordinary(__global VTYPE *restrict const zg,
                                             __global const uint_32 *restrict const wg,
                                             __global uchar *par)
{
    const int lm = LN_SZ - LVSIZE - 8;
    const unsigned int s = 256 / 4;
    __local VTYPE Z[4 * B_256 * CHUNK256];
    DECLARE_VAR(B_256, CHUNK256);
    DECLARE_VAR_FORWARD();
    forward_4i_parity_ordinary(pq, g_f0_ordinary[lid], B_256 * CHUNK256, &Z[i], B_256 << lm, zi, zg,
                               par);
    FORWARD_256();
}

#define BACKWARD_256()                                                                             \
    __local VTYPE Z[4 * B_256 * CHUNK256];                                                         \
    BACKWARD_I(B_256, CHUNK256);                                                                   \
    const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);                            \
    backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sji / 4);                                 \
    const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);                         \
    backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sji / 16);

__kernel ATTR_256() void backward256_0(__global VTYPE *restrict const zg,
                                       __global const uint_32 *restrict const wg)
{
    const int lm = LN_SZ - LVSIZE - 8;
    const unsigned int s = 256 / 4;
    BACKWARD_256();
    backward_4o_0(pq, g_b0[lid], B_256 << lm, zo, B_256 * CHUNK256, &Z[i]);
}

// -----------------

// -----------------

#define L1024S (1024 / VSIZE)

// if BLK1024 != 1 then const sz_t i1024 = (local_id & ~(L1024S / 4 - 1)) * 4, i256 = local_id %
// (L1024S / 4); if BLK1024 = 1 then const sz_t i1024 = 0, i256 = local_id;

INLINE uint4_32 tail_xor1(const uint4_32 v)
{
    uint_32 x0, x1, x2, x3;
    asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 31, -1;" : "=r"(x0) : "r"(v.s0));
    asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 31, -1;" : "=r"(x1) : "r"(v.s1));
    asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 31, -1;" : "=r"(x2) : "r"(v.s2));
    asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 31, -1;" : "=r"(x3) : "r"(v.s3));
    return (uint4_32)(x0, x1, x2, x3);
}

INLINE void tail_commute_1_0(uint4_32 a[4], const sz_t lane)
{
    const bool high = (lane & 1) != 0;
    {
        const uint4_32 x = a[0], y = a[2], v = high ? x : y, z = tail_xor1(v);
        a[0] = high ? z : x;
        a[2] = high ? y : z;
    }
    {
        const uint4_32 x = a[1], y = a[3], v = high ? x : y, z = tail_xor1(v);
        a[1] = high ? z : x;
        a[3] = high ? y : z;
    }
}
INLINE void tail_to_leaf(uint4_32 a[4], const sz_t lane)
{
    tail_commute_1_0(a, lane);
    const uint4_32 tmp = a[1];
    a[1] = a[2];
    a[2] = tmp;
}
INLINE void tail_from_leaf(uint4_32 a[4], const sz_t lane)
{
    const uint4_32 tmp = a[1];
    a[1] = a[2];
    a[2] = tmp;
    tail_commute_1_0(a, lane);
}
INLINE void warp_tail(const uint2_32 pq, __local uint4_32 *Z, __global const uint_32 *w,
                      __global const uint_32 *wi, const sz_t lane, const sz_t sj, const sz_t sji)
{
    uint4_32 a[4];
    const sz_t base = ((4 * lane) & ~7) + (lane % 2);
    barrier(CLK_LOCAL_MEM_FENCE);
    _loadl4(4, a, &Z[base], 2);
    {
        DECLARE_W12(sj / 2);
        _forward4(pq, a, w1, w2);
    }
    tail_to_leaf(a, lane);
    {
        DECLARE_W1_2(sj);
        DECLARE_WIN1_2(sji);
        quartic_square8(pq, &a[0], w1[0], win1[0]);
        quartic_square8(pq, &a[2], w1[1], win1[1]);
    }
    tail_from_leaf(a, lane);
    {
        DECLARE_WIN12(sji / 2);
        _backward4(pq, a, win1, win2);
    }
    _storel4(4, &Z[base], 2, a);
}

#define L2048S (2048 / VSIZE)

#define DECLARE_VAR_2048()                                                                         \
    __local VTYPE Z[L2048S];                                                                       \
    DECLARE_VAR_REG();                                                                             \
    const sz_t local_id = id % (L2048S / 4), group_id = id / (L2048S / 4);                         \
    const sz_t j = id, sj = N_SZ / 4 / VSIZE + j;                                                  \
    DECLARE_IVAR(N_SZ / 4 / VSIZE, j);                                                             \
    const sz_t i512 = local_id, k2048 = group_id * L2048S + i512;                                  \
    __global VTYPE *restrict const zk = &z[k2048];                                                 \
    __local VTYPE *const Zi512 = &Z[i512];                                                         \
    const sz_t i128 = ((4 * i512) & ~(4 * (L2048S / 16) - 1)) + (i512 % (L2048S / 16));            \
    __local VTYPE *const Zi128 = &Z[i128];                                                         \
    const sz_t i32 = ((4 * i512) & ~(4 * (L2048S / 64) - 1)) + (i512 % (L2048S / 64));             \
    __local VTYPE *const Zi32 = &Z[i32];                                                           \
    const sz_t i8 = ((4 * i512) & ~(4 * (L2048S / 256) - 1)) + (i512 % (L2048S / 256));            \
    __local VTYPE *const Zi8 = &Z[i8];                                                             \
    const sz_t i2 = ((4 * i512) & ~(4 * 2 - 1)) + (i512 % 2);                                      \
    __local VTYPE *const Zi2 = &Z[i2];                                                             \
    __local VTYPE *const Z4 = &Z[4 * i512];

__kernel __attribute__((reqd_work_group_size(L2048S / 4, 1, 1))) void square2048(
    __global VTYPE *restrict const zg, __global const uint_32 *restrict const wg,
    __global uchar *aux)
{
    DECLARE_VAR_2048();
    if (get_global_id(0) == 0)
    {
        volatile __global uint *m =
            (volatile __global uint *)(aux + (512 / CHUNK256) * 2048 + ((512 / CHUNK256) + 4) * 4);
        m[0] = 0;
        m[1] = m[1] + 1;
    }

    forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
    forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
    forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
    warp_tail(pq, Z, w, wi, local_id, sj, sji);
    backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));
    backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));
    backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));
}

// -----------------

__kernel __attribute__((reqd_work_group_size(L2048S / 4, 1, 1))) void fwd2048p(
    __global VTYPE *restrict const zg, __global const uint_32 *restrict const wg)
{
    DECLARE_VAR_2048();

    forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
    forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
    forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
    forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
    fwd8_write(pq, L2048S / 4, zk, Z4, w, sj);
}

// -----------------

__kernel __attribute__((reqd_work_group_size(L2048S / 4, 1, 1))) void mul2048(
    __global VTYPE *restrict const zg, __global const VTYPE *restrict const zpg,
    __global const uint_32 *restrict const wg)
{
    DECLARE_VAR_2048();
    DECLARE_VARP_REG();
    __global const VTYPE *restrict const zpk = &zp[k2048];

    forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
    forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
    forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
    forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
    mul_8(pq, Z4, L2048S / 4, zpk, w, wi, sj, sji);
    backward_4(pq, L2048S / 256, Zi8, wi, sji / (L2048S / 256));
    backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));
    backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));
    backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));
}

// -----------------

INLINE uint_32 barrett(const uint_64 a, const uint_32 b, const uint_32 b_inv, const int b_s,
                       uint_32 *a_p)
{
    // Using notations of Modular SIMD arithmetic in Mathemagix, Joris van der Hoeven, Grégoire
    // Lecerf, Guillaume Quintin, 2014, HAL. n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 =
    // 32 => h = 1. b < 2^31, alpha = 2^29 => a < 2^29 b 2^{r-1} < b <= 2^r then a < 2^{r + 29} =
    // 2^{s + 31} and (a >> s) < 2^31 b_inv = [2^{s + 32} / b] b_inv < 2^{s + 32} / b < 2^{s + 32} /
    // 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31 Let h be the number of iterations in Barrett's
    // reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32]. h = ([a/b] - a/b) + a/2^{s + 32}
    // (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s]
    // b_inv / 2^32]) Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1, 0
    // <= h < 1 + 1/2 + 1/2 => h = 1.

    const uint_32 d = mul_hi((uint_32)(a >> b_s), b_inv), r = (uint_32)(a)-d * b;
    const bool o = (r >= b);
    *a_p = d + (o ? 1 : 0);
    return r - (o ? b : 0);
}

INLINE int_32 reduce64(int_64 *f, const uint_32 b, const uint_32 b_inv, const int b_s)
{
    // 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
    // 2- t < 2^23 b^2 => t_h < b^2 / 2^6. If 2 <= b < 32 then t_h < 32^2 / 2^6 = 16 < 2^29 b
    const uint_64 t = abs(*f);
    const uint_64 t_h = t >> 29;
    const uint_32 t_l = (uint_32)(t) % (1u << 29);

    uint_32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
    uint_32 d_l, r_l = barrett(((uint_64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
    const uint_64 d = ((uint_64)(d_h) << 29) | d_l;

    const bool s = (*f < 0);
    *f = s ? -(int_64)(d) : (int_64)(d);
    int_32 rem = s ? -(int_32)(r_l) : (int_32)(r_l);
    if (rem > (int_32)(b / 2))
    {
        rem -= b;
        ++*f;
    }
    if (rem < -(int_32)(b / 2))
    {
        rem += b;
        --*f;
    }
    return rem;
}

INLINE int_64 garner2(const uint_32 r1, const uint_32 r2)
{
    const uint_64 P1P2 = P1 * (uint_64)(P2);
    uint_32 u12 = mulmod(submod(r1, r2, P1), INVP2_P1, PQ1); // P2 < P1
    const uint_64 n = r2 + u12 * (uint_64)(P2);
    const bool b = (n > P1P2 / 2);
    return (int_64)(n - (b ? P1P2 : 0));
}

INLINE void write_rns(__global uint4_32 *restrict const zi, const int4_32 r)
{
    uint4_32 zo1, zo2;
    zo1.s0 = set_int(r.s0, P1);
    zo2.s0 = set_int(r.s0, P2);
    zo1.s1 = set_int(r.s1, P1);
    zo2.s1 = set_int(r.s1, P2);
    zo1.s2 = set_int(r.s2, P1);
    zo2.s2 = set_int(r.s2, P2);
    zo1.s3 = set_int(r.s3, P1);
    zo2.s3 = set_int(r.s3, P2);

    zi[0 * N_SZ / 4] = zo1;
    zi[1 * N_SZ / 4] = zo2;
}

INLINE int_64 extend_parity(int_64 r, sz_t i, __global const uchar *par)
{
    const uint_32 expected =
        (i & 1) ? 0 : (((par[i / 8] ^ par[i / 8 + N_SZ / 8]) >> ((i / 2) & 3)) & 1);
    const int_64 modulus = (int_64)((uint_64)P1 * P2);
    if ((((uint_64)r) & 1) != expected)
        r += (r < 0) ? modulus : -modulus;
    return r;
}

INLINE int_32 repair_first(int_64 *f, const uint_32 b, const uint_32 inv, const int bs);

INLINE int_32 direct_wide_divide(int_64 *f, uint_32 b, uint_32 inv, int bs)
{
    uint_32 q, r = barrett(abs(*f), b, inv, bs, &q);
    // abs(f)<2^30*b: the estimate can be low by two, instead of one.
    if (r >= b)
    {
        r -= b;
        ++q;
    }
    bool negative = *f < 0;
    *f = negative ? -(int_64)q : (int_64)q;
    int_32 rem = negative ? -(int_32)r : (int_32)r;
    if (rem > (int_32)(b / 2))
    {
        rem -= b;
        ++*f;
    }
    if (rem < -(int_32)(b / 2))
    {
        rem += b;
        --*f;
    }
    return rem;
}
INLINE int_32 direct_coefficient(uint_32 u1, uint_32 u2, sz_t i, __global const uchar *par,
                                 int useParity, bool dup, int_64 *carry, uint_32 b, uint_32 inv,
                                 int bs, int_32 qp, int_32 rp, int_64 qm, int_32 rm)
{
    uint_32 k = mulmod(submod(u1, u2, P1), INVP2_P1, PQ1);
    uint lift;
    if (useParity)
    {
        uint expected = (i & 1) ? 0 : (((par[i / 8] ^ par[i / 8 + N_SZ / 8]) >> ((i / 2) & 3)) & 1);
        lift = (u2 ^ k ^ expected) & 1;
    }
    else
        lift = (k > P1 / 2 || (k == P1 / 2 && u2 > P2 / 2));
    int_64 small = (int_64)u2 + (int_64)rp * k - (lift ? (int_64)rm : 0);
    int_64 large = (int_64)qp * k - (lift ? qm : 0);
    if (dup)
    {
        small += small;
        large += large;
    }
    small += *carry;
    int_32 r = direct_wide_divide(&small, b, inv, bs);
    *carry = large + small;
    return r;
}

INLINE int_64 fused_crt_center(const uint_32 k, const uint_32 v)
{
    const uint_64 m = (uint_64)P1 * P2, n = (uint_64)k * P2 + v;
    return (int_64)(n - (n > m / 2 ? m : 0));
}
INLINE int_32 direct_coefficient_k(uint_32 k, uint_32 u2, sz_t i, __global const uchar *par,
                                   int useParity, bool dup, int_64 *carry, uint_32 b, uint_32 inv,
                                   int bs, int_32 qp, int_32 rp, int_64 qm, int_32 rm)
{

    uint lift;
    if (useParity)
    {
        uint expected = (i & 1) ? 0 : (((par[i / 8] ^ par[i / 8 + N_SZ / 8]) >> ((i / 2) & 3)) & 1);
        lift = (u2 ^ k ^ expected) & 1;
    }
    else
        lift = (k > P1 / 2 || (k == P1 / 2 && u2 > P2 / 2));
    int_64 small = (int_64)u2 + (int_64)rp * k - (lift ? (int_64)rm : 0);
    int_64 large = (int_64)qp * k - (lift ? qm : 0);
    if (dup)
    {
        small += small;
        large += large;
    }
    small += *carry;
    int_32 r = direct_wide_divide(&small, b, inv, bs);
    *carry = large + small;
    return r;
}

INLINE uint2_32 ordinary_spectrum_crt(const uint_32 a, const uint_32 d)
{
    // Ordinary forward state becomes R^-1 after the Montgomery quartic square.
    // Its required square normalization is R^2/(N/2), so the ordinary scale is2^14.
    uint_32 q = mul_hi(d, 133549u), v = (d << 15) - q * P2;
    if (v >= P2)
    {
        v -= P2;
        ++q;
    }
    const uint_32 diff = a + 2 * P1 - d, h = mul_hi(diff, 53687091u);
    uint_32 k = diff * 13205504u - h * P1 + q;
    if (k >= 2 * P1)
        k -= 2 * P1;
    if (k >= P1)
        k -= P1;
    return (uint2_32)(k, v);
}
INLINE void ordinary_spectrum_crt4(const uint4_32 a, const uint4_32 d, __private uint4_32 *k,
                                   __private uint4_32 *v)
{
    const uint2_32 x0 = ordinary_spectrum_crt(a.s0, d.s0), x1 = ordinary_spectrum_crt(a.s1, d.s1),
                   x2 = ordinary_spectrum_crt(a.s2, d.s2), x3 = ordinary_spectrum_crt(a.s3, d.s3);
    *k = (uint4_32)(x0.s0, x1.s0, x2.s0, x3.s0);
    *v = (uint4_32)(x0.s1, x1.s1, x2.s1, x3.s1);
}
INLINE int4_32 normalize_1(__global uint4_32 *restrict const zi, __global int_64 *restrict const c,
                           __local int_64 *const cl, const sz_t gid, const sz_t lid,
                           const uint_32 b, const uint_32 b_inv, const int b_s, const bool dup,
                           __global const uchar *par, const int useParity)
{
    const uint4_32 u1 = mulmod4(zi[0 * N_SZ / 4], NORM1, PQ1),
                   u2 = mulmod4(zi[1 * N_SZ / 4], NORM2, PQ2);
    int4_32 r;

    int_64 f;
    if (b >= (1u << 22))
    {
        __global const int_64 *conf = c + N_SZ / 4 - 4;
        const int_32 qp = (int_32)conf[0], rp = (int_32)conf[1], rm = (int_32)conf[3];
        const int_64 qm = conf[2];
        f = 0;
        r.s0 = direct_coefficient(u1.s0, u2.s0, 4 * gid, par, useParity, dup, &f, b, b_inv, b_s, qp,
                                  rp, qm, rm);
        r.s1 = direct_coefficient(u1.s1, u2.s1, 4 * gid + 1, par, useParity, dup, &f, b, b_inv, b_s,
                                  qp, rp, qm, rm);
        r.s2 = direct_coefficient(u1.s2, u2.s2, 4 * gid + 2, par, useParity, dup, &f, b, b_inv, b_s,
                                  qp, rp, qm, rm);
        r.s3 = direct_coefficient(u1.s3, u2.s3, 4 * gid + 3, par, useParity, dup, &f, b, b_inv, b_s,
                                  qp, rp, qm, rm);
    }
    else
    {
        int4_64 l = (int4_64)(garner2(u1.s0, u2.s0), garner2(u1.s1, u2.s1), garner2(u1.s2, u2.s2),
                              garner2(u1.s3, u2.s3));
        if (useParity)
        {
            l.s0 = extend_parity(l.s0, 4 * gid, par);
            l.s1 = extend_parity(l.s1, 4 * gid + 1, par);
            l.s2 = extend_parity(l.s2, 4 * gid + 2, par);
            l.s3 = extend_parity(l.s3, 4 * gid + 3, par);
        }
        if (dup)
            l += l;

        f = l.s0;
        r.s0 = reduce64(&f, b, b_inv, b_s);
        f += l.s1;
        r.s1 = reduce64(&f, b, b_inv, b_s);
        f += l.s2;
        r.s2 = reduce64(&f, b, b_inv, b_s);
        f += l.s3;
        r.s3 = reduce64(&f, b, b_inv, b_s);
    }

    cl[lid] = f;

    if (lid == NORM_WG_SZ - 1)
    {
        const sz_t i = (gid / NORM_WG_SZ + 1) % (N_SZ / 4 / NORM_WG_SZ);
        c[i] = (i == 0) ? -f : f;
    }

    return r;
}

// For b >= 2^22 and |coefficient| < 2^63, a normalized carry is
// bounded by ceil((2^63 + b/2)/(b-1)). Its first repair input is
// below 2^29*b; after one division its magnitude is less than b.
INLINE int_32 repair_first(int_64 *f, const uint_32 b, const uint_32 b_inv, const int b_s)
{
    const bool neg = (*f < 0);
    uint_32 q;
    const uint_32 r = barrett(abs(*f), b, b_inv, b_s, &q);
    *f = neg ? -(int_64)q : (int_64)q;
    int_32 rem = neg ? -(int_32)r : (int_32)r;
    if (rem > (int_32)(b / 2))
    {
        rem -= b;
        ++*f;
    }
    if (rem < -(int_32)(b / 2))
    {
        rem += b;
        --*f;
    }
    return rem;
}
INLINE int_32 repair_small(int_64 *f, const uint_32 b)
{
    const int_64 x = *f, h = (int_64)(b / 2);
    const int_32 q = (x > h) ? 1 : ((x < -h) ? -1 : 0);
    *f = q;
    return (int_32)(x - (int_64)q * b);
}

INLINE void normalize_2(__global uint4_32 *restrict const zi, __local int_64 *const cl,
                        const sz_t lid, const int4_32 r, const uint_32 b, const uint_32 b_inv,
                        const int b_s)
{
    int_64 f = (lid == 0) ? 0 : cl[lid - 1];
    int4_32 ro;
    if (b >= (1u << 22))
    {
        f += r.s0;
        ro.s0 = repair_first(&f, b, b_inv, b_s);
        f += r.s1;
        ro.s1 = repair_small(&f, b);
        f += r.s2;
        ro.s2 = repair_small(&f, b);
    }
    else
    {
        f += r.s0;
        ro.s0 = reduce64(&f, b, b_inv, b_s);
        f += r.s1;
        ro.s1 = reduce64(&f, b, b_inv, b_s);
        f += r.s2;
        ro.s2 = reduce64(&f, b, b_inv, b_s);
    }
    f += r.s3;
    ro.s3 = (sz_t)(f);

    write_rns(zi, ro);
}

__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1))) void normalize1(
    __global uint4_32 *restrict const z, __global int_64 *restrict const c, const uint_32 b,
    const uint_32 b_inv, const int b_s, const int_32 dup, __global const uchar *par,
    const int useParity)
{
    const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;
    __global uint4_32 *restrict const zi = &z[gid];
    __local int_64 cl[NORM_WG_SZ];

    const int4_32 r = normalize_1(zi, c, cl, gid, lid, b, b_inv, b_s, dup != 0, par, useParity);

    barrier(CLK_LOCAL_MEM_FENCE);

    normalize_2(zi, cl, lid, r, b, b_inv, b_s);
}

__kernel void normalize2(__global uint4_32 *restrict const z,
                         __global const int_64 *restrict const c, const uint_32 b,
                         const uint_32 b_inv, const int b_s)
{
    const sz_t gid = (sz_t)get_global_id(0);
    __global uint4_32 *restrict const zi = &z[NORM_WG_SZ * gid];

    const uint4_32 u1 = zi[0 * N_SZ / 4];
    int4_32 r;

    int_64 f = c[gid] + get_int(u1.s0, P1);
    r.s0 = reduce64(&f, b, b_inv, b_s);
    f += get_int(u1.s1, P1);
    r.s1 = reduce64(&f, b, b_inv, b_s);
    f += get_int(u1.s2, P1);
    r.s2 = reduce64(&f, b, b_inv, b_s);
    f += get_int(u1.s3, P1);
    r.s3 = (int_32)(f);

    write_rns(zi, r);
}

__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1))) void mulscalar(
    __global uint4_32 *restrict const z, __global int_64 *restrict const c, const uint_32 b,
    const uint_32 b_inv, const int b_s, const int_32 a)
{
    const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;
    __global uint4_32 *restrict const zi = &z[gid];
    __local int_64 cl[NORM_WG_SZ];

    const uint4_32 u1 = zi[0 * N_SZ / 4];
    int4_32 r;

    int_64 f = get_int(u1.s0, P1) * (int_64)(a);
    r.s0 = reduce64(&f, b, b_inv, b_s);
    f += get_int(u1.s1, P1) * (int_64)(a);
    r.s1 = reduce64(&f, b, b_inv, b_s);
    f += get_int(u1.s2, P1) * (int_64)(a);
    r.s2 = reduce64(&f, b, b_inv, b_s);
    f += get_int(u1.s3, P1) * (int_64)(a);
    r.s3 = reduce64(&f, b, b_inv, b_s);

    cl[lid] = f;

    if (lid == NORM_WG_SZ - 1)
    {
        const sz_t i = (gid / NORM_WG_SZ + 1) % (N_SZ / 4 / NORM_WG_SZ);
        c[i] = (i == 0) ? -f : f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    normalize_2(zi, cl, lid, r, b, b_inv, b_s);
}

__kernel void set(__global uint4_32 *restrict const z, const uint_32 a)
{
    const sz_t gid = (sz_t)get_global_id(0);
    z[gid] = (gid % (N_SZ / 4) == 0) ? (uint4_32)(a, 0, 0, 0) : (uint4_32)(0, 0, 0, 0);
}

__kernel void copy(__global uint4_32 *restrict const z, const sz_t dst, const sz_t src)
{
    const sz_t gid = (sz_t)get_global_id(0);
    z[dst + gid] = z[src + gid];
}

__kernel void copyp(__global uint4_32 *restrict const zp, __global const uint4_32 *restrict const z,
                    const sz_t src)
{
    const sz_t gid = (sz_t)get_global_id(0);
    zp[gid] = z[src + gid];
}

// Independent Genefer prototype: acyclic carry publication with a duplicate
// root strip. Dynamic logical IDs prevent waiting on unstarted predecessors.
// Algorithmic reference only: PRPLL carryFused, GPL-3.0; no code copied.
// Publish carry packets with a device-wide fence before their ready flags.
INLINE void sw_device_fence(void)
{
    __asm__ volatile("membar.gl;" : : : "memory");
}
#define SW_P (64 * CHUNK256)
#define SW_G (512 / CHUNK256)
#define SW_READY (SW_G * 2048)
#define SW_META (SW_READY + (SW_G + 4) * 4)
#define SW_SHADOW (SW_META + 16)
#define STAIR_OPAQUE 1
#undef DECLARE_VAR_REG
#define DECLARE_VAR_REG()                                                                          \
    const sz_t lid = rns, mid = rns * (N_SZ / 16), id = group * SW_P + (get_local_id(0) % SW_P);   \
    const uint2_32 pq = g_pq[lid];                                                                 \
    __global VTYPE *restrict const z = &zg[4 * mid];                                               \
    __global const uint_32 *restrict const w = &wg[lid * W_SHFT];
INLINE void sw_inverse(const sz_t rns, const sz_t group, __global VTYPE *zg,
                       __global const uint_32 *wg, __local VTYPE *Z, __private VTYPE *out)
{
    const int lm = 9;
    const unsigned int s = 64;
    BACKWARD_I(B_256, CHUNK256);
    const sz_t k4 = ((4 * threadIdx) & ~15) + (threadIdx % 4);
    backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sji / 4);
    const sz_t k16 = ((4 * threadIdx) & ~63) + (threadIdx % 16);
    backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sji / 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    _loadl(4, out, &Z[i], SW_P);
    _backward4_0(pq, g_b0[lid], out);
    barrier(CLK_LOCAL_MEM_FENCE);
}
INLINE void sw_forward(const sz_t rns, const sz_t group, __global VTYPE *zg,
                       __global const uint_32 *wg, __local VTYPE *Z,
                       __private const int4_32 *digits)
{
    const int lm = 9;
    const unsigned int s = 64;
    DECLARE_VAR(B_256, CHUNK256);
    DECLARE_VAR_FORWARD();
    VTYPE out[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        int4_32 v = digits[q];
        out[q] = (VTYPE)(set_int(v.s0, pq.s0), set_int(v.s1, pq.s0), set_int(v.s2, pq.s0),
                         set_int(v.s3, pq.s0));
    }
    forward4_ordinary(pq, g_f0_ordinary[lid], out);
    _storel(4, &Z[i], SW_P, out);
    FORWARD_256();
    barrier(CLK_LOCAL_MEM_FENCE);
}
__kernel __attribute__((reqd_work_group_size(SW_P * (1 + STAIR_CONCURRENT), 1, 1))) void
boundary_bridge(__global VTYPE *zg, __global const uint_32 *wg, __global uchar *par,
                __global uchar *aux, const uint_32 b, const uint_32 b_inv, const int b_s,
                const int dup)
{
    const sz_t localIndex = get_local_id(0), lane = localIndex % SW_P, rns = localIndex / SW_P;
    __local VTYPE Z[4 * SW_P * (1 + STAIR_CONCURRENT)];
    __local uint logical;
    volatile __global uint *meta = (volatile __global uint *)(aux + SW_META);
    volatile __global uint *ready = (volatile __global uint *)(aux + SW_READY);
    volatile __global int_64 *carry = (volatile __global int_64 *)aux;

    if (!localIndex)
        logical = atomic_inc(&meta[0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const sz_t group = (logical == SW_G) ? 0 : logical, chunk = lane % CHUNK256,
               row0 = lane / CHUNK256;
    const uint epoch = meta[1];
    __global const uchar *oldPar = par + ((epoch - 1) & 1) * (N_SZ / 4);
    __global uchar *newPar = par + (epoch & 1) * (N_SZ / 4);
    VTYPE a[4], d[4];
    int4_32 digits[4];
    sw_inverse(0, group, zg, wg, Z, a);
    sw_inverse(1, group, zg, wg, Z, d);
    __local int_64 *cl = (__local int_64 *)Z;
    if (!rns)
    {
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const sz_t row = row0 + 64 * q, gidx = row * 512 + group * CHUNK256 + chunk;
            int_64 f;
            int4_32 r;
            uint4_32 u, v;
            ordinary_spectrum_crt4(a[q], d[q], &u, &v);
            if (b >= (1u << 22))
            {
                __global const int_64 *cfg = (__global const int_64 *)(aux + SW_SHADOW);
                const int_32 qp = (int_32)cfg[0], rp = (int_32)cfg[1], rm = (int_32)cfg[3];
                const int_64 qm = cfg[2];
                f = 0;
                r.s0 = direct_coefficient_k(u.s0, v.s0, 4 * gidx, oldPar, 1, dup, &f, b, b_inv, b_s,
                                            qp, rp, qm, rm);
                r.s1 = direct_coefficient_k(u.s1, v.s1, 4 * gidx + 1, oldPar, 1, dup, &f, b, b_inv,
                                            b_s, qp, rp, qm, rm);
                r.s2 = direct_coefficient_k(u.s2, v.s2, 4 * gidx + 2, oldPar, 1, dup, &f, b, b_inv,
                                            b_s, qp, rp, qm, rm);
                r.s3 = direct_coefficient_k(u.s3, v.s3, 4 * gidx + 3, oldPar, 1, dup, &f, b, b_inv,
                                            b_s, qp, rp, qm, rm);
            }
            else
            {
                int4_64 l = (int4_64)(fused_crt_center(u.s0, v.s0), fused_crt_center(u.s1, v.s1),
                                      fused_crt_center(u.s2, v.s2), fused_crt_center(u.s3, v.s3));
                l.s0 = extend_parity(l.s0, 4 * gidx, oldPar);
                l.s1 = extend_parity(l.s1, 4 * gidx + 1, oldPar);
                l.s2 = extend_parity(l.s2, 4 * gidx + 2, oldPar);
                l.s3 = extend_parity(l.s3, 4 * gidx + 3, oldPar);
                if (dup)
                    l += l;
                f = l.s0;
                r.s0 = reduce64(&f, b, b_inv, b_s);
                f += l.s1;
                r.s1 = reduce64(&f, b, b_inv, b_s);
                f += l.s2;
                r.s2 = reduce64(&f, b, b_inv, b_s);
                f += l.s3;
                r.s3 = reduce64(&f, b, b_inv, b_s);
            }
            digits[q] = r;
            cl[q * SW_P + lane] = f;
            if (chunk == CHUNK256 - 1 && logical < SW_G)
                carry[row * SW_G + group] = f;
        }
    }
    sw_device_fence();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (!localIndex && logical < SW_G)
        atomic_xchg(&ready[group], epoch);
    // The root publishes its carries and exits. The duplicate closes the
    // negacyclic wrap after every original group has been assigned a ticket.
    if (logical == 0)
        return;
    if (!localIndex)
    {
        while (atomic_add(&ready[logical - 1], 0) < epoch)
        {
        }
        // The duplicate must not overwrite root input until root completed reads.
        if (logical == SW_G)
            while (atomic_add(&ready[0], 0) < epoch)
            {
            }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    sw_device_fence();
    if (!rns)
    {
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const sz_t row = row0 + 64 * q, gidx = row * 512 + group * CHUNK256 + chunk;
            int4_32 r = digits[q], ro;
            int_64 f =
                chunk ? cl[q * SW_P + lane - 1]
                      : (group ? carry[row * SW_G + group - 1]
                               : (row ? 1 : -1) * carry[((row + 255) % 256) * SW_G + SW_G - 1]);
            if (b >= (1u << 22))
            {
                f += r.s0;
                ro.s0 = repair_first(&f, b, b_inv, b_s);
                f += r.s1;
                ro.s1 = repair_small(&f, b);
                f += r.s2;
                ro.s2 = repair_small(&f, b);
            }
            else
            {
                f += r.s0;
                ro.s0 = reduce64(&f, b, b_inv, b_s);
                f += r.s1;
                ro.s1 = reduce64(&f, b, b_inv, b_s);
                f += r.s2;
                ro.s2 = reduce64(&f, b, b_inv, b_s);
            }
            f += r.s3;
            ro.s3 = (int_32)f;
            digits[q] = ro;
            newPar[gidx] = (uchar)(((uint)ro.s0 & 1) | (((uint)ro.s1 & 1) << 1) |
                                   (((uint)ro.s2 & 1) << 2) | (((uint)ro.s3 & 1) << 3));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const sz_t opaque = STAIR_OPAQUE ? (group >> 16) : 0;
    sw_forward(0, group, zg, wg + opaque, Z, digits);
    sw_forward(1, group, zg, wg + opaque, Z, digits);
}

INLINE uint_32 pc_shoup_root(uint_32 a, ulong root, uint2_32 pq)
{
    const uint_32 rq = (uint_32)root, w = (uint_32)(root >> 32), q = mul_hi(a, rq);
    return a * w - q * pq.s0;
}
INLINE uint2_32 pc_shoup_root2(uint2_32 a, ulong root, uint2_32 pq)
{
    return (uint2_32)(pc_shoup_root(a.s0, root, pq), pc_shoup_root(a.s1, root, pq));
}
INLINE uint4_32 pc_shoup_root4(uint4_32 a, ulong root, uint2_32 pq)
{
    return (uint4_32)(pc_shoup_root2(a.s01, root, pq), pc_shoup_root2(a.s23, root, pq));
}
#undef ROOTMUL4
#define ROOTMUL4(a, w, pq) pc_shoup_root4(a, (ulong)(w), pq)
INLINE void pc_forward4x4(const uint2_32 pq, uint4_32 z[4], const ulong w1, const ulong w2[2])
{
    FWD2v4(z[0], z[2], w1);
    FWD2v4(z[1], z[3], w1);
    FWD2v4(z[0], z[1], w2[0]);
    FWD2v4(z[2], z[3], w2[1]);
}

INLINE void pc_backward4x4(const uint2_32 pq, uint4_32 z[4], const ulong win1, const ulong win2[2])
{
    BCK2v4(z[0], z[1], win2[0]);
    BCK2v4(z[2], z[3], win2[1]);
    BCK2v4(z[0], z[2], win1);
    BCK2v4(z[1], z[3], win1);
}

#undef ROOTMUL4
#define ROOTMUL4(a, w, pq)                                                                         \
    ((sizeof(w) == 4) ? lscalar4(a, (uint_32)(w), pq) : shoup_root4(a, (ulong)(w), pq))
// Root-cache experiment: only the accepted 256-row, four-column bridge.
INLINE void rc_fwd4(const uint2_32 pq, const sz_t stride, __local VTYPE *z, __local const ulong *w,
                    const sz_t sj)
{
    const ulong w1 = w[sj];
    const ulong w2[2] = {w[2 * sj], w[2 * sj + 1]};
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE v[4];
    _loadl(4, v, z, stride);
    pc_forward4x4(pq, v, w1, w2);
    _storel(4, z, stride, v);
}
INLINE void rc_bck4(const uint2_32 pq, const sz_t stride, __local VTYPE *z, __local const ulong *w,
                    const sz_t sj)
{
    const ulong w1 = w[sj];
    const ulong w2[2] = {w[2 * sj + 1], w[2 * sj]};
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE v[4];
    _loadl(4, v, z, stride);
    pc_backward4x4(pq, v, w1, w2);
    _storel(4, z, stride, v);
}
INLINE void rc_inverse(const sz_t rns, const sz_t group, __global VTYPE *zg,
                       __local const ulong *roots, __local VTYPE *Z, __private VTYPE *out)
{
    const sz_t lane = get_local_id(0), c = lane % CHUNK256, t = lane / CHUNK256;
    const uint2_32 pq = g_pq[rns];
    __local const ulong *w = roots + 256 * rns;
    __local VTYPE *Zi = Z + c;
    const sz_t sj = 127 - t, ko = group * CHUNK256 + c + (t << 11);
    const ulong w1 = w[sj];
    const ulong w2[2] = {w[2 * sj + 1], w[2 * sj]};
    VTYPE v[4];
    _loadg(4, v, zg + rns * (N_SZ / 4) + ko, 512);
    pc_backward4x4(pq, v, w1, w2);
    _storel(4, &Zi[CHUNK256 * 4 * t], CHUNK256, v);
    const sz_t k4 = ((4 * t) & ~15) + (t % 4);
    rc_bck4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
    const sz_t k16 = ((4 * t) & ~63) + (t % 16);
    rc_bck4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    _loadl(4, out, &Z[lane], SW_P);
    _backward4_0(pq, g_b0[rns], out);
    barrier(CLK_LOCAL_MEM_FENCE);
}
INLINE void rc_forward(const sz_t rns, const sz_t group, __global VTYPE *zg,
                       __local const ulong *roots, __local VTYPE *Z,
                       __private const int4_32 *digits)
{
    const sz_t lane = get_local_id(0), c = lane % CHUNK256, t = lane / CHUNK256;
    const uint2_32 pq = g_pq[rns];
    __local const ulong *w = roots + 256 * rns;
    __local VTYPE *Zi = Z + c;
    VTYPE v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        const int4_32 a = digits[q];
        v[q] = (VTYPE)(set_int(a.s0, pq.s0), set_int(a.s1, pq.s0), set_int(a.s2, pq.s0),
                       set_int(a.s3, pq.s0));
    }
    forward4_ordinary(pq, g_f0_ordinary[rns], v);
    _storel(4, &Z[lane], SW_P, v);
    const sz_t sj = 64 + t, k16 = ((4 * t) & ~63) + (t % 16);
    rc_fwd4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
    const sz_t k4 = ((4 * t) & ~15) + (t % 4);
    rc_fwd4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
    const ulong w1 = w[sj];
    const ulong w2[2] = {w[2 * sj], w[2 * sj + 1]};
    barrier(CLK_LOCAL_MEM_FENCE);
    _loadl(4, v, &Zi[CHUNK256 * 4 * t], CHUNK256);
    pc_forward4x4(pq, v, w1, w2);
    const sz_t ko = group * CHUNK256 + c + (t << 11);
    _storeg(4, zg + rns * (N_SZ / 4) + ko, 512, v);
    barrier(CLK_LOCAL_MEM_FENCE);
}

// All low-word multiply/subtract operations intentionally wrap unsigned.
INLINE int cp_small(int x, uint b, uint reciprocal, uint shift)
{
    const uint biased = (uint)(x + (int)(7 * b / 2));
    return (int)(mul_hi(biased, reciprocal) >> shift) - 3;
}
INLINE void cp_split(uint k, uint v, sz_t i, __global const uchar *par, bool dup, uint b, int qp,
                     int rp, int rw, int vw, int tw, int *r, int *u, int *t)
{
    const uint expected =
        (i & 1) ? 0 : (((par[i / 8] ^ par[i / 8 + N_SZ / 8]) >> ((i / 2) & 3)) & 1);
    int sk = as_int(k - (((v ^ k ^ expected) & 1) ? P1 : 0u)), sv = (int)v;
    if (dup)
    {
        sk += sk;
        sv += sv;
    }
    const int a = mul_hi(sk, rw) + mul_hi(sv, vw) + 1;
    *t = mul_hi(sk, tw);
    *r = as_int((uint)sv + (uint)sk * (uint)rp - b * (uint)a);
    *u = as_int((uint)sk * (uint)qp + (uint)a - b * (uint)*t);
}
// Pack the three signed predecessor carry components into 48 bits.
INLINE ulong cp_pack(int u, int t, int h)
{
    return ((ulong)((uint)u & 0x7ffffffu)) | ((ulong)((uint)t & 0x3ffffu) << 27) |
           ((ulong)((uint)h & 7u) << 45);
}
INLINE int cp_unpack_u(ulong f)
{
    return as_int((uint)f << 5) >> 5;
}
INLINE int cp_unpack_t(ulong f)
{
    return as_int((uint)(f >> 27) << 14) >> 14;
}
INLINE int cp_unpack_h(ulong f)
{
    return as_int((uint)(f >> 45) << 29) >> 29;
}
__kernel __attribute__((reqd_work_group_size(SW_P * (1 + STAIR_CONCURRENT), 1, 1))) void
boundary_bridge_signed(__global VTYPE *zg, __global const uint_32 *wg, __global uchar *par,
                       __global uchar *aux, const uint_32 b, const uint_32 b_inv, const int b_s,
                       const int dup)
{
    const sz_t localIndex = get_local_id(0), lane = localIndex % SW_P, rns = localIndex / SW_P;
    __local VTYPE Z[4 * SW_P * (1 + STAIR_CONCURRENT)];
    __local uint logical;
    __local ulong cachedRoots[512];
    volatile __global uint *meta = (volatile __global uint *)(aux + SW_META);
    volatile __global uint *ready = (volatile __global uint *)(aux + SW_READY);
    volatile __global int_64 *carry = (volatile __global int_64 *)aux;

#pragma unroll
    for (sz_t rootWord = localIndex; rootWord < 512; rootWord += SW_P)
    {
        const uint_32 q = wg[(rootWord / 256) * W_SHFT + rootWord % 256];
        const uint_32 p = g_pq[rootWord / 256].s0, w = mul_hi(q, p) + (q != 0);
        cachedRoots[rootWord] = ((ulong)w << 32) | q;
    }
    if (!localIndex)
        logical = atomic_inc(&meta[0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const sz_t group = (logical == SW_G) ? 0 : logical, chunk = lane % CHUNK256,
               row0 = lane / CHUNK256;
    const uint epoch = meta[1];
    __global const uchar *oldPar = par + ((epoch - 1) & 1) * (N_SZ / 4);
    __global uchar *newPar = par + (epoch & 1) * (N_SZ / 4);
    VTYPE a[4], d[4];
    int4_32 digits[4];
    rc_inverse(0, group, zg, cachedRoots, Z, a);
    rc_inverse(1, group, zg, cachedRoots, Z, d);
    __local int_64 *cl = (__local int_64 *)Z;
    __global const uint *cfg = (__global const uint *)(aux + SW_SHADOW);
    const int qp = as_int(cfg[0]), rp = as_int(cfg[1]), rw = as_int(cfg[2]);
    const int vw = as_int(cfg[3]), tw = as_int(cfg[4]);
    const uint shift = cfg[5], reciprocal = cfg[6];

    if (!rns)
    {
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const sz_t row = row0 + 64 * q, gidx = row * 512 + group * CHUNK256 + chunk;
            uint4_32 k, v;
            ordinary_spectrum_crt4(a[q], d[q], &k, &v);
            int4_32 r, u, t;
            int r0, u0, t0;
            cp_split(k.s0, v.s0, 4 * gidx, oldPar, dup, b, qp, rp, rw, vw, tw, &r0, &u0, &t0);
            r.s0 = r0;
            u.s0 = u0;
            t.s0 = t0;
            int r1, u1, t1;
            cp_split(k.s1, v.s1, 4 * gidx + 1, oldPar, dup, b, qp, rp, rw, vw, tw, &r1, &u1, &t1);
            r.s1 = r1;
            u.s1 = u1;
            t.s1 = t1;
            int r2, u2, t2;
            cp_split(k.s2, v.s2, 4 * gidx + 2, oldPar, dup, b, qp, rp, rw, vw, tw, &r2, &u2, &t2);
            r.s2 = r2;
            u.s2 = u2;
            t.s2 = t2;
            int r3, u3, t3;
            cp_split(k.s3, v.s3, 4 * gidx + 3, oldPar, dup, b, qp, rp, rw, vw, tw, &r3, &u3, &t3);
            r.s3 = r3;
            u.s3 = u3;
            t.s3 = t3;
            r.s1 += u.s0;
            r.s2 += u.s1 + t.s0;
            r.s3 += u.s2 + t.s1;
            const int h2 = cp_small(r.s2, b, reciprocal, shift),
                      h3 = cp_small(r.s3, b, reciprocal, shift);
            r.s2 -= ((int)b) * h2;
            r.s3 = r.s3 - ((int)b) * h3 + h2;
            const int_64 f = as_long(cp_pack(u.s3 + t.s2, t.s3, h3));
            digits[q] = r;
            cl[q * SW_P + lane] = f;
            if (chunk == CHUNK256 - 1 && logical < SW_G)
                carry[row * SW_G + group] = f;
        }
    }
    sw_device_fence();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (!localIndex && logical < SW_G)
        atomic_xchg(&ready[group], epoch);
    // The root publishes its carries and exits. The duplicate closes the
    // negacyclic wrap after every original group has been assigned a ticket.
    if (logical == 0)
        return;
    if (!localIndex)
    {
        while (atomic_add(&ready[logical - 1], 0) < epoch)
        {
        }
        // The duplicate must not overwrite root input until root completed reads.
        if (logical == SW_G)
            while (atomic_add(&ready[0], 0) < epoch)
            {
            }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    sw_device_fence();
    if (!rns)
    {
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const sz_t row = row0 + 64 * q, gidx = row * 512 + group * CHUNK256 + chunk;
            int4_32 r = digits[q], ro;
            const ulong packet =
                as_ulong(chunk ? cl[q * SW_P + lane - 1]
                               : (group ? carry[row * SW_G + group - 1]
                                        : carry[((row + 255) % 256) * SW_G + SW_G - 1]));
            int iu = cp_unpack_u(packet), it = cp_unpack_t(packet), ih = cp_unpack_h(packet);
            if (!chunk && !group && !row)
            {
                iu = -iu;
                it = -it;
                ih = -ih;
            }
            const int e0 = r.s0 + iu, e1 = r.s1 + it;
            const int h0 = cp_small(e0, b, reciprocal, shift),
                      h1 = cp_small(e1, b, reciprocal, shift);
            ro = (int4_32)(e0 - (int)b * h0 + ih, e1 - (int)b * h1 + h0, r.s2 + h1, r.s3);
            digits[q] = ro;
            newPar[gidx] = (uchar)(((uint)ro.s0 & 1) | (((uint)ro.s1 & 1) << 1) |
                                   (((uint)ro.s2 & 1) << 2) | (((uint)ro.s3 & 1) << 3));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const sz_t opaque = STAIR_OPAQUE ? (group >> 16) : 0;
    rc_forward(0, group, zg, cachedRoots + opaque, Z, digits);
    rc_forward(1, group, zg, cachedRoots + opaque, Z, digits);
}

// Ordinary digits are materialized only for readback, then transformed again.
// packed-square NORM=2/N, while inverse outer scale is1/256.
__kernel void materialize_stairway(__global uint4_32 *zg, const int phase)
{
    const sz_t gid = get_global_id(0), rns = gid / (N_SZ / 4);
    if (!phase)
    {
        zg[gid] = fold24(zg[gid], g_pq[rns].s0);
        return;
    }
    const uint_32 scale =
        (rns == 0)
            ? (uint_32)(((((uint_64)NORM1 * (N_SZ / 512)) % P1) * (((uint_64)1 << 32) % P1)) % P1)
            : (uint_32)(((((uint_64)NORM2 * (N_SZ / 512)) % P2) * (((uint_64)1 << 32) % P2)) % P2);
    zg[gid] = mulmod4(zg[gid], scale, g_pq[rns]);
}
