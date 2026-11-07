/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or
modify it. Please give feedback to the authors if improvement is realized. It is distributed in the
hope that it will be useful.
*/

#if __OPENCL_VERSION__ >= 120
#define INLINE static inline
#else
#define INLINE
#endif

#if defined(__NV_CL_C_VERSION)
#define PTX_ASM 1
#endif

#define N_SZ 65536u
#define LN_SZ 16
#define RNS_SZ PG_K
#define VSIZE 4
#define LVSIZE 2
#define W_SHFT 32768u
#define WI_SHFT 32768u
#define MAX_WG_SZ 1024

typedef uint sz_t;
typedef uint uint_32;
typedef int int_32;
typedef ulong uint_64;
typedef long int_64;
typedef uint2 uint2_32;
typedef uint4 uint4_32;
typedef int4 int4_32;
typedef long4 int4_64;

// 32-bit tables store a Shoup quotient; 64-bit tables store value and quotient.
#if PG_ROOT_BITS == 64
typedef ulong root_word_t;
typedef ulong2 root2_t;
typedef ulong4 root4_t;
#else
typedef uint_32 root_word_t;
typedef uint2_32 root2_t;
typedef uint4_32 root4_t;
#endif

// --- modular arithmetic

#define PQ1 (uint2_32)(P1, Q1)
#define PQ2 (uint2_32)(P2, Q2)
#define PQ3 (uint2_32)(P3, Q3)

/* @G_PQ@ */
/* @G_B0@ */

INLINE uint_32 addmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p) {
#if defined(IS32)
    return lhs + rhs - ((lhs >= p - rhs) ? p : 0);
#else
    const uint_32 t = lhs + rhs;
    return t - ((t >= p) ? p : 0);
#endif
}

INLINE uint_32 submod(const uint_32 lhs, const uint_32 rhs, const uint_32 p) {
#if defined(IS32)
    return lhs - rhs + ((lhs < rhs) ? p : 0);
#else
    const uint_32 t = lhs - rhs;
    return t + (((int_32)(t) < 0) ? p : 0);
#endif
}

// 2 mul + 2 mul_hi
INLINE uint_32 mulmod(const uint_32 lhs, const uint_32 rhs, const uint2_32 pq) {
    const uint_64 t = lhs * (uint_64)(rhs);
    const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
    const uint_32 mp = mul_hi(lo * pq.s1, pq.s0);
    return submod(hi, mp, pq.s0);
}

INLINE uint_32 sqrmod(const uint_32 lhs, const uint2_32 pq) { return mulmod(lhs, lhs, pq); }

INLINE int_32 get_int(const uint_32 n, const uint_32 p) {
    return (int_32)(n - ((n >= p / 2) ? p : 0));
}
INLINE uint_32 set_int(const int_32 i, const uint_32 p) { return (uint_32)(i + ((i < 0) ? p : 0)); }

// --- v2

INLINE uint2_32 addmod2(const uint2_32 lhs, const uint2_32 rhs, const uint_32 p) {
    return (uint2_32)(addmod(lhs.s0, rhs.s0, p), addmod(lhs.s1, rhs.s1, p));
}

INLINE uint2_32 submod2(const uint2_32 lhs, const uint2_32 rhs, const uint_32 p) {
    return (uint2_32)(submod(lhs.s0, rhs.s0, p), submod(lhs.s1, rhs.s1, p));
}

INLINE uint2_32 mulmod2(const uint2_32 lhs, const uint2_32 rhs, const uint2_32 pq) {
    return (uint2_32)(mulmod(lhs.s0, rhs.s0, pq), mulmod(lhs.s1, rhs.s1, pq));
}

// --- v4

INLINE uint4_32 addmod4(const uint4_32 lhs, const uint4_32 rhs, const uint_32 p) {
    return (uint4_32)(addmod2(lhs.s01, rhs.s01, p), addmod2(lhs.s23, rhs.s23, p));
}

INLINE uint4_32 submod4(const uint4_32 lhs, const uint4_32 rhs, const uint_32 p) {
    return (uint4_32)(submod2(lhs.s01, rhs.s01, p), submod2(lhs.s23, rhs.s23, p));
}

INLINE uint4_32 mulmod4(const uint4_32 lhs, const uint4_32 rhs, const uint2_32 pq) {
    return (uint4_32)(mulmod2(lhs.s01, rhs.s01, pq), mulmod2(lhs.s23, rhs.s23, pq));
}

INLINE uint_32 fold2(uint_32 a, uint_32 p) { return a - ((a >= 2 * p) ? 2 * p : 0); }
INLINE uint_32 iadd(uint_32 a, uint_32 b, uint_32 p) {
    uint_32 t = a + b;
    return t - ((t >= 2 * p) ? 2 * p : 0);
}
INLINE uint_32 ladd(uint_32 a, uint_32 b, uint_32 p) { return iadd(fold2(a, p), fold2(b, p), p); }
INLINE uint_32 lsub(uint_32 a, uint_32 b, uint_32 p) {
    a = fold2(a, p);
    b = fold2(b, p);
    uint_32 t = a + 2 * p - b;
    return t - ((t >= 2 * p) ? 2 * p : 0);
}
INLINE uint_32 lscalar(uint_32 a, uint_32 b, uint2_32 pq) {
    ulong t = a * (ulong)b;
    uint_32 lo = (uint_32)t, hi = (uint_32)(t >> 32), mp = mul_hi(lo * pq.s1, pq.s0);
    return hi + pq.s0 - mp;
}
INLINE uint_32 lmul(uint_32 a, uint_32 b, uint2_32 pq) {
    return lscalar(fold2(a, pq.s0), fold2(b, pq.s0), pq);
}
INLINE uint_32 lsqr(uint_32 a, uint2_32 pq) {
    a = fold2(a, pq.s0);
    return lscalar(a, a, pq);
}
INLINE uint2_32 fold22(uint2_32 a, uint_32 p) { return (uint2_32)(fold2(a.s0, p), fold2(a.s1, p)); }
INLINE uint2_32 iadd2(uint2_32 a, uint2_32 b, uint_32 p) {
    return (uint2_32)(iadd(a.s0, b.s0, p), iadd(a.s1, b.s1, p));
}
INLINE uint2_32 ladd2(uint2_32 a, uint2_32 b, uint_32 p) {
    return (uint2_32)(ladd(a.s0, b.s0, p), ladd(a.s1, b.s1, p));
}
INLINE uint2_32 lsub2(uint2_32 a, uint2_32 b, uint_32 p) {
    return (uint2_32)(lsub(a.s0, b.s0, p), lsub(a.s1, b.s1, p));
}
INLINE uint2_32 lmul2(uint2_32 a, uint2_32 b, uint2_32 pq) {
    return (uint2_32)(lmul(a.s0, b.s0, pq), lmul(a.s1, b.s1, pq));
}
INLINE uint2_32 lscalar2(uint2_32 a, uint_32 b, uint2_32 pq) {
    return (uint2_32)(lscalar(a.s0, b, pq), lscalar(a.s1, b, pq));
}
INLINE uint4_32 fold24(uint4_32 a, uint_32 p) {
    return (uint4_32)(fold2(a.s0, p), fold2(a.s1, p), fold2(a.s2, p), fold2(a.s3, p));
}
INLINE uint4_32 iadd4(uint4_32 a, uint4_32 b, uint_32 p) {
    return (uint4_32)(iadd(a.s0, b.s0, p), iadd(a.s1, b.s1, p), iadd(a.s2, b.s2, p),
                      iadd(a.s3, b.s3, p));
}
INLINE uint4_32 ladd4(uint4_32 a, uint4_32 b, uint_32 p) {
    return (uint4_32)(ladd(a.s0, b.s0, p), ladd(a.s1, b.s1, p), ladd(a.s2, b.s2, p),
                      ladd(a.s3, b.s3, p));
}
INLINE uint4_32 lsub4(uint4_32 a, uint4_32 b, uint_32 p) {
    return (uint4_32)(lsub(a.s0, b.s0, p), lsub(a.s1, b.s1, p), lsub(a.s2, b.s2, p),
                      lsub(a.s3, b.s3, p));
}
INLINE uint4_32 lmul4(uint4_32 a, uint4_32 b, uint2_32 pq) {
    return (uint4_32)(lmul(a.s0, b.s0, pq), lmul(a.s1, b.s1, pq), lmul(a.s2, b.s2, pq),
                      lmul(a.s3, b.s3, pq));
}
INLINE uint4_32 lscalar4(uint4_32 a, uint_32 b, uint2_32 pq) {
    return (uint4_32)(lscalar(a.s0, b, pq), lscalar(a.s1, b, pq), lscalar(a.s2, b, pq),
                      lscalar(a.s3, b, pq));
}

INLINE uint_32 shoup_root(uint_32 a, ulong root, uint2_32 pq) {
#if PG_ROOT_BITS == 64
    uint_32 w = (uint_32)root, q = mul_hi(a, (uint_32)(root >> 32));
#else
    uint_32 rq = (uint_32)root, w = mul_hi(rq, pq.s0) + (rq != 0), q = mul_hi(a, rq);
#endif
    uint_32 r = a * w - q * pq.s0;
    return r;
}
INLINE uint2_32 shoup_root2(uint2_32 a, ulong root, uint2_32 pq) {
    return (uint2_32)(shoup_root(a.s0, root, pq), shoup_root(a.s1, root, pq));
}
INLINE uint4_32 shoup_root4(uint4_32 a, ulong root, uint2_32 pq) {
    return (uint4_32)(shoup_root2(a.s01, root, pq), shoup_root2(a.s23, root, pq));
}
#define ROOTMUL(a, w, pq)                                                                          \
    ((sizeof(w) == 4) ? lscalar(a, (uint_32)(w), pq) : shoup_root(a, (ulong)(w), pq))
#define ROOTMUL2(a, w, pq)                                                                         \
    ((sizeof(w) == 4) ? lscalar2(a, (uint_32)(w), pq) : shoup_root2(a, (ulong)(w), pq))
#define ROOTMUL4(a, w, pq)                                                                         \
    ((sizeof(w) == 4) ? lscalar4(a, (uint_32)(w), pq) : shoup_root4(a, (ulong)(w), pq))

// --- transform/macro ---

#define FWD2(z0, z1, w)                                                                            \
    {                                                                                              \
        const uint_32 t = ROOTMUL(z1, w, pq);                                                      \
        z0 = fold2(z0, pq.s0);                                                                     \
        z1 = z0 + 2 * pq.s0 - t;                                                                   \
        z0 = z0 + t;                                                                               \
    }

#define BCK2(z0, z1, win)                                                                          \
    {                                                                                              \
        const uint_32 t = z1 + 2 * pq.s0 - z0;                                                     \
        z0 = iadd(z0, z1, pq.s0);                                                                  \
        z1 = ROOTMUL(t, win, pq);                                                                  \
    }

#define SQR2(z0, z1, w)                                                                            \
    {                                                                                              \
        const uint_32 t = ROOTMUL(lsqr(z1, pq), w, pq);                                            \
        z1 = lmul(ladd(z0, z0, pq.s0), z1, pq);                                                    \
        z0 = ladd(lsqr(z0, pq), t, pq.s0);                                                         \
    }

#define SQR2N(z0, z1, w)                                                                           \
    {                                                                                              \
        const uint_32 t = ROOTMUL(lsqr(z1, pq), w, pq);                                            \
        z1 = lmul(ladd(z0, z0, pq.s0), z1, pq);                                                    \
        z0 = lsub(lsqr(z0, pq), t, pq.s0);                                                         \
    }

#define MUL2(z0, z1, zp0, zp1, w)                                                                  \
    {                                                                                              \
        const uint_32 t = ROOTMUL(lmul(z1, zp1, pq), w, pq);                                       \
        z1 = ladd(lmul(z0, zp1, pq), lmul(zp0, z1, pq), pq.s0);                                    \
        z0 = ladd(lmul(z0, zp0, pq), t, pq.s0);                                                    \
    }

#define MUL2N(z0, z1, zp0, zp1, w)                                                                 \
    {                                                                                              \
        const uint_32 t = ROOTMUL(lmul(z1, zp1, pq), w, pq);                                       \
        z1 = ladd(lmul(z0, zp1, pq), lmul(zp0, z1, pq), pq.s0);                                    \
        z0 = lsub(lmul(z0, zp0, pq), t, pq.s0);                                                    \
    }

#define FWD2v2(z0, z1, w)                                                                          \
    {                                                                                              \
        const uint2_32 t = ROOTMUL2(z1, w, pq);                                                    \
        z0 = fold22(z0, pq.s0);                                                                    \
        z1 = z0 + 2 * pq.s0 - t;                                                                   \
        z0 = z0 + t;                                                                               \
    }

#define BCK2v2(z0, z1, win)                                                                        \
    {                                                                                              \
        const uint2_32 t = z1 + 2 * pq.s0 - z0;                                                    \
        z0 = iadd2(z0, z1, pq.s0);                                                                 \
        z1 = ROOTMUL2(t, win, pq);                                                                 \
    }

#define FWD2v4(z0, z1, w)                                                                          \
    {                                                                                              \
        const uint4_32 t = ROOTMUL4(z1, w, pq);                                                    \
        z0 = fold24(z0, pq.s0);                                                                    \
        z1 = z0 + 2 * pq.s0 - t;                                                                   \
        z0 = z0 + t;                                                                               \
    }

#define BCK2v4(z0, z1, win)                                                                        \
    {                                                                                              \
        const uint4_32 t = z1 + 2 * pq.s0 - z0;                                                    \
        z0 = iadd4(z0, z1, pq.s0);                                                                 \
        z1 = ROOTMUL4(t, win, pq);                                                                 \
    }

INLINE void _loadg1(const sz_t n, uint_32 *const zl, __global const uint_32 *restrict const z,
                    const sz_t s) {
    for (sz_t l = 0; l < n; ++l)
        zl[l] = z[l * s];
}
INLINE void _loadl1(const sz_t n, uint_32 *const zl, __local const uint_32 *restrict const Z,
                    const sz_t s) {
    for (sz_t l = 0; l < n; ++l)
        zl[l] = Z[l * s];
}
INLINE void _storeg1(const sz_t n, __global uint_32 *restrict const z, const sz_t s,
                     const uint_32 *const zl) {
    for (sz_t l = 0; l < n; ++l)
        z[l * s] = zl[l];
}
INLINE void _storel1(const sz_t n, __local uint_32 *restrict const Z, const sz_t s,
                     const uint_32 *const zl) {
    for (sz_t l = 0; l < n; ++l)
        Z[l * s] = zl[l];
}

INLINE void _loadg2(const sz_t n, uint2_32 *const zl, __global const uint2_32 *restrict const z,
                    const sz_t s) {
    for (sz_t l = 0; l < n; ++l)
        zl[l] = z[l * s];
}
INLINE void _loadl2(const sz_t n, uint2_32 *const zl, __local const uint2_32 *restrict const Z,
                    const sz_t s) {
    for (sz_t l = 0; l < n; ++l)
        zl[l] = Z[l * s];
}
INLINE void _storeg2(const sz_t n, __global uint2_32 *restrict const z, const sz_t s,
                     const uint2_32 *const zl) {
    for (sz_t l = 0; l < n; ++l)
        z[l * s] = zl[l];
}
INLINE void _storel2(const sz_t n, __local uint2_32 *restrict const Z, const sz_t s,
                     const uint2_32 *const zl) {
    for (sz_t l = 0; l < n; ++l)
        Z[l * s] = zl[l];
}

INLINE void _loadg4(const sz_t n, uint4_32 *const zl, __global const uint4_32 *restrict const z,
                    const sz_t s) {
    for (sz_t l = 0; l < n; ++l)
        zl[l] = z[l * s];
}
INLINE void _loadl4(const sz_t n, uint4_32 *const zl, __local const uint4_32 *restrict const Z,
                    const sz_t s) {
    for (sz_t l = 0; l < n; ++l)
        zl[l] = Z[l * s];
}
INLINE void _storeg4(const sz_t n, __global uint4_32 *restrict const z, const sz_t s,
                     const uint4_32 *const zl) {
    for (sz_t l = 0; l < n; ++l)
        z[l * s] = zl[l];
}
INLINE void _storel4(const sz_t n, __local uint4_32 *restrict const Z, const sz_t s,
                     const uint4_32 *const zl) {
    for (sz_t l = 0; l < n; ++l)
        Z[l * s] = zl[l];
}

// ---

INLINE void _forward4x1(const uint2_32 pq, uint_32 z[4], const ulong w1, const ulong w2[2]) {
    FWD2(z[0], z[2], w1);
    FWD2(z[1], z[3], w1);
    FWD2(z[0], z[1], w2[0]);
    FWD2(z[2], z[3], w2[1]);
}

INLINE void _backward4x1(const uint2_32 pq, uint_32 z[4], const ulong win1, const ulong win2[2]) {
    BCK2(z[0], z[1], win2[0]);
    BCK2(z[2], z[3], win2[1]);
    BCK2(z[0], z[2], win1);
    BCK2(z[1], z[3], win1);
}

INLINE void _forward4x1_0(const uint2_32 pq, const uint4_32 f0, uint_32 z[4]) {
    const uint_32 rsq = f0.s0, mfim = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
    z[0] = lmul(z[0], rsq, pq);
    z[1] = lmul(z[1], rsq, pq);
    FWD2(z[0], z[2], mfim);
    FWD2(z[1], z[3], mfim);
    FWD2(z[0], z[1], sqrti);
    FWD2(z[2], z[3], isqrti);
}

INLINE void _backward4x1_0(const uint2_32 pq, const uint4_32 b0, uint_32 z[4]) {
    const uint_32 isqrti = b0.s0, sqrti = b0.s1, im = b0.s2;
    BCK2(z[0], z[1], isqrti);
    BCK2(z[2], z[3], sqrti);
    BCK2(z[0], z[2], im);
    BCK2(z[1], z[3], im);
}

INLINE void _square2x2(const uint2_32 pq, uint_32 z[4], const ulong w) {
    SQR2(z[0], z[1], w);
    SQR2N(z[2], z[3], w);
}

INLINE void _square4(const uint2_32 pq, uint_32 z[4], const ulong w, const ulong win) {
    FWD2(z[0], z[2], w);
    FWD2(z[1], z[3], w);
    _square2x2(pq, z, w);
    BCK2(z[0], z[2], win);
    BCK2(z[1], z[3], win);
}

INLINE void _fwd4(const uint2_32 pq, uint_32 z[4], const ulong w) {
    FWD2(z[0], z[2], w);
    FWD2(z[1], z[3], w);
}

INLINE void _mul2x2(const uint2_32 pq, uint_32 z[4], const uint_32 zp[4], const ulong w) {
    MUL2(z[0], z[1], zp[0], zp[1], w);
    MUL2N(z[2], z[3], zp[2], zp[3], w);
}

INLINE void _mul4(const uint2_32 pq, uint_32 z[4], const uint_32 zp[4], const ulong w,
                  const ulong win) {
    _fwd4(pq, z, w);
    _mul2x2(pq, z, zp, w);
    BCK2(z[0], z[2], win);
    BCK2(z[1], z[3], win);
}

// --- v2

INLINE void _forward4x2(const uint2_32 pq, uint2_32 z[4], const ulong w1, const ulong w2[2]) {
    FWD2v2(z[0], z[2], w1);
    FWD2v2(z[1], z[3], w1);
    FWD2v2(z[0], z[1], w2[0]);
    FWD2v2(z[2], z[3], w2[1]);
}

INLINE void _backward4x2(const uint2_32 pq, uint2_32 z[4], const ulong win1, const ulong win2[2]) {
    BCK2v2(z[0], z[1], win2[0]);
    BCK2v2(z[2], z[3], win2[1]);
    BCK2v2(z[0], z[2], win1);
    BCK2v2(z[1], z[3], win1);
}

INLINE void _forward4x2_0(const uint2_32 pq, const uint4_32 f0, uint2_32 z[4]) {
    const uint_32 rsq = f0.s0, mfim = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
    z[0] = lmul2(z[0], rsq, pq);
    z[1] = lmul2(z[1], rsq, pq);
    FWD2v2(z[0], z[2], mfim);
    FWD2v2(z[1], z[3], mfim);
    FWD2v2(z[0], z[1], sqrti);
    FWD2v2(z[2], z[3], isqrti);
}

INLINE void _backward4x2_0(const uint2_32 pq, const uint4_32 b0, uint2_32 z[4]) {
    const uint_32 isqrti = b0.s0, sqrti = b0.s1, im = b0.s2;
    BCK2v2(z[0], z[1], isqrti);
    BCK2v2(z[2], z[3], sqrti);
    BCK2v2(z[0], z[2], im);
    BCK2v2(z[1], z[3], im);
}

INLINE void _square4x2(const uint2_32 pq, uint2_32 z[4], const ulong w2[2], const ulong win2[2]) {
    FWD2v2(z[0], z[1], w2[0]);
    FWD2v2(z[2], z[3], w2[1]);
    SQR2(z[0].s0, z[0].s1, w2[0]);
    SQR2N(z[1].s0, z[1].s1, w2[0]);
    SQR2(z[2].s0, z[2].s1, w2[1]);
    SQR2N(z[3].s0, z[3].s1, w2[1]);
    BCK2v2(z[0], z[1], win2[0]);
    BCK2v2(z[2], z[3], win2[1]);
}

INLINE void _square8(const uint2_32 pq, uint2_32 z[4], const ulong w1, const ulong win1,
                     const ulong w2[2], const ulong win2[2]) {
    FWD2v2(z[0], z[2], w1);
    FWD2v2(z[1], z[3], w1);
    _square4x2(pq, z, w2, win2);
    BCK2v2(z[0], z[2], win1);
    BCK2v2(z[1], z[3], win1);
}

INLINE void _fwd4x2(const uint2_32 pq, uint2_32 z[4], const ulong w2[2]) {
    FWD2v2(z[0], z[1], w2[0]);
    FWD2v2(z[2], z[3], w2[1]);
}

INLINE void _fwd8(const uint2_32 pq, uint2_32 z[4], const ulong w1, const ulong w2[2]) {
    FWD2v2(z[0], z[2], w1);
    FWD2v2(z[1], z[3], w1);
    _fwd4x2(pq, z, w2);
}

INLINE void _mul4x2(const uint2_32 pq, uint2_32 z[4], const uint2_32 zp[4], const ulong w2[2],
                    const ulong win2[2]) {
    FWD2v2(z[0], z[1], w2[0]);
    FWD2v2(z[2], z[3], w2[1]);
    MUL2(z[0].s0, z[0].s1, zp[0].s0, zp[0].s1, w2[0]);
    MUL2N(z[1].s0, z[1].s1, zp[1].s0, zp[1].s1, w2[0]);
    MUL2(z[2].s0, z[2].s1, zp[2].s0, zp[2].s1, w2[1]);
    MUL2N(z[3].s0, z[3].s1, zp[3].s0, zp[3].s1, w2[1]);
    BCK2v2(z[0], z[1], win2[0]);
    BCK2v2(z[2], z[3], win2[1]);
}

INLINE void _mul8(const uint2_32 pq, uint2_32 z[4], const uint2_32 zp[4], const ulong w1,
                  const ulong win1, const ulong w2[2], const ulong win2[2]) {
    FWD2v2(z[0], z[2], w1);
    FWD2v2(z[1], z[3], w1);
    _mul4x2(pq, z, zp, w2, win2);
    BCK2v2(z[0], z[2], win1);
    BCK2v2(z[1], z[3], win1);
}

// --- v4

INLINE void _forward4x4(const uint2_32 pq, uint4_32 z[4], const ulong w1, const ulong w2[2]) {
    FWD2v4(z[0], z[2], w1);
    FWD2v4(z[1], z[3], w1);
    FWD2v4(z[0], z[1], w2[0]);
    FWD2v4(z[2], z[3], w2[1]);
}

INLINE void _backward4x4(const uint2_32 pq, uint4_32 z[4], const ulong win1, const ulong win2[2]) {
    BCK2v4(z[0], z[1], win2[0]);
    BCK2v4(z[2], z[3], win2[1]);
    BCK2v4(z[0], z[2], win1);
    BCK2v4(z[1], z[3], win1);
}

INLINE void _forward4x4_0(const uint2_32 pq, const uint4_32 f0, uint4_32 z[4]) {
    const uint_32 rsq = f0.s0, mfim = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
    z[0] = lmul4(z[0], rsq, pq);
    z[1] = lmul4(z[1], rsq, pq);
    FWD2v4(z[0], z[2], mfim);
    FWD2v4(z[1], z[3], mfim);
    FWD2v4(z[0], z[1], sqrti);
    FWD2v4(z[2], z[3], isqrti);
}

INLINE void _backward4x4_0(const uint2_32 pq, const uint4_32 b0, uint4_32 z[4]) {
    const uint_32 isqrti = b0.s0, sqrti = b0.s1, im = b0.s2;
    BCK2v4(z[0], z[1], isqrti);
    BCK2v4(z[2], z[3], sqrti);
    BCK2v4(z[0], z[2], im);
    BCK2v4(z[1], z[3], im);
}

INLINE void _square4x2v4(const uint2_32 pq, uint4_32 z[2], const ulong w2[2], const ulong win2[2]) {
    for (sz_t i = 0; i < 2; ++i) {
        FWD2v2(z[i].s01, z[i].s23, w2[i]);
        SQR2(z[i].s0, z[i].s1, w2[i]);
        SQR2N(z[i].s2, z[i].s3, w2[i]);
        BCK2v2(z[i].s01, z[i].s23, win2[i]);
    }
}

INLINE void _square8v4(const uint2_32 pq, uint4_32 z[2], const ulong w1, const ulong win1,
                       const ulong w2[2], const ulong win2[2]) {
    FWD2v4(z[0], z[1], w1);
    _square4x2v4(pq, z, w2, win2);
    BCK2v4(z[0], z[1], win1);
}

INLINE void _fwd4x2v4(const uint2_32 pq, uint4_32 z[2], const ulong w2[2]) {
    for (sz_t i = 0; i < 2; ++i)
        FWD2v2(z[i].s01, z[i].s23, w2[i]);
}

INLINE void _fwd8v4(const uint2_32 pq, uint4_32 z[2], const ulong w1, const ulong w2[2]) {
    FWD2v4(z[0], z[1], w1);
    _fwd4x2v4(pq, z, w2);
}

INLINE void _mul4x2v4(const uint2_32 pq, uint4_32 z[2], const uint4_32 zp[2], const ulong w2[2],
                      const ulong win2[2]) {
    for (sz_t i = 0; i < 2; ++i) {
        FWD2v2(z[i].s01, z[i].s23, w2[i]);
        MUL2(z[i].s0, z[i].s1, zp[i].s0, zp[i].s1, w2[i]);
        MUL2N(z[i].s2, z[i].s3, zp[i].s2, zp[i].s3, w2[i]);
        BCK2v2(z[i].s01, z[i].s23, win2[i]);
    }
}

INLINE void _mul8v4(const uint2_32 pq, uint4_32 z[2], const uint4_32 zp[2], const ulong w1,
                    const ulong win1, const ulong w2[2], const ulong win2[2]) {
    FWD2v4(z[0], z[1], w1);
    _mul4x2v4(pq, z, zp, w2, win2);
    BCK2v4(z[0], z[1], win1);
}

// --- inverse of roots is wi[s + j] or w[s + s - j - 1] ---

#define DECLARE_W1(sj) const ulong w1 = w[sj];
#define DECLARE_W2(sj)                                                                             \
    ulong w2[2];                                                                                   \
    {                                                                                              \
        const root2_t t = ((__global const root2_t *)w)[sj];                                       \
        w2[0] = t.s0;                                                                              \
        w2[1] = t.s1;                                                                              \
    }
#define DECLARE_W12(sj)                                                                            \
    DECLARE_W1(sj);                                                                                \
    DECLARE_W2(sj);
#define DECLARE_W1_2(sj)                                                                           \
    ulong w1[2];                                                                                   \
    {                                                                                              \
        const root2_t t = ((__global const root2_t *)w)[sj];                                       \
        w1[0] = t.s0;                                                                              \
        w1[1] = t.s1;                                                                              \
    }
#define DECLARE_W2_4(sj)                                                                           \
    ulong w2[4];                                                                                   \
    {                                                                                              \
        const root4_t t = ((__global const root4_t *)w)[sj];                                       \
        w2[0] = t.s0;                                                                              \
        w2[1] = t.s1;                                                                              \
        w2[2] = t.s2;                                                                              \
        w2[3] = t.s3;                                                                              \
    }
#define DECLARE_W12_24(sj)                                                                         \
    DECLARE_W1_2(sj);                                                                              \
    DECLARE_W2_4(sj);

#define DECLARE_WIN1(sji) const ulong win1 = wi[sji];
#if defined(USE_WI)
#define DECLARE_IVAR(s, j)                                                                         \
    const sz_t sji = s + j;                                                                        \
    __global const root_word_t *restrict const wi = &w[WI_SHFT];
#define DECLARE_WIN2(sji)                                                                          \
    ulong win2[2];                                                                                 \
    {                                                                                              \
        const root2_t t = ((__global const root2_t *)wi)[sji];                                     \
        win2[0] = t.s0;                                                                            \
        win2[1] = t.s1;                                                                            \
    }
#define DECLARE_WIN1_2(sji)                                                                        \
    ulong win1[2];                                                                                 \
    {                                                                                              \
        const root2_t t = ((__global const root2_t *)wi)[sji];                                     \
        win1[0] = t.s0;                                                                            \
        win1[1] = t.s1;                                                                            \
    }
#define DECLARE_WIN2_4(sji)                                                                        \
    ulong win2[4];                                                                                 \
    {                                                                                              \
        const root4_t t = ((__global const root4_t *)wi)[sji];                                     \
        win2[0] = t.s0;                                                                            \
        win2[1] = t.s1;                                                                            \
        win2[2] = t.s2;                                                                            \
        win2[3] = t.s3;                                                                            \
    }
#else
#define DECLARE_IVAR(s, j)                                                                         \
    const sz_t sji = s + s - j - 1;                                                                \
    __global const root_word_t *restrict const wi = w;
#define DECLARE_WIN2(sji)                                                                          \
    ulong win2[2];                                                                                 \
    {                                                                                              \
        const root2_t t = ((__global const root2_t *)wi)[sji];                                     \
        win2[0] = t.s1;                                                                            \
        win2[1] = t.s0;                                                                            \
    }
#define DECLARE_WIN1_2(sji)                                                                        \
    ulong win1[2];                                                                                 \
    {                                                                                              \
        const root2_t t = ((__global const root2_t *)wi)[sji];                                     \
        win1[0] = t.s1;                                                                            \
        win1[1] = t.s0;                                                                            \
    }
#define DECLARE_WIN2_4(sji)                                                                        \
    ulong win2[4];                                                                                 \
    {                                                                                              \
        const root4_t t = ((__global const root4_t *)wi)[sji];                                     \
        win2[0] = t.s3;                                                                            \
        win2[1] = t.s2;                                                                            \
        win2[2] = t.s1;                                                                            \
        win2[3] = t.s0;                                                                            \
    }
#endif
#define DECLARE_WIN12(sj)                                                                          \
    DECLARE_WIN1(sj);                                                                              \
    DECLARE_WIN2(sj);
#define DECLARE_WIN12_24(sj)                                                                       \
    DECLARE_WIN1_2(sj);                                                                            \
    DECLARE_WIN2_4(sj);

// --- vector size (1, 2 or 4) ---

#if VSIZE == 4
#define VTYPE uint4_32
#define _loadg _loadg4
#define _loadl _loadl4
#define _storeg _storeg4
#define _storel _storel4
#define _forward4 _forward4x4
#define _backward4 _backward4x4
#define _forward4_0 _forward4x4_0
#define _backward4_0 _backward4x4_0
#elif VSIZE == 2
#define VTYPE uint2_32
#define _loadg _loadg2
#define _loadl _loadl2
#define _storeg _storeg2
#define _storel _storel2
#define _forward4 _forward4x2
#define _backward4 _backward4x2
#define _forward4_0 _forward4x2_0
#define _backward4_0 _backward4x2_0
#else
#define VTYPE uint_32
#define _loadg _loadg1
#define _loadl _loadl1
#define _storeg _storeg1
#define _storel _storel1
#define _forward4 _forward4x1
#define _backward4 _backward4x1
#define _forward4_0 _forward4x1_0
#define _backward4_0 _backward4x1_0
#endif

// --- transform/inline global mem ---

#if defined(ALL_FUNC)

INLINE void forward4io(const uint2_32 pq, const sz_t m, __global VTYPE *restrict const z,
                       __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12(sj);
    VTYPE zl[4];
    _loadg(4, zl, z, m);
    _forward4(pq, zl, w1, w2);
    _storeg(4, z, m, zl);
}

INLINE void backward4io(const uint2_32 pq, const sz_t m, __global VTYPE *restrict const z,
                        __global const root_word_t *restrict const wi, const sz_t sji) {
    DECLARE_WIN12(sji);
    VTYPE zl[4];
    _loadg(4, zl, z, m);
    _backward4(pq, zl, win1, win2);
    _storeg(4, z, m, zl);
}

INLINE void forward4io_0(const uint2_32 pq, const uint4_32 f0, __global VTYPE *restrict const z) {
    const sz_t m = N_SZ / 4 / VSIZE;
    VTYPE zl[4];
    _loadg(4, zl, z, m);
    _forward4_0(pq, f0, zl);
    _storeg(4, z, m, zl);
}

INLINE void backwardio_0(const uint2_32 pq, const uint4_32 b0, __global VTYPE *restrict const z) {
    const sz_t m = N_SZ / 4 / VSIZE;
    VTYPE zl[4];
    _loadg(4, zl, z, m);
    _backward4_0(pq, b0, zl);
    _storeg(4, z, m, zl);
}

// --- v1

INLINE void square2x2io(const uint2_32 pq, __global uint_32 *restrict const z, const ulong w) {
    uint_32 zl[4];
    _loadg1(4, zl, z, 1);
    _square2x2(pq, zl, w);
    _storeg1(4, z, 1, zl);
}

INLINE void square4x1io(const uint2_32 pq, __global uint_32 *restrict const z, const ulong w,
                        const ulong win) {
    uint_32 zl[4];
    _loadg1(4, zl, z, 1);
    _square4(pq, zl, w, win);
    _storeg1(4, z, 1, zl);
}

INLINE void fwd4x1io(const uint2_32 pq, __global uint_32 *restrict const z, const ulong w) {
    uint_32 zl[4];
    _loadg1(4, zl, z, 1);
    _fwd4(pq, zl, w);
    _storeg1(4, z, 1, zl);
}

INLINE void mul2x2io(const uint2_32 pq, __global uint_32 *restrict const z,
                     const __global uint_32 *restrict const zp, const ulong w) {
    uint_32 zpl[4];
    _loadg1(4, zpl, zp, 1);
    uint_32 zl[4];
    _loadg1(4, zl, z, 1);
    _mul2x2(pq, zl, zpl, w);
    _storeg1(4, z, 1, zl);
}

INLINE void mul4x1io(const uint2_32 pq, __global uint_32 *restrict const z,
                     const __global uint_32 *restrict const zp, const ulong w, const ulong win) {
    uint_32 zpl[4];
    _loadg1(4, zpl, zp, 1);
    uint_32 zl[4];
    _loadg1(4, zl, z, 1);
    _mul4(pq, zl, zpl, w, win);
    _storeg1(4, z, 1, zl);
}

// --- v2

INLINE void square4x2io(const uint2_32 pq, __global uint2_32 *restrict const z,
                        __global const root_word_t *restrict const w,
                        __global const root_word_t *restrict const wi, const sz_t sj,
                        const sz_t sji) {
    DECLARE_W2(sj);
    DECLARE_WIN2(sji);
    uint2_32 zl[4];
    _loadg2(4, zl, z, 1);
    _square4x2(pq, zl, w2, win2);
    _storeg2(4, z, 1, zl);
}

INLINE void square8x1io(const uint2_32 pq, __global uint2_32 *restrict const z,
                        __global const root_word_t *restrict const w,
                        __global const root_word_t *restrict const wi, const sz_t sj,
                        const sz_t sji) {
    DECLARE_W12(sj);
    DECLARE_WIN12(sji);
    uint2_32 zl[4];
    _loadg2(4, zl, z, 1);
    _square8(pq, zl, w1, win1, w2, win2);
    _storeg2(4, z, 1, zl);
}

INLINE void fwd4x2io(const uint2_32 pq, __global uint2_32 *restrict const z,
                     __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W2(sj);
    uint2_32 zl[4];
    _loadg2(4, zl, z, 1);
    _fwd4x2(pq, zl, w2);
    _storeg2(4, z, 1, zl);
}

INLINE void fwd8x1io(const uint2_32 pq, __global uint2_32 *restrict const z,
                     __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12(sj);
    uint2_32 zl[4];
    _loadg2(4, zl, z, 1);
    _fwd8(pq, zl, w1, w2);
    _storeg2(4, z, 1, zl);
}

INLINE void mul4x2io(const uint2_32 pq, __global uint2_32 *restrict const z,
                     const __global uint2_32 *restrict const zp,
                     __global const root_word_t *restrict const w,
                     __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
    DECLARE_W2(sj);
    DECLARE_WIN2(sji);
    uint2_32 zpl[4];
    _loadg2(4, zpl, zp, 1);
    uint2_32 zl[4];
    _loadg2(4, zl, z, 1);
    _mul4x2(pq, zl, zpl, w2, win2);
    _storeg2(4, z, 1, zl);
}

INLINE void mul8x1io(const uint2_32 pq, __global uint2_32 *restrict const z,
                     const __global uint2_32 *restrict const zp,
                     __global const root_word_t *restrict const w,
                     __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
    DECLARE_W12(sj);
    DECLARE_WIN12(sji);
    uint2_32 zpl[4];
    _loadg2(4, zpl, zp, 1);
    uint2_32 zl[4];
    _loadg2(4, zl, z, 1);
    _mul8(pq, zl, zpl, w1, win1, w2, win2);
    _storeg2(4, z, 1, zl);
}

// --- v4

INLINE void square4x4io(const uint2_32 pq, __global uint4_32 *restrict const z,
                        __global const root_word_t *restrict const w,
                        __global const root_word_t *restrict const wi, const sz_t sj,
                        const sz_t sji) {
    DECLARE_W2_4(sj);
    DECLARE_WIN2_4(sji);
    uint4_32 zl[4];
    _loadg4(4, zl, z, 1);
    _square4x2v4(pq, &zl[0], &w2[0], &win2[0]);
    _square4x2v4(pq, &zl[2], &w2[2], &win2[2]);
    _storeg4(4, z, 1, zl);
}

INLINE void square8x2io(const uint2_32 pq, __global uint4_32 *restrict const z,
                        __global const root_word_t *restrict const w,
                        __global const root_word_t *restrict const wi, const sz_t sj,
                        const sz_t sji) {
    DECLARE_W12_24(sj);
    DECLARE_WIN12_24(sji);
    uint4_32 zl[4];
    _loadg4(4, zl, z, 1);
    _square8v4(pq, &zl[0], w1[0], win1[0], &w2[0], &win2[0]);
    _square8v4(pq, &zl[2], w1[1], win1[1], &w2[2], &win2[2]);
    _storeg4(4, z, 1, zl);
}

INLINE void fwd4x4io(const uint2_32 pq, __global uint4_32 *restrict const z,
                     __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W2_4(sj);
    uint4_32 zl[4];
    _loadg4(4, zl, z, 1);
    _fwd4x2v4(pq, &zl[0], &w2[0]);
    _fwd4x2v4(pq, &zl[2], &w2[2]);
    _storeg4(4, z, 1, zl);
}

INLINE void fwd8x2io(const uint2_32 pq, __global uint4_32 *restrict const z,
                     __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12_24(sj);
    uint4_32 zl[4];
    _loadg4(4, zl, z, 1);
    _fwd8v4(pq, &zl[0], w1[0], &w2[0]);
    _fwd8v4(pq, &zl[2], w1[1], &w2[2]);
    _storeg4(4, z, 1, zl);
}

INLINE void mul4x4io(const uint2_32 pq, __global uint4_32 *restrict const z,
                     const __global uint4_32 *restrict const zp,
                     __global const root_word_t *restrict const w,
                     __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
    DECLARE_W2_4(sj);
    DECLARE_WIN2_4(sji);
    uint4_32 zpl[4];
    _loadg4(4, zpl, zp, 1);
    uint4_32 zl[4];
    _loadg4(4, zl, z, 1);
    _mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);
    _mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);
    _storeg4(4, z, 1, zl);
}

INLINE void mul8x2io(const uint2_32 pq, __global uint4_32 *restrict const z,
                     const __global uint4_32 *restrict const zp,
                     __global const root_word_t *restrict const w,
                     __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
    DECLARE_W12_24(sj);
    DECLARE_WIN12_24(sji);
    uint4_32 zpl[4];
    _loadg4(4, zpl, zp, 1);
    uint4_32 zl[4];
    _loadg4(4, zl, z, 1);
    _mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);
    _mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);
    _storeg4(4, z, 1, zl);
}

// --- v1, v2, v4

INLINE void square4io(const uint2_32 pq, __global VTYPE *restrict const z,
                      __global const root_word_t *restrict const w,
                      __global const root_word_t *restrict const wi, const sz_t sj,
                      const sz_t sji) {
#if VSIZE == 4
    square4x4io(pq, z, w, wi, sj, sji);
#elif VSIZE == 2
    square4x2io(pq, z, w, wi, sj, sji);
#else
    square4x1io(pq, z, w[sj], wi[sji]);
#endif
}

INLINE void fwd4io(const uint2_32 pq, __global VTYPE *restrict const z,
                   __global const root_word_t *restrict const w, const sz_t sj) {
#if VSIZE == 4
    fwd4x4io(pq, z, w, sj);
#elif VSIZE == 2
    fwd4x2io(pq, z, w, sj);
#else
    fwd4x1io(pq, z, w[sj]);
#endif
}

INLINE void mul4io(const uint2_32 pq, __global VTYPE *restrict const z,
                   const __global VTYPE *restrict const zp,
                   __global const root_word_t *restrict const w,
                   __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
#if VSIZE == 4
    mul4x4io(pq, z, zp, w, wi, sj, sji);
#elif VSIZE == 2
    mul4x2io(pq, z, zp, w, wi, sj, sji);
#else
    mul4x1io(pq, z, zp, w[sj], wi[sji]);
#endif
}

// --- v2, v4

INLINE void square8io(const uint2_32 pq, __global VTYPE *restrict const z,
                      __global const root_word_t *restrict const w,
                      __global const root_word_t *restrict const wi, const sz_t sj,
                      const sz_t sji) {
#if VSIZE == 4
    square8x2io(pq, z, w, wi, sj, sji);
#elif VSIZE == 2
    square8x1io(pq, z, w, wi, sj, sji);
#endif
}

INLINE void fwd8io(const uint2_32 pq, __global VTYPE *restrict const z,
                   __global const root_word_t *restrict const w, const sz_t sj) {
#if VSIZE == 4
    fwd8x2io(pq, z, w, sj);
#elif VSIZE == 2
    fwd8x1io(pq, z, w, sj);
#endif
}

INLINE void mul8io(const uint2_32 pq, __global VTYPE *restrict const z,
                   const __global VTYPE *restrict const zp,
                   __global const root_word_t *restrict const w,
                   __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
#if VSIZE == 4
    mul8x2io(pq, z, zp, w, wi, sj, sji);
#elif VSIZE == 2
    mul8x1io(pq, z, zp, w, wi, sj, sji);
#endif
}

#endif // ALL_FUNC

// --- transform/inline local & global mem ---

INLINE void forward_4(const uint2_32 pq, const sz_t m, __local VTYPE *restrict const Z,
                      __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE zl[4];
    _loadl(4, zl, Z, m);
    _forward4(pq, zl, w1, w2);
    _storel(4, Z, m, zl);
}

INLINE void forward_4i(const uint2_32 pq, const sz_t ml, __local VTYPE *restrict const Z,
                       const sz_t mg, __global const VTYPE *restrict const z,
                       __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12(sj);
    VTYPE zl[4];
    _loadg(4, zl, z, mg);
    _forward4(pq, zl, w1, w2);
    _storel(4, Z, ml, zl);
}

/* @G_F0_ORDINARY@ */
INLINE void forward4_ordinary(const uint2_32 pq, const uint4_32 f0, uint4_32 z[4]) {
    FWD2v4(z[0], z[2], f0.s1);
    FWD2v4(z[1], z[3], f0.s1);
    FWD2v4(z[0], z[1], f0.s2);
    FWD2v4(z[2], z[3], f0.s3);
}

INLINE void forward_4i_0(const uint2_32 pq, const uint4_32 f0, const sz_t ml,
                         __local VTYPE *restrict const Z, const sz_t mg,
                         __global const VTYPE *restrict const z) {
    VTYPE zl[4];
    _loadg(4, zl, z, mg);
    _forward4_0(pq, f0, zl);
    _storel(4, Z, ml, zl);
}

INLINE void forward_4o(const uint2_32 pq, const sz_t mg, __global VTYPE *restrict const z,
                       const sz_t ml, __local const VTYPE *restrict const Z,
                       __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE zl[4];
    _loadl(4, zl, Z, ml);
    _forward4(pq, zl, w1, w2);
    _storeg(4, z, mg, zl);
}

INLINE void backward_4(const uint2_32 pq, const sz_t m, __local VTYPE *restrict const Z,
                       __global const root_word_t *restrict const wi, const sz_t sji) {
    DECLARE_WIN12(sji);
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE zl[4];
    _loadl(4, zl, Z, m);
    _backward4(pq, zl, win1, win2);
    _storel(4, Z, m, zl);
}

INLINE void backward_4i(const uint2_32 pq, const sz_t ml, __local VTYPE *restrict const Z,
                        const sz_t mg, __global const VTYPE *restrict const z,
                        __global const root_word_t *restrict const wi, const sz_t sji) {
    DECLARE_WIN12(sji);
    VTYPE zl[4];
    _loadg(4, zl, z, mg);
    _backward4(pq, zl, win1, win2);
    _storel(4, Z, ml, zl);
}

INLINE void backward_4o(const uint2_32 pq, const sz_t mg, __global VTYPE *restrict const z,
                        const sz_t ml, __local const VTYPE *restrict const Z,
                        __global const root_word_t *restrict const wi, const sz_t sji) {
    DECLARE_WIN12(sji);
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE zl[4];
    _loadl(4, zl, Z, ml);
    _backward4(pq, zl, win1, win2);
    _storeg(4, z, mg, zl);
}

INLINE void backward_4o_0(const uint2_32 pq, const uint4_32 b0, const sz_t mg,
                          __global VTYPE *restrict const z, const sz_t ml,
                          __local const VTYPE *restrict const Z) {
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE zl[4];
    _loadl(4, zl, Z, ml);
    _backward4_0(pq, b0, zl);
    _storeg(4, z, mg, zl);
}

// --- v1

INLINE void square_2x2(const uint2_32 pq, __local uint_32 *restrict const Z, const ulong w) {
    barrier(CLK_LOCAL_MEM_FENCE);
    uint_32 zl[4];
    _loadl1(4, zl, Z, 1);
    _square2x2(pq, zl, w);
    _storel1(4, Z, 1, zl);
}

INLINE void square_4x1(const uint2_32 pq, __local uint_32 *restrict const Z, const ulong w,
                       const ulong win) {
    barrier(CLK_LOCAL_MEM_FENCE);
    uint_32 zl[4];
    _loadl1(4, zl, Z, 1);
    _square4(pq, zl, w, win);
    _storel1(4, Z, 1, zl);
}

INLINE void write_4(const sz_t mg, __global VTYPE *restrict const z,
                    __local const VTYPE *restrict const Z) {
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE zl[4];
    _loadl(4, zl, Z, 1);
    _storeg(4, z, mg, zl);
}

INLINE void fwd4x1_write(const uint2_32 pq, const sz_t mg, __global uint_32 *restrict const z,
                         __local const uint_32 *restrict const Z, const ulong w) {
    barrier(CLK_LOCAL_MEM_FENCE);
    uint_32 zl[4];
    _loadl1(4, zl, Z, 1);
    _fwd4(pq, zl, w);
    _storeg1(4, z, mg, zl);
}

INLINE void mul_2x2(const uint2_32 pq, __local uint_32 *restrict const Z, const sz_t mg,
                    __global const uint_32 *restrict const zp, const ulong w) {
    uint_32 zpl[4];
    _loadg1(4, zpl, zp, mg);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint_32 zl[4];
    _loadl1(4, zl, Z, 1);
    _mul2x2(pq, zl, zpl, w);
    _storel1(4, Z, 1, zl);
}

INLINE void mul_4x1(const uint2_32 pq, __local uint_32 *restrict const Z, const sz_t mg,
                    __global const uint_32 *restrict const zp, const ulong w, const ulong win) {
    uint_32 zpl[4];
    _loadg1(4, zpl, zp, mg);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint_32 zl[4];
    _loadl1(4, zl, Z, 1);
    _mul4(pq, zl, zpl, w, win);
    _storel1(4, Z, 1, zl);
}

// --- v2

INLINE void square_4x2(const uint2_32 pq, __local uint2_32 *restrict const Z,
                       __global const root_word_t *restrict const w,
                       __global const root_word_t *restrict const wi, const sz_t sj,
                       const sz_t sji) {
    DECLARE_W2(sj);
    DECLARE_WIN2(sji);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint2_32 zl[4];
    _loadl2(4, zl, Z, 1);
    _square4x2(pq, zl, w2, win2);
    _storel2(4, Z, 1, zl);
}

INLINE void square_8x1(const uint2_32 pq, __local uint2_32 *restrict const Z,
                       __global const root_word_t *restrict const w,
                       __global const root_word_t *restrict const wi, const sz_t sj,
                       const sz_t sji) {
    DECLARE_W12(sj);
    DECLARE_WIN12(sji);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint2_32 zl[4];
    _loadl2(4, zl, Z, 1);
    _square8(pq, zl, w1, win1, w2, win2);
    _storel2(4, Z, 1, zl);
}

INLINE void fwd4x2_write(const uint2_32 pq, const sz_t mg, __global uint2_32 *restrict const z,
                         __local const uint2_32 *restrict const Z,
                         __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W2(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint2_32 zl[4];
    _loadl2(4, zl, Z, 1);
    _fwd4x2(pq, zl, w2);
    _storeg2(4, z, mg, zl);
}

INLINE void fwd8x1_write(const uint2_32 pq, const sz_t mg, __global uint2_32 *restrict const z,
                         __local const uint2_32 *restrict const Z,
                         __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint2_32 zl[4];
    _loadl2(4, zl, Z, 1);
    _fwd8(pq, zl, w1, w2);
    _storeg2(4, z, mg, zl);
}

INLINE void mul_4x2(const uint2_32 pq, __local uint2_32 *restrict const Z, const sz_t mg,
                    const __global uint2_32 *restrict const zp,
                    __global const root_word_t *restrict const w,
                    __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
    DECLARE_W2(sj);
    DECLARE_WIN2(sji);
    uint2_32 zpl[4];
    _loadg2(4, zpl, zp, mg);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint2_32 zl[4];
    _loadl2(4, zl, Z, 1);
    _mul4x2(pq, zl, zpl, w2, win2);
    _storel2(4, Z, 1, zl);
}

INLINE void mul_8x1(const uint2_32 pq, __local uint2_32 *restrict const Z, const sz_t mg,
                    const __global uint2_32 *restrict const zp,
                    __global const root_word_t *restrict const w,
                    __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
    DECLARE_W12(sj);
    DECLARE_WIN12(sji);
    uint2_32 zpl[4];
    _loadg2(4, zpl, zp, mg);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint2_32 zl[4];
    _loadl2(4, zl, Z, 1);
    _mul8(pq, zl, zpl, w1, win1, w2, win2);
    _storel2(4, Z, 1, zl);
}

// --- v4

INLINE uint4_32 quartic_dot(const uint2_32 pq, const uint4_32 input, const uint_32 gamma,
                            const uint_32 qgamma);
INLINE void square_4x4(const uint2_32 pq, __local uint4 *Z, __global const root_word_t *w,
                       __global const root_word_t *wi, const sz_t sj, const sz_t sji) {
    DECLARE_W1_2(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint4 a[4];
    _loadl4(4, a, Z, 1);
    for (int i = 0; i < 4; ++i) {
#if PG_ROOT_BITS == 64
        uint q = (uint)(w1[i / 2] >> 32), gamma = (uint)w1[i / 2];
#else
        uint q = (uint)w1[i / 2], gamma = mul_hi(q, pq.s0) + (q != 0);
#endif
        if (i & 1) {
            q = ~q;
            gamma = pq.s0 - gamma;
        }
        a[i] = quartic_dot(pq, a[i], gamma, q);
    }
    _storel4(4, Z, 1, a);
}

INLINE uint_32 quartic_redc(const ulong t, const uint2_32 pq) {
    const uint_32 h = (uint_32)(t >> 32), m = mul_hi((uint_32)t * pq.s1, pq.s0);
    return fold2(h + pq.s0 - m, pq.s0);
}
INLINE uint4_32 quartic_dot(const uint2_32 pq, const uint4_32 input, const uint_32 gamma,
                            const uint_32 qgamma) {
    const uint_32 p = pq.s0;
    uint4_32 a = fold24(input, p);
    a.s0 -= a.s0 >= p ? p : 0;
    a.s1 -= a.s1 >= p ? p : 0;
    a.s2 -= a.s2 >= p ? p : 0;
    a.s3 -= a.s3 >= p ? p : 0;
    uint_32 u2 = a.s2 * gamma - mul_hi(a.s2, qgamma) * p,
            u3 = a.s3 * gamma - mul_hi(a.s3, qgamma) * p;
    u2 -= u2 >= p ? p : 0;
    u3 -= u3 >= p ? p : 0;
    const ulong t0 = (ulong)(2 * a.s0) * a.s0 + (ulong)(4 * a.s1) * u3 + (ulong)(2 * a.s2) * u2;
    const ulong t1 = (ulong)(4 * a.s0) * a.s1 + (ulong)(4 * a.s2) * u3;
    const ulong t2 = (ulong)(4 * a.s0) * a.s2 + (ulong)(2 * a.s1) * a.s1 + (ulong)(2 * a.s3) * u3;
    const ulong t3 = (ulong)(4 * a.s0) * a.s3 + (ulong)(4 * a.s1) * a.s2;
    return (uint4_32)(quartic_redc(t0, pq), quartic_redc(t1, pq), quartic_redc(t2, pq),
                      quartic_redc(t3, pq));
}

INLINE void quartic_square8(const uint2_32 pq, uint4_32 z[2], const ulong w1, const ulong win1) {

#if PG_ROOT_BITS == 64
    const uint_32 q = (uint_32)(w1 >> 32), gamma = (uint_32)w1;
#else
    const uint_32 q = (uint_32)w1, gamma = mul_hi(q, pq.s0) + (q != 0);
#endif

    FWD2v4(z[0], z[1], w1);
    z[0] = quartic_dot(pq, z[0], gamma, q);
    z[1] = quartic_dot(pq, z[1], pq.s0 - gamma, ~q);
    BCK2v4(z[0], z[1], win1);
}

INLINE void square_8x2(const uint2_32 pq, __local uint4_32 *restrict const Z,
                       __global const root_word_t *restrict const w,
                       __global const root_word_t *restrict const wi, const sz_t sj,
                       const sz_t sji) {
    DECLARE_W1_2(sj);
    DECLARE_WIN1_2(sji);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint4_32 zl[4];
    _loadl4(4, zl, Z, 1);
    quartic_square8(pq, &zl[0], w1[0], win1[0]);
    quartic_square8(pq, &zl[2], w1[1], win1[1]);
    _storel4(4, Z, 1, zl);
}

INLINE void fwd4x4_write(const uint2_32 pq, const sz_t mg, __global uint4_32 *restrict const z,
                         __local const uint4_32 *restrict const Z,
                         __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W2_4(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint4_32 zl[4];
    _loadl4(4, zl, Z, 1);
    _fwd4x2v4(pq, &zl[0], &w2[0]);
    _fwd4x2v4(pq, &zl[2], &w2[2]);
    _storeg4(4, z, mg, zl);
}

INLINE void fwd8x2_write(const uint2_32 pq, const sz_t mg, __global uint4_32 *restrict const z,
                         __local const uint4_32 *restrict const Z,
                         __global const root_word_t *restrict const w, const sz_t sj) {
    DECLARE_W12_24(sj);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint4_32 zl[4];
    _loadl4(4, zl, Z, 1);
    _fwd8v4(pq, &zl[0], w1[0], &w2[0]);
    _fwd8v4(pq, &zl[2], w1[1], &w2[2]);
    _storeg4(4, z, mg, zl);
}

INLINE void mul_4x4(const uint2_32 pq, __local uint4_32 *restrict const Z, const sz_t mg,
                    const __global uint4_32 *restrict const zp,
                    __global const root_word_t *restrict const w,
                    __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
    DECLARE_W2_4(sj);
    DECLARE_WIN2_4(sji);
    uint4_32 zpl[4];
    _loadg4(4, zpl, zp, mg);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint4_32 zl[4];
    _loadl4(4, zl, Z, 1);
    _mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);
    _mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);
    _storel4(4, Z, 1, zl);
}

INLINE void mul_8x2(const uint2_32 pq, __local uint4_32 *restrict const Z, const sz_t mg,
                    const __global uint4_32 *restrict const zp,
                    __global const root_word_t *restrict const w,
                    __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
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

INLINE void square_4(const uint2_32 pq, __local VTYPE *restrict const Z,
                     __global const root_word_t *restrict const w,
                     __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
#if VSIZE == 4
    square_4x4(pq, Z, w, wi, sj, sji);
#elif VSIZE == 2
    square_4x2(pq, Z, w, wi, sj, sji);
#else
    square_4x1(pq, Z, w[sj], wi[sji]);
#endif
}

INLINE void fwd4_write(const uint2_32 pq, const sz_t mg, __global VTYPE *restrict const z,
                       __local const VTYPE *restrict const Z,
                       __global const root_word_t *restrict const w, const sz_t sj) {
#if VSIZE == 4
    fwd4x4_write(pq, mg, z, Z, w, sj);
#elif VSIZE == 2
    fwd4x2_write(pq, mg, z, Z, w, sj);
#else
    fwd4x1_write(pq, mg, z, Z, w[sj]);
#endif
}

INLINE void mul_4(const uint2_32 pq, __local VTYPE *restrict const Z, const sz_t mg,
                  const __global VTYPE *restrict const zp,
                  __global const root_word_t *restrict const w,
                  __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
#if VSIZE == 4
    mul_4x4(pq, Z, mg, zp, w, wi, sj, sji);
#elif VSIZE == 2
    mul_4x2(pq, Z, mg, zp, w, wi, sj, sji);
#else
    mul_4x1(pq, Z, mg, zp, w[sj], wi[sji]);
#endif
}

// --- v2, v4 -- no barrier

INLINE void square_8(const uint2_32 pq, __local VTYPE *restrict const Z,
                     __global const root_word_t *restrict const w,
                     __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
#if VSIZE == 4
    square_8x2(pq, Z, w, wi, sj, sji);
#elif VSIZE == 2
    square_8x1(pq, Z, w, wi, sj, sji);
#endif
}

INLINE void fwd8_write(const uint2_32 pq, const sz_t mg, __global VTYPE *restrict const z,
                       __local const VTYPE *restrict const Z,
                       __global const root_word_t *restrict const w, const sz_t sj) {
#if VSIZE == 4
    fwd8x2_write(pq, mg, z, Z, w, sj);
#elif VSIZE == 2
    fwd8x1_write(pq, mg, z, Z, w, sj);
#endif
}

INLINE void mul_8(const uint2_32 pq, __local VTYPE *restrict const Z, const sz_t mg,
                  const __global VTYPE *restrict const zp,
                  __global const root_word_t *restrict const w,
                  __global const root_word_t *restrict const wi, const sz_t sj, const sz_t sji) {
#if VSIZE == 4
    mul_8x2(pq, Z, mg, zp, w, wi, sj, sji);
#elif VSIZE == 2
    mul_8x1(pq, Z, mg, zp, w, wi, sj, sji);
#endif
}

// --- transform/macro ---

#define DECLARE_VAR_REGv1()                                                                        \
    const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 2),                             \
               mid = gid & ~((N_SZ / 4) - 1), id = gid % (N_SZ / 4);                               \
    const uint2_32 pq = g_pq[lid];                                                                 \
    __global uint_32 *restrict const z = &zg[4 * mid];                                             \
    __global const root_word_t *restrict const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv1() __global const uint_32 *restrict const zp = &zpg[4 * mid];

#define DECLARE_VAR_REGv2()                                                                        \
    const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 3),                             \
               mid = gid & ~((N_SZ / 8) - 1), id = gid % (N_SZ / 8);                               \
    const uint2_32 pq = g_pq[lid];                                                                 \
    __global uint2_32 *restrict const z = &zg[4 * mid];                                            \
    __global const root_word_t *restrict const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv2() __global const uint2_32 *restrict const zp = &zpg[4 * mid];

#define DECLARE_VAR_REGv4()                                                                        \
    const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 4),                             \
               mid = gid & ~((N_SZ / 16) - 1), id = gid % (N_SZ / 16);                             \
    const uint2_32 pq = g_pq[lid];                                                                 \
    __global uint4_32 *restrict const z = &zg[4 * mid];                                            \
    __global const root_word_t *restrict const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv4() __global const uint4_32 *restrict const zp = &zpg[4 * mid];

#if VSIZE == 4
#define DECLARE_VAR_REG DECLARE_VAR_REGv4
#define DECLARE_VARP_REG DECLARE_VARP_REGv4
#elif VSIZE == 2
#define DECLARE_VAR_REG DECLARE_VAR_REGv2
#define DECLARE_VARP_REG DECLARE_VARP_REGv2
#else
#define DECLARE_VAR_REG DECLARE_VAR_REGv1
#define DECLARE_VARP_REG DECLARE_VARP_REGv1
#endif

#define DECLARE_VAR(B_N, CHUNK_N)                                                                  \
    /* threadIdx < B_N */                                                                          \
    DECLARE_VAR_REG();                                                                             \
    const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N);                   \
    const sz_t i = local_id, chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N,                     \
               blockIdx = group_id * CHUNK_N + chunk_idx;                                          \
    __local VTYPE *const Zi = &Z[chunk_idx];                                                       \
                                                                                                   \
    const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx;                  \
    const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm;                               \
                                                                                                   \
    const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm,                               \
               ko = blockIdx - blockIdx_mm + idx_mm * 4;                                           \
                                                                                                   \
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
                                                                                                   \
    backward_4i(pq, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sji / 1);

__kernel __attribute__((reqd_work_group_size(PG_OUTER_THREADS, 1, 1))) void
packed_forward(__global VTYPE *restrict const zg, __global const root_word_t *restrict const wg) {
    const int lm = LN_SZ - LVSIZE - 6;
    const unsigned s = 16;
    __local VTYPE Z[64 * PG_CHUNK];
    DECLARE_VAR(16, PG_CHUNK);
    DECLARE_VAR_FORWARD();
    VTYPE v[4];
    _loadg(4, v, zi, 16 << lm);
    forward4_ordinary(pq, g_f0_ordinary[lid], v);
    _storel(4, &Z[i], 16 * PG_CHUNK, v);
    const sz_t k4 = ((4 * threadIdx) & ~(16 - 1)) + (threadIdx % 4);
    forward_4(pq, 4 * PG_CHUNK, &Zi[PG_CHUNK * k4], w, sj / 4);
    forward_4o(pq, (sz_t)1 << lm, zo, PG_CHUNK, &Zi[PG_CHUNK * 4 * threadIdx], w, sj);
}
/* @G_PACKED_NORM@ */

__kernel __attribute__((reqd_work_group_size(PG_OUTER_THREADS, 1, 1))) void
packed_backward(__global VTYPE *restrict const zg, __global const root_word_t *restrict const wg) {
    const int lm = LN_SZ - LVSIZE - 6;
    const unsigned s = 16;
    __local VTYPE Z[64 * PG_CHUNK];
    DECLARE_VAR(16, PG_CHUNK);
    DECLARE_VAR_BACKWARD();
    backward_4i(pq, PG_CHUNK, &Zi[PG_CHUNK * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sji);
    const sz_t k4 = ((4 * threadIdx) & ~(16 - 1)) + (threadIdx % 4);
    backward_4(pq, 4 * PG_CHUNK, &Zi[PG_CHUNK * k4], wi, sji / 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    VTYPE v[4];
    _loadl(4, v, &Z[i], 16 * PG_CHUNK);
    _backward4_0(pq, g_b0[lid], v);
    for (int k = 0; k < 4; ++k)
        v[k] = lscalar4(fold24(v[k], pq.s0), g_packed_norm[lid], pq);
    _storeg(4, zo, 16 << lm, v);
}
#define BLK1024 1
#define L1024S (1024 / VSIZE)

// if BLK1024 != 1 then const sz_t i1024 = (local_id & ~(L1024S / 4 - 1)) * 4, i256 = local_id %
// (L1024S / 4); if BLK1024 = 1 then const sz_t i1024 = 0, i256 = local_id;
#define DECLARE_VAR_1024()                                                                         \
    __local VTYPE Z[L1024S * BLK1024];                                                             \
                                                                                                   \
    DECLARE_VAR_REG();                                                                             \
    const sz_t local_id = id % (L1024S / 4 * BLK1024), group_id = id / (L1024S / 4 * BLK1024);     \
    const sz_t j = id, sj = N_SZ / 4 / VSIZE + j;                                                  \
    DECLARE_IVAR(N_SZ / 4 / VSIZE, j);                                                             \
                                                                                                   \
    const sz_t i1024 = 0, i256 = local_id;                                                         \
    const sz_t k1024 = group_id * L1024S * BLK1024 + i1024 + i256;                                 \
                                                                                                   \
    __global VTYPE *restrict const zk = &z[k1024];                                                 \
    __local VTYPE *const Z1024 = &Z[i1024];                                                        \
    __local VTYPE *const Zi256 = &Z1024[i256];                                                     \
    const sz_t i64 = ((4 * i256) & ~(4 * (L1024S / 16) - 1)) + (i256 % (L1024S / 16));             \
    __local VTYPE *const Zi64 = &Z1024[i64];                                                       \
    const sz_t i16 = ((4 * i256) & ~(4 * (L1024S / 64) - 1)) + (i256 % (L1024S / 64));             \
    __local VTYPE *const Zi16 = &Z1024[i16];                                                       \
    const sz_t i4 = ((4 * i256) & ~(4 * (L1024S / 256) - 1)) + (i256 % (L1024S / 256));            \
    __local VTYPE *const Zi4 = &Z1024[i4];                                                         \
    __local VTYPE *const Z4 = &Z1024[4 * i256];

__kernel
#if MAX_WG_SZ >= L1024S / 4 * BLK1024
    __attribute__((reqd_work_group_size(L1024S / 4 * BLK1024, 1, 1)))
#endif
    void packed_square1024(__global VTYPE *restrict const zg,
                           __global const root_word_t *restrict const wg) {
    DECLARE_VAR_1024();

    forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));
    forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));
    forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));
    forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));
    square_4(pq, Z4, w, wi, sj, sji);
    backward_4(pq, L1024S / 256, Zi4, wi, sji / (L1024S / 256));
    backward_4(pq, L1024S / 64, Zi16, wi, sji / (L1024S / 64));
    backward_4(pq, L1024S / 16, Zi64, wi, sji / (L1024S / 16));
    backward_4o(pq, L1024S / 4, zk, L1024S / 4, Zi256, wi, sji / (L1024S / 4));
}
