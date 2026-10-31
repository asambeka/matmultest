// Bit-exact building blocks: Philox RNG, distributions, number formats, XXH64.
//
// Everything here must give identical bits on every compiler, CPU and SIMD
// width. Rules: no platform math library calls (log, exp, cos), no fast-math,
// no floating-point contraction (build with -ffp-contract=off). Only IEEE
// operations that are correctly rounded everywhere (+ - * / sqrt) are used.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mg {

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ---------------------------------------------------------------------------
// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3")
// ---------------------------------------------------------------------------

constexpr uint32_t kPhM0 = 0xD2511F53u, kPhM1 = 0xCD9E8D57u;
constexpr uint32_t kPhW0 = 0x9E3779B9u, kPhW1 = 0xBB67AE85u;

inline void philox4x32_10(const uint32_t in[4], uint32_t k0, uint32_t k1, uint32_t out[4]) {
    uint32_t c0 = in[0], c1 = in[1], c2 = in[2], c3 = in[3];
    for (int r = 0; r < 10; ++r) {
        uint64_t p0 = (uint64_t)kPhM0 * c0;
        uint64_t p1 = (uint64_t)kPhM1 * c2;
        uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
        uint32_t n1 = (uint32_t)p1;
        uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
        uint32_t n3 = (uint32_t)p0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += kPhW0; k1 += kPhW1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

// Fills out[0 .. 4*nblocks) with Philox blocks for counters
// {block, block >> 32, stream, 0}, block = block0 .. block0+nblocks-1.
inline void philox_blocks(uint64_t key, uint32_t stream, uint64_t block0, size_t nblocks,
                          uint32_t* out) {
    const uint32_t key0 = (uint32_t)key, key1 = (uint32_t)(key >> 32);
    for (size_t b = 0; b < nblocks; ++b) {
        uint64_t blk = block0 + b;
        uint32_t c0 = (uint32_t)blk, c1 = (uint32_t)(blk >> 32), c2 = stream, c3 = 0;
        uint32_t k0 = key0, k1 = key1;
        for (int r = 0; r < 10; ++r) {
            uint64_t p0 = (uint64_t)kPhM0 * c0;
            uint64_t p1 = (uint64_t)kPhM1 * c2;
            uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
            uint32_t n1 = (uint32_t)p1;
            uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
            uint32_t n3 = (uint32_t)p0;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
            k0 += kPhW0; k1 += kPhW1;
        }
        out[4 * b + 0] = c0; out[4 * b + 1] = c1; out[4 * b + 2] = c2; out[4 * b + 3] = c3;
    }
}

// One 32-bit random word per element: element e uses lane e%4 of block e/4.
// tmp must hold at least n + 8 words.
inline void philox_range(uint64_t key, uint32_t stream, uint64_t e0, size_t n, uint32_t* out,
                         uint32_t* tmp) {
    const uint64_t b0 = e0 / 4;
    const size_t off = (size_t)(e0 % 4);
    const size_t nb = (off + n + 3) / 4;
    philox_blocks(key, stream, b0, nb, tmp);
    std::memcpy(out, tmp + off, n * sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// Distributions from one 32-bit word. The top 24 bits give k in [0, 2^24);
// the uniform is p = (k + 0.5) / 2^24, which is exact in float and never 0 or 1.
// ---------------------------------------------------------------------------

// Natural log for positive normal floats, polynomial only (deterministic).
inline float det_logf(float x) {
    uint32_t b = f2u(x);
    int32_t e = (int32_t)((b >> 23) & 0xFF) - 127;
    float m = u2f((b & 0x7FFFFFu) | 0x3F800000u);  // [1, 2)
    int32_t big = m > 1.41421356f ? 1 : 0;
    m = big ? m * 0.5f : m;                        // [sqrt(1/2), sqrt(2))
    e += big;
    float s = (m - 1.0f) / (m + 1.0f);
    float s2 = s * s;
    float p = s2 * (0.333333343f + s2 * (0.2f + s2 * (0.142857149f + s2 * 0.111111112f)));
    return (float)e * 0.693147182f + (2.0f * s + 2.0f * s * p);
}

// Standard normal via Acklam's rational approximation of the inverse CDF.
// Range is bounded at about +-5.3 (the most extreme 24-bit uniforms).
inline float normal_from_u32(uint32_t u) {
    const int32_t k = (int32_t)(u >> 8);
    const int32_t km = 16777215 - k;
    const int32_t kt = k < km ? k : km;
    const float pt = ((float)kt + 0.5f) * 5.9604644775390625e-8f;          // min(p, 1-p), exact
    // Central region in double: the rational function cancels heavily, and
    // float evaluation would make the transform visibly non-monotonic.
    const double q = (double)(2 * k + 1 - 16777216) * 2.98023223876953125e-8;  // p - 0.5, exact
    const double r = q * q;
    const double num = (((((-3.969683028665376e+01 * r + 2.209460984245205e+02) * r
                           - 2.759285104469687e+02) * r + 1.383577518672690e+02) * r
                         - 3.066479806614716e+01) * r + 2.506628277459239e+00) * q;
    const double den = (((((-5.447609879822406e+01 * r + 1.615858368580409e+02) * r
                           - 1.556989798598866e+02) * r + 6.680131188771972e+01) * r
                         - 1.328068155288572e+01) * r + 1.0);
    const float central = (float)(num / den);

    const float t = __builtin_sqrtf(-2.0f * det_logf(pt));
    const float tn = ((((-7.784894002430293e-03f * t - 3.223964580411365e-01f) * t
                        - 2.400758277161838e+00f) * t - 2.549732539343734e+00f) * t
                      + 4.374664141464968e+00f) * t + 2.938163982698783e+00f;
    const float td = (((7.784695709041462e-03f * t + 3.224671290700398e-01f) * t
                       + 2.445134137142996e+00f) * t + 3.754408661907416e+00f) * t + 1.0f;
    float tail = tn / td;  // lower-tail value (negative)
    tail = k < km ? tail : -tail;
    return pt < 0.02425f ? tail : central;
}

// Laplace with unit variance (scale b = 1/sqrt(2)).
inline float laplace_from_u32(uint32_t u) {
    const int32_t k = (int32_t)(u >> 8);
    const int32_t km = 16777215 - k;
    const int32_t kt = k < km ? k : km;
    const float pt = ((float)kt + 0.5f) * 5.9604644775390625e-8f;
    const float a = -det_logf(2.0f * pt) * 0.707106769f;
    return k < km ? -a : a;
}

// Round to nearest integer, ties to even, for |v| < 2^22.
inline float rne_int(float v) { return (v + 12582912.0f) - 12582912.0f; }

// ---------------------------------------------------------------------------
// Number formats. All encoders round to nearest, ties to even.
// ---------------------------------------------------------------------------

inline uint16_t f32_to_f16(float x) {
    uint32_t f = f2u(x);
    const uint32_t sign = (f >> 16) & 0x8000u;
    f &= 0x7FFFFFFFu;
    const uint32_t odd = (f >> 13) & 1u;
    const uint32_t nrm = (f + 0xC8000FFFu + odd) >> 13;           // rebias exponent, round
    const uint32_t sub = f2u(u2f(f) + 0.5f) - 0x3F000000u;          // ulp(0.5) = 2^-24
    uint32_t o = f < 0x38800000u ? sub : nrm;                       // below 2^-14: subnormal
    o = f >= 0x47800000u ? 0x7C00u : o;                             // overflow -> Inf
    o = f > 0x7F800000u ? 0x7E00u : o;                              // NaN
    return (uint16_t)(o | sign);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) {
        float v = (float)m * 5.9604644775390625e-8f;  // m * 2^-24
        return u2f(f2u(v) | sign);
    }
    if (e == 31) return u2f(sign | 0x7F800000u | (m << 13));
    return u2f(sign | ((e + 112) << 23) | (m << 13));
}

inline uint16_t f32_to_bf16(float x) {
    const uint32_t f = f2u(x);
    const uint32_t r = (f + 0x7FFFu + ((f >> 16) & 1u)) >> 16;
    const uint32_t nan = ((f >> 16) & 0x8000u) | 0x7FC0u;
    return (uint16_t)((f & 0x7FFFFFFFu) > 0x7F800000u ? nan : r);
}

inline float bf16_to_f32(uint16_t b) { return u2f((uint32_t)b << 16); }

// Generic small-float encoder with saturation to the largest finite value.
//   MB: mantissa bits, BIAS: exponent bias, kMagic: 2^((1-BIAS) - MB + 23),
//   max_val/max_code: largest finite value and its code, nan_code: NaN code.
template <int MB, int BIAS>
inline uint8_t f32_to_f8(float x, float magic, float max_val, uint32_t max_code,
                         uint32_t nan_code) {
    uint32_t f = f2u(x);
    const uint32_t sign = (f >> 24) & 0x80u;
    f &= 0x7FFFFFFFu;
    const float a = u2f(f);
    constexpr int SH = 23 - MB;
    const uint32_t odd = (f >> SH) & 1u;
    const uint32_t nrm =
        (f + ((uint32_t)(BIAS - 127) << 23) + ((1u << (SH - 1)) - 1u) + odd) >> SH;
    const uint32_t sub = f2u(a + magic) - f2u(magic);
    constexpr uint32_t min_normal_bits = (uint32_t)(127 + 1 - BIAS) << 23;
    uint32_t o = f < min_normal_bits ? sub : nrm;
    o = a >= max_val ? max_code : o;
    o = f > 0x7F800000u ? nan_code : o;
    return (uint8_t)(o | sign);
}

// FP8 E4M3 "fn": bias 7, max 448 (0x7E), NaN 0x7F, no Inf.
inline uint8_t f32_to_e4m3(float x) { return f32_to_f8<3, 7>(x, 16384.0f, 448.0f, 0x7Eu, 0x7Fu); }
// FP8 E5M2: bias 15, max 57344 (0x7B), saturating; NaN 0x7E.
inline uint8_t f32_to_e5m2(float x) { return f32_to_f8<2, 15>(x, 128.0f, 57344.0f, 0x7Bu, 0x7Eu); }

template <int MB, int BIAS>
inline float f8_to_f32_generic(uint8_t c) {
    const float sign = (c & 0x80) ? -1.0f : 1.0f;
    const int e = (c & 0x7F) >> MB, m = c & ((1 << MB) - 1);
    float v;
    if (e == 0) {
        v = (float)m;
        for (int i = 0; i < (BIAS - 1) + MB; ++i) v *= 0.5f;
    } else {
        v = 1.0f + (float)m / (float)(1 << MB);
        int s = e - BIAS;
        for (; s > 0; --s) v *= 2.0f;
        for (; s < 0; ++s) v *= 0.5f;
    }
    return sign * v;
}
inline bool e4m3_is_nan(uint8_t c) { return (c & 0x7F) == 0x7F; }
inline float e4m3_to_f32(uint8_t c) { return e4m3_is_nan(c) ? u2f(0x7FC00000u) : f8_to_f32_generic<3, 7>(c); }
inline bool e5m2_is_special(uint8_t c) { return ((c >> 2) & 0x1F) == 0x1F; }
inline float e5m2_to_f32(uint8_t c) {
    if (e5m2_is_special(c)) {
        if ((c & 3) == 0) return u2f((c & 0x80 ? 0xFF800000u : 0x7F800000u));
        return u2f(0x7FC00000u);
    }
    return f8_to_f32_generic<2, 15>(c);
}

// ---------------------------------------------------------------------------
// XXH64 (https://github.com/Cyan4973/xxHash), one-shot.
// ---------------------------------------------------------------------------

constexpr uint64_t kXP1 = 0x9E3779B185EBCA87ull, kXP2 = 0xC2B2AE3D27D4EB4Full,
                   kXP3 = 0x165667B19E3779F9ull, kXP4 = 0x85EBCA77C2B2AE63ull,
                   kXP5 = 0x27D4EB2F165667C5ull;
inline uint64_t xrotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t xround(uint64_t acc, uint64_t in) { acc += in * kXP2; acc = xrotl(acc, 31); return acc * kXP1; }
inline uint64_t xmerge(uint64_t acc, uint64_t v) { acc ^= xround(0, v); return acc * kXP1 + kXP4; }
inline uint64_t xread64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t xread32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

inline uint64_t xxh64(const void* data, size_t len, uint64_t seed = 0) {
    const uint8_t* p = (const uint8_t*)data;
    const uint8_t* end = p + len;
    uint64_t h;
    if (len >= 32) {
        uint64_t v1 = seed + kXP1 + kXP2, v2 = seed + kXP2, v3 = seed, v4 = seed - kXP1;
        const uint8_t* limit = end - 32;
        do {
            v1 = xround(v1, xread64(p)); v2 = xround(v2, xread64(p + 8));
            v3 = xround(v3, xread64(p + 16)); v4 = xround(v4, xread64(p + 24));
            p += 32;
        } while (p <= limit);
        h = xrotl(v1, 1) + xrotl(v2, 7) + xrotl(v3, 12) + xrotl(v4, 18);
        h = xmerge(h, v1); h = xmerge(h, v2); h = xmerge(h, v3); h = xmerge(h, v4);
    } else {
        h = seed + kXP5;
    }
    h += (uint64_t)len;
    while (p + 8 <= end) { h ^= xround(0, xread64(p)); h = xrotl(h, 27) * kXP1 + kXP4; p += 8; }
    if (p + 4 <= end) { h ^= (uint64_t)xread32(p) * kXP1; h = xrotl(h, 23) * kXP2 + kXP3; p += 4; }
    while (p < end) { h ^= (uint64_t)(*p) * kXP5; h = xrotl(h, 11) * kXP1; ++p; }
    h ^= h >> 33; h *= kXP2; h ^= h >> 29; h *= kXP3; h ^= h >> 32;
    return h;
}

// SplitMix64 finalizer, used for seed derivation.
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

}  // namespace mg
