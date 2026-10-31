// Unit tests. Run: ./matgen_tests [--quick]
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

#if defined(__F16C__)
#include <immintrin.h>
#endif

#include "core.hpp"
#include "dataset.hpp"
#include "json.hpp"
#include "matgen.h"
#include "spec.hpp"

using namespace mg;

static int g_failures = 0, g_checks = 0;
static bool g_quick = false;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            ++g_failures;                                                           \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                           \
    } while (0)

static void test_philox_kat() {
    // Known-answer vectors from Random123 (kat_vectors, philox4x32 10 rounds).
    struct V { uint32_t c[4], k[2], o[4]; } v[] = {
        {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
        {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, {0xffffffff, 0xffffffff},
         {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
        {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, {0xa4093822, 0x299f31d0},
         {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
    };
    for (auto& t : v) {
        uint32_t o[4];
        philox4x32_10(t.c, t.k[0], t.k[1], o);
        for (int i = 0; i < 4; ++i) CHECK(o[i] == t.o[i]);
    }
    // philox_blocks agrees with the reference function.
    uint32_t blocks[16];
    const uint64_t key = 0x0123456789abcdefull;
    philox_blocks(key, 7, 0xfffffffeull, 4, blocks);
    for (int b = 0; b < 4; ++b) {
        uint64_t blk = 0xfffffffeull + (uint64_t)b;
        uint32_t c[4] = {(uint32_t)blk, (uint32_t)(blk >> 32), 7, 0}, o[4];
        philox4x32_10(c, (uint32_t)key, (uint32_t)(key >> 32), o);
        for (int i = 0; i < 4; ++i) CHECK(blocks[4 * b + i] == o[i]);
    }
}

static void test_xxh64() {
    CHECK(xxh64("", 0) == 0xEF46DB3751D8E999ull);
    CHECK(xxh64("abc", 3) == 0x44BC2CF5AD770999ull);
    const char* s = "Nobody inspects the spammish repetition";
    CHECK(xxh64(s, std::strlen(s)) == 0xFBCEA83C8A378BF1ull);
}

static void test_f16() {
#if defined(__F16C__)
    // Exhaustive over every non-NaN float32 bit pattern, against the CPU's F16C.
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int64_t> bad{0};
    const uint64_t step = g_quick ? 97 : 1;
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            int64_t local = 0;
            for (uint64_t u = (uint64_t)t * step; u < (1ull << 32); u += (uint64_t)nt * step) {
                uint32_t b = (uint32_t)u;
                if ((b & 0x7FFFFFFFu) > 0x7F800000u) continue;
                float x = u2f(b);
                uint16_t ref = (uint16_t)_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT);
                if (f32_to_f16(x) != ref) ++local;
            }
            bad += local;
        });
    }
    for (auto& x : th) x.join();
    std::printf("  f16 %s check vs F16C: %lld mismatches\n", g_quick ? "sampled" : "exhaustive", (long long)bad.load());
    CHECK(bad == 0);
#else
    std::printf("  f16: F16C not available, comparing against decode round-trip only\n");
#endif
    // Every half value round-trips.
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF)) continue;
        CHECK(f32_to_f16(f16_to_f32((uint16_t)h)) == h);
    }
    CHECK((f32_to_f16(u2f(0x7FC00000u)) & 0x7FFF) > 0x7C00);  // NaN stays NaN
}

static void test_bf16() {
    std::mt19937 rng(1);
    for (int i = 0; i < 2000000; ++i) {
        uint32_t b = rng();
        if ((b & 0x7FFFFFFFu) > 0x7F800000u) continue;
        float x = u2f(b);
        // Reference: pick the nearer of truncation and truncation + 1 ulp, ties to even.
        uint32_t lo = b & 0xFFFF0000u, hi = lo + 0x10000u;
        double dx = x, dl = u2f(lo), dh = u2f(hi);
        uint16_t ref;
        if ((b & 0xFFFF) == 0) ref = (uint16_t)(lo >> 16);
        else if (std::isinf(dh) && !std::isinf(dx)) {
            // Rounding up overflows to Inf only past the midpoint, same rule.
            ref = (b & 0xFFFF) > 0x8000 || ((b & 0xFFFF) == 0x8000 && ((lo >> 16) & 1)) ? (uint16_t)(hi >> 16) : (uint16_t)(lo >> 16);
        } else {
            double el = std::fabs(dx - dl), eh = std::fabs(dh - dx);
            ref = el < eh ? (uint16_t)(lo >> 16) : eh < el ? (uint16_t)(hi >> 16) : (uint16_t)((((lo >> 16) & 1) ? hi : lo) >> 16);
        }
        CHECK(f32_to_bf16(x) == ref);
        if (f32_to_bf16(x) != ref) break;
    }
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7F80) == 0x7F80 && (h & 0x7F)) continue;
        CHECK(f32_to_bf16(bf16_to_f32((uint16_t)h)) == h);
    }
}

// Brute-force nearest finite code, ties to even code, saturating.
static uint8_t ref_f8(float x, const std::function<float(uint8_t)>& dec, const std::function<bool(uint8_t)>& special) {
    const bool neg = std::signbit(x);
    const double a = std::fabs((double)x);
    int best = -1, maxc = 0;
    double bd = 0;
    for (int c = 0; c < 128; ++c) if (!special((uint8_t)c)) maxc = c;
    if (a >= (double)dec((uint8_t)maxc)) return (uint8_t)(maxc | (neg ? 0x80 : 0));  // saturate
    for (int c = 0; c < 128; ++c) {
        if (special((uint8_t)c)) continue;
        double d = std::fabs((double)dec((uint8_t)c) - a);
        if (best < 0 || d < bd || (d == bd && (c & 1) == 0)) { best = c; bd = d; }
    }
    return (uint8_t)(best | (neg ? 0x80 : 0));
}

static void test_f8_one(const char* name, uint8_t (*enc)(float), float (*dec)(uint8_t), bool (*special)(uint8_t)) {
    std::function<float(uint8_t)> D = dec;
    std::function<bool(uint8_t)> S = special;
    int64_t bad = 0, n = 0;
    auto check = [&](float x) {
        ++n;
        if (enc(x) != ref_f8(x, D, S)) {
            if (bad < 3) std::printf("  %s mismatch at %a: got %02x want %02x\n", name, x, enc(x), ref_f8(x, D, S));
            ++bad;
        }
    };
    // Every code, and every midpoint between neighbouring finite codes.
    std::vector<float> vals;
    for (int c = 0; c < 128; ++c) if (!special((uint8_t)c)) vals.push_back(dec((uint8_t)c));
    for (size_t i = 0; i < vals.size(); ++i) {
        check(vals[i]); check(-vals[i]);
        if (i + 1 < vals.size()) {
            float mid = (float)(((double)vals[i] + (double)vals[i + 1]) / 2);
            check(mid); check(-mid);
            check(std::nextafter(mid, 0.0f)); check(std::nextafter(mid, 1e30f));
        }
    }
    // Random magnitudes across and beyond the format's range.
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> ex(-24.0f, 20.0f);
    for (int i = 0; i < (g_quick ? 20000 : 200000); ++i) {
        float x = std::exp2(ex(rng)) * (rng() & 1 ? -1.0f : 1.0f);
        check(x);
    }
    check(1e30f); check(-1e30f); check(u2f(0x7F800000u));
    std::printf("  %s: %lld values checked, %lld mismatches\n", name, (long long)n, (long long)bad);
    CHECK(bad == 0);
    CHECK((enc(u2f(0x7FC00000u)) & 0x7F) == (std::strcmp(name, "f8e4m3") == 0 ? 0x7F : 0x7E));
}

static void test_f8() {
    test_f8_one("f8e4m3", f32_to_e4m3, e4m3_to_f32, e4m3_is_nan);
    test_f8_one("f8e5m2", f32_to_e5m2, e5m2_to_f32, e5m2_is_special);
}

static void test_distributions() {
    const int n = 1 << 22;
    std::vector<uint32_t> u(n), tmp(n + 8);
    philox_range(99, 0, 0, n, u.data(), tmp.data());
    double s = 0, s2 = 0, mx = 0;
    int64_t beyond = 0;
    for (int i = 0; i < n; ++i) {
        double z = normal_from_u32(u[i]);
        s += z; s2 += z * z;
        mx = std::max(mx, std::fabs(z));
        beyond += std::fabs(z) > 1.959964;
    }
    double mean = s / n, sd = std::sqrt(s2 / n - mean * mean), frac = (double)beyond / n;
    std::printf("  normal: mean %.5f std %.5f P(|z|>1.96) %.5f max|z| %.3f\n", mean, sd, frac, mx);
    CHECK(std::fabs(mean) < 0.003);
    CHECK(std::fabs(sd - 1.0) < 0.003);
    CHECK(std::fabs(frac - 0.05) < 0.001);
    CHECK(std::fabs(-normal_from_u32(0) - normal_from_u32(0xFFFFFFFFu)) < 1e-6);
    CHECK(normal_from_u32(0x80000000u) > 0 && normal_from_u32(0x7FFFFFFFu) < 0);
    // Monotonic in the uniform, over all 2^24 inputs.
    float prev = -1e9f;
    bool mono = true;
    for (uint32_t k = 0; k < (1u << 24); ++k) {
        float z = normal_from_u32(k << 8);
        mono &= z >= prev;
        prev = z;
    }
    CHECK(mono);

    s = s2 = 0;
    for (int i = 0; i < n; ++i) { double z = laplace_from_u32(u[i]); s += z; s2 += z * z; }
    mean = s / n; sd = std::sqrt(s2 / n - mean * mean);
    std::printf("  laplace: mean %.5f std %.5f\n", mean, sd);
    CHECK(std::fabs(mean) < 0.003);
    CHECK(std::fabs(sd - 1.0) < 0.01);

    // det_logf against libm (accuracy only; determinism does not depend on libm).
    double worst = 0;
    for (uint32_t k = 0; k < (1u << 24); k += 97) {
        float p = ((float)k + 0.5f) * 5.9604644775390625e-8f;
        worst = std::max(worst, std::fabs((double)det_logf(p) - std::log((double)p)));
    }
    std::printf("  det_logf max abs error: %.2e\n", worst);
    CHECK(worst < 2e-6);
}

static OperandSpec make_spec(DType dt, Dist dist, Scale sc, int64_t rows, int64_t cols) {
    OperandSpec s;
    s.id = std::string("t.") + dtype_name(dt) + "." + dist_name(dist);
    s.rows = rows; s.cols = cols; s.dtype = dt; s.dist = dist; s.scale = sc;
    s.group_size = sc == Scale::Group ? 128 : 0;
    s.stddev = dist == Dist::Normal ? 0.03125f : 1.0f;
    s.outlier_frac = 0.01f; s.outlier_mult = 50.0f;
    s.seed = 7;
    s.finalize();
    return s;
}

static void test_determinism() {
    const OperandSpec specs[] = {
        make_spec(DType::F16, Dist::LaplaceOutlier, Scale::None, 301, 1003),
        make_spec(DType::BF16, Dist::Normal, Scale::None, 301, 1003),
        make_spec(DType::F8E4M3, Dist::Normal, Scale::Tensor, 301, 1003),
        make_spec(DType::F8E5M2, Dist::Normal, Scale::Tensor, 301, 1003),
        make_spec(DType::I8, Dist::Normal, Scale::Row, 301, 1003),
        make_spec(DType::I4, Dist::Normal, Scale::Group, 301, 1024),
        make_spec(DType::I8, Dist::UniformInt, Scale::None, 301, 1003),
        make_spec(DType::I4, Dist::UniformInt, Scale::None, 301, 1002),
    };
    for (const auto& s : specs) {
        const size_t db = (size_t)s.data_bytes(), sb = (size_t)s.rows * s.scales_per_row();
        std::vector<uint8_t> full(db), part(db), par(db);
        std::vector<float> sf(sb + 1), sp(sb + 1), spar(sb + 1);
        generate_rows(s, 0, s.rows, full.data(), sf.data());
        // Uneven pieces.
        const int64_t cuts[] = {0, 1, 7, 150, 151, 301};
        for (int i = 0; i + 1 < 6; ++i)
            generate_rows(s, cuts[i], cuts[i + 1] - cuts[i], part.data() + (size_t)cuts[i] * s.row_bytes(),
                          sf.size() > 1 ? sp.data() + (size_t)cuts[i] * s.scales_per_row() : nullptr);
        CHECK(full == part);
        if (sb) CHECK(std::memcmp(sf.data(), sp.data(), sb * 4) == 0);
        for (int t : {2, 3, 6}) {
            generate_rows_parallel(s, 0, s.rows, par.data(), sb ? spar.data() : nullptr, t);
            CHECK(full == par);
        }
        // A different seed gives different data.
        OperandSpec s2 = s;
        s2.seed = 8;
        s2.finalize();
        std::vector<uint8_t> other(db);
        generate_rows(s2, 0, 1, other.data(), sb ? spar.data() : nullptr);
        CHECK(std::memcmp(other.data(), full.data(), s.row_bytes()) != 0);
    }
}

static void test_quantization() {
    // INT4 group and INT8 row quantization: dequantized error <= scale/2.
    for (DType dt : {DType::I4, DType::I8}) {
        OperandSpec s = make_spec(dt, Dist::Normal, dt == DType::I4 ? Scale::Group : Scale::Row, 16, 1024);
        std::vector<uint8_t> d((size_t)s.data_bytes());
        std::vector<float> sc((size_t)s.rows * s.scales_per_row());
        generate_rows(s, 0, s.rows, d.data(), sc.data());
        // Recreate the source values.
        std::vector<uint32_t> u((size_t)s.cols), tmp((size_t)s.cols + 8);
        double worst = 0;
        bool range_ok = true;
        for (int64_t r = 0; r < s.rows; ++r) {
            philox_range(s.key, kStreamValues, (uint64_t)(r * s.cols), (size_t)s.cols, u.data(), tmp.data());
            for (int64_t c = 0; c < s.cols; ++c) {
                float x = normal_from_u32(u[(size_t)c]) * s.stddev;
                int q;
                float scale;
                if (dt == DType::I4) {
                    uint8_t byte = d[(size_t)(r * s.cols + c) / 2];
                    int nib = (c & 1) ? byte >> 4 : byte & 0xF;
                    q = nib >= 8 ? nib - 16 : nib;
                    scale = sc[(size_t)(r * (s.cols / 128) + c / 128)];
                    range_ok &= q >= -7 && q <= 7;
                } else {
                    q = (int8_t)d[(size_t)(r * s.cols + c)];
                    scale = sc[(size_t)r];
                    range_ok &= q >= -127 && q <= 127;
                }
                worst = std::max(worst, std::fabs(q * (double)scale - x) / scale);
            }
        }
        std::printf("  %s: max |dequant - source| = %.4f scale\n", dtype_name(dt), worst);
        CHECK(range_ok);
        CHECK(worst <= 0.5 + 1e-4);
    }
    // FP8 tensor scale: largest generated magnitude maps to <= format max, no saturation of real data.
    OperandSpec s = make_spec(DType::F8E4M3, Dist::Normal, Scale::Tensor, 4, 64);
    CHECK(std::fabs(-normal_from_u32(0) * s.stddev / s.tensor_scale - 448.0f) < 0.01f);
}

static void test_json() {
    Json j = Json::object();
    j.set("a", 1).set("b", "x\"y\n").set("c", 0.1f).set("big", (int64_t)67108864000ll);
    Json arr = Json::array();
    arr.push(1).push(2.5).push(true).push(Json());
    j.set("arr", arr);
    Json k = json_parse(json_dump(j));
    CHECK(k["a"].i64() == 1);
    CHECK(k["b"].str() == "x\"y\n");
    CHECK(k["c"].f32() == 0.1f);
    CHECK(k["big"].i64() == 67108864000ll);
    CHECK(k["arr"].size() == 4 && k["arr"][2].b && k["arr"][3].type == Json::Null);
}

static void test_dataset_roundtrip() {
    BuildConfig cfg;
    cfg.profile = {"tiny", 256, 1, 1000, 4096};
    cfg.weights = {"i8", "i4", "f8e4m3", "i8-raw"};
    cfg.lmhead = {"f16", "i8"};
    cfg.act = "bf16";
    cfg.mode = "hybrid";
    cfg.threads = 3;
    Plan plan = plan_classic(cfg, false);
    CHECK(plan.cases.size() == 7 * 4 + 2);
    const std::string dir = "test_dataset_tmp";
    std::string err;
    CHECK(write_dataset(plan, dir, Progress{}, err) == 0);

    char e[256];
    mg_dataset* ds = mg_open(dir.c_str(), e, sizeof e);
    CHECK(ds != nullptr);
    if (!ds) { std::printf("  mg_open: %s\n", e); return; }
    CHECK(mg_num_operands(ds) == plan.operands.size());
    CHECK(mg_num_cases(ds) == plan.cases.size());
    for (size_t i = 0; i < mg_num_operands(ds); ++i) {
        mg_operand_info info;
        CHECK(mg_operand_at(ds, i, &info) == 0);
        const size_t bytes = (size_t)info.rows * info.row_bytes;
        std::vector<uint8_t> a(bytes), b(bytes);
        std::vector<float> sa((size_t)info.rows * info.scales_per_row + 1), sb(sa.size());
        CHECK(mg_load_rows(ds, info.id, 0, info.rows, a.data(), sa.data(), 2) == 0);
        CHECK(mg_generate_rows(ds, info.id, 0, info.rows, b.data(), sb.data(), 1) == 0);
        CHECK(a == b);
        CHECK(sa == sb);
        CHECK(info.stored == (std::string(info.cls) != "C"));
        // Partial row ranges match the full load.
        if (info.rows > 10) {
            std::vector<uint8_t> c(5 * info.row_bytes);
            CHECK(mg_load_rows(ds, info.id, 3, 5, c.data(), nullptr, 1) == 0);
            CHECK(std::memcmp(c.data(), a.data() + 3 * info.row_bytes, c.size()) == 0);
        }
    }
    mg_case_info ci;
    CHECK(mg_case_at(ds, 0, &ci) == 0);
    CHECK(ci.m == 4096 && ci.n == 256 && ci.k == 256 && ci.trans_b == 1);
    CHECK(mg_load_rows(ds, "x.act", 4000, 200, nullptr, nullptr, 1) != 0);  // out of range
    mg_close(ds);
}

int main(int argc, char** argv) {
    g_quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    struct T { const char* name; void (*fn)(); } tests[] = {
        {"philox known-answer vectors", test_philox_kat},
        {"xxh64 known values", test_xxh64},
        {"fp16 encoder", test_f16},
        {"bf16 encoder", test_bf16},
        {"fp8 encoders", test_f8},
        {"distributions", test_distributions},
        {"determinism across splits and threads", test_determinism},
        {"quantization", test_quantization},
        {"json", test_json},
        {"dataset write + C API round trip", test_dataset_roundtrip},
    };
    for (auto& t : tests) {
        std::printf("[ RUN ] %s\n", t.name);
        int before = g_failures;
        t.fn();
        std::printf("[ %s ] %s\n", g_failures == before ? " OK " : "FAIL", t.name);
    }
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
