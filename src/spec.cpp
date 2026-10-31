#include "spec.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "core.hpp"

namespace mg {

const char* dtype_name(DType d) {
    switch (d) {
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::F8E4M3: return "f8e4m3";
        case DType::F8E5M2: return "f8e5m2";
        case DType::I8: return "i8";
        case DType::I4: return "i4";
    }
    return "?";
}

bool parse_dtype(const std::string& s, DType& d) {
    if (s == "f16" || s == "fp16") d = DType::F16;
    else if (s == "bf16") d = DType::BF16;
    else if (s == "f8e4m3" || s == "fp8" || s == "fp8e4m3" || s == "e4m3") d = DType::F8E4M3;
    else if (s == "f8e5m2" || s == "fp8e5m2" || s == "e5m2") d = DType::F8E5M2;
    else if (s == "i8" || s == "int8") d = DType::I8;
    else if (s == "i4" || s == "int4") d = DType::I4;
    else return false;
    return true;
}

const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Normal: return "normal";
        case Dist::LaplaceOutlier: return "laplace_outlier";
        case Dist::UniformInt: return "uniform_int";
    }
    return "?";
}

const char* scale_name(Scale s) {
    switch (s) {
        case Scale::None: return "none";
        case Scale::Tensor: return "tensor";
        case Scale::Row: return "row";
        case Scale::Group: return "group";
    }
    return "?";
}

static Dist parse_dist(const std::string& s) {
    if (s == "normal") return Dist::Normal;
    if (s == "laplace_outlier") return Dist::LaplaceOutlier;
    if (s == "uniform_int") return Dist::UniformInt;
    throw std::runtime_error("unknown distribution '" + s + "'");
}

static Scale parse_scale(const std::string& s) {
    if (s == "none") return Scale::None;
    if (s == "tensor") return Scale::Tensor;
    if (s == "row") return Scale::Row;
    if (s == "group") return Scale::Group;
    throw std::runtime_error("unknown scale mode '" + s + "'");
}

size_t OperandSpec::row_bytes() const {
    switch (dtype) {
        case DType::F16:
        case DType::BF16: return (size_t)cols * 2;
        case DType::F8E4M3:
        case DType::F8E5M2:
        case DType::I8: return (size_t)cols;
        case DType::I4: return (size_t)cols / 2;
    }
    return 0;
}

size_t OperandSpec::scales_per_row() const {
    if (scale == Scale::Row) return 1;
    if (scale == Scale::Group) return (size_t)(cols / group_size);
    return 0;
}

uint64_t derive_key(uint64_t seed, const std::string& id) {
    uint64_t h = xxh64(id.data(), id.size(), 0x6D617467656E31ull);  // "matgen1"
    return splitmix64(seed ^ splitmix64(h));
}

void OperandSpec::finalize() {
    if (rows <= 0 || cols <= 0) throw std::runtime_error(id + ": rows and cols must be positive");
    const bool is_int = dtype == DType::I8 || dtype == DType::I4;
    const bool is_f8 = dtype == DType::F8E4M3 || dtype == DType::F8E5M2;
    if (dtype == DType::I4 && cols % 2) throw std::runtime_error(id + ": INT4 needs an even number of columns");
    if (dist == Dist::UniformInt && !is_int) throw std::runtime_error(id + ": uniform_int needs an integer format");
    if (is_int && dist != Dist::UniformInt && scale != Scale::Row && scale != Scale::Group)
        throw std::runtime_error(id + ": quantized integers need row or group scales");
    if (scale == Scale::Group) {
        if (group_size <= 0 || cols % group_size || group_size % 2)
            throw std::runtime_error(id + ": cols must be a multiple of an even group size");
    }
    if (scale == Scale::Tensor) {
        if (!is_f8) throw std::runtime_error(id + ": tensor scale is only used for FP8");
        if (dist != Dist::Normal) throw std::runtime_error(id + ": tensor scale needs a normal distribution");
        // The generator's extreme value is known exactly, so one pass suffices:
        // the largest generated magnitude maps to the format's largest value.
        const float zmax = -normal_from_u32(0);
        const float fmt_max = dtype == DType::F8E4M3 ? 448.0f : 57344.0f;
        tensor_scale = (stddev * zmax) / fmt_max;
    }
    key = derive_key(seed, id);
    if (tile_rows <= 0) {
        const int64_t target = 16ll << 20;  // ~16 MiB of data per tile
        tile_rows = std::max<int64_t>(1, target / (int64_t)row_bytes());
        tile_rows = std::min(tile_rows, rows);
    }
}

// Packs n (even) signed 4-bit values, two per byte, low nibble first.
static void pack_int4(const int8_t* q, int64_t n, uint8_t* out) {
    for (int64_t j = 0; j < n / 2; ++j) {
        uint16_t v;
        std::memcpy(&v, q + 2 * j, 2);  // little-endian: low byte is q[2j]
        out[j] = (uint8_t)((v & 0x0Fu) | ((v >> 4) & 0xF0u));
    }
}

void generate_rows(const OperandSpec& s, int64_t row0, int64_t nrows, uint8_t* data,
                   float* scales, Stats* stats) {
    const int64_t C = s.cols;
    thread_local std::vector<uint32_t> u, tmp;
    thread_local std::vector<float> x, colscale;
    thread_local std::vector<int8_t> qbuf;
    u.resize((size_t)C);
    tmp.resize((size_t)C + 8);
    x.resize((size_t)C);
    if (s.dtype == DType::I4) qbuf.resize((size_t)C);

    if (s.dist == Dist::LaplaceOutlier) {
        colscale.resize((size_t)C);
        philox_range(s.key, kStreamOutlierCols, 0, (size_t)C, u.data(), tmp.data());
        const double thr = (double)s.outlier_frac * 4294967296.0;
        for (int64_t c = 0; c < C; ++c)
            colscale[(size_t)c] = s.stddev * ((double)u[(size_t)c] < thr ? s.outlier_mult : 1.0f);
    }

    const size_t rb = s.row_bytes(), spr = s.scales_per_row();
    Stats st;
    for (int64_t r = 0; r < nrows; ++r) {
        const int64_t row = row0 + r;
        philox_range(s.key, kStreamValues, (uint64_t)row * (uint64_t)C, (size_t)C, u.data(), tmp.data());
        uint8_t* out = data + (size_t)r * rb;
        float* sc = scales ? scales + (size_t)r * spr : nullptr;
        const uint32_t* U = u.data();
        float* X = x.data();

        if (s.dist == Dist::UniformInt) {
            if (s.dtype == DType::I8) {
                int8_t* o = (int8_t*)out;
                for (int64_t c = 0; c < C; ++c) o[c] = (int8_t)(U[c] >> 24);
            } else {
                int8_t* qi = qbuf.data();
                for (int64_t c = 0; c < C; ++c) qi[c] = (int8_t)((int32_t)(U[c] >> 28) - 8);
                pack_int4(qi, C, out);
            }
            if (stats) {
                for (int64_t c = 0; c < C; ++c) {
                    double v = s.dtype == DType::I8 ? (double)(int8_t)(U[c] >> 24) : (double)((int)(U[c] >> 28) - 8);
                    st.sum += v; st.sum2 += v * v;
                    if (v < st.mn) st.mn = v;
                    if (v > st.mx) st.mx = v;
                }
                st.n += C;
            }
            continue;
        }

        if (s.dist == Dist::Normal) {
            const float sd = s.stddev;
            for (int64_t c = 0; c < C; ++c) X[c] = normal_from_u32(U[c]) * sd;
        } else {
            const float* cs = colscale.data();
            for (int64_t c = 0; c < C; ++c) X[c] = laplace_from_u32(U[c]) * cs[c];
        }

        if (stats) {
            for (int64_t c = 0; c < C; ++c) {
                double v = X[c];
                st.sum += v; st.sum2 += v * v;
                if (v < st.mn) st.mn = v;
                if (v > st.mx) st.mx = v;
            }
            st.n += C;
        }

        switch (s.dtype) {
            case DType::F16: {
                uint16_t* o = (uint16_t*)out;
                for (int64_t c = 0; c < C; ++c) o[c] = f32_to_f16(X[c]);
                break;
            }
            case DType::BF16: {
                uint16_t* o = (uint16_t*)out;
                for (int64_t c = 0; c < C; ++c) o[c] = f32_to_bf16(X[c]);
                break;
            }
            case DType::F8E4M3:
            case DType::F8E5M2: {
                const float inv = s.scale == Scale::Tensor ? 1.0f / s.tensor_scale : 1.0f;
                if (s.dtype == DType::F8E4M3)
                    for (int64_t c = 0; c < C; ++c) out[c] = f32_to_e4m3(X[c] * inv);
                else
                    for (int64_t c = 0; c < C; ++c) out[c] = f32_to_e5m2(X[c] * inv);
                break;
            }
            case DType::I8: {
                // Symmetric per-row scale: real = q * scale, q in [-127, 127].
                float m = 0.0f;
                for (int64_t c = 0; c < C; ++c) { float a = std::fabs(X[c]); m = a > m ? a : m; }
                const float scale = m > 0.0f ? m / 127.0f : 1.0f;
                const float inv = m > 0.0f ? 127.0f / m : 0.0f;
                int8_t* o = (int8_t*)out;
                for (int64_t c = 0; c < C; ++c) {
                    float q = rne_int(X[c] * inv);
                    q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
                    o[c] = (int8_t)(int32_t)q;
                }
                if (sc) sc[0] = scale;
                break;
            }
            case DType::I4: {
                // Symmetric per-group scale along the row: real = q * scale, q in [-7, 7].
                const int64_t G = s.group_size;
                int8_t* qi = qbuf.data();
                for (int64_t g = 0; g < C / G; ++g) {
                    const float* xg = X + g * G;
                    float m = 0.0f;
                    for (int64_t c = 0; c < G; ++c) { float a = std::fabs(xg[c]); m = a > m ? a : m; }
                    const float scale = m > 0.0f ? m / 7.0f : 1.0f;
                    const float inv = m > 0.0f ? 7.0f / m : 0.0f;
                    int8_t* qg = qi + g * G;
                    for (int64_t c = 0; c < G; ++c) {
                        float q = rne_int(xg[c] * inv);
                        q = q > 7.0f ? 7.0f : (q < -7.0f ? -7.0f : q);
                        qg[c] = (int8_t)(int32_t)q;
                    }
                    if (sc) sc[g] = scale;
                }
                pack_int4(qi, C, out);
                break;
            }
        }
    }
    if (stats) stats->merge(st);
}

int default_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 1;
}

void generate_rows_parallel(const OperandSpec& s, int64_t row0, int64_t nrows, uint8_t* data,
                            float* scales, int nthreads, Stats* stats) {
    if (nthreads <= 0) nthreads = default_threads();
    const size_t rb = s.row_bytes(), spr = s.scales_per_row();
    const int64_t chunk = std::max<int64_t>(1, (4ll << 20) / (int64_t)rb);  // ~4 MiB per task
    const int64_t nchunks = (nrows + chunk - 1) / chunk;
    nthreads = (int)std::min<int64_t>(nthreads, std::max<int64_t>(1, nchunks));
    if (nthreads == 1) { generate_rows(s, row0, nrows, data, scales, stats); return; }

    std::atomic<int64_t> next{0};
    std::mutex mu;
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&] {
            Stats local;
            for (;;) {
                int64_t i = next.fetch_add(1);
                if (i >= nchunks) break;
                int64_t r = i * chunk, n = std::min(chunk, nrows - r);
                generate_rows(s, row0 + r, n, data + (size_t)r * rb,
                              scales && spr ? scales + (size_t)r * spr : nullptr,
                              stats ? &local : nullptr);
            }
            if (stats) { std::lock_guard<std::mutex> g(mu); stats->merge(local); }
        });
    }
    for (auto& th : pool) th.join();
}

std::string hex64(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)v);
    return buf;
}

uint64_t parse_hex64(const std::string& s) { return std::stoull(s, nullptr, 16); }

Json spec_to_json(const OperandSpec& s) {
    Json j = Json::object();
    j.set("operand_id", s.id).set("class", s.cls).set("role", s.role);
    Json shape = Json::array();
    shape.push(s.rows).push(s.cols);
    j.set("shape", shape);
    j.set("dtype", dtype_name(s.dtype));
    Json layout = Json::object();
    layout.set("order", "row_major").set("row_bytes", (uint64_t)s.row_bytes()).set("endian", "little");
    if (s.dtype == DType::I4) layout.set("int4_packing", "low_nibble_first").set("int4_signed", true);
    j.set("layout", layout);

    Json dist = Json::object();
    dist.set("name", dist_name(s.dist));
    if (s.dist != Dist::UniformInt) dist.set("std", s.stddev);
    if (s.dist == Dist::LaplaceOutlier)
        dist.set("outlier_frac", s.outlier_frac).set("outlier_mult", s.outlier_mult);
    j.set("distribution", dist);

    Json q = Json::object();
    q.set("mode", scale_name(s.scale));
    if (s.scale == Scale::Group) q.set("group_size", s.group_size).set("group_axis", "row");
    if (s.scale == Scale::Tensor) q.set("tensor_scale", s.tensor_scale);
    if (s.scale == Scale::Row || s.scale == Scale::Group) q.set("scale_dtype", "f32").set("symmetric", true);
    j.set("quantization", q);

    Json seed = Json::object();
    seed.set("global", s.seed).set("derived_key", hex64(s.key));
    j.set("seed", seed);
    Json gen = Json::object();
    gen.set("tool", kToolVersion).set("algo_version", kAlgoVersion).set("rng", "philox4x32-10");
    j.set("generator", gen);

    Json st = Json::object();
    st.set("stored", s.stored);
    if (s.stored) {
        st.set("data_file", s.id + ".bin");
        if (s.scales_per_row()) st.set("scales_file", s.id + ".scales.bin");
    }
    st.set("data_bytes", s.data_bytes()).set("scales_bytes", s.scales_bytes());
    st.set("tile_rows", s.tile_rows);
    Json th = Json::array();
    for (uint64_t h : s.tile_hash) th.push(hex64(h));
    st.set("tile_xxh64", th);
    if (s.scales_per_row()) st.set("scales_xxh64", hex64(s.scales_hash));
    j.set("storage", st);

    Json stats = Json::object();
    stats.set("min", s.stat_min).set("max", s.stat_max).set("mean", s.stat_mean).set("std", s.stat_std);
    j.set("source_stats", stats);
    return j;
}

OperandSpec spec_from_json(const Json& j) {
    OperandSpec s;
    s.id = j["operand_id"].str();
    s.cls = j["class"].str();
    s.role = j["role"].str();
    s.rows = j["shape"][0].i64();
    s.cols = j["shape"][1].i64();
    if (!parse_dtype(j["dtype"].str(), s.dtype)) throw std::runtime_error("unknown dtype");
    const Json& d = j["distribution"];
    s.dist = parse_dist(d["name"].str());
    if (d.has("std")) s.stddev = d["std"].f32();
    if (d.has("outlier_frac")) s.outlier_frac = d["outlier_frac"].f32();
    if (d.has("outlier_mult")) s.outlier_mult = d["outlier_mult"].f32();
    const Json& q = j["quantization"];
    s.scale = parse_scale(q["mode"].str());
    if (q.has("group_size")) s.group_size = (int32_t)q["group_size"].i64();
    s.seed = (uint64_t)j["seed"]["global"].n;
    const Json& st = j["storage"];
    s.stored = st["stored"].b;
    s.tile_rows = st["tile_rows"].i64();
    for (size_t i = 0; i < st["tile_xxh64"].size(); ++i) s.tile_hash.push_back(parse_hex64(st["tile_xxh64"][i].str()));
    if (st.has("scales_xxh64")) s.scales_hash = parse_hex64(st["scales_xxh64"].str());
    const Json& ss = j["source_stats"];
    s.stat_min = ss["min"].n; s.stat_max = ss["max"].n; s.stat_mean = ss["mean"].n; s.stat_std = ss["std"].n;
    s.finalize();  // recomputes key and tensor_scale exactly
    if (hex64(s.key) != j["seed"]["derived_key"].str())
        throw std::runtime_error(s.id + ": derived key mismatch (manifest from a different algorithm version?)");
    return s;
}

}  // namespace mg
