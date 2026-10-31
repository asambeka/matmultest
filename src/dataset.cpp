#include "dataset.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core.hpp"
#include "hw.hpp"
#include "matgen.h"

namespace mg {

bool find_profile(const std::string& name, Profile& p) {
    if (name == "8b") { p = {"8b", 4096, 32}; return true; }
    if (name == "40b" || name == "30b") { p = {"40b", 8192, 60}; return true; }
    if (name == "70b" || name == "70b+") { p = {"70b", 16384, 80}; return true; }
    return false;
}

int64_t Plan::stored_bytes() const {
    int64_t b = 0;
    for (auto& s : operands) if (s.stored) b += s.data_bytes() + s.scales_bytes();
    return b;
}

static DType resolve_16(const std::string& f, bool cpu_bf16) {
    if (f == "auto16") return cpu_bf16 ? DType::BF16 : DType::F16;
    DType d;
    if (!parse_dtype(f, d) || (d != DType::F16 && d != DType::BF16))
        throw std::runtime_error("expected a 16-bit format (auto16, f16, bf16), got '" + f + "'");
    return d;
}

// Configures format-specific quantization for a normally distributed weight.
static void set_weight_format(OperandSpec& s, const std::string& fmt, bool cpu_bf16) {
    std::string base = fmt;
    bool raw = false;
    if (base.size() > 4 && base.compare(base.size() - 4, 4, "-raw") == 0) { raw = true; base.resize(base.size() - 4); }
    if (base == "auto16") { s.dtype = resolve_16(base, cpu_bf16); }
    else if (!parse_dtype(base, s.dtype)) throw std::runtime_error("unknown weight format '" + fmt + "'");
    s.dist = Dist::Normal;
    switch (s.dtype) {
        case DType::I8: s.scale = Scale::Row; break;
        case DType::I4: s.scale = Scale::Group; s.group_size = 128; break;
        case DType::F8E4M3:
        case DType::F8E5M2: s.scale = Scale::Tensor; break;
        default: s.scale = Scale::None; break;
    }
    if (raw) {
        if (s.dtype != DType::I8 && s.dtype != DType::I4) throw std::runtime_error("-raw is only for i8 and i4");
        s.dist = Dist::UniformInt;
        s.scale = Scale::None;
        s.group_size = 0;
    }
}

static std::string fmt_tag(const OperandSpec& s) {
    std::string t = dtype_name(s.dtype);
    if (s.dist == Dist::UniformInt) t += "-raw";
    return t;
}

Plan plan_classic(const BuildConfig& cfg, bool cpu_bf16) {
    Plan p;
    p.cfg = cfg;
    const Profile& pr = cfg.profile;
    if (pr.H <= 0 || pr.L <= 0 || pr.V <= 0 || pr.T <= 0) throw std::runtime_error("profile parameters must be positive");
    if (cfg.mode != "hybrid" && cfg.mode != "materialize" && cfg.mode != "spec")
        throw std::runtime_error("mode must be hybrid, materialize or spec");
    p.half = resolve_16(cfg.act, cpu_bf16);
    const float wstd = (float)(1.0 / std::sqrt((double)pr.H));

    OperandSpec x;
    x.id = "x.act"; x.cls = "C"; x.role = "activation";
    x.rows = pr.T; x.cols = pr.H; x.dtype = p.half;
    x.dist = Dist::LaplaceOutlier; x.stddev = 1.0f; x.outlier_frac = 0.005f; x.outlier_mult = 50.0f;
    x.seed = cfg.seed;
    x.stored = cfg.mode == "materialize";
    x.finalize();
    p.operands.push_back(x);

    static const char* roles[7] = {"q", "k", "v", "o", "gate", "up", "down"};
    for (const std::string& wf : cfg.weights) {
        for (int64_t l = 0; l < pr.L; ++l) {
            for (const char* role : roles) {
                OperandSpec w;
                w.cls = "A"; w.rows = pr.H; w.cols = pr.H; w.stddev = wstd; w.seed = cfg.seed;
                set_weight_format(w, wf, cpu_bf16);
                char buf[64];
                std::snprintf(buf, sizeof buf, "L%02lld.%s", (long long)l, role);
                w.id = "w." + fmt_tag(w) + "." + buf;
                w.role = std::string("weight.") + buf;
                w.stored = cfg.mode != "spec";
                w.finalize();
                p.operands.push_back(w);
                CaseDef c;
                c.id = fmt_tag(w) + "." + buf; c.group = fmt_tag(w);
                c.a = x.id; c.b = w.id; c.M = pr.T; c.N = pr.H; c.K = pr.H;
                p.cases.push_back(c);
            }
        }
    }
    for (const std::string& lf : cfg.lmhead) {
        OperandSpec b;
        b.cls = "B"; b.role = "lm_head"; b.rows = pr.V; b.cols = pr.H; b.stddev = wstd; b.seed = cfg.seed;
        set_weight_format(b, lf, cpu_bf16);
        b.id = "lm." + fmt_tag(b);
        b.stored = cfg.mode != "spec";
        b.finalize();
        p.operands.push_back(b);
        CaseDef c;
        c.id = "lm." + fmt_tag(b); c.group = c.id;
        c.a = x.id; c.b = b.id; c.M = pr.T; c.N = pr.V; c.K = pr.H;
        p.cases.push_back(c);
    }
    std::unordered_map<std::string, int> seen;
    for (auto& s : p.operands)
        if (seen[s.id]++) throw std::runtime_error("duplicate operand '" + s.id + "' (format listed twice?)");
    return p;
}

double case_flops(const CaseDef& c) { return 2.0 * (double)c.M * (double)c.N * (double)c.K; }

double case_bytes_min(const CaseDef& c, const OperandSpec& a, const OperandSpec& b) {
    return (double)(a.data_bytes() + a.scales_bytes()) + (double)(b.data_bytes() + b.scales_bytes()) +
           (double)c.M * (double)c.N * 2.0;
}

static bool write_all(int fd, const void* buf, size_t n, off_t off) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) {
        ssize_t w = pwrite(fd, p, n, off);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += w; n -= (size_t)w; off += w;
    }
    return true;
}

static bool write_file(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
    return (bool)f;
}

namespace {
struct OpState {
    std::once_flag once;
    int fd = -1;
    std::vector<float> scales;
    std::atomic<int64_t> remaining{0};
    std::mutex mu;
    Stats stats;
    std::chrono::steady_clock::time_point t0;
};
}  // namespace

static bool mkdirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i < path.size() && path[i] != '/') continue;
        const std::string p = path.substr(0, i);
        if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return true;
}

int write_dataset(Plan& plan, const std::string& out_dir, const Progress& progress, std::string& err) {
    const std::string opdir = out_dir + "/operands";
    if (!mkdirs(opdir)) { err = "cannot create " + opdir + ": " + std::strerror(errno); return -1; }

    auto& ops = plan.operands;
    struct Task { size_t op; int64_t tile; };
    std::vector<Task> tasks;
    std::vector<std::unique_ptr<OpState>> state;
    size_t max_tile_bytes = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
        auto& s = ops[i];
        state.emplace_back(new OpState());
        const bool work = s.stored || plan.cfg.hash_spec_operands;
        s.tile_hash.assign(work ? (size_t)s.num_tiles() : 0, 0);
        if (!work) continue;
        state[i]->remaining = s.num_tiles();
        for (int64_t t = 0; t < s.num_tiles(); ++t) tasks.push_back({i, t});
        max_tile_bytes = std::max(max_tile_bytes, (size_t)s.tile_rows * s.row_bytes());
    }

    std::atomic<size_t> next{0}, done_ops{0};
    std::atomic<bool> failed{false};
    std::mutex err_mu, prog_mu;
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> g(err_mu);
        if (!failed.exchange(true)) err = m;
    };
    size_t work_ops = 0;
    for (auto& st : state) work_ops += st->remaining > 0;

    auto finalize_op = [&](size_t i) {
        auto& s = ops[i];
        auto& st = *state[i];
        const std::string base = opdir + "/" + s.id;
        if (s.stored) {
            if (fsync(st.fd) != 0 || close(st.fd) != 0) { fail("write failed for " + s.id); return; }
            if (rename((base + ".bin.partial").c_str(), (base + ".bin").c_str()) != 0) { fail("rename failed for " + s.id); return; }
        }
        if (!st.scales.empty()) {
            s.scales_hash = xxh64(st.scales.data(), st.scales.size() * 4);
            if (s.stored) {
                int fd = open((base + ".scales.bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                bool ok = fd >= 0 && write_all(fd, st.scales.data(), st.scales.size() * 4, 0) && fsync(fd) == 0;
                if (fd >= 0) close(fd);
                if (!ok) { fail("cannot write scales for " + s.id); return; }
            }
            std::vector<float>().swap(st.scales);
        }
        const Stats& a = st.stats;
        if (a.n) {
            s.stat_min = a.mn; s.stat_max = a.mx; s.stat_mean = a.sum / (double)a.n;
            s.stat_std = std::sqrt(std::max(0.0, a.sum2 / (double)a.n - s.stat_mean * s.stat_mean));
        }
        if (!write_file(base + ".json", json_dump(spec_to_json(s)))) { fail("cannot write manifest for " + s.id); return; }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - st.t0).count();
        std::lock_guard<std::mutex> g(prog_mu);
        size_t k = ++done_ops;
        if (progress.on_operand) progress.on_operand(k, work_ops, s, sec);
    };

    int nthreads = plan.cfg.threads > 0 ? plan.cfg.threads : default_threads();
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&] {
            std::vector<uint8_t> buf(max_tile_bytes);
            for (;;) {
                if (failed) return;
                size_t ti = next.fetch_add(1);
                if (ti >= tasks.size()) return;
                const size_t i = tasks[ti].op;
                const int64_t tile = tasks[ti].tile;
                auto& s = ops[i];
                auto& st = *state[i];
                std::call_once(st.once, [&] {
                    st.t0 = std::chrono::steady_clock::now();
                    if (s.scales_per_row()) st.scales.assign((size_t)s.rows * s.scales_per_row(), 0.0f);
                    if (s.stored) {
                        std::string path = opdir + "/" + s.id + ".bin.partial";
                        st.fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                        if (st.fd < 0) { fail("cannot create " + path + ": " + std::strerror(errno)); return; }
                        int rc = posix_fallocate(st.fd, 0, (off_t)s.data_bytes());
                        if (rc != 0) fail("cannot allocate " + path + ": " + std::strerror(rc));
                    }
                });
                if (failed) return;
                const int64_t r0 = tile * s.tile_rows;
                const int64_t n = std::min(s.tile_rows, s.rows - r0);
                const size_t bytes = (size_t)n * s.row_bytes();
                float* sc = st.scales.empty() ? nullptr : st.scales.data() + (size_t)r0 * s.scales_per_row();
                Stats local;
                generate_rows(s, r0, n, buf.data(), sc, &local);
                s.tile_hash[(size_t)tile] = xxh64(buf.data(), bytes);
                if (s.stored && !write_all(st.fd, buf.data(), bytes, (off_t)(r0 * (int64_t)s.row_bytes()))) {
                    fail("write failed for " + s.id + ": " + std::strerror(errno));
                    return;
                }
                { std::lock_guard<std::mutex> g(st.mu); st.stats.merge(local); }
                if (st.remaining.fetch_sub(1) == 1) finalize_op(i);
            }
        });
    }
    for (auto& th : pool) th.join();
    if (failed) return -1;

    // Spec-only operands that were not hashed still need a manifest.
    for (size_t i = 0; i < ops.size(); ++i)
        if (state[i]->remaining == 0 && ops[i].tile_hash.empty())
            write_file(opdir + "/" + ops[i].id + ".json", json_dump(spec_to_json(ops[i])));

    std::unordered_map<std::string, const OperandSpec*> byid;
    for (auto& s : ops) byid[s.id] = &s;
    Json cases = Json::array();
    for (auto& c : plan.cases) {
        Json j = Json::object();
        const double fl = case_flops(c), by = case_bytes_min(c, *byid[c.a], *byid[c.b]);
        j.set("case_id", c.id).set("group", c.group).set("a", c.a).set("b", c.b);
        j.set("M", c.M).set("N", c.N).set("K", c.K).set("trans_a", false).set("trans_b", true);
        j.set("format_a", dtype_name(byid[c.a]->dtype)).set("format_b", dtype_name(byid[c.b]->dtype));
        j.set("flops", fl).set("bytes_min", by).set("arithmetic_intensity", fl / by);
        cases.push(j);
    }
    Json cj = Json::object();
    cj.set("cases", cases);
    if (!write_file(out_dir + "/cases.json", json_dump(cj))) { err = "cannot write cases.json"; return -1; }

    const Profile& pr = plan.cfg.profile;
    Json d = Json::object();
    d.set("format", "matgen-dataset").set("format_version", 1);
    d.set("tool", kToolVersion).set("algo_version", kAlgoVersion);
    d.set("scenario", "classic");
    Json params = Json::object();
    params.set("profile", pr.name).set("H", pr.H).set("L", pr.L).set("V", pr.V).set("T", pr.T);
    d.set("params", params);
    d.set("seed", plan.cfg.seed).set("mode", plan.cfg.mode).set("half_format", dtype_name(plan.half));
    d.set("generated_on", cpu_to_json(detect_cpu()));
    std::time_t now = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    d.set("created", ts);
    Json ids = Json::array();
    for (auto& s : ops) ids.push(s.id);
    d.set("operands", ids);
    d.set("stored_bytes", plan.stored_bytes());
    if (!write_file(out_dir + "/dataset.json", json_dump(d))) { err = "cannot write dataset.json"; return -1; }
    return 0;
}

}  // namespace mg

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

struct mg_dataset {
    std::string dir;
    std::vector<mg::OperandSpec> ops;
    std::unordered_map<std::string, size_t> index;
    std::vector<mg::CaseDef> cases;
    std::vector<double> case_bytes;
    mutable std::string err;
};

static std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

extern "C" {

mg_dataset* mg_open(const char* dir, char* err, size_t errlen) {
    auto ds = std::make_unique<mg_dataset>();
    try {
        ds->dir = dir;
        mg::Json d = mg::json_parse(read_text(ds->dir + "/dataset.json"));
        if (d["format"].str() != "matgen-dataset") throw std::runtime_error("not a matgen dataset");
        if (d["algo_version"].i64() != mg::kAlgoVersion)
            throw std::runtime_error("dataset uses generator algorithm v" + std::to_string(d["algo_version"].i64()) +
                                     ", this library implements v" + std::to_string(mg::kAlgoVersion));
        const mg::Json& ids = d["operands"];
        for (size_t i = 0; i < ids.size(); ++i) {
            const std::string& id = ids[i].str();
            ds->ops.push_back(mg::spec_from_json(mg::json_parse(read_text(ds->dir + "/operands/" + id + ".json"))));
            ds->index[id] = i;
        }
        mg::Json c = mg::json_parse(read_text(ds->dir + "/cases.json"));
        const mg::Json& cs = c["cases"];
        for (size_t i = 0; i < cs.size(); ++i) {
            mg::CaseDef cd;
            cd.id = cs[i]["case_id"].str(); cd.group = cs[i]["group"].str();
            cd.a = cs[i]["a"].str(); cd.b = cs[i]["b"].str();
            cd.M = cs[i]["M"].i64(); cd.N = cs[i]["N"].i64(); cd.K = cs[i]["K"].i64();
            ds->cases.push_back(cd);
            ds->case_bytes.push_back(cs[i]["bytes_min"].n);
        }
    } catch (const std::exception& e) {
        if (err && errlen) std::snprintf(err, errlen, "%s", e.what());
        return nullptr;
    }
    return ds.release();
}

void mg_close(mg_dataset* ds) { delete ds; }

size_t mg_num_operands(const mg_dataset* ds) { return ds->ops.size(); }

static void fill_info(const mg::OperandSpec& s, mg_operand_info* o) {
    o->id = s.id.c_str(); o->cls = s.cls.c_str(); o->role = s.role.c_str();
    o->rows = s.rows; o->cols = s.cols;
    o->dtype = (mg_dtype)(int)s.dtype;
    o->row_bytes = s.row_bytes();
    o->scale_mode = (mg_scale_mode)(int)s.scale;
    o->group_size = s.group_size;
    o->scales_per_row = s.scales_per_row();
    o->tensor_scale = s.tensor_scale;
    o->stored = s.stored ? 1 : 0;
}

int mg_operand_at(const mg_dataset* ds, size_t i, mg_operand_info* out) {
    if (i >= ds->ops.size()) { ds->err = "operand index out of range"; return -1; }
    fill_info(ds->ops[i], out);
    return 0;
}

int mg_operand(const mg_dataset* ds, const char* id, mg_operand_info* out) {
    auto it = ds->index.find(id);
    if (it == ds->index.end()) { ds->err = std::string("no operand '") + id + "'"; return -1; }
    fill_info(ds->ops[it->second], out);
    return 0;
}

size_t mg_num_cases(const mg_dataset* ds) { return ds->cases.size(); }

int mg_case_at(const mg_dataset* ds, size_t i, mg_case_info* out) {
    if (i >= ds->cases.size()) { ds->err = "case index out of range"; return -1; }
    const mg::CaseDef& c = ds->cases[i];
    out->id = c.id.c_str(); out->group = c.group.c_str(); out->a = c.a.c_str(); out->b = c.b.c_str();
    out->m = c.M; out->n = c.N; out->k = c.K;
    out->trans_a = 0; out->trans_b = 1;
    out->flops = mg::case_flops(c);
    out->bytes_min = ds->case_bytes[i];
    return 0;
}

static const mg::OperandSpec* find_rows(const mg_dataset* ds, const char* id, int64_t row0, int64_t nrows) {
    auto it = ds->index.find(id);
    if (it == ds->index.end()) { ds->err = std::string("no operand '") + id + "'"; return nullptr; }
    const mg::OperandSpec& s = ds->ops[it->second];
    if (row0 < 0 || nrows < 0 || row0 + nrows > s.rows) { ds->err = "row range out of bounds"; return nullptr; }
    return &s;
}

static bool read_all(int fd, void* buf, size_t n, off_t off) {
    uint8_t* p = (uint8_t*)buf;
    while (n) {
        ssize_t r = pread(fd, p, n, off);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        p += r; n -= (size_t)r; off += r;
    }
    return true;
}

int mg_generate_rows(const mg_dataset* ds, const char* id, int64_t row0, int64_t nrows, void* data,
                     float* scales, int nthreads) {
    const mg::OperandSpec* s = find_rows(ds, id, row0, nrows);
    if (!s) return -1;
    if (scales && !s->scales_per_row()) scales = nullptr;
    if (!scales && s->scales_per_row()) {
        std::vector<float> tmp((size_t)nrows * s->scales_per_row());
        mg::generate_rows_parallel(*s, row0, nrows, (uint8_t*)data, tmp.data(), nthreads);
    } else {
        mg::generate_rows_parallel(*s, row0, nrows, (uint8_t*)data, scales, nthreads);
    }
    return 0;
}

int mg_load_rows(const mg_dataset* ds, const char* id, int64_t row0, int64_t nrows, void* data,
                 float* scales, int nthreads) {
    const mg::OperandSpec* s = find_rows(ds, id, row0, nrows);
    if (!s) return -1;
    if (!s->stored) return mg_generate_rows(ds, id, row0, nrows, data, scales, nthreads);
    const std::string base = ds->dir + "/operands/" + s->id;
    int fd = open((base + ".bin").c_str(), O_RDONLY);
    if (fd < 0) { ds->err = "cannot open " + base + ".bin"; return -1; }
    bool ok = read_all(fd, data, (size_t)nrows * s->row_bytes(), (off_t)(row0 * (int64_t)s->row_bytes()));
    close(fd);
    if (!ok) { ds->err = "read failed for " + base + ".bin"; return -1; }
    if (scales && s->scales_per_row()) {
        fd = open((base + ".scales.bin").c_str(), O_RDONLY);
        if (fd < 0) { ds->err = "cannot open " + base + ".scales.bin"; return -1; }
        const size_t spr = s->scales_per_row();
        ok = read_all(fd, scales, (size_t)nrows * spr * 4, (off_t)(row0 * (int64_t)spr * 4));
        close(fd);
        if (!ok) { ds->err = "read failed for " + base + ".scales.bin"; return -1; }
    }
    return 0;
}

void* mg_alloc(size_t bytes) {
    const size_t huge = 2u << 20;
    const size_t len = (bytes + huge - 1) / huge * huge;
    void* p = mmap(nullptr, len + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    // Align to 2 MiB and release the slack so huge pages can back the region.
    uintptr_t a = ((uintptr_t)p + huge - 1) & ~(uintptr_t)(huge - 1);
    size_t head = a - (uintptr_t)p, tail = huge - head;
    if (head) munmap(p, head);
    if (tail) munmap((void*)(a + len), tail);
#ifdef MADV_HUGEPAGE
    madvise((void*)a, len, MADV_HUGEPAGE);
#endif
    std::memset((void*)a, 0, len);  // pre-fault
    return (void*)a;
}

void mg_free(void* p, size_t bytes) {
    if (!p) return;
    const size_t huge = 2u << 20;
    munmap(p, (bytes + huge - 1) / huge * huge);
}

const char* mg_dtype_name(mg_dtype d) { return mg::dtype_name((mg::DType)(int)d); }

const char* mg_last_error(const mg_dataset* ds) { return ds->err.c_str(); }

}  // extern "C"
