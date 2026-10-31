// matgen command-line tool.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core.hpp"
#include "dataset.hpp"
#include "hw.hpp"
#include "spec.hpp"

using namespace mg;
using Clock = std::chrono::steady_clock;

static double secs_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

static std::string human(double bytes) {
    const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int i = 0;
    while (bytes >= 1024.0 && i < 4) { bytes /= 1024.0; ++i; }
    char buf[32];
    std::snprintf(buf, sizeof buf, i ? "%.2f %s" : "%.0f %s", bytes, u[i]);
    return buf;
}

static std::string human_time(double s) {
    char buf[32];
    if (s < 60) std::snprintf(buf, sizeof buf, "%.1f s", s);
    else if (s < 3600) std::snprintf(buf, sizeof buf, "%.1f min", s / 60);
    else std::snprintf(buf, sizeof buf, "%.1f h", s / 3600);
    return buf;
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) if (!cur.empty()) out.push_back(cur);
    return out;
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Args {
    std::vector<std::string> pos;
    std::map<std::string, std::string> opt;
    std::set<std::string> used;
    Args(int argc, char** argv, int start, const std::set<std::string>& flags) {
        for (int i = start; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                std::string k = a.substr(2), v;
                auto eq = k.find('=');
                if (eq != std::string::npos) { v = k.substr(eq + 1); k = k.substr(0, eq); }
                else if (flags.count(k)) v = "1";
                else if (i + 1 < argc) v = argv[++i];
                else throw std::runtime_error("missing value for --" + k);
                opt[k] = v;
            } else {
                pos.push_back(a);
            }
        }
    }
    bool has(const std::string& k) { used.insert(k); return opt.count(k) > 0; }
    std::string get(const std::string& k, const std::string& def = "") {
        used.insert(k);
        auto it = opt.find(k);
        return it == opt.end() ? def : it->second;
    }
    void check_unused() {
        for (auto& kv : opt)
            if (!used.count(kv.first)) throw std::runtime_error("unknown option --" + kv.first);
    }
};

static void usage() {
    std::printf(
        "matgen %s: LLM-like matrix multiplication test data\n\n"
        "Usage:\n"
        "  matgen generate --profile 8b|40b|70b --out DIR [options]\n"
        "      --set H=..,L=..,V=..,T=..   override profile parameters\n"
        "      --T N                       tokens (activation rows), default 1024000\n"
        "      --weights LIST              weight formats: i8,i4,f8e4m3,f8e5m2,f16,bf16,i8-raw,i4-raw (default i8)\n"
        "      --lmhead LIST               LM-head formats: auto16,f16,bf16,i8,f8e4m3,... (default auto16)\n"
        "      --act FMT                   activation format: auto16|f16|bf16 (default auto16)\n"
        "      --mode hybrid|materialize|spec  what to write to disk (default hybrid: weights only)\n"
        "      --seed N --threads N --dry-run --force --no-hash-spec --disk-mbps N\n"
        "  matgen verify DIR [--regen N] [--threads N]   check checksums and regeneration\n"
        "  matgen bench [--threads N] [--elements N]     in-memory generation speed per format\n"
        "  matgen info DIR                               summarize a dataset\n"
        "  matgen hw                                     show detected CPU features\n",
        kToolVersion);
}

static int cmd_hw() {
    CpuInfo c = detect_cpu();
    std::printf("%s", json_dump(cpu_to_json(c)).c_str());
    std::printf("16-bit format chosen by auto16: %s\n", c.bf16 ? "bf16" : "f16 (no native BF16)");
    return 0;
}

static int64_t free_space(const std::string& dir) {
    std::string d = dir;
    struct stat st;
    while (!d.empty() && stat(d.c_str(), &st) != 0) {
        auto p = d.find_last_of('/');
        d = p == std::string::npos ? "." : (p == 0 ? "/" : d.substr(0, p));
    }
    struct statvfs v;
    if (statvfs(d.empty() ? "." : d.c_str(), &v) != 0) return -1;
    return (int64_t)v.f_bavail * (int64_t)v.f_frsize;
}

static int cmd_generate(Args& a) {
    BuildConfig cfg;
    const std::string prof = a.get("profile", "8b");
    if (!find_profile(prof, cfg.profile)) throw std::runtime_error("unknown profile '" + prof + "' (8b, 40b, 70b)");
    if (a.has("T")) cfg.profile.T = std::stoll(a.get("T"));
    for (const std::string& kv : split(a.get("set"), ',')) {
        auto eq = kv.find('=');
        if (eq == std::string::npos) throw std::runtime_error("--set expects KEY=VALUE");
        std::string k = kv.substr(0, eq);
        int64_t v = std::stoll(kv.substr(eq + 1));
        if (k == "H") cfg.profile.H = v;
        else if (k == "L") cfg.profile.L = v;
        else if (k == "V") cfg.profile.V = v;
        else if (k == "T") cfg.profile.T = v;
        else throw std::runtime_error("unknown parameter '" + k + "' (H, L, V, T)");
    }
    if (a.has("weights")) cfg.weights = split(a.get("weights"), ',');
    if (a.has("lmhead")) cfg.lmhead = split(a.get("lmhead"), ',');
    cfg.act = a.get("act", cfg.act);
    cfg.mode = a.get("mode", cfg.mode);
    cfg.seed = std::stoull(a.get("seed", "42"));
    cfg.threads = std::stoi(a.get("threads", "0"));
    cfg.hash_spec_operands = !a.has("no-hash-spec");
    const bool dry = a.has("dry-run"), force = a.has("force");
    const double disk_mbps = std::stod(a.get("disk-mbps", "95"));
    const std::string out = a.get("out");
    a.check_unused();
    if (out.empty() && !dry) throw std::runtime_error("--out DIR is required");
    if (cfg.seed >= (1ull << 53)) throw std::runtime_error("--seed must be below 2^53");

    CpuInfo cpu = detect_cpu();
    Plan plan = plan_classic(cfg, cpu.bf16);
    const Profile& p = cfg.profile;

    std::map<std::string, std::pair<int, int64_t>> by_class;  // count, stored bytes
    int64_t gen_elems = 0;
    for (auto& s : plan.operands) {
        auto& e = by_class[s.cls];
        e.first++;
        if (s.stored) e.second += s.data_bytes() + s.scales_bytes();
        if (s.stored || cfg.hash_spec_operands) gen_elems += s.elements();
    }
    const int64_t stored = plan.stored_bytes();
    const OperandSpec& x = plan.operands[0];
    std::printf("Profile %s: H=%lld L=%lld V=%lld T=%lld, mode %s, 16-bit format %s%s\n", p.name.c_str(),
                (long long)p.H, (long long)p.L, (long long)p.V, (long long)p.T, cfg.mode.c_str(),
                dtype_name(plan.half), cpu.bf16 ? "" : " (CPU has no native BF16)");
    std::printf("Operands: %d weights (A), %d LM head (B), 1 activation (C %lld x %lld, %s, %s)\n",
                by_class["A"].first, by_class["B"].first, (long long)x.rows, (long long)x.cols,
                human((double)x.data_bytes()).c_str(), x.stored ? "stored" : "regenerated on load");
    std::printf("GEMM cases: %zu\n", plan.cases.size());
    std::printf("Stored on disk: %s (A %s, B %s, C %s); estimated write time %s at %.0f MB/s\n",
                human((double)stored).c_str(), human((double)by_class["A"].second).c_str(),
                human((double)by_class["B"].second).c_str(), human((double)by_class["C"].second).c_str(),
                human_time((double)stored / (disk_mbps * 1e6)).c_str(), disk_mbps);
    const double ws = (double)x.data_bytes() * 2 + (double)plan.operands[1].data_bytes();
    std::printf("Largest projection GEMM working set (A + B + output): %s; RAM available: %s\n",
                human(ws).c_str(), human((double)cpu.mem_available_bytes).c_str());
    if (dry) return 0;

    const int64_t avail = free_space(out);
    if (avail >= 0 && stored > avail && !force) {
        std::fprintf(stderr, "error: needs %s but only %s is free at %s (use --force to try anyway)\n",
                     human((double)stored).c_str(), human((double)avail).c_str(), out.c_str());
        return 2;
    }

    const bool tty = isatty(fileno(stdout));
    const auto t0 = Clock::now();
    Progress prog;
    prog.on_operand = [&](size_t k, size_t total, const OperandSpec& s, double sec) {
        const double el = secs_since(t0);
        const bool big = s.data_bytes() >= (1ll << 30);
        if (tty && !big) {
            std::printf("\r  [%zu/%zu] %-24s %10s  elapsed %s   ", k, total, s.id.c_str(),
                        human((double)s.data_bytes()).c_str(), human_time(el).c_str());
            std::fflush(stdout);
        } else if (big || k == total || k % 50 == 0) {
            std::printf("%s  [%zu/%zu] %-24s %10s  %s  %s in %s\n", tty ? "\r" : "", k, total, s.id.c_str(),
                        human((double)s.data_bytes()).c_str(), s.stored ? "written" : "generated+hashed",
                        dtype_name(s.dtype), human_time(sec).c_str());
        }
    };
    std::string err;
    if (write_dataset(plan, out, prog, err) != 0) {
        std::fprintf(stderr, "\nerror: %s\n", err.c_str());
        return 1;
    }
    const double total = secs_since(t0);
    std::printf("%sDone in %s: wrote %s (%.0f MB/s), generated %.2f G elements (%.0f M elements/s overall)\n",
                tty ? "\n" : "", human_time(total).c_str(), human((double)stored).c_str(),
                (double)stored / total / 1e6, (double)gen_elems / 1e9, (double)gen_elems / total / 1e6);
    std::printf("Dataset: %s\n", out.c_str());
    return 0;
}

static std::vector<OperandSpec> load_specs(const std::string& dir, Json& d) {
    d = json_parse(read_text(dir + "/dataset.json"));
    std::vector<OperandSpec> ops;
    for (size_t i = 0; i < d["operands"].size(); ++i)
        ops.push_back(spec_from_json(json_parse(read_text(dir + "/operands/" + d["operands"][i].str() + ".json"))));
    return ops;
}

static int cmd_verify(Args& a) {
    if (a.pos.empty()) throw std::runtime_error("usage: matgen verify DIR");
    const std::string dir = a.pos[0];
    const int regen = std::stoi(a.get("regen", "2"));
    const int threads = std::stoi(a.get("threads", "0"));
    a.check_unused();
    Json d;
    auto ops = load_specs(dir, d);
    std::mt19937_64 rng(1234);
    int64_t tiles_hashed = 0, tiles_regen = 0, failures = 0;
    std::vector<uint8_t> buf, gen;
    const auto t0 = Clock::now();
    for (auto& s : ops) {
        if (s.tile_hash.empty()) continue;
        const std::string base = dir + "/operands/" + s.id;
        int fd = -1;
        if (s.stored) {
            fd = open((base + ".bin").c_str(), O_RDONLY);
            if (fd < 0) { std::printf("FAIL %s: missing data file\n", s.id.c_str()); ++failures; continue; }
            struct stat st;
            fstat(fd, &st);
            if (st.st_size != s.data_bytes()) { std::printf("FAIL %s: size %lld, expected %lld\n", s.id.c_str(), (long long)st.st_size, (long long)s.data_bytes()); ++failures; close(fd); continue; }
        }
        // Tiles to regenerate: all for spec-only operands, else first, last and random ones.
        std::set<int64_t> regen_tiles;
        const int64_t nt = s.num_tiles();
        if (!s.stored) for (int64_t t = 0; t < nt; ++t) regen_tiles.insert(t);
        else if (regen > 0) {
            regen_tiles.insert(0);
            regen_tiles.insert(nt - 1);
            for (int i = 2; i < regen; ++i) regen_tiles.insert((int64_t)(rng() % (uint64_t)nt));
        }
        bool ok = true;
        for (int64_t t = 0; t < nt && ok; ++t) {
            const int64_t r0 = t * s.tile_rows, n = std::min(s.tile_rows, s.rows - r0);
            const size_t bytes = (size_t)n * s.row_bytes();
            buf.resize(bytes);
            if (s.stored) {
                if (pread(fd, buf.data(), bytes, (off_t)(r0 * (int64_t)s.row_bytes())) != (ssize_t)bytes) ok = false;
                else if (xxh64(buf.data(), bytes) != s.tile_hash[(size_t)t]) { std::printf("FAIL %s tile %lld: checksum\n", s.id.c_str(), (long long)t); ok = false; }
                ++tiles_hashed;
            }
            if (ok && regen_tiles.count(t)) {
                gen.resize(bytes);
                std::vector<float> sc((size_t)n * s.scales_per_row());
                generate_rows_parallel(s, r0, n, gen.data(), sc.empty() ? nullptr : sc.data(), threads);
                if (s.stored ? std::memcmp(gen.data(), buf.data(), bytes) != 0
                             : xxh64(gen.data(), bytes) != s.tile_hash[(size_t)t]) {
                    std::printf("FAIL %s tile %lld: regenerated bytes differ\n", s.id.c_str(), (long long)t);
                    ok = false;
                }
                ++tiles_regen;
            }
        }
        if (fd >= 0) close(fd);
        if (ok && s.stored && s.scales_per_row()) {
            std::vector<float> sc((size_t)s.rows * s.scales_per_row());
            std::ifstream f(base + ".scales.bin", std::ios::binary);
            f.read((char*)sc.data(), (std::streamsize)(sc.size() * 4));
            if (!f || xxh64(sc.data(), sc.size() * 4) != s.scales_hash) { std::printf("FAIL %s: scales checksum\n", s.id.c_str()); ok = false; }
        }
        if (!ok) ++failures;
    }
    std::printf("%s: %zu operands, %lld stored tiles checksummed, %lld tiles regenerated and compared, %lld failures (%s)\n",
                failures ? "FAILED" : "OK", ops.size(), (long long)tiles_hashed, (long long)tiles_regen,
                (long long)failures, human_time(secs_since(t0)).c_str());
    return failures ? 1 : 0;
}

static int cmd_bench(Args& a) {
    int threads = std::stoi(a.get("threads", "0"));
    if (threads <= 0) threads = default_threads();
    const int64_t elements = std::stoll(a.get("elements", "268435456"));
    a.check_unused();
    struct B { const char* name; DType dt; Dist dist; Scale sc; };
    const B cases[] = {
        {"activation laplace+outliers -> f16", DType::F16, Dist::LaplaceOutlier, Scale::None},
        {"activation laplace+outliers -> bf16", DType::BF16, Dist::LaplaceOutlier, Scale::None},
        {"weight normal -> f16", DType::F16, Dist::Normal, Scale::None},
        {"weight normal -> i8 (row scales)", DType::I8, Dist::Normal, Scale::Row},
        {"weight normal -> i4 (group-128 scales)", DType::I4, Dist::Normal, Scale::Group},
        {"weight normal -> f8e4m3 (tensor scale)", DType::F8E4M3, Dist::Normal, Scale::Tensor},
        {"raw integers -> i8 (RNG only)", DType::I8, Dist::UniformInt, Scale::None},
    };
    CpuInfo cpu = detect_cpu();
    std::printf("CPU: %s, %d threads; %lld elements per run (4096 columns)\n\n", cpu.model.c_str(), cpu.threads,
                (long long)elements);
    std::printf("%-40s %14s %14s %12s\n", "", "1 thread", "all threads", "output");
    for (const B& b : cases) {
        OperandSpec s;
        s.id = "bench"; s.cols = 4096; s.rows = elements / 4096;
        s.dtype = b.dt; s.dist = b.dist; s.scale = b.sc;
        s.group_size = b.sc == Scale::Group ? 128 : 0;
        s.stddev = b.dist == Dist::Normal ? 0.015625f : 1.0f;
        s.outlier_frac = 0.005f; s.outlier_mult = 50.0f;
        s.finalize();
        std::vector<uint8_t> data((size_t)s.data_bytes());
        std::vector<float> sc((size_t)s.rows * s.scales_per_row() + 1);
        std::memset(data.data(), 0, data.size());
        const int64_t rows1 = std::max<int64_t>(1, s.rows / std::max(1, threads));
        auto t = Clock::now();
        generate_rows(s, 0, rows1, data.data(), sc.data());
        const double r1 = (double)(rows1 * s.cols) / secs_since(t);
        t = Clock::now();
        generate_rows_parallel(s, 0, s.rows, data.data(), sc.data(), threads);
        const double rn = (double)s.elements() / secs_since(t);
        const double bpe = (double)s.row_bytes() / (double)s.cols;
        std::printf("%-40s %9.0f M/s %9.0f M/s %7.2f GB/s\n", b.name, r1 / 1e6, rn / 1e6, rn * bpe / 1e9);
    }
    std::printf("\nM/s = million elements generated per second, in memory (no disk).\n");
    return 0;
}

static int cmd_info(Args& a) {
    if (a.pos.empty()) throw std::runtime_error("usage: matgen info DIR");
    a.check_unused();
    Json d;
    auto ops = load_specs(a.pos[0], d);
    const Json& p = d["params"];
    std::printf("Profile %s (H=%lld L=%lld V=%lld T=%lld), mode %s, 16-bit %s, created %s\n", p["profile"].str().c_str(),
                (long long)p["H"].i64(), (long long)p["L"].i64(), (long long)p["V"].i64(), (long long)p["T"].i64(),
                d["mode"].str().c_str(), d["half_format"].str().c_str(), d["created"].str().c_str());
    std::map<std::string, std::tuple<int, int64_t, int64_t>> g;  // count, bytes, stored bytes
    for (auto& s : ops) {
        std::string k = s.cls + " " + dtype_name(s.dtype) + " " + std::to_string(s.rows) + "x" + std::to_string(s.cols);
        auto& e = g[k];
        std::get<0>(e)++;
        std::get<1>(e) += s.data_bytes() + s.scales_bytes();
        if (s.stored) std::get<2>(e) += s.data_bytes() + s.scales_bytes();
    }
    std::printf("%-34s %6s %12s %12s\n", "class dtype shape", "count", "size", "on disk");
    for (auto& kv : g)
        std::printf("%-34s %6d %12s %12s\n", kv.first.c_str(), std::get<0>(kv.second),
                    human((double)std::get<1>(kv.second)).c_str(), human((double)std::get<2>(kv.second)).c_str());
    Json c = json_parse(read_text(a.pos[0] + "/cases.json"));
    std::printf("GEMM cases: %zu\n", c["cases"].size());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2 || !std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help")) { usage(); return argc < 2; }
    const std::string cmd = argv[1];
    try {
        Args a(argc, argv, 2, {"dry-run", "force", "no-hash-spec"});
        if (cmd == "generate") return cmd_generate(a);
        if (cmd == "verify") return cmd_verify(a);
        if (cmd == "bench") return cmd_bench(a);
        if (cmd == "info") return cmd_info(a);
        if (cmd == "hw") return cmd_hw();
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
