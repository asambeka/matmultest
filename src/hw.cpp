#include "hw.hpp"

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#include <sys/utsname.h>

namespace mg {

CpuInfo detect_cpu() {
    CpuInfo c;
    c.threads = (int)std::thread::hardware_concurrency();
    struct utsname un;
    if (uname(&un) == 0) c.arch = un.machine;

    std::set<std::string> flags;
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string k = line.substr(0, colon), v = line.substr(colon + 1);
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
        if (c.model.empty() && (k == "model name" || k == "Model")) {
            c.model = v.substr(v.find_first_not_of(' '));
        }
        if (flags.empty() && (k == "flags" || k == "Features")) {
            std::istringstream is(v);
            std::string t;
            while (is >> t) flags.insert(t);
        }
    }
    auto has = [&](const char* x) { return flags.count(x) > 0; };
    c.avx2 = has("avx2");
    c.avx512 = has("avx512f");
    c.amx = has("amx_tile");
    c.f16c = has("f16c") || has("fphp");
    c.bf16 = has("avx512_bf16") || has("amx_bf16") || has("bf16");
    c.fp16 = has("avx512_fp16") || has("amx_fp16") || has("asimdhp");
    c.int8_dot = has("avx512_vnni") || has("avx_vnni") || has("amx_int8") || has("asimddp") || has("i8mm");

    std::ifstream m("/proc/meminfo");
    while (std::getline(m, line)) {
        long long kb = 0;
        if (sscanf(line.c_str(), "MemTotal: %lld kB", &kb) == 1) c.mem_total_bytes = kb * 1024;
        if (sscanf(line.c_str(), "MemAvailable: %lld kB", &kb) == 1) c.mem_available_bytes = kb * 1024;
    }
    return c;
}

Json cpu_to_json(const CpuInfo& c) {
    Json j = Json::object();
    j.set("model", c.model).set("arch", c.arch).set("threads", c.threads);
    j.set("mem_total_bytes", c.mem_total_bytes);
    Json f = Json::object();
    f.set("bf16", c.bf16).set("fp16", c.fp16).set("f16c", c.f16c).set("int8_dot", c.int8_dot);
    f.set("avx2", c.avx2).set("avx512", c.avx512).set("amx", c.amx);
    j.set("features", f);
    return j;
}

}  // namespace mg
