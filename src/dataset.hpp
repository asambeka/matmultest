// Dataset planning (profiles, classic scenario), writing and loading.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "spec.hpp"

namespace mg {

struct Profile {
    std::string name;
    int64_t H = 0, L = 0, V = 128256, T = 1024000;
};

// Built-in profiles: 8b, 40b (alias 30b), 70b (alias 70b+).
bool find_profile(const std::string& name, Profile& p);

struct BuildConfig {
    Profile profile;
    std::vector<std::string> weights = {"i8"};  // i8, i4, f8e4m3, f8e5m2, f16, bf16, i8-raw, i4-raw
    std::vector<std::string> lmhead = {"auto16"};
    std::string act = "auto16";                // auto16, f16, bf16
    std::string mode = "hybrid";               // hybrid, materialize, spec
    uint64_t seed = 42;
    int threads = 0;
    bool hash_spec_operands = true;            // hash regenerated operands too (enables verify)
};

struct CaseDef {
    std::string id, group, a, b;
    int64_t M = 0, N = 0, K = 0;
};

struct Plan {
    BuildConfig cfg;
    DType half = DType::F16;  // resolved 16-bit format
    std::vector<OperandSpec> operands;
    std::vector<CaseDef> cases;
    int64_t stored_bytes() const;
};

// Expands the classic scenario for a profile: 7*L square weights per weight
// format, one LM head per LM-head format, one shared activation.
Plan plan_classic(const BuildConfig& cfg, bool cpu_has_bf16);

struct Progress {
    std::function<void(size_t index, size_t total, const OperandSpec& s, double seconds)> on_operand;
};

// Generates and writes the dataset. Returns 0 on success.
int write_dataset(Plan& plan, const std::string& out_dir, const Progress& progress, std::string& err);

double case_flops(const CaseDef& c);
double case_bytes_min(const CaseDef& c, const OperandSpec& a, const OperandSpec& b);

}  // namespace mg
