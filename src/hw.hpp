// CPU capability detection (Linux).
#pragma once

#include <cstdint>
#include <string>

#include "json.hpp"

namespace mg {

struct CpuInfo {
    std::string model;
    std::string arch;
    int threads = 0;
    int64_t mem_total_bytes = 0, mem_available_bytes = 0;
    bool bf16 = false;       // native BF16 arithmetic (AVX512_BF16, AMX-BF16 or Arm BF16)
    bool fp16 = false;       // native FP16 arithmetic (AVX512_FP16 or Arm FP16)
    bool f16c = false;       // FP16 <-> FP32 conversion only
    bool int8_dot = false;   // VNNI, AMX-INT8, Arm dot product / I8MM
    bool avx2 = false, avx512 = false, amx = false;
};

CpuInfo detect_cpu();
Json cpu_to_json(const CpuInfo& c);

}  // namespace mg
