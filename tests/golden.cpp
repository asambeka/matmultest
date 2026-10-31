// Prints checksums of generated data for every format. Built twice, once
// vectorized for the build CPU and once without vectorization for a baseline
// CPU; the outputs must be identical (see tests/compare.cmake).
#include <cstdio>
#include <vector>

#include "core.hpp"
#include "spec.hpp"

using namespace mg;

int main() {
    struct C { DType dt; Dist dist; Scale sc; int64_t cols; };
    const C cases[] = {
        {DType::F16, Dist::LaplaceOutlier, Scale::None, 4099},
        {DType::BF16, Dist::LaplaceOutlier, Scale::None, 4096},
        {DType::F16, Dist::Normal, Scale::None, 4096},
        {DType::F8E4M3, Dist::Normal, Scale::Tensor, 4096},
        {DType::F8E5M2, Dist::Normal, Scale::Tensor, 4097},
        {DType::I8, Dist::Normal, Scale::Row, 4096},
        {DType::I4, Dist::Normal, Scale::Group, 4096},
        {DType::I8, Dist::UniformInt, Scale::None, 4096},
        {DType::I4, Dist::UniformInt, Scale::None, 4096},
    };
    for (const C& c : cases) {
        OperandSpec s;
        s.id = std::string("golden.") + dtype_name(c.dt);
        s.rows = 257; s.cols = c.cols; s.dtype = c.dt; s.dist = c.dist; s.scale = c.sc;
        s.group_size = c.sc == Scale::Group ? 128 : 0;
        s.stddev = c.dist == Dist::Normal ? 0.015625f : 1.0f;
        s.outlier_frac = 0.005f; s.outlier_mult = 50.0f;
        s.seed = 42;
        s.finalize();
        std::vector<uint8_t> d((size_t)s.data_bytes());
        std::vector<float> sc((size_t)s.rows * s.scales_per_row() + 1);
        generate_rows(s, 0, s.rows, d.data(), sc.data());
        std::printf("%-8s %-16s %-7s data %016llx scales %016llx\n", dtype_name(c.dt), dist_name(c.dist),
                    scale_name(c.sc), (unsigned long long)xxh64(d.data(), d.size()),
                    (unsigned long long)xxh64(sc.data(), (sc.size() - 1) * 4));
    }
    return 0;
}
