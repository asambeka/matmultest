# MatGen

Generates LLM-like test data for measuring matrix multiplication (GEMM)
performance on CPUs. Every matrix is a reproducible function of a small spec,
so large matrices can be regenerated in memory instead of being stored.

Design and plan: [PROJECT_SCOPE_PLAN_STRATEGY.md](PROJECT_SCOPE_PLAN_STRATEGY.md).

## Build

Needs GCC ≥ 9 or Clang ≥ 10 and CMake ≥ 3.16. No other dependencies.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
(cd build && ctest --output-on-failure)   # unit tests + vectorized-vs-baseline byte check
```

## Generate

```bash
# 8B profile: INT8 and INT4 weights, INT8 and 16-bit LM head; weights stored, activation regenerated
./build/matgen generate --profile 8b --weights i8,i4 --lmhead i8,auto16 --out data/8b

# 40B profile, INT8 weights
./build/matgen generate --profile 40b --weights i8 --lmhead i8,auto16 --out data/40b

# Show sizes and estimated time without writing anything
./build/matgen generate --profile 40b --weights i8,i4 --dry-run
```

| Option | Meaning |
|---|---|
| `--profile 8b\|40b\|70b` | Hidden size H = 4096 / 8192 / 16384, layers L = 32 / 60 / 80 |
| `--set H=..,L=..,V=..,T=..` | Override profile parameters |
| `--T N` | Activation rows (tokens), default 1,024,000 |
| `--weights LIST` | `i8`, `i4`, `f8e4m3`, `f8e5m2`, `f16`, `bf16`, `i8-raw`, `i4-raw` |
| `--lmhead LIST` | Same formats plus `auto16` |
| `--act auto16\|f16\|bf16` | Activation format; `auto16` picks BF16 if the CPU supports it |
| `--mode hybrid\|materialize\|spec` | `hybrid` (default) stores weights only; `materialize` also stores the activation; `spec` stores nothing but manifests |
| `--seed N`, `--threads N` | Seed (default 42) and thread count (default: all) |

Other commands: `matgen verify DIR`, `matgen info DIR`, `matgen bench`, `matgen hw`.

## What a dataset contains

The classic scenario has 7 × L projection GEMMs per weight format plus one
LM-head GEMM per LM-head format. Each GEMM is `C = A · Bᵀ`, where A is the
activation (T × H) and B is a weight stored as N × K row-major.

```
data/8b/
├── dataset.json            profile, parameters, CPU, operand list
├── cases.json              every GEMM: operands, M/N/K, FLOPs, bytes, intensity
└── operands/
    ├── w.i8.L00.q.json     manifest: shape, format, distribution, seed, checksums, stats
    ├── w.i8.L00.q.bin      data
    ├── w.i8.L00.q.scales.bin
    ├── ...
    └── x.act.json          activation: manifest only in hybrid mode (regenerated on load)
```

File format: row-major, little-endian, no header. INT4 is two values per
byte, low nibble first, signed. Quantized integers have float32 scales
(`real = q × scale`): one per row for INT8, one per 128-element group for
INT4. FP8 weights use one tensor scale given in the manifest.

## Read from C or C++

```c
#include "matgen.h"

char err[256];
mg_dataset* ds = mg_open("data/8b", err, sizeof err);

mg_operand_info w;
mg_operand(ds, "w.i8.L00.q", &w);
void*  wdata   = mg_alloc((size_t)w.rows * w.row_bytes);          // 2 MiB aligned, pre-faulted
float* wscales = malloc(sizeof(float) * w.rows * w.scales_per_row);
mg_load_rows(ds, w.id, 0, w.rows, wdata, wscales, 0);             // from disk

mg_operand_info x;
mg_operand(ds, "x.act", &x);
void* chunk = mg_alloc((size_t)262144 * x.row_bytes);
mg_load_rows(ds, x.id, 0, 262144, chunk, NULL, 0);                // regenerated in memory, ~2.7 GB/s
```

Link with `build/libmatgen.a` and `-lpthread` using a C++ linker. See
[examples/read_example.c](examples/read_example.c).

## Reproducibility

The same spec and seed give identical bytes on any machine, thread count or
SIMD width: Philox4x32-10 counter-based RNG, no platform math library in
generation, no FMA contraction, no fast-math. `matgen verify` checks stored
files against their checksums and regenerates tiles to compare them.
