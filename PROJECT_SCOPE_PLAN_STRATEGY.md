# MatGen: LLM-Like Test Data for Matrix Multiplication Performance Testing

**Project Scope, Plan and Strategy**

| | |
|---|---|
| **Document status** | Draft v0.6 for review |
| **Date** | 2026-10-31 |
| **Owner** | Ashwin Ambekar |
| **Working name** | `matgen` |

### Revision history

| Version | Changes |
|---|---|
| v0.1 | Initial draft |
| v0.2 | Class C width follows the model profile (4096 / 8192 / 16384). Profiles renamed 8B / 30–40B / 70B+. 4-bit is INT4 only; 8-bit is FP8 and INT8; 16-bit is BF16 with FP16 fallback. Class A gains 16384. |
| v0.3 | **Purpose clarified:** the data is for testing matrix multiplication *algorithm performance*, not for testing LLMs. Added the Test Case Model (§5): dimensionality parameters, a use-case scenario catalog, dimension sweeps, performance-relevant data properties, layout controls, performance metadata and test suites. The original Class A/B/C shapes become one built-in scenario. Verification reframed as a correctness guard for performance runs. Plan, risks and open questions updated. |
| v0.4 | **Profile-driven builds:** a dataset is built for the profile chosen at input (`8b`, `40b`, `70b` or a custom profile), and every shape is derived from that profile's parameters, so Class A size is a profile setting rather than a fixed choice (resolves the 16384-vs-12288 question). **Memory compression** is a priority data property, with new compressibility controls and measurement (§5.6). **Target is CPU:** CPU format support and auto-detection, CPU-specific layouts, NUMA and huge-page staging, RAM working-set and runtime planning, and a C/C++ loader; GPU backend removed. |
| v0.5 | **Generator implemented in C++17** (SIMD, multithreaded) as a library plus CLI; the same library links into C/C++ benchmark harnesses to regenerate data in memory. Python becomes optional bindings. Added starter-dataset sizes and generation-time estimates for `8b` and `40b` (§6.5). |
| v0.6 | **First implementation (matgen 0.1.0) built and measured.** §6.5 now has measured generation speeds and dataset times on the development machine, and lists what v0.1.0 implements and how it differs from this plan. |

---

## Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Problem Statement](#2-background-and-problem-statement)
3. [Project Scope](#3-project-scope)
4. [Requirements](#4-requirements)
5. [Test Case Model](#5-test-case-model)
6. [Sizing Analysis](#6-sizing-analysis)
7. [Technical Strategy](#7-technical-strategy)
8. [Architecture](#8-architecture)
9. [Project Plan](#9-project-plan)
10. [Validation and Quality Strategy](#10-validation-and-quality-strategy)
11. [Risks and Mitigations](#11-risks-and-mitigations)
12. [Decisions, Assumptions and Open Questions](#12-decisions-assumptions-and-open-questions)
13. [Glossary](#13-glossary)

---

## 1. Executive Summary

MatGen generates test data for **measuring the performance of matrix multiplication (GEMM) algorithms**. The data is shaped and distributed like the matrices found in large language model (LLM) workloads, but MatGen does not test LLMs and does not reproduce any particular model. "LLM-like" means:

- **Dimensions** like real LLM matmuls: square projections, wide vocabulary projections, very tall activation matrices, skinny decode GEMMs, batched attention GEMMs, grouped mixture-of-experts GEMMs, low-rank adapters, and the transposed shapes of training.
- **Number formats** used for LLMs: INT4, INT8, FP8 and BF16 (FP16 for hardware without BF16).
- **Value statistics** like real tensors: weight-like distributions, activation outlier channels, quantization scales and sparsity. These matter for performance even in dense GEMM, because they affect power draw and clock speed, subnormal slow paths, sparsity shortcuts, memory compression and dequantization cost (§2.2).

The core mechanism is a **parametric test case generator**. Users describe the dimensionality (hidden size, token count, batch, heads, experts, rank and so on) and choose a **use-case scenario** (prefill, decode, attention, MoE, training backward and others). MatGen expands that into concrete GEMM test cases, sweeps dimensions when asked, applies the selected data properties and layouts, and attaches performance metadata (FLOPs, bytes, arithmetic intensity) to each case.

**Every build is driven by its input.** The user picks a profile (`8b`, `40b`, `70b`, or a custom one) and MatGen builds only that profile's dataset. Every shape is derived from the profile's parameters, so changing one value (for example the hidden size) changes all related matrices consistently.

**The target is CPU.** The algorithms under test run on CPUs, so MatGen detects the CPU's format support (e.g. whether BF16 is available), offers CPU-oriented layouts, stages data with NUMA and huge-page awareness, checks that each case fits in RAM, and provides a C/C++ loader for native benchmark harnesses. **Memory compression** is a priority: MatGen can target, and always measures, how compressible each matrix is.

Scale is the main engineering problem. A single activation matrix can take 125 GiB, and one 70B+-like step contains 561 GEMMs. Every matrix is therefore a **reproducible function of a small specification**, regenerable exactly, tile by tile, on any machine. Writing to disk is optional.

---

## 2. Background and Problem Statement

### 2.1 Context

GEMM implementations (vendor libraries, custom kernels, compilers, new algorithms, accelerators) are tuned and compared at the shapes and precisions that dominate LLM workloads. Benchmarks that use only square matrices of random or zero data miss the regimes that matter in practice: skinny memory-bound decode GEMMs, long-K weight-gradient GEMMs, uneven grouped MoE GEMMs, dimensions that don't divide the kernel tile size, and low-precision formats that need dequantization.

### 2.2 Why the data values matter for performance

Dense GEMM performs the same number of operations whatever the values are, but measured performance still depends on the data:

| Effect | Why it changes measured performance | MatGen control |
|---|---|---|
| **Power and clock throttling** | Bit switching activity in multipliers and data paths depends on the values. All-zero or constant matrices draw much less power than realistic data, so sustained clocks and reported throughput look better than they are. | Realistic distributions by default; explicit low/high toggle-density patterns |
| **Subnormal slow paths** | Many CPUs and some accelerators process subnormal floats much more slowly unless flush-to-zero is on. | Subnormal density control |
| **Sparsity** | Sparse or structured-sparse kernels (e.g. 2:4) and zero-skipping hardware get faster with more zeros. | Unstructured, 2:4, block and row/column sparsity |
| **Memory compression** (priority) | When data is compressed in memory (by hardware, the operating system or the algorithm itself), well-compressing data moves faster and takes less space. Random data hardly compresses, constant data compresses almost completely, and real LLM tensors are in between. | Compressibility targets and measurement (§5.6) |
| **Dequantization cost** | INT4/INT8/FP8 kernels pay for scales and zero points; cost depends on group size and layout. | Scale granularity, zero points, scale layout |
| **Outlier handling** | Mixed-precision decomposition algorithms take separate paths for outlier channels. | Outlier fraction and magnitude |
| **Special values** | NaN and Inf can trigger exception or slow paths. | Special-value injection |
| **Algorithm structure** | Fast or approximate algorithms (Strassen-like, low-rank, randomized) behave differently on low-rank or structured inputs. | Rank and structure controls |

### 2.3 How the baseline matmul counts map to model structure

The baseline per-step matmul counts match a decoder-only transformer with 7 weight matmuls per layer (Q, K, V, O, gate, up, down) plus one output (LM-head) projection:

| Profile | Layers (L) | 7 × L + 1 | GEMMs per step |
|---|---|---|---|
| 8B     | 32 | 7 × 32 + 1 | **225** |
| 30–40B | 60 | 7 × 60 + 1 | **421** |
| 70B+   | 80 | 7 × 80 + 1 | **561** |

### 2.4 Baseline matrix classes and profiles

The originally specified shapes are kept as the built-in **`classic`** scenario (§5.3):

| Class | Shape | LLM-like role | Formats |
|---|---|---|---|
| **A** | N × N, N ∈ {4096, 8192, 16384}; 12288 standalone | Square hidden-to-hidden weights | INT4; FP8, INT8 |
| **B** | 128256 × H, H ∈ {4096, 8192, 16384}; 5120 standalone | Vocabulary embedding / LM-head weights | FP8, INT8; BF16 (FP16 fallback) |
| **C** | T × H, T ∈ {1,024,000; 2,048,000; 4,096,000} | Activations: T tokens with hidden size H | BF16 (FP16 fallback) |

| Profile | Hidden size H | Class A | Class B | Class C | GEMMs / step |
|---|---|---|---|---|---|
| **8B**     | 4,096  | 4096²  | 128256 × 4096  | T × 4096  | 225 |
| **30–40B** | 8,192  | 8192²  | 128256 × 8192  | T × 8192  | 421 |
| **70B+**   | 16,384 | 16384² | 128256 × 16384 | T × 16384 | 561 |

Profiles are named size points. They provide default dimensionality parameters (§5.2) and do not try to match real models exactly. A build targets the profile(s) selected at input, and every class is derived from the profile's hidden size H, so Class A, B and C always have compatible shapes. Other sizes, such as Class A 12288 or Class B × 5120, are obtained by overriding H or by defining a custom profile (§5.2).

### 2.5 Problem statement

> Engineers measuring GEMM algorithm performance need a fast, reproducible and storage-efficient way to produce test data whose dimensions, formats and value statistics resemble LLM workloads, for many use cases and dimension ranges, with the metadata needed to interpret the results and a way to confirm the computation was correct.

---

## 3. Project Scope

### 3.1 Objectives

| # | Objective | Success measure |
|---|---|---|
| O1 | Generate LLM-like GEMM test cases for any dimensionality and use case | Every scenario in §5.3 expands correctly for all three profiles and for custom parameters |
| O2 | Control the data properties that affect performance | Every property in §5.5 is configurable, measured and reported in the manifest |
| O3 | Support dimension sweeps for performance curves | Grid, range, log-spaced and alignment-stress sweeps produce valid, deduplicated case lists |
| O4 | Attach performance metadata to every case | FLOPs, minimum bytes, arithmetic intensity and expected regime present and correct |
| O5 | Be bit-exactly reproducible | Same spec and seed give identical bytes regardless of machine, worker count or tile order |
| O6 | Run in bounded memory at any scale | Peak generator RAM of 4 GiB or less for any matrix, including 125 GiB activations |
| O7 | Let performance runs confirm correctness cheaply | Every case has an exact, sampled or probabilistic check that costs much less than the GEMM |

### 3.2 In scope

- **Test case model**: GEMM case definitions, dimensionality parameters, use-case scenarios, sweeps and named suites (§5).
- **Profile-driven builds**: select `8b`, `40b`, `70b` or a custom profile at input; override any parameter.
- **CPU target support**: CPU capability detection, CPU-oriented layouts (including pre-packed layouts for VNNI/AMX-style kernels), NUMA- and huge-page-aware staging, RAM and runtime planning, and a C/C++ loader.
- **Memory compressibility**: target compression ratios, control how compressibility is produced, and measure it per block with reference codecs (§5.6).
- **Number formats**: INT4 (signed, packed, optional group scales); FP8 (E4M3, E5M2) and INT8 (optional scales); BF16 by default and FP16 when the target lacks BF16.
- **Data properties**: value distributions, outliers, compressibility, sparsity patterns, subnormal and special-value density, toggle density, rank and structure.
- **Layout controls**: row/column major, transposes, leading-dimension padding, alignment offsets, batched layouts, INT4 packing and scale layouts.
- **Performance metadata** per case and a roofline classification for a user-supplied hardware description.
- **Benchmark staging support**: rotating operand copies to defeat cache reuse, and pre-generation so generation never runs inside a timed region.
- **Output modes**: materialize to disk (sharded), stream tiles, or emit specs only.
- **Correctness guards**: exact integer references, sampled references, Freivalds' checks and error metrics for approximate algorithms.
- **C++ generator library and CLI** (optional Python bindings), documentation and an automated test suite.

### 3.3 Out of scope

- Testing LLMs, or matching any specific real model's architecture.
- Timing GEMMs or providing the benchmark harness. MatGen produces data and metadata; the user's harness runs and times the algorithms. (A minimal reference runner for smoke tests is optional, FR-22.)
- Real model weights, or training or fine-tuning models.
- FP4 and other 4-bit floating-point formats.
- GPU and accelerator targets, and GPU-accelerated generation. (The data formats are not CPU-specific, so other targets can be added later.)
- Implementing a memory compression scheme. MatGen controls and measures how compressible the data is; the compression itself belongs to the system under test.
- Non-GEMM operators (softmax, normalization, convolution).
- A GUI or web service.
- Distributed storage provisioning.

### 3.4 Stakeholders and users

| Role | Interest |
|---|---|
| GEMM algorithm / kernel developers | Shapes and data that expose real-world performance regimes |
| Performance engineers | Reproducible sweeps, performance metadata, fair benchmark staging |
| Hardware evaluation teams | Realistic power and data behavior at scale; format coverage |
| Numerics researchers | Control over distributions, scaling, rank and formats; error metrics |

### 3.5 Deliverables

1. `libmatgen` C++17 library and `matgen` CLI: generation, planning, loading and verification.
2. C API header (`matgen.h`) for loading operands and regenerating any tile in memory from C or C++ harnesses.
3. Scenario catalog, profile files (`8b`, `40b`, `70b`) and example suites.
4. Manifest and test-case schema specifications (the contract with benchmark harnesses).
5. Test suite: format conformance, determinism, data-property accuracy (including compressibility) and verification.
6. User guide covering profiles, scenarios, sweeps, data properties and CPU benchmark staging.
7. Benchmark report on MatGen's own generation throughput and memory usage.

---

## 4. Requirements

### 4.1 Functional requirements

**Test case generation**

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Define GEMM cases by (batch, M, N, K, transposes, formats, layout, data properties) and expand them from use-case scenarios (§5.3) | Must |
| FR-2 | Build for the profile(s) selected at input (`8b`, `40b`, `70b` or custom); derive every shape from profile parameters; allow any parameter to be overridden from the CLI or suite file (§5.2) | Must |
| FR-3 | Built-in `classic` scenario reproducing Class A/B/C shapes and 225 / 421 / 561 GEMMs per step | Must |
| FR-4 | Dimension sweeps: explicit list, linear range, log-spaced, grid, alignment-stress, random sampling within bounds | Must |
| FR-5 | Named, versioned test suites combining scenarios, sweeps and data properties | Must |
| FR-6 | Custom scenarios defined in configuration without code changes | Should |
| FR-7 | Size budget per suite (max bytes / max cases) with clear reporting of what was cut | Should |

**Data and formats**

| ID | Requirement | Priority |
|---|---|---|
| FR-8 | INT4, INT8, FP8 (E4M3, E5M2), BF16 and FP16 encoding with explicit rounding and saturation | Must |
| FR-9 | Integer quantization metadata: group / channel / tensor scales, optional zero points, or raw integers | Must |
| FR-10 | 16-bit format selection: BF16 when the target CPU supports it (auto-detected or declared), otherwise FP16 | Must |
| FR-11 | Value distributions and outlier channels (§5.5) | Must |
| FR-12 | Memory compressibility: target ratio, generation mechanism, block granularity, and per-block measurement with reference codecs (§5.6) | Must |
| FR-13 | Sparsity: unstructured, 2:4 structured, block, zero rows/columns | Should |
| FR-14 | Subnormal, special-value, toggle-density, rank and structure controls | Should |
| FR-15 | Layout controls: row/column major, transposes, leading-dimension padding, alignment offsets, batched layouts, CPU pre-packed layouts (§5.7) | Must |

**Performance support and output**

| ID | Requirement | Priority |
|---|---|---|
| FR-16 | Per-case performance metadata: FLOPs/OPs, minimum bytes, arithmetic intensity | Must |
| FR-17 | Roofline classification (memory- vs compute-bound), cache-fit and RAM-fit checks, and estimated runtime, from a CPU description (auto-detected, with user-supplied peak rates and bandwidth) | Should |
| FR-18 | Rotating operand copies (N distinct copies per operand) to defeat cache reuse between iterations | Must |
| FR-19 | Deterministic tiled generation; sharded output plus JSON manifests; streaming and spec-only modes | Must |
| FR-20 | Per-shard checksums and per-matrix measured statistics (including achieved sparsity, subnormal and outlier counts) | Must |
| FR-21 | Correctness guards: exact integer references, sampled references, Freivalds vectors, error metrics | Should |
| FR-22 | Minimal CPU reference GEMM (C++) for smoke-testing a suite | Could |
| FR-23 | Export to `.npy` and `safetensors` where the dtype is supported | Should |
| FR-24 | Resume interrupted generation | Should |
| FR-25 | CPU staging: NUMA placement (bind, interleave, first-touch per socket), huge pages, aligned buffers, page-cache-free loading | Must |
| FR-26 | C API that reads manifests, loads operands into aligned, NUMA-placed buffers, and regenerates any row range in memory (no disk) | Must |
| FR-27 | Split a case along M (`split_m`) when its working set exceeds available RAM, with metadata linking the parts | Should |
| FR-28 | Python bindings over the C API (NumPy arrays) | Could |

### 4.2 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Reproducibility**: bit-identical output across runs, machines and degrees of parallelism |
| NFR-2 | **Bounded memory**: generator memory use does not depend on matrix size |
| NFR-3 | **Throughput**: scales linearly with cores until storage is the bottleneck |
| NFR-4 | **64-bit safety**: every index and offset uses 64-bit integers (activations reach 6.7 × 10¹⁰ elements) |
| NFR-5 | **Portability**: Linux first, x86-64 and ARM64, C++17 with GCC or Clang; SIMD paths (AVX2, AVX-512, NEON) chosen at run time with a scalar fallback; generation never depends on the host supporting BF16, FP8 or INT4 natively |
| NFR-9 | **Generation speed**: in-memory generation limited by CPU and memory bandwidth, not by the implementation language (target ≥ 300 M elements/s per core with AVX2) |
| NFR-6 | **Auditability**: every case traces to a suite, scenario, parameters, seed and tool version |
| NFR-7 | **Atomicity**: no consumer ever sees a partially written shard as complete |
| NFR-8 | **Measurement neutrality**: MatGen never requires generation work inside a timed benchmark region |

---

## 5. Test Case Model

This section describes the mechanisms that generate dimension-specific and use-case-specific test data.

### 5.1 The GEMM test case

Every test case is one GEMM, `C = op(A) · op(B)`, described by:

| Field | Meaning |
|---|---|
| `batch` | Number of independent GEMMs of the same shape (1 for a plain GEMM) |
| `M`, `N`, `K` | Output rows, output columns, reduction dimension |
| `trans_a`, `trans_b` | Whether each operand is stored transposed |
| `format_a`, `format_b`, `format_out`, `accumulate` | Number formats, e.g. BF16 × INT4 → BF16 with FP32 accumulation |
| `layout` | Storage order, leading dimensions, alignment, pre-packing (§5.7) |
| `data_a`, `data_b` | Data property profiles for each operand (§5.5) |
| `group` | For grouped GEMMs (MoE): the list of per-group M values sharing N and K |
| `tags` | Scenario, profile, role (e.g. `decode`, `70b+`, `mlp.down`) |

Scenarios and sweeps produce lists of these cases. Benchmark harnesses only need to understand this one structure.

### 5.2 Dimensionality parameters

Scenarios are written in terms of named parameters rather than fixed numbers. Profiles supply defaults; users override any of them.

| Parameter | Meaning | `8b` | `40b` | `70b` |
|---|---|---|---|---|
| `H` | Hidden size | 4,096 | 8,192 | 16,384 |
| `L` | Layers | 32 | 60 | 80 |
| `I` | MLP intermediate size (default 3.5 × H) | 14,336 | 28,672 | 57,344 |
| `d_head` | Attention head dimension | 128 | 128 | 128 |
| `n_heads` | Query heads (H / d_head) | 32 | 64 | 128 |
| `n_kv` | Key/value heads (grouped-query attention) | 8 | 8 | 8 |
| `V` | Vocabulary size | 128,256 | 128,256 | 128,256 |
| `T` | Tokens per prefill GEMM (batch × sequence) | 1,024,000 – 4,096,000 | same | same |
| `B_dec` | Decode batch sizes | 1 – 256 | same | same |
| `S` | Sequence length for attention | 2,048 – 131,072 | same | same |
| `E`, `top_k`, `I_e` | Experts, experts per token, expert intermediate size | off by default | off | off |
| `r` | Low-rank adapter rank | 8 – 64 | same | same |

The `classic` scenario uses only `H`, `L`, `V` and `T`, with all weights square (H × H), matching the original specification.

**Selecting and customizing a profile.** The profile is the main input of a build. Profile IDs are `8b`, `40b` (alias `30b`) and `70b` (alias `70b+`); the document's labels 8B, 30–40B and 70B+ refer to the same profiles. Only the selected profiles are built.

```bash
matgen generate --builtin classic --profile 8b  --out ./data/8b      # 8B dataset only
matgen generate --builtin classic --profile 40b --out ./data/40b     # 40B dataset only
matgen generate --builtin classic --profile 70b --set H=12288 ...    # 70B with 12288 weights
matgen generate --builtin classic --profile ./profiles/my-55b.yaml   # custom profile
```

A custom profile is a small YAML file. Any parameter it does not set comes from the profile it extends:

```yaml
profile: my-55b
extends: 70b
H: 12288
L: 64
T: [1024000, 2048000]
```

Because every shape is derived from parameters, overriding `H` changes the Class A, B and C shapes together and keeps them compatible. The resolved parameters are recorded in every manifest.

### 5.3 Use-case scenario catalog

A scenario is a template that turns dimensionality parameters into GEMM cases. Built-in scenarios:

| Scenario | GEMMs generated (M × N × K) | Performance regime it exercises |
|---|---|---|
| **`classic`** | Per layer 7 × [T × H × H]; per step 1 × [T × V × H]. 225 / 421 / 561 per step | Large compute-bound GEMMs; original A/B/C specification |
| **`prefill`** | Q/O: T × H × H; K/V: T × (n_kv·d_head) × H; gate/up: T × I × H; down: T × H × I; LM head: T × V × H | Compute-bound, realistic non-square projections |
| **`decode`** | Same projections with M = B_dec (1 – 256) | Memory-bound skinny GEMMs and near-GEMV; weight bandwidth dominates |
| **`attention`** | Batched QKᵀ: batch = B·n_heads, S × S × d_head; PV: S × d_head × S | Batched small-K and large-K GEMMs |
| **`attention-decode`** | Batched per head: 1 × S × d_head and 1 × d_head × S (grouped by n_heads / n_kv) | Memory-bound batched GEMV over KV cache |
| **`moe`** | Grouped GEMMs: E groups of [m_e × I_e × H] and [m_e × H × I_e], with Σ m_e = T·top_k and m_e drawn from a token-routing distribution (uniform, Zipf-skewed, capacity-limited) | Uneven grouped GEMM, load imbalance, small-M tails |
| **`lm-head`** | T × V × H (prefill) and B_dec × V × H (decode) | Very wide N; large outputs |
| **`training-backward`** | For each forward Y = X·W: dX = dY·Wᵀ (M × K × N) and dW = Xᵀ·dY (K × N × M) | Transposed operands; long-K reductions (K = T) |
| **`lora`** | X·A_r: T × r × H and (X·A_r)·B_r: T × H × r | Extremely skinny inner or outer dimensions |
| **`square-baseline`** | N × N × N for the Class A sizes | Reference point for comparison with standard benchmarks |

Every scenario supports all formats in §7.3 and all data properties in §5.5. Custom scenarios are YAML templates with expressions over the parameters, for example:

```yaml
scenario: my-wide-proj
cases:
  - role: proj.wide
    M: T
    N: 4 * H
    K: H
    format_a: bf16
    format_b: int4
```

### 5.4 Dimension sweeps

Sweeps vary one or more parameters to produce performance curves.

| Sweep type | Example | Purpose |
|---|---|---|
| `list` | `B_dec: [1, 2, 4, 8, 16, 32, 64, 128, 256]` | Specific points of interest |
| `range` | `M: {start: 1024, stop: 65536, step: 1024}` | Linear scans |
| `log` | `T: {start: 1024, stop: 4096000, per_octave: 2}` | Wide ranges with even coverage on a log scale |
| `grid` | Cartesian product of several sweeps | Multi-dimensional performance surfaces |
| `alignment-stress` | Each dimension at multiples of 128, ±1, ±7, +64, and primes near them | Kernel tile-boundary and padding behavior |
| `aspect` | Fixed FLOPs, varying M:N:K ratios | Shape sensitivity independent of work |
| `random` | Seeded uniform or log-uniform samples within bounds | Broad coverage, regression testing |

Sweeps can be constrained (e.g. `M * N * K <= 2e12`, `bytes <= 64 GiB`) and are deduplicated. Each generated case records which sweep point produced it.

### 5.5 Data property controls

Each operand gets a **data profile**. Defaults are LLM-like; every property can be changed.

| Property | Options | Default (weights / activations) |
|---|---|---|
| **Distribution** | Normal, truncated normal, Laplace, uniform, Student-t (heavy tail), integer-uniform, constant | Normal σ = 1/√fan_in / Laplace |
| **Outlier channels** | Fraction of columns (0 – 5%), magnitude multiplier (1 – 100×), fixed or random positions | none / 0.5% at 50× |
| **Sparsity** | Fraction of zeros; pattern: unstructured, 2:4 (or N:M), block (b × b), zero rows, zero columns | 0% / 0% |
| **Subnormal density** | Fraction of values placed in the subnormal range (float formats) | 0% |
| **Special values** | Density of NaN, ±Inf, ±0, max-finite (where the format allows) | 0% |
| **Toggle density** | Low (repeated values), natural, high (alternating bit patterns) | natural |
| **Compressibility** | Natural (measured only) or a target ratio with a chosen mechanism (§5.6) | natural |
| **Rank / structure** | Full rank, rank-r, identity, diagonal, banded, small integers | full rank |
| **Quantization** | Scale granularity (tensor / channel / group of 32, 64, 128, 256), symmetric or asymmetric, scale format | group 128 (INT4), channel (INT8) |

Named presets bundle common combinations:

| Preset | Meaning |
|---|---|
| `llm-weights` | Normal weights quantized per channel or group (the default for weights) |
| `llm-activations` | Laplace activations with outlier channels (the default for activations) |
| `relu-sparse` | Activations with 50–90% zeros, as after ReLU-family activations |
| `pruned-2of4` | Weights with 2:4 structured sparsity |
| `max-power` | High toggle density, full entropy, for power and thermal worst cases |
| `min-power` | Low toggle density, highly compressible (shows best-case clocks) |
| `numerics-stress` | Subnormals, special values and boundary magnitudes |
| `low-rank` | Rank-r matrices for fast and approximate algorithm testing |
| `incompressible` | Full-entropy data; worst case for memory compression |
| `highly-compressible` | Data that compresses at a high target ratio (e.g. 4×) |
| `compress-mixed` | A mix of compressible and incompressible blocks |

After generation, MatGen **measures** each property (achieved sparsity, outlier count, subnormal count, compression ratio) and records it in the manifest, so results can be related to the data actually used.

### 5.6 Memory compressibility controls

Memory compression is a priority for this project, so compressibility gets its own controls. Compression can come from the operating system (e.g. zswap or zram when memory is under pressure), from hardware compression engines on some server CPUs, or from the GEMM algorithm itself storing operands compressed. In every case, performance depends on how well the data compresses at the scheme's block size. §12.3 Q2 asks which of these applies; the controls below cover all of them.

**Why LLM-like data compresses in a particular way.** LLM tensors cluster near zero, so BF16 values use a narrow band of exponents: the exponent bits are highly predictable while the low mantissa bits look random. Real tensors therefore compress partially. Plain random data in a wide range compresses very little, and constant or zero data compresses almost completely. Both extremes give misleading compression benchmarks, which is why the default is `natural`: LLM-like values, with compressibility measured rather than forced.

**Controls**

| Control | Options | Notes |
|---|---|---|
| `target_ratio` | `natural`, or a ratio such as 1.0× – 8× | Achieved ratio is reported next to the target |
| `codec` | `lz4`, `zstd:<level>`, `entropy` (codec-independent byte-plane entropy bound), or a user-supplied measurement hook | The hook lets users measure with the same codec as their system |
| `block_size` | 64 B (cache line), 4 KiB (page), 64 KiB, or whole shard | Must match the granularity of the compression scheme being tested |
| `mechanism` | `exponent-spread` (narrow or widen the value range), `mantissa-truncation` (zero low mantissa bits), `quant-levels` (fewer distinct values), `zero-runs` (zeros in runs), `repeated-blocks` (duplicate blocks) | Each mechanism changes the data differently; choose the one that best fits the scheme under test |
| `spatial_mix` | `uniform`, or a fraction of compressible vs incompressible blocks, arranged `random`, `banded` or `per-row` | Tests schemes whose cost varies per block |
| `measure_sample` | Fraction of blocks measured for very large operands (default 100% up to 4 GiB, then 1%) | Keeps measurement time reasonable |

**How a target is reached.** MatGen searches the chosen mechanism's parameter on a deterministic sample of tiles until the measured ratio is within tolerance (default ±5%), then applies that parameter to the whole matrix. Generation stays reproducible because the parameter is recorded in the manifest.

**What is measured and recorded** for every operand, whether or not a target was set:

- compression ratio per block: mean, 5th / 50th / 95th percentile, histogram;
- fraction of all-zero blocks and of duplicate blocks;
- byte-plane entropy (e.g. BF16 high byte vs low byte), which shows *where* the redundancy is;
- codec name, version, level and block size used.

**Trade-off.** Forcing a compression ratio changes the value statistics (e.g. truncating mantissas reduces precision). The manifest records both the data profile and the measured statistics so these effects stay visible.

### 5.7 Layout and memory controls

| Control | Options | Why it matters |
|---|---|---|
| Storage order | Row-major, column-major, per operand | Kernel variants and transpose paths |
| Transposes | `trans_a`, `trans_b` | NN / NT / TN / TT kernel selection |
| Leading dimension | Tight, padded to a multiple, padded by a fixed amount | Strided access; padding behavior |
| Base alignment | 64 B (cache line) by default; 16 B to 2 MiB, or deliberately misaligned offsets | SIMD load paths; split cache-line loads |
| Leading-dimension aliasing | Leading dimensions that are large powers of two (e.g. 4096 × element size) vs padded | Cache-set conflicts on CPUs, a common hidden slowdown |
| Batched layout | Strided batch (one buffer) or array of separate buffers | Batched GEMM API variants |
| INT4 packing | Low or high nibble first; packing along K or along N | Matches the kernel's expected packing |
| Scale layout | Row-major or column-major scale tensors; interleaved with data or separate | Dequantization access pattern |
| CPU pre-packed layouts | K-interleaved layouts as consumed by dot-product instructions: groups of 4 along K for INT8 (VNNI-style), groups of 2 for BF16, plus blocked panels (e.g. 16 × 64 tiles) for matrix-extension (AMX-style) kernels | Lets kernels that expect pre-packed weights be tested without a packing step; can be compared with packing inside the timed region |

### 5.8 Performance metadata

Every case carries metadata computed from its definition, so benchmark results can be interpreted without recomputing them:

| Field | Formula / meaning |
|---|---|
| `flops` | 2 · M · N · K · batch (called OPs for integer formats) |
| `bytes_min` | Each operand and the output read or written once, including scale and zero-point tensors |
| `arithmetic_intensity` | flops / bytes_min |
| `regime` | `memory-bound` or `compute-bound`, given a CPU description (peak FLOP/s per format and memory bandwidth) |
| `roofline_bound` | Maximum achievable FLOP/s for this case on that CPU |
| `working_set_bytes` | Inputs + output + scales + all rotating copies; checked against RAM per NUMA node and in total |
| `fits_in` | Smallest level that holds the case's operands: `L2`, `L3`, `NUMA node`, `RAM`, or `exceeds RAM` |
| `est_time_at_peak` | flops / peak FLOP/s, a lower bound on runtime for planning |
| `compression` | Measured compression ratios of each operand (§5.6) |

Examples (BF16 activations, 8-bit weights unless noted, BF16 output):

| Case | M × N × K | FLOPs | Min bytes | Intensity |
|---|---|---|---|---|
| Decode, 70B+ projection | 1 × 16384 × 16384 | 0.54 G | 0.25 GiB | **2** (memory-bound) |
| Decode, batch 64, 70B+ | 64 × 16384 × 16384 | 34 G | 0.25 GiB | 126 |
| Decode, batch 256, 30–40B, INT4 weights | 256 × 8192 × 8192 | 34 G | 0.04 GiB | 819 |
| Decode LM head, batch 32, 30–40B (BF16) | 32 × 128256 × 8192 | 67 G | 1.97 GiB | 32 |
| Attention QKᵀ, S = 8192, 64 heads (BF16) | 64 × [8192 × 8192 × 128] | 1.1 T | 8.25 GiB | 124 |
| Prefill, 2048 tokens, 30–40B | 2048 × 8192 × 8192 | 275 G | 0.13 GiB | 2,048 |
| MLP up, 8192 tokens, 30–40B | 8192 × 28672 × 8192 | 3.8 T | 0.78 GiB | 4,588 |
| Weight gradient, long K (BF16 in, FP32 out) | 8192 × 8192 × 1,024,000 | 137 T | 31.5 GiB | 4,063 |
| Prefill, 1.024M tokens, 70B+ | 1,024,000 × 16384 × 16384 | 550 T | 62.8 GiB | 8,159 |

Intensity ranges over four orders of magnitude across these use cases. This is why a benchmark built only on large square matrices does not describe LLM-like performance. On CPUs, whose memory bandwidth is low relative to their compute, most decode cases are firmly memory-bound, which is also where memory compression has the most effect.

### 5.9 Test suites

A **suite** is a named, versioned YAML file combining profiles, scenarios, sweeps, formats, data profiles, layouts and benchmark staging. It is the unit users create, share and rerun.

```yaml
suite: decode-int4-sweep
version: 1
seed: 42
target: { half_format: auto, hardware: hw/my-cpu.yaml }   # auto: BF16 if the CPU supports it
profiles: [8b]                                            # build only the 8B dataset
overrides: { }                                            # e.g. { H: 12288 }
scenario: decode
sweep:
  B_dec: { type: list, values: [1, 2, 4, 8, 16, 32, 64, 128, 256] }
formats: { activations: bf16, weights: int4, out: bf16, accumulate: fp32 }
data:
  activations: llm-activations
  weights: { preset: llm-weights, quant: { granularity: group, group_size: 128 },
             compression: { target_ratio: natural, codec: lz4, block_size: 4KiB } }
layout: { weights: { order: col_major, int4_packing: low_nibble_first, alignment: 64 } }
staging: { rotate_copies: 4, numa: interleave, huge_pages: 2MiB }
output: { mode: hybrid, budget: 500GiB }
verify: [freivalds, sampled]
```

Built-in example suites (each takes `--profile`): `classic`, `decode-sweep`, `prefill-sweep`, `compression-sweep` (the same GEMM at compression ratios from 1× to 8×), `attention-long-context`, `moe-imbalance`, `training-backward`, `alignment-stress`, `format-matrix` (every format pairing on one shape), `power-worst-case`.

---

## 6. Sizing Analysis

1 MiB = 2²⁰ bytes, 1 GiB = 2³⁰ bytes, 1 TiB = 2⁴⁰ bytes. The figures below are for the `classic` scenario, which contains the largest matrices. Other scenarios use the same mechanisms and are usually much smaller (decode and attention cases are MiB to a few GiB).

### 6.1 Per-matrix size

**Class A (N × N)**

| N | Profile | Elements | INT4* | FP8 / INT8 |
|---|---|---|---|---|
| 4,096  | 8B         | 16,777,216  | 8 MiB   | 16 MiB  |
| 8,192  | 30–40B     | 67,108,864  | 32 MiB  | 64 MiB  |
| 12,288 | standalone | 150,994,944 | 72 MiB  | 144 MiB |
| 16,384 | 70B+       | 268,435,456 | 128 MiB | 256 MiB |

\*Plus scales if scaled: with BF16 scales per 128-element group, about 3% of the INT4 data.

**Class B (128,256 × H)**

| H | Profile | Elements | FP8 / INT8 | BF16 / FP16 |
|---|---|---|---|---|
| 4,096  | 8B         | 525,336,576   | 501 MiB    | 0.98 GiB |
| 5,120  | standalone | 656,670,720   | 626.25 MiB | 1.22 GiB |
| 8,192  | 30–40B     | 1,050,673,152 | 1,002 MiB  | 1.96 GiB |
| 16,384 | 70B+       | 2,101,346,304 | 2,004 MiB  | 3.91 GiB |

128256 × 16384 has 2.10 × 10⁹ elements, just under the signed 32-bit limit; byte offsets overflow 32 bits.

**Class C (T × H, BF16/FP16)**

| T (tokens) | H = 4,096 | H = 8,192 | H = 16,384 |
|---|---|---|---|
| 1,024,000 | 7.81 GiB  | 15.63 GiB | 31.25 GiB |
| 2,048,000 | 15.63 GiB | 31.25 GiB | 62.5 GiB  |
| 4,096,000 | 31.25 GiB | 62.5 GiB  | **125 GiB** |

### 6.2 Large outputs

The LM-head output (T × 128256, BF16) is 244.6 GiB at T = 1.024M and 978.5 GiB at T = 4.096M, regardless of H. Full reference outputs are therefore not practical for the largest cases; §10.2 describes cheaper checks.

### 6.3 Per-step footprint if every operand were stored (`classic`)

| Profile | 7·L Class A weights (8-bit) | Class B (BF16) | Activations, T = 1.024M | Activations, T = 4.096M |
|---|---|---|---|---|
| 8B     | 3.5 GiB  | 0.98 GiB | 1.72 TiB  | 6.87 TiB  |
| 30–40B | 26.3 GiB | 1.96 GiB | 6.42 TiB  | 25.7 TiB  |
| 70B+   | 140 GiB  | 3.91 GiB | **17.1 TiB** | **68.5 TiB** |

Storing every operand is impractical as the default, so generation is treated as computation rather than storage (§7). Rotating copies for benchmark staging (§7.6) multiply the stored size of the operands they apply to; the size budget (FR-7) accounts for this.

### 6.4 CPU RAM and runtime planning

On a CPU, a case's working set must fit in RAM, ideally within one NUMA node, and large cases take a long time to run. The planner reports both before anything is generated.

**Largest single GEMM working set (`classic`, activation in + output + 8-bit weight):**

| Profile | T = 1.024M | T = 4.096M |
|---|---|---|
| 8B  | 15.6 GiB | 62.5 GiB |
| 40B | 31.3 GiB | 125 GiB |
| 70B | 62.8 GiB | **250 GiB** |

The LM-head case is larger still because of its output (up to 978.5 GiB, §6.2), and exceeds the RAM of most servers. The planner flags such cases and can split them along M (`split_m`, FR-27) into parts that fit.

**Compute per classic step and lower-bound runtime** at an assumed sustained rate (illustrative; actual CPU rates come from the CPU description):

| Profile | T | FLOPs per step | at 10 TFLOP/s | at 50 TFLOP/s |
|---|---|---|---|---|
| 8B  | 1.024M | 8.8 P  | 0.24 h | 0.05 h |
| 8B  | 4.096M | 35.1 P | 0.97 h | 0.19 h |
| 40B | 1.024M | 59.9 P | 1.7 h  | 0.33 h |
| 40B | 4.096M | 240 P  | 6.7 h  | 1.3 h  |
| 70B | 1.024M | 312 P  | 8.7 h  | 1.7 h  |
| 70B | 4.096M | 1,249 P | 34.7 h | 6.9 h |

Full classic steps at the largest token counts are therefore long-running soak tests on CPUs. For routine performance work, the `prefill-sweep` and `decode-sweep` suites with smaller T give performance curves in minutes, and the planner's `est_time_at_peak` lets users fit a suite to a time budget.

### 6.5 Starter datasets and generation time (`8b`, `40b`)

**Measured** with matgen 0.1.0 on the development machine: Intel Core i5-9400 (6 cores, AVX2/FMA/F16C, **no BF16**, so 16-bit data is FP16), 14 GiB RAM, SATA SSD. Raw disk write speed: 95 MB/s (2 GiB with `fdatasync`).

**In-memory generation speed** (`matgen bench`, million elements per second):

| Data | 1 thread | 6 threads | Output rate |
|---|---|---|---|
| Activation, Laplace + outliers → FP16 | 257 | 1,321 | 2.6 GB/s |
| Activation, Laplace + outliers → BF16 | 273 | 1,448 | 2.9 GB/s |
| Weight, normal → FP16 | 166 | 821 | 1.6 GB/s |
| Weight, normal → INT8 (row scales) | 143 | 712 | 0.7 GB/s |
| Weight, normal → INT4 (group-128 scales) | 141 | 719 | 0.36 GB/s |
| Weight, normal → FP8 E4M3 (tensor scale) | 160 | 830 | 0.8 GB/s |
| Random-number generator only (raw INT8) | 412 | 2,013 | 2.0 GB/s |

This is below the earlier estimate of 2.4 G elements/s. The random-number generator alone reaches 2.0 G/s; the distribution transforms take the rest of the time. Further SIMD work could raise it (§9, Phase 7), but on this machine disk writes are the bottleneck for stored data anyway.

**Starter datasets (classic scenario, T = 1.024M, `hybrid` mode: weights stored, activation regenerated):**

| Dataset | Contents | On disk | Time to generate and write | Verify |
|---|---|---|---|---|
| `8b` | 448 weights (224 INT8 + 224 INT4), LM head INT8 + FP16, 1 activation 1,024,000 × 4096 | 6.83 GiB | **1.7 min** (74 MB/s average) | 32 s |
| `40b` | 420 INT8 weights, LM head INT8 + FP16, 1 activation 1,024,000 × 8192 | 29.2 GiB | **6.4 min** (81 MB/s average) | 1.7 min |

**One activation matrix, ways of getting it into memory:**

| Activation | Write to disk (95 MB/s) | Regenerate in memory via the C API (measured 2.7 GB/s) |
|---|---|---|
| `8b`, T = 1.024M (7.8 GiB) | ~88 s | **~3 s** |
| `40b`, T = 1.024M (15.6 GiB) | ~177 s | **~6 s** |

**Conclusions for this machine:**

- Writing to disk is the bottleneck for stored data. Average write speed including file creation and `fsync` was 74–81 MB/s.
- With the C++ library linked into the harness, **regenerating an activation (2.7 GB/s) is several times faster than reading it from this SATA SSD**, and needs no disk space. The recommended setup is to store the weights (small) and regenerate activations in memory (`hybrid` mode).
- RAM, not disk, limits the size of a single run: a classic GEMM at T = 1.024M needs 15.6 GiB (`8b`) or 31.3 GiB (`40b`), more than the ~10 GiB free. Activations are processed in row chunks: 262,144 rows for `8b` and 131,072 rows for `40b`, each about 4 GiB for input plus output. The LM-head output is processed in 16,384-row chunks (4 GiB each).

**What matgen 0.1.0 implements, and differences from this plan:**

- Implemented: `classic` scenario for `8b`, `40b`, `70b` and `--set` overrides; INT4, INT8, FP8 E4M3/E5M2, BF16 and FP16 (auto-selected from CPU features); quantized and raw integer modes; normal weights and Laplace activations with outlier channels; `hybrid`, `materialize` and `spec` modes; per-tile XXH64 checksums; `generate`, `verify`, `bench`, `info`, `hw` commands; C API (`include/matgen.h`) with load, in-memory regeneration and huge-page allocation.
- Verified: Philox and XXH64 known-answer tests; FP16 encoder exhaustive over all 2³² inputs against the CPU's F16C instructions; FP8 encoders against brute-force search; normal transform monotonic over all 2²⁴ inputs; byte-identical output across thread counts, row splits, and vectorized (AVX2) vs non-vectorized baseline builds.
- Differences: scales are stored as FP32 (the plan's default was BF16); each matrix is one file rather than 1–4 GiB shards; no external libraries are used yet (manifests use a built-in JSON reader/writer; suites, YAML, compression measurement, NUMA placement and other scenarios are later phases).

---

## 7. Technical Strategy

### 7.1 Guiding principles

1. **Describe, then expand.** Users describe dimensionality and use case; MatGen expands them into concrete cases.
2. **Specify, don't store.** A matrix is defined by a small spec. Its bytes can always be recomputed.
3. **Tile everything.** No code path holds a whole large matrix in memory.
4. **Counter-based randomness.** Any tile can be generated independently and in any order.
5. **Formats and data properties are first-class.** They are implemented explicitly, measured after generation and recorded.
6. **Stay out of the timed region.** Generation happens before measurement; the harness only reads prepared data.

### 7.2 Deterministic generation

- **RNG:** a counter-based generator (Philox-4x32-10), implemented in C++ with scalar and SIMD (AVX2, AVX-512, NEON) paths that produce identical bits. Tile *i* is generated by setting the counter directly. Output is checked against the published Philox test vectors.
- **Bit-exact floating point:** normal and other distributions are produced without platform math libraries (whose `log`/`cos` results differ between systems and between scalar and vector code). Transforms use fixed polynomial or table-based methods with a fixed order of operations and explicit FMA, and the code is built without fast-math options. This keeps output identical across compilers, CPUs and SIMD widths.
- **Seed derivation:** `key = H(suite_seed, case_id, operand, copy_index)`, with H a stable hash such as BLAKE2b truncated to 64 bits. Tile *t* starts at counter offset `t × tile_elements`.
- **Guarantee:** output depends only on the spec and the tool version, not on worker count, scheduling or hardware.
- **Versioning:** the generator algorithm version is recorded in every manifest and bumped whenever output bytes would change.
- **Structured properties** (2:4 sparsity, outlier channel positions, low rank) are derived from the same keyed streams, so they are reproducible too. Low-rank matrices are generated as products of two thin factors, tile by tile.

### 7.3 Number formats

| Width | Format | Encoding | Range | Scaling | Implementation |
|---|---|---|---|---|---|
| 4-bit | **INT4** | Signed two's complement, 2 per byte | −8 … 7 | Optional per-group scale (default group 128 along K), optional zero point | C++ quantize + pack |
| 8-bit | **INT8** | Signed two's complement | −128 … 127 (−127 … 127 symmetric) | Optional per-channel or per-tensor scale, optional zero point | `int8_t` |
| 8-bit | **FP8 E4M3** | 1-4-3, "fn" variant | max 448, **no Inf** | Optional per-tensor FP32 scale | C++ bit-level encoder |
| 8-bit | **FP8 E5M2** | 1-5-2 | max 57344, has Inf/NaN | Optional per-tensor FP32 scale | C++ bit-level encoder |
| 16-bit | **BF16** (default) | 1-8-7 | max ≈ 3.39 × 10³⁸ | none | C++ round-to-nearest-even from FP32 (AVX-512 BF16 instructions when present) |
| 16-bit | **FP16** (fallback) | IEEE binary16, 1-5-10 | max 65504 | none | F16C / NEON conversion, scalar fallback |

All encoders are tested exhaustively (every FP32 input class and every INT4/INT8/FP8 code point) against the `ml_dtypes` Python package as an independent reference, used only in the test suite.

Design decisions:

- **16-bit selection** is a property of the target CPU. `half_format: auto | bf16 | fp16` defaults to `auto`, which reads the target CPU's capabilities (§7.8): BF16 if supported, FP16 otherwise. When the data is generated on a different machine from the one running the benchmark, a CPU description file for the target is supplied instead of detecting the local CPU. The choice is recorded in each manifest.
- **FP16 fallback clamps** distributions and outliers to FP16's range; saturation counts are reported.
- **Integer generation modes:** `quantized` (FP32 values → scales → rounded and clamped integers plus a scale sidecar) and `raw` (integers drawn directly; used for pure integer GEMMs and exact verification).
- **Floating-point generation:** FP32 values converted with explicit rounding (round-to-nearest-even by default, stochastic optional) and explicit overflow handling.
- **Integer accumulation** fits INT32 at every profile size: INT8 products are at most 2¹⁴ and K ≤ 2¹⁴ in the profile weights, giving at most 2²⁸. The long-K `training-backward` scenario (K = T up to 4,096,000 ≈ 2²²) can exceed INT32 with INT8 inputs; the composer flags these cases, and `numerics-stress` can target accumulator overflow on purpose.

### 7.4 Scenario and sweep engine

- Scenarios and sweeps are data (YAML templates with arithmetic expressions over named parameters). Expressions are evaluated with a small, safe expression evaluator (no arbitrary code).
- Expansion is a pure function: suite file → ordered list of cases with stable IDs. The case list can be produced and inspected (`matgen expand`) before any data is generated.
- The engine validates every case: positive dimensions, compatible inner dimensions, supported format pairings, alignment and packing constraints (e.g. INT4 packing needs an even dimension along the packing axis; group scales need K divisible by the group size or an explicit padding rule).
- **Operand deduplication:** cases that share an operand spec (e.g. the same weight matrix across a decode batch-size sweep) reference one generated matrix instead of producing copies.

### 7.5 Output modes

| Mode | Description | When to use |
|---|---|---|
| `spec` | Manifests only; the harness regenerates data by calling `libmatgen` | Very large suites; the fastest option when the harness links the library (§6.5) |
| `stream` | Tiles delivered to a callback (C API) or a pipe | Loading data straight into the harness's memory without touching disk |
| `materialize` | Sharded binary files plus manifests | Fixed datasets; harnesses that can't link the library; archival |
| `hybrid` | Materialize small operands; spec or stream the large ones | **Recommended default** for large suites |

### 7.6 Benchmark staging support

Performance results are only meaningful if the data is staged fairly. MatGen supports this without running the benchmark itself:

- **Rotating copies.** `rotate_copies: N` generates N distinct versions of an operand (different seeds, same properties). The harness cycles through them between iterations, so the operand can't stay resident in caches. This matters most on CPUs with large last-level caches (hundreds of MiB on some server parts), which can hold an entire 8B decode weight matrix. MatGen picks a default N so the rotation bytes exceed the detected L3 size by a margin, and records the result.
- **Pre-generation.** All modes deliver complete operands before timing starts. The C API (`mg_load_operand`, `mg_generate_rows`) returns ready buffers; the streaming mode is for staging into memory, not for timed regions.
- **NUMA placement.** On multi-socket CPUs, where memory lives matters as much as how much there is. The loaders allocate buffers with an explicit policy: bound to one node, interleaved across nodes, or first-touched by the threads that will use them. The policy is recorded with the results.
- **Huge pages.** Large operands are loaded into 2 MiB (or 1 GiB) huge pages when available, so TLB misses don't distort the measurement. MatGen reports when huge pages were requested but not granted.
- **No page-cache effects.** Mapping shard files directly can turn the first iterations into disk reads, and leaves data in the OS page cache. The loaders copy into anonymous, aligned, pre-faulted memory by default; direct mapping is an explicit option.
- **Warm-up guidance.** The manifest suggests warm-up iterations so first-run effects (page faults, frequency ramp-up) can be excluded.

### 7.7 Generator performance

- **Language:** C++17. Generation is CPU-bound in memory and I/O-bound on disk; native SIMD code removes the CPU bound as a practical limit (§6.5).
- **Parallelism:** a thread pool, one task per tile; tiles are independent. Threads can be pinned to cores and NUMA nodes so generated data is first-touched where it will be used.
- **SIMD:** fused per-tile pipeline (Philox → distribution → data properties → quantize/convert → pack) that keeps data in registers and cache, written once to the destination. AVX2, AVX-512 and NEON paths are selected at run time.
- **Tile size:** row blocks of about 4–64 MiB, sized to stay cache-friendly, aligned to quantization groups and sparsity patterns.
- **I/O:** preallocated shards, positional writes (`pwrite`), double-buffered so generation overlaps writing, `fsync`, atomic rename from `*.partial`. Optional `O_DIRECT` to avoid filling the page cache.
- **Shards:** 1–4 GiB each.
- **Capacity check:** total bytes computed before writing; generation refuses to start without enough free space unless forced.

### 7.8 CPU target considerations

**Format support on CPUs.** CPU support for LLM formats varies by generation, so MatGen detects what the target CPU has and records it with each build. Typical support:

| Format | x86-64 instruction support (examples) | Arm64 instruction support (examples) | Implication for test data |
|---|---|---|---|
| INT8 | AVX-512 VNNI, AVX-VNNI, AMX-INT8 | Dot product (SDOT/UDOT), I8MM | Native on most current server CPUs; pre-packed K-groups of 4 are useful |
| BF16 | AVX-512 BF16, AMX-BF16 | BF16 extension (BFDOT, BFMMLA) | `auto` picks BF16 when present, FP16 otherwise |
| FP16 | AVX-512 FP16, AMX-FP16 (newer parts); conversion-only (F16C) on older | FP16 arithmetic (Armv8.2-A and later) | Fallback format |
| FP8 | Little or no native arithmetic on most CPUs | Little or no native arithmetic | Kernels convert to a wider type; tests measure conversion cost |
| INT4 | No native arithmetic | No native arithmetic | Kernels unpack to INT8; packing layout matters |

**Capability detection.** `matgen hw detect` reads the CPU's feature flags (e.g. `avx512_bf16`, `amx_bf16`, `amx_int8`, `avx512_vnni`, `avx512_fp16` on x86; `bf16`, `i8mm`, `asimddp` on Arm), core and socket counts, NUMA layout, cache sizes and RAM per node from `/proc/cpuinfo` and `/sys`, and writes a CPU description file. Peak FLOP/s per format and memory bandwidth are added by the user (or from a separate bandwidth measurement), because they cannot be read reliably from the system.

**Denormals.** CPUs often handle subnormal floats far more slowly unless flush-to-zero is enabled. The `numerics-stress` preset and the subnormal density control make this measurable, and the manifest records the subnormal count.

**Accumulation.** CPU INT8 dot-product instructions accumulate into INT32, so the INT32 overflow flagging in §7.3 applies directly.

### 7.9 Technology stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | C++17 (GCC ≥ 9 or Clang ≥ 10) | Generation speed; links directly into C/C++ harnesses |
| Build | CMake | Standard; easy to embed in other projects |
| Public API | C header (`matgen.h`) over the C++ library | Callable from C, C++ and, through bindings, Python |
| SIMD | Intrinsics with run-time dispatch (AVX2, AVX-512, NEON) | One binary for many CPUs |
| Threads | `std::thread` pool; optional `libnuma` for placement | No heavy dependencies |
| Compression measurement | `liblz4`, `libzstd` | Standard reference codecs |
| CPU detection | `cpuid` / `/proc/cpuinfo`, `/sys/devices/system` | No extra dependencies |
| Config and manifests | YAML suites (`yaml-cpp`), JSON manifests (`nlohmann/json`, header-only) | Human-editable input, machine-readable output |
| CLI | C++ (`CLI11`, header-only) | Simple, self-documenting |
| Checksums | xxHash-64 (header-only) + optional SHA-256 | Speed and strength |
| Testing | GoogleTest; Python + `ml_dtypes` used only as an independent reference for format tests | Unit tests plus cross-checked format conformance |
| Optional bindings | `pybind11` | Python access if needed later |

---

## 8. Architecture

### 8.1 Component overview

```
     ┌──────────────────────────────────────────────────────────┐
     │      matgen CLI  /  C API (libmatgen, C++17)             │
     └───────────────────────────┬──────────────────────────────┘
                                 │  suite.yaml
     ┌───────────────────────────▼──────────────────────────────┐
     │  Suite Loader: selected profile(s) + overrides,          │
     │  target CPU description, data presets                    │
     └───────────────────────────┬──────────────────────────────┘
                                 │
     ┌───────────────────────────▼──────────────────────────────┐
     │  Scenario & Sweep Engine                                 │
     │  parameters → scenarios → sweeps → validated GEMM cases  │
     │  + performance metadata + operand deduplication          │
     └───────────────────────────┬──────────────────────────────┘
                                 │  OperandSpec[] (incl. rotating copies)
     ┌───────────────────────────▼──────────────────────────────┐
     │  Tile Scheduler: tiles, shards, budget                   │
     └───────────┬──────────────────────────────┬───────────────┘
                 │                              │
     ┌───────────▼───────────┐      ┌───────────▼───────────┐
     │ Generator Core        │ ...  │ Generator Core        │  (thread pool, SIMD)
     │ RNG → distribution →  │      │                       │
     │ properties (outliers, │      │                       │
     │ compressibility,      │      │                       │
     │ sparsity, rank) →     │      │                       │
     │ quantize → layout/pack│      │                       │
     └───────────┬───────────┘      └───────────┬───────────┘
                 │                              │
     ┌───────────▼──────────────────────────────▼───────────────┐
     │  Sink: disk shards | stream | spec-only                  │
     └───────────────────────────┬──────────────────────────────┘
                                 │
     ┌───────────────────────────▼──────────────────────────────┐
     │  Manifests, measured statistics (incl. compression),     │
     │  checksums, verification artifacts                       │
     └───────────────────────────┬──────────────────────────────┘
                                 │
     ┌───────────────────────────▼──────────────────────────────┐
     │  C API in the harness: load from disk or regenerate in   │
     │  memory into aligned, NUMA-placed, huge-page buffers     │
     └──────────────────────────────────────────────────────────┘
```

### 8.2 Core abstractions

| Abstraction | Responsibility |
|---|---|
| `Suite` | Top-level, versioned description of what to generate |
| `Profile` | Named set of dimensionality parameters (`8b`, `40b`, `70b` or custom), with overrides applied |
| `Scenario` | Template from parameters to GEMM cases |
| `Sweep` | Rule for varying parameters |
| `GemmCase` | One test case (§5.1) plus performance metadata |
| `OperandSpec` | Shape, format, layout, data profile, seed key, copy index |
| `DataProfile` | Distribution and data properties (§5.5), including compressibility (§5.6) |
| `Format` | Encode/decode, rounding, saturation, packing, pre-packed layouts, scales |
| `Target` | CPU description: ISA features (BF16, INT8 dot product, matrix extensions), sockets, NUMA nodes, cache sizes, RAM, peak rates, bandwidth |
| `Loader` | C API functions that load or regenerate operands into aligned, NUMA-placed memory |
| `Tile` | (operand, row range) → data bytes plus scale bytes |
| `Sink` | Destination for tiles |
| `Manifest` | Machine-readable record of cases and operands |

### 8.3 Example CLI

```bash
# Describe the target CPU (features, NUMA, caches, RAM); add peak rates by hand
matgen hw detect --out hw/my-cpu.yaml

# Show the cases a suite would produce, with sizes and metadata; nothing generated
matgen expand suites/decode-int4-sweep.yaml --profile 40b --table

# Size, RAM-fit and runtime estimate against budgets
matgen plan suites/decode-int4-sweep.yaml --profile 40b --hw hw/my-cpu.yaml

# Generate a suite for one profile
matgen generate suites/decode-int4-sweep.yaml --profile 40b --out ./data/decode-int4-40b

# Same GEMM at several compression ratios
matgen generate --builtin compression-sweep --profile 8b --codec lz4 --block 4KiB \
    --ratios 1,1.5,2,4,8 --out ./data/compress-8b

# Quick ad-hoc case without a suite file
matgen case --M 64 --N 16384 --K 16384 --a bf16 --b int8 \
    --data-a llm-activations --data-b llm-weights --rotate 4 --out ./data/adhoc

# The original specification: classic step for the 70B profile
matgen generate --builtin classic --profile 70b --mode hybrid --out ./data/classic70

# Check a consumer's output for one case
matgen verify ./data/decode-int4 --case decode.70b+.B64.q --output result.bin
```

### 8.4 Output layout

```
decode-int4/
├── suite.yaml                  # copy of the input suite
├── cases.json                  # expanded case list + performance metadata
├── operands/
│   ├── w.70b+.q.copy0.manifest.json
│   ├── w.70b+.q.copy0.shard-0000.bin
│   ├── w.70b+.q.copy0.scales.bin
│   ├── ...
│   └── x.70b+.B64.manifest.json
└── verify/
    └── decode.70b+.B64.q.freivalds.bin
```

### 8.5 Case and operand manifests (abridged)

```json
{
  "case_id": "decode.70b+.B64.q",
  "suite": "decode-int4-sweep@1",
  "tags": { "scenario": "decode", "profile": "70b+", "role": "attn.q", "sweep": { "B_dec": 64 } },
  "gemm": { "batch": 1, "M": 64, "N": 16384, "K": 16384, "trans_a": false, "trans_b": true },
  "formats": { "a": "bf16", "b": "int4", "out": "bf16", "accumulate": "fp32" },
  "operands": { "a": "x.70b+.B64", "b": ["w.70b+.q.copy0", "w.70b+.q.copy1", "w.70b+.q.copy2", "w.70b+.q.copy3"] },
  "perf": { "flops": 34359738368, "bytes_min": 142606336, "arithmetic_intensity": 240.9,
            "regime": "memory-bound", "rotation_bytes": 553648128 },
  "verify": { "methods": ["freivalds", "sampled"], "tolerance": { "rel": 0.01 } }
}
```

```json
{
  "operand_id": "w.70b+.q.copy0",
  "shape": [16384, 16384],
  "layout": { "order": "col_major", "ld": 16384, "alignment": 4096 },
  "format": { "name": "int4", "signed": true, "packing": "low_nibble_first",
              "quantization": { "mode": "quantized", "granularity": "group", "group_size": 128,
                                "group_axis": "K", "scale_dtype": "bf16" } },
  "data_profile": { "preset": "llm-weights", "distribution": "normal", "std": 0.0078125 },
  "seed": { "suite": 42, "derived_key": "0x9f3c1a7e5b20d814" },
  "measured": { "sparsity": 0.061, "outlier_channels": 0, "clamped": 36183,
                "compression": { "codec": "lz4-1.9.4", "block_size": 4096,
                                 "ratio_mean": 1.12, "ratio_p5": 1.09, "ratio_p95": 1.15,
                                 "zero_blocks": 0.0, "duplicate_blocks": 0.0 } },
  "shards": [ { "file": "w.70b+.q.copy0.shard-0000.bin", "bytes": 134217728, "xxh64": "…" } ]
}
```

---

## 9. Project Plan

Each phase ends at a usable state. The test case model comes early because it is what users interact with; the largest-scale work follows once the model is stable.

### Phase 0: Foundations

- Confirm the open questions in §12.
- Repository, packaging, CI, test framework.
- Schemas: suite, profile, case, operand manifest, CPU description.
- **Exit criteria:** open questions answered; CI green on a skeleton package.

### Phase 1: Formats and core generator

- C++ project skeleton (CMake, GoogleTest, CI), C API header.
- `Format` for INT4, INT8, FP8 (E4M3, E5M2), BF16 and FP16, including packing, scales and zero points.
- Philox-based tiled generation with seed derivation; scalar and AVX2 paths with bit-identical output.
- Basic distributions; `quantized` and `raw` integer modes.
- A first `matgen generate` that writes classic `8b` and `40b` matrices with manifests, so data generation can start at the end of this phase.
- **Exit criteria:** conformance tests pass against `ml_dtypes` and exhaustive INT4/INT8/FP8 code-point tables; Philox matches published test vectors; determinism across 1, 4 and 16 threads and across scalar/AVX2 paths; measured generation rate recorded and §6.5 estimates updated.

### Phase 2: Test case model

- `GemmCase`, profiles (`8b`, `40b`, `70b`), overrides and custom profiles.
- Scenario engine with `classic`, `prefill`, `decode` and `square-baseline`.
- Sweep engine: list, range, log, grid, alignment-stress.
- `matgen hw detect`; performance metadata, roofline, cache-fit, RAM-fit and runtime estimates.
- `matgen expand`, `matgen plan` and `matgen case`.
- **Exit criteria:** `classic` expands to exactly 225 / 421 / 561 GEMMs for `8b` / `40b` / `70b`; an `H` override changes all three classes consistently; metadata matches hand-computed values; every expanded case validates.

### Phase 3: Scale and output

- Tile scheduler, thread pool, sharded disk sink, resume, checksums.
- Stream, spec and hybrid modes; size budget; operand deduplication; rotating copies sized from the detected L3; `split_m`.
- C API loaders and in-memory regeneration with NUMA placement, huge pages and pre-faulted aligned buffers.
- **Exit criteria:** a 125 GiB activation is generated with bounded memory; the `8b` and `40b` starter datasets (§6.5) are produced in hybrid mode; interrupted runs resume correctly; the C API reads every format, regenerates row ranges bit-identically to the stored files, and reports the NUMA and huge-page placement actually obtained.

### Phase 4: Data properties and layouts

Memory compressibility is done first in this phase because it is the priority property.

- **Compressibility:** measurement (lz4, zstd, entropy, user hook) at 64 B / 4 KiB / 64 KiB blocks; target ratios through each mechanism; spatial mixes; presets; `compression-sweep` suite.
- Outliers, subnormals, special values, toggle density, rank and structure; sparsity patterns after these.
- Data presets; measured statistics in manifests.
- Layout controls: order, transposes, leading-dimension padding and aliasing, alignment, batched layouts, INT4 packing, scale layouts, CPU pre-packed layouts.
- **Exit criteria:** achieved compression ratios are within ±5% of target for every mechanism and block size; other measured properties match targets within tolerance; every layout option round-trips through both loaders.

### Phase 5: Remaining scenarios and suites

- `attention`, `attention-decode`, `moe` (with routing distributions), `lm-head`, `training-backward`, `lora`.
- Custom YAML scenarios.
- Built-in example suites (§5.9).
- **Exit criteria:** every built-in suite expands, plans and generates; custom scenario example works without code changes.

### Phase 6: Correctness guards

- Exact integer references; sampled FP32/FP64 references; Freivalds vectors; closed-form results for structured data.
- Error metrics (max abs, relative, ULP-based) for approximate or reduced-precision algorithms.
- `matgen verify`; optional C++ reference GEMM.
- **Exit criteria:** injected faults are detected (§10.3); clean outputs pass.

### Phase 7: Hardening and release

- Cross-platform checks on x86-64 and Arm64 CPUs; `.npy` / `safetensors` export.
- Documentation, user guide, generator benchmark report.
- **Exit criteria:** v1.0 tagged; documentation complete.

### Milestone summary

| Milestone | Delivered capability |
|---|---|
| M1 (Phase 1) | Correct, reproducible data in every format |
| M2 (Phase 2) | Users pick a profile, describe use cases and get case lists with performance, RAM and runtime metadata |
| M3 (Phase 3) | Any case at full scale, in bounded memory, loaded into CPU memory fairly (NUMA, huge pages, rotation) |
| M4 (Phase 5) | Compressibility controls first, then all data properties, layouts, scenarios and example suites |
| M5 (Phase 6) | Correctness checks for performance runs |
| M6 (Phase 7) | v1.0 release |

### Dependency view

```
Phase 0 ─► Phase 1 ─► Phase 2 ─► Phase 3 ─► Phase 5 ─► Phase 6 ─► Phase 7
                                   │           ▲
                                   └► Phase 4 ─┘
```

Phase 4 can run in parallel with Phase 5 once Phase 3 is complete. A first usable release for decode and prefill performance testing on CPUs is possible at M3, and compression testing becomes available early in Phase 4.

---

## 10. Validation and Quality Strategy

### 10.1 Testing the generator

| Test type | What it proves |
|---|---|
| **Format conformance** | Every INT4/INT8/FP8 code point encodes and decodes correctly; INT4 packing and sign extension; rounding and saturation; FP8 NaN handling; BF16/FP16 rounding from FP32 |
| **Quantization** | Scales and zero points reproduce source values within expected error; `raw` mode covers the full integer range |
| **Scenario expansion** | Every scenario produces the expected cases for known parameters (golden case lists); `classic` gives 225 / 421 / 561 |
| **Sweeps** | Correct points, constraints applied, duplicates removed, stable case IDs |
| **Performance metadata** | FLOPs, bytes and intensity match independent calculations |
| **Profiles** | Each profile builds only its own dataset; overrides and custom profiles produce consistent, compatible shapes |
| **Compressibility** | Achieved ratios within tolerance for each mechanism, codec and block size; measurement matches an independent run of the codec; `natural` data compresses between the incompressible and highly compressible presets |
| **Data properties** | Measured sparsity, outlier fraction, subnormal density and rank match targets within tolerance |
| **Layout** | Every order, padding, alignment, packing and pre-packed layout round-trips through the C API |
| **CPU detection** | Feature flags, NUMA layout and cache sizes match `lscpu` output on x86-64 and Arm64 test machines |
| **Determinism and tile independence** | Byte-identical output across runs, thread counts, tile orders, SIMD paths and CPU architectures; any tile alone equals the slice of the full matrix |
| **Scale / soak** | Largest matrices complete with bounded memory; resume works |

### 10.2 Correctness guards for performance runs

A fast result is only useful if it is correct. MatGen provides checks that cost much less than the GEMM itself:

1. **Exact integer.** INT4 × INT4 and INT8 × INT8 GEMMs in `raw` mode have exact integer results. Any mismatch is a bug.
2. **Exact structured.** Identity, diagonal, small-integer and low-rank inputs have closed-form products.
3. **Sampled reference.** Selected output tiles are computed in FP32/FP64 and compared with a tolerance based on K, the input formats (including dequantization) and the accumulator precision.
4. **Freivalds' algorithm (O(n²)).** For C = A·B, check A·(B·r) ≈ C·r for random vectors r. A few rounds make false acceptance very unlikely; with integer r and INT64 arithmetic the check is exact for integer GEMMs.
5. **Error metrics.** For fast or approximate algorithms, which may legitimately differ from the reference (e.g. Strassen-like methods or reduced-precision accumulation), MatGen reports max absolute error, relative error and ULP distance against the sampled reference, so accuracy can be reported next to performance.

### 10.3 Detection tests for the verifiers

Inject known faults into outputs (bit flips, a transposed tile, a dropped row block, swapped INT4 nibbles, wrong sign extension, wrong scale handling, a skipped MoE group) and confirm each verification level detects them.

---

## 11. Risks and Mitigations

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Benchmarks run on unrepresentative data (zeros, constants) and overstate performance | High | Medium | LLM-like defaults; `min-power` / `max-power` presets show the range; measured properties in manifests |
| R2 | Cache reuse makes small cases look faster than they are | High | High | Rotating copies with rotation-bytes metadata; staging guidance |
| R3 | Generation or I/O leaks into timed regions | High | Medium | Pre-generation design; `load_case` returns ready buffers; documentation |
| R4 | Storage exceeds what is available (up to ~68 TiB per classic 70B+ step) | High | High | Spec/stream/hybrid modes; deduplication; size budget; `matgen plan` |
| R5 | Scenario catalog misses a use case users need | Medium | Medium | Custom YAML scenarios; catalog extended based on feedback |
| R6 | Combinatorial explosion of sweeps × formats × properties | Medium | High | `matgen expand` preview; constraints; budgets; random sampling sweeps |
| R7 | Quantization or packing conventions differ from the kernel under test | High | Medium | All conventions configurable and recorded; confirm in Phase 0 |
| R8 | FP16 fallback saturates outliers that are valid in BF16 | Medium | Medium | Clamping with reported saturation counts |
| R9 | INT32 accumulator overflow in long-K integer cases | Medium | Medium | Composer flags these cases; documented |
| R10 | 32-bit index or offset overflow | High | Medium | 64-bit indexing everywhere; tests above 2³¹ and 2³² elements |
| R11 | Generator too slow for large suites | Medium | Medium | C++ SIMD generator, threads across all cores; deduplication; in-memory regeneration instead of disk |
| R18 | Bit-exactness breaks across compilers, SIMD widths or CPUs | High | Medium | No platform math library in generation; fixed operation order and explicit FMA; no fast-math; CI compares scalar, AVX2, AVX-512 and NEON output |
| R12 | Non-determinism across platforms | High | Low | Integer RNG path; explicit rounding; cross-platform CI |
| R13 | Reference codecs compress differently from the compression scheme under test | High | Medium | Block size and codec configurable; user measurement hook; byte-plane entropy as a codec-independent measure |
| R14 | Forcing compressibility distorts the LLM-like value statistics | Medium | Medium | `natural` default; choice of mechanism; both data profile and measured statistics recorded |
| R15 | Case working set exceeds RAM or a NUMA node | High | High | RAM-fit check in `matgen plan`; `split_m`; per-node placement policies |
| R16 | NUMA placement, page cache or missing huge pages distort CPU measurements | High | Medium | Loaders control placement, pre-fault memory and report the placement actually obtained |
| R17 | Full classic steps take hours to days on CPUs | Medium | High | Runtime estimates; time budgets; smaller-T sweep suites for routine use |

---

## 12. Decisions, Assumptions and Open Questions

### 12.1 Decisions

| # | Decision |
|---|---|
| D1 | Class C width follows the profile: 4096 (8B), 8192 (30–40B), 16384 (70B+). |
| D2 | Profiles are 8B, 30–40B and 70B+, with 225, 421 and 561 GEMMs per classic step. Exact model sizes are representative only. |
| D3 | 16-bit format is BF16 when the target supports it, otherwise FP16. |
| D4 | 4-bit format is INT4 only. |
| D5 | 8-bit formats are FP8 and INT8. |
| D6 | **Purpose:** the data is for testing GEMM algorithm performance, not LLMs. It must look like LLM data and be generated for any dimensionality and use case. |
| D7 | The original Class A/B/C specification is kept as the built-in `classic` scenario; other scenarios add realistic non-square, skinny, batched and grouped shapes. |
| D8 | **Builds are configurable by input profile** (`8b`, `40b`, `70b` or custom). All shapes are derived from the selected profile's parameters, which can be overridden. Defaults: H = 4096 / 8192 / 16384; other sizes such as 12288 are obtained by override or custom profile. |
| D9 | MatGen prepares data and metadata; timing is done by the user's benchmark harness. |
| D10 | **The target is CPU.** CPU capability detection, CPU layouts, NUMA/huge-page staging, RAM and runtime planning, and a C/C++ loader are in scope; GPU targets are out of scope. |
| D11 | **Memory compression is the priority data property.** Compressibility controls and measurement are Must-have and are built first among the data properties. |
| D12 | **The generator is written in C++17**, as a library (`libmatgen`) with a C API and a CLI. Benchmark harnesses link the library to load data or regenerate it in memory. Python is limited to optional bindings and test references. |
| D13 | **First datasets are `8b` and `40b`**; `70b` is deferred. |

### 12.2 Working assumptions

- **A1.** Class C rows are 1,024,000, 2,048,000 and 4,096,000 tokens in the `classic` scenario; other scenarios use their own token counts.
- **A2.** INT4 and INT8 are signed by default, with unsigned + zero point available.
- **A3.** INT4 defaults to symmetric group-128 scales (BF16); INT8 to symmetric per-channel scales.
- **A4.** FP8 defaults to E4M3 (fn variant), with E5M2 available.
- **A5.** Profile defaults in §5.2 (I = 3.5 × H, d_head = 128, n_kv = 8, V = 128,256) are acceptable as representative values.
- **A6.** Benchmark harnesses are C or C++ and link `libmatgen`, or read the raw little-endian files described by the JSON manifests directly.
- **A7.** Target CPUs run Linux on x86-64 or Arm64.
- **A8.** Scenario priority, unless changed: classic, decode, prefill, attention, MoE, training-backward, LoRA.

### 12.3 Open questions

| # | Question | Why it matters |
|---|---|---|
| Q1 | Is the scenario priority in A8 right? | Phase 2 and Phase 5 priorities |
| Q2 | Which kind of memory compression is being tested: operating-system level (zswap/zram), a hardware compression engine, or compression inside the GEMM algorithm? And at what block size? | Default codec, block size and mechanism (§5.6) |
| Q3 | Which CPUs will be tested (vendor, generation, sockets, RAM per node)? | Format defaults, pre-packed layouts, RAM-fit limits |
| Q4 | Is Python access to the data needed at all, or is C/C++ enough? | Whether to build the Python bindings (FR-28) |
| Q5 | Do the kernels under test expect specific INT4 packing, pre-packed layouts or quantization conventions? | Bit-level compatibility |
| Q6 | Can operands be shared between cases, and how many rotating copies are needed? | Storage and RAM budget |
| Q7 | What storage capacity is available for materialized suites? | Default modes and budgets |
| Q8 | Is a minimal reference runner wanted (FR-22), or will the harness be entirely in-house? | Scope of Phase 6 |

---

## 13. Glossary

| Term | Definition |
|---|---|
| **GEMM** | General matrix multiplication, C = op(A) · op(B) |
| **M, N, K** | Output rows, output columns and reduction (inner) dimension of a GEMM |
| **Arithmetic intensity** | FLOPs per byte of memory traffic; decides whether a GEMM is memory- or compute-bound |
| **Roofline** | Performance bound given by min(peak FLOP/s, intensity × bandwidth) |
| **Scenario** | Template that turns dimensionality parameters into GEMM cases for one use case |
| **Sweep** | Rule for varying parameters to produce many cases |
| **Suite** | Named, versioned file combining scenarios, sweeps, formats, data properties and staging |
| **Profile** | Named set of dimensionality parameters selected at build time (`8b`, `40b`, `70b` or custom) |
| **Compression ratio** | Uncompressed size ÷ compressed size for a block of data under a given codec |
| **Byte-plane entropy** | Entropy of each byte position separately (e.g. high vs low byte of BF16); shows where the redundancy is |
| **NUMA** | Non-uniform memory access: on multi-socket CPUs, memory attached to the local socket is faster |
| **Huge pages** | 2 MiB or 1 GiB memory pages that reduce TLB misses for large buffers |
| **VNNI / AMX** | x86 instructions for INT8/BF16 dot products (VNNI) and tile-based matrix multiplication (AMX) |
| **I8MM / BFMMLA** | Arm instructions for INT8 and BF16 matrix multiply-accumulate |
| **Prefill / decode** | Processing many tokens at once (large M) / one token per sequence (small M) |
| **GQA** | Grouped-query attention: fewer key/value heads than query heads |
| **MoE** | Mixture of experts: tokens routed to a few of many expert weight matrices; gives grouped GEMMs |
| **LoRA** | Low-rank adapter: two thin matrices of rank r |
| **Class A / B / C** | Square weights / vocabulary × hidden weights / tokens × hidden activations (classic scenario) |
| **BF16 / FP16** | 16-bit floats with 8-bit / 5-bit exponents |
| **E4M3, E5M2** | FP8 formats named by exponent and mantissa bit counts |
| **INT4 / INT8** | 4-bit and 8-bit two's-complement integers |
| **Group scale / zero point** | Shared scale for a block of elements / integer offset in asymmetric quantization |
| **2:4 sparsity** | At most 2 non-zero values in every group of 4 consecutive elements |
| **Toggle density** | How often bits change between consecutive values; drives dynamic power |
| **Rotating copies** | Distinct versions of an operand cycled between benchmark iterations to avoid cache reuse |
| **Philox** | Counter-based pseudorandom generator with random access to any position |
| **Freivalds' algorithm** | Randomized O(n²) check of a matrix product |
| **Manifest** | JSON metadata describing a case or operand |
| **Shard / tile** | Fixed-size file holding part of a matrix / unit of generation work |
