/*
 * MatGen C API: read MatGen datasets and regenerate operands in memory.
 *
 * Operands are row-major, little-endian. INT4 packs two values per byte, low
 * nibble first, signed two's complement. Quantized integers carry float32
 * scales: real = q * scale, with one scale per row (MG_SCALE_ROW) or per
 * group of `group_size` consecutive elements in a row (MG_SCALE_GROUP).
 * FP8 operands with MG_SCALE_TENSOR use real = decoded_value * tensor_scale.
 *
 * Bytes returned by mg_load_rows and mg_generate_rows are identical: an
 * operand that is not stored on disk is regenerated, bit for bit.
 */
#ifndef MATGEN_H
#define MATGEN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mg_dataset mg_dataset;

typedef enum { MG_F16 = 0, MG_BF16, MG_F8E4M3, MG_F8E5M2, MG_I8, MG_I4 } mg_dtype;
typedef enum { MG_SCALE_NONE = 0, MG_SCALE_TENSOR, MG_SCALE_ROW, MG_SCALE_GROUP } mg_scale_mode;

typedef struct {
    const char* id;
    const char* cls;   /* "A", "B" or "C" */
    const char* role;  /* e.g. "weight.L00.q", "lm_head", "activation" */
    int64_t rows, cols;
    mg_dtype dtype;
    size_t row_bytes;      /* bytes per row of data */
    mg_scale_mode scale_mode;
    int32_t group_size;
    size_t scales_per_row; /* floats per row in the scales array (0 if none) */
    float tensor_scale;
    int stored;            /* 1: data file on disk; 0: regenerated on load */
} mg_operand_info;

typedef struct {
    const char* id;
    const char* group;  /* weight format set the case belongs to, e.g. "i8" or "lm.bf16" */
    const char* a;      /* operand id of A (M x K, row-major) */
    const char* b;      /* operand id of B (stored N x K, row-major; used transposed) */
    int64_t m, n, k;
    int trans_a, trans_b;
    double flops;
    double bytes_min;
} mg_case_info;

/* Opens a dataset directory. Returns NULL on error and writes a message to err. */
mg_dataset* mg_open(const char* dir, char* err, size_t errlen);
void mg_close(mg_dataset* ds);

size_t mg_num_operands(const mg_dataset* ds);
int mg_operand_at(const mg_dataset* ds, size_t index, mg_operand_info* out);
int mg_operand(const mg_dataset* ds, const char* id, mg_operand_info* out);

size_t mg_num_cases(const mg_dataset* ds);
int mg_case_at(const mg_dataset* ds, size_t index, mg_case_info* out);

/* Loads rows [row0, row0 + nrows): reads the file if stored, otherwise
 * regenerates. data: nrows * row_bytes bytes. scales: nrows * scales_per_row
 * floats, or NULL to skip. nthreads <= 0 uses all hardware threads. */
int mg_load_rows(const mg_dataset* ds, const char* id, int64_t row0, int64_t nrows,
                 void* data, float* scales, int nthreads);

/* Always regenerates in memory, even if the operand is stored. */
int mg_generate_rows(const mg_dataset* ds, const char* id, int64_t row0, int64_t nrows,
                     void* data, float* scales, int nthreads);

/* 2 MiB-aligned allocation, backed by huge pages when the kernel allows it,
 * and pre-faulted so the first benchmark iteration doesn't pay page faults. */
void* mg_alloc(size_t bytes);
void mg_free(void* p, size_t bytes);

const char* mg_dtype_name(mg_dtype d);
const char* mg_last_error(const mg_dataset* ds);

#ifdef __cplusplus
}
#endif

#endif /* MATGEN_H */
