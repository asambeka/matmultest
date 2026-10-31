/* Example: open a MatGen dataset from C, list its cases, load a weight and
 * regenerate a chunk of the activation in memory, timing both.
 *
 *   ./read_example DATASET_DIR
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matgen.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s DATASET_DIR\n", argv[0]); return 1; }
    char err[512];
    mg_dataset* ds = mg_open(argv[1], err, sizeof err);
    if (!ds) { fprintf(stderr, "mg_open: %s\n", err); return 1; }

    printf("%zu operands, %zu GEMM cases\n", mg_num_operands(ds), mg_num_cases(ds));
    for (size_t i = 0; i < mg_num_cases(ds) && i < 3; ++i) {
        mg_case_info c;
        mg_case_at(ds, i, &c);
        printf("  case %-14s A=%s (%lld x %lld)  B=%s (%lld x %lld, transposed)  %.1f TFLOP\n", c.id, c.a,
               (long long)c.m, (long long)c.k, c.b, (long long)c.n, (long long)c.k, c.flops / 1e12);
    }

    /* Load the B operand of the first case (a weight, stored on disk). */
    mg_case_info c0;
    mg_case_at(ds, 0, &c0);
    mg_operand_info w;
    mg_operand(ds, c0.b, &w);
    size_t wbytes = (size_t)w.rows * w.row_bytes;
    void* wdata = mg_alloc(wbytes);
    float* wscales = w.scales_per_row ? malloc(sizeof(float) * w.rows * w.scales_per_row) : NULL;
    double t = now();
    if (mg_load_rows(ds, w.id, 0, w.rows, wdata, wscales, 0) != 0) { fprintf(stderr, "%s\n", mg_last_error(ds)); return 1; }
    printf("loaded %s: %lld x %lld %s, %.1f MiB in %.3f s (%s)\n", w.id, (long long)w.rows, (long long)w.cols,
           mg_dtype_name(w.dtype), wbytes / 1048576.0, now() - t, w.stored ? "from disk" : "regenerated");

    /* Regenerate the first chunk of activation rows in memory. */
    mg_operand_info x;
    mg_operand(ds, c0.a, &x);
    int64_t rows = x.rows < 262144 ? x.rows : 262144;
    size_t xbytes = (size_t)rows * x.row_bytes;
    void* xdata = mg_alloc(xbytes);
    t = now();
    if (mg_load_rows(ds, x.id, 0, rows, xdata, NULL, 0) != 0) { fprintf(stderr, "%s\n", mg_last_error(ds)); return 1; }
    double dt = now() - t;
    printf("activation %s rows [0, %lld): %.1f MiB in %.3f s (%.2f GB/s, %s)\n", x.id, (long long)rows,
           xbytes / 1048576.0, dt, xbytes / dt / 1e9, x.stored ? "from disk" : "regenerated");

    mg_free(xdata, xbytes);
    mg_free(wdata, wbytes);
    free(wscales);
    mg_close(ds);
    return 0;
}
