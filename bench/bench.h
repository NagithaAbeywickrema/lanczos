#include "../src/kernels.h"
#include "../src/lanczos.h"
#include "../src/matrix-util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>


#define BLOCK_SIZE 32
#define TRAILS 1000
#define WARMUP 100
#define MAX_SOURCE_SIZE (0x100000)

#define tcalloc(T, n) (T *)calloc(n, sizeof(T))
#define tfree(p) free_((void **)p)
void free_(void **p);
double *create_host_vec(int size);
int inc(int i);
FILE *open_file(char *suffix);

void lanczos_bench(int argc, char *argv[]);

void vec_sclr_mul_bench();
void vec_sclr_div_bench();
void calc_w_bench();
void vec_norm_bench();
void vec_dot_bench();
void mtx_col_copy_bench();
void create_roofline();
void spmv_bench();
