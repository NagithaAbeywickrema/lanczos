#include "../src/kernels.h"
#include "../src/lanczos.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX 10
#define EPS 1e-12
#define MAX_SOURCE_SIZE (0x100000)

#define tcalloc(T, n) (T *)calloc(n, sizeof(T))
#define tfree(p) free_((void **)p)
void free_(void **p) { free(*p), *p = NULL; }

double *create_host_vec(const unsigned size) {
  double *x = tcalloc(double, size);
  for (unsigned i = 0; i < size; i++)
    x[i] = (rand() + 1.0) / RAND_MAX;

  return x;
}

unsigned inc(const unsigned i) {
  if (i < 1000)
    return i + 1;
  else
    return (unsigned)(1.03 * i);
}

FILE *open_file(const char *suffix) {
  char fname[2 * BUFSIZ];
  strncpy(fname, "lanczos", BUFSIZ);
  strncat(fname, "_", 2);
  strncat(fname, suffix, BUFSIZ);
  strncat(fname, ".txt", 5);

  FILE *fp = fopen(fname, "a");
  if (!fp)
    printf("Not found \n");
  return fp;
}

void vec_norm_bench() {
  FILE *fp = open_file("vec-norm");
  for (unsigned i = 1; i < 1e6; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *d_a;
    cudaMalloc((void **)&d_a, i * sizeof(double));
    cudaMemcpy(d_a, h_a, i * sizeof(double), cudaMemcpyHostToDevice);

    // Warmup runs
    for (int j = 0; j < 1000; j++)
      cuda_vec_norm(d_a, i);

    // Measure time
    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      cuda_vec_norm(d_a, i);
    t = clock() - t;

    fprintf(fp, "%s,%s,%u,%e\n", "vec-norm", "cuda", i,
            (double)t / (CLOCKS_PER_SEC * 1000));

    cudaFree(d_a);
    tfree(&h_a);
  }
  fclose(fp);
}

void vec_sclr_div_bench() {
  FILE *fp = open_file("vec-sclr-div");
  for (unsigned i = 1; i < 1e6; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *d_a, *d_b;
    cudaMalloc((void **)&d_a, i * sizeof(double));
    cudaMalloc((void **)&d_b, i * sizeof(double));
    cudaMemcpy(d_a, h_a, i * sizeof(double), cudaMemcpyHostToDevice);

    // Warmup runs
    for (int j = 0; j < 1000; j++)
      cuda_vec_sclr_div(d_a, d_b, 10, i);

    // Measure time
    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      cuda_vec_sclr_div(d_a, d_b, 10, i);
    t = clock() - t;

    fprintf(fp, "%s,%s,%u,%e\n", "vec-sclr-div", "cuda", i,
            (double)t / (CLOCKS_PER_SEC * 1000));

    cudaFree(d_a), cudaFree(d_b);
    tfree(&h_a);
  }
  fclose(fp);
}

void mtx_col_copy_bench() {
  FILE *fp = open_file("mtx-col-copy");
  for (unsigned i = 1; i < 1e4; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *d_a, *d_b;
    cudaMalloc((void **)&d_a, i * sizeof(double));
    cudaMalloc((void **)&d_b, i * i * sizeof(double));
    cudaMemcpy(d_a, h_a, i * sizeof(double), cudaMemcpyHostToDevice);
    // Warmup runs
    for (int j = 0; j < 1000; j++)
      cuda_mtx_col_copy(d_a, d_b, 0, i);

    // Measure time
    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      cuda_mtx_col_copy(d_a, d_b, 0, i);
    t = clock() - t;

    fprintf(fp, "%s,%s,%u,%e\n", "mtx-col-copy", "cuda", i,
            (double)t / (CLOCKS_PER_SEC * 1000));

    cudaFree(d_a), cudaFree(d_b);
    tfree(&h_a);
  }
  fclose(fp);
}

void calc_w_bench() {
  FILE *fp = open_file("calc-w");

  for (unsigned i = 2; i < 1e5; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = create_host_vec(i * i);
    double *d_a, *d_b;
    cudaMalloc((void **)&d_a, i * sizeof(double));
    cudaMalloc((void **)&d_b, i * i * sizeof(double));
    cudaMemcpy(d_a, h_a, i * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, i * i * sizeof(double), cudaMemcpyHostToDevice);

    // Warmup runs
    for (int j = 0; j < 1000; j++)
      cuda_calc_w(d_a, 2, d_b, 2, 1, i);

    // Measure time
    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      cuda_calc_w(d_a, 2, d_b, 2, 1, i);
    t = clock() - t;

    fprintf(fp, "%s,%s,%u,%e\n", "calc-w", "cuda", i,
            (double)t / (CLOCKS_PER_SEC * 1000));

    cudaFree(d_a), cudaFree(d_b);
    tfree(&h_a), tfree(&h_b);
  }

  fclose(fp);
}

void lanczos_bench(int argc, char *argv[]) {
  // Benchmark kernels
  vec_norm_bench();
  vec_sclr_div_bench();
  mtx_col_copy_bench();
  calc_w_bench();
}