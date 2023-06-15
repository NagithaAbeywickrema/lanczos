#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "../src/kernels.h"
#include "../src/lanczos.h"
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
    return (unsigned)(1.01 * i);
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
void vec_sclr_div_bench() {
    FILE *fp = open_file("vec-sclr-mul-cuda-sync");
    for(unsigned i =1e4; i < 1e7; i = inc(i)) {
        double *h_a = create_host_vec(i);
        // double *h_b = (double *)malloc(sizeof(double) * i);

        double *d_a, *d_b;

  // Allocate device memory
        cudaMalloc((void **)&d_a, (i) * sizeof(double));
        cudaMalloc((void **)&d_b, (i) * sizeof(double));

        cudaMemcpy(d_a, h_a, (i) * sizeof(double),
             cudaMemcpyHostToDevice);

        //Warmup d2d
        for(int j = 0; j< 1000; j++)
            cuda_d2d_mem_cpy(d_a, d_b, i);
 cudaDeviceSynchronize();
        //Warmup
        for(int j = 0; j< 1000; j++)
            cuda_vec_sclr_div(d_a, d_b, 1/10, i);

     cudaDeviceSynchronize();  

        clock_t t1 = clock();
        for(int j=0; j <1000; j++)
            cuda_d2d_mem_cpy(d_a, d_b, i);
         cudaDeviceSynchronize();
        t1 = clock() - t1;

        clock_t t = clock();
        for (int j=0; j < 1000; j++)
            cuda_vec_sclr_div(d_a, d_b, 1/10, i);
           cudaDeviceSynchronize();
        t = clock() - t;

        cudaFree(d_a), cudaFree(d_b);
        fprintf(fp, "%s,%s,%u,%u,%e,%e\n", "vec-sclr-mul","cuda", 32, i, (double)t1 / (CLOCKS_PER_SEC*1000),(double)t / (CLOCKS_PER_SEC*1000));
        tfree(&h_a);
    }
    fclose(fp);
}


void lanczos_bench(int argc, char *argv[]) {
    vec_sclr_div_bench();
    // vec_norm_bench();
    // calc_w_bench();
}
