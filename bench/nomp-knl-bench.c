#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "../src/kernels.h"

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
    return i + 1;
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
    for(unsigned i = 1e4; i < 1e6; i = inc(i)) {
        double *h_a = create_host_vec(i);
        #pragma nomp update(to: h_a[0, i])

        // Warmup
        for(int j = 0; j< 1000; j++)
            nomp_vec_norm(h_a, i);

        clock_t t = clock();
        for (int j=0; j < 1000; j++)
            nomp_vec_norm(h_a, i);
        t = clock() - t;

        fprintf(fp, "%s,%s,%u,%e\n", "vec-norm","nomp",i, (double)t / (CLOCKS_PER_SEC*1000));
        #pragma nomp update(free: h_a[0, i])
        tfree(&h_a);
    }
    fclose(fp);
}

void vec_dot_bench() {
    FILE *fp = open_file("vec-dot-prod-01");
    for(unsigned i = 1e4; i < 1e7; i = inc(i)) {
        double *h_a = create_host_vec(i);
        double *h_b = create_host_vec(i);
        double *h_c = create_host_vec(32);
        #pragma nomp update(to: h_a[0, i], h_b[0, i],h_c[0,32])

        //Warmup d2d
        for(int j = 0; j< 1000; j++)
            nomp_d2d_mem_cpy(h_a, h_b, i);

        // Warmup
        for(int j = 0; j< 1000; j++)
            nomp_vec_dot(h_a, h_b, i);

        clock_t t1 = clock();
        for(int j=0; j <1000; j++){
            nomp_d2d_mem_cpy(h_a, h_b, i);
            nomp_d2d_mem_cpy(h_a, h_b, i);
        }
        #pragma nomp update(from: h_c[0, 32])
        t1 = clock() - t1;

        clock_t t = clock();
        for (int j=0; j < 1000; j++)
            nomp_vec_dot(h_a, h_b, i);
        t = clock() - t;

        fprintf(fp, "%s,%s,%u,%u,%e,%e\n", "vec-dot-prod","nomp", 32, i, (double)t1 / (CLOCKS_PER_SEC*1000),(double)t / (CLOCKS_PER_SEC*1000));
        #pragma nomp update(free: h_a[0, i], h_b[0, i])
        tfree(&h_a), tfree(&h_b);
    }
    fclose(fp);
}

void vec_sclr_div_bench() {
    FILE *fp = open_file("vec-sclr-mul-sync");
    for(unsigned i =1e4; i < 1e7; i = inc(i)) {
        double *h_a = create_host_vec(i);
        double *h_b = (double *)malloc(sizeof(double) * i);
        #pragma nomp update(to: h_a[0, i])
        #pragma nomp update(alloc: h_b[0, i])

        //Warmup d2d
        for(int j = 0; j< 1000; j++)
            nomp_d2d_mem_cpy(h_a, h_b, i);
  #pragma nomp sync
        //Warmup
        for(int j = 0; j< 1000; j++)
            nomp_vec_sclr_div(h_a, h_b, 1/10, i);
  #pragma nomp sync
        clock_t t1 = clock();
        for(int j=0; j <1000; j++)
            nomp_d2d_mem_cpy(h_a, h_b, i);
      #pragma nomp sync
        t1 = clock() - t1;

        clock_t t = clock();
        for (int j=0; j < 1000; j++)
            nomp_vec_sclr_div(h_a, h_b, 1/10, i);
    #pragma nomp sync
        t = clock() - t;

        fprintf(fp, "%s,%s,%u,%u,%e,%e\n", "vec-sclr-mul","nomp", 32, i, (double)t1 / (CLOCKS_PER_SEC*1000),(double)t / (CLOCKS_PER_SEC*1000));
        // fprintf(fp, "%s,%s,%u,%e\n", "vec-sclr-div","nomp",i, (double)t / (CLOCKS_PER_SEC*1000));
        #pragma nomp update(free: h_a[0, i], h_b[0,i])
        tfree(&h_a);
        free(h_b);
    }
    fclose(fp);
}


void vec_add_bench() {
    FILE *fp = open_file("vec-add");
    for(unsigned i =1e4; i < 1e7; i = inc(i)) {
        double *h_a = create_host_vec(i);
        double *h_c = create_host_vec(i);
        double *h_b = (double *)malloc(sizeof(double) * i);
        #pragma nomp update(to: h_a[0, i], h_c[0,i])
        #pragma nomp update(alloc: h_b[0, i])

        //Warmup d2d
        for(int j = 0; j< 1000; j++)
            nomp_d2d_mem_cpy(h_a, h_b, i);

        //Warmup
        for(int j = 0; j< 1000; j++)
            nomp_vec_add(h_a, h_b, h_c, i);

        clock_t t1 = clock();
        for(int j=0; j <1000; j++)
            nomp_d2d_mem_cpy(h_a, h_b, i);
        
        t1 = clock() - t1;

        clock_t t = clock();
        for (int j=0; j < 1000; j++)
            nomp_vec_sclr_div(h_a, h_b, 10, i);
        t = clock() - t;

        fprintf(fp, "%s,%s,%u,%u,%e,%e\n", "vec-sclr-div-15","nomp", 32, i, (double)t1 / (CLOCKS_PER_SEC*1000),(double)t / (CLOCKS_PER_SEC*1000));
        // fprintf(fp, "%s,%s,%u,%e\n", "vec-sclr-div","nomp",i, (double)t / (CLOCKS_PER_SEC*1000));
        #pragma nomp update(free: h_a[0, i], h_b[0,i])
        tfree(&h_a);
        free(h_b);
    }
    fclose(fp);
}

void mtx_col_copy_bench() {
    FILE *fp = open_file("mtx-col-copy");
    for(unsigned i =1; i<1e4; i = inc(i)) {
        double *h_a = (double *)calloc(i*i, sizeof(double));
        double *h_b = create_host_vec(i);

        #pragma nomp update(to: h_b[0, i])
        #pragma nomp update(alloc: h_a[0, i*i])

        //Warmup
        for(int j = 0; j< 1000; j++)
            nomp_mtx_col_copy(h_b, h_a, 0, i); // col_index

        clock_t t = clock();
        for (int j=0; j < 1000; j++)
            nomp_mtx_col_copy(h_b, h_a, 0, i);
        t = clock() - t;

        fprintf(fp, "%s,%s,%u,%e\n", "mtx-col-copy","nomp", i, (double)t / (CLOCKS_PER_SEC*1000));
        #pragma nomp update(free: h_a[0, i*i], h_b[0,i])
        tfree(&h_b);
        free(h_a);
    }
}

void calc_w_bench() {
    FILE *fp = open_file("calc-w");
    for(unsigned i =100; i<1e7; i = inc(i)) {
        double *h_a = create_host_vec(i);
        double *h_b = create_host_vec(i*i);

        #pragma nomp update(to: h_a[0, i], h_b[0, i*i])

        //Warmup
        for(int j = 0; j< 1000; j++)
            nomp_calc_w(h_a, 2, h_b, 2, 1, i); // col_index

        clock_t t = clock();
        for (int j=0; j < 1000; j++)
            nomp_calc_w(h_a, 2, h_b, 2, 1, i);
        t = clock() - t;

        fprintf(fp, "%s,%s,%u,%e\n", "calc-w","nomp", i, (double)t / (CLOCKS_PER_SEC*1000));
        #pragma nomp update(free: h_a[0, i], h_b[0, i*i])
        tfree(&h_a), tfree(&h_b);
    }
}

void lanczos_bench(int argc, char *argv[]) {
    #pragma nomp init(argc, argv)
    vec_sclr_div_bench();
    // vec_norm_bench();
    // vec_dot_bench();
    // calc_w_bench();
    #pragma nomp finalize
}
