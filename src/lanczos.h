#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX 1000000
#define EPS 1e-8
#define TRIALS 100

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double time;
  long no_of_itt;
} time_data;

typedef struct {
  time_data *vec_norm, *vec_dot, *vec_sclr_mul, *vec_copy, *spmv, *calc_w,
      *calc_w_init;
} time_struct;

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count, int size,
             int m, double *eigvals, double *eigvecs, time_struct *time_measure,
             int argc, char *argv[]);
void lanczos_bench(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif
