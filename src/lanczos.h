#include "print-helper.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX 1000000
#define EPS 1e-8
#define TRIALS 2

#ifdef __cplusplus
extern "C" {
#endif

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count, int size,
             int m, double *eigvals, double *eigvecs, int argc, char *argv[]);
void lanczos_bench(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif
