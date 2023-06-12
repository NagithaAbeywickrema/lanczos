#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double vec_norm, vec_dot, vec_sclr_div, mtx_col_copy, spmv, calc_w;
} time_struct;

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count,
             const unsigned size, const unsigned m, double *eigvals,
             double *eigvecs, int argc, char *argv[],
             time_struct *time_measure);

#ifdef __cplusplus
}
#endif
