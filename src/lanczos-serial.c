#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

#define MAX 1000000
#define EPS 1e-12

void lanczos_algo(int *row_ptrs, int *columns, double *vals, double *alpha,
                  double *beta, double *w_vec, double *orth_vec,
                  double *orth_mtx, const unsigned m, const unsigned size) {
  for (unsigned i = 0; i < m; i++) {
    beta[i] = serial_vec_norm(w_vec, size);

    if (fabs(beta[i] - 0) > EPS) {
      serial_vec_sclr_div(w_vec, orth_vec, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++)
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      double norm_val = serial_vec_norm(orth_vec, size);
      serial_vec_sclr_div(orth_vec, orth_vec, norm_val, size);
    }

    serial_mtx_col_copy(orth_vec, orth_mtx, i, size);

    // serial_mtx_vec_mul(lap, orth_vec, w_vec, size, size);
    serial_spmv(row_ptrs, columns, vals, orth_vec, w_vec, size, size);

    alpha[i] = serial_vec_dot(orth_vec, w_vec, size);

    if (i == 0) {
      serial_calc_w_init(w_vec, alpha[i], orth_mtx, i, size);
    } else {
      serial_calc_w(w_vec, alpha[i], orth_mtx, beta[i], i, size);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count,
             const unsigned size, const unsigned m, double *eigvals,
             double *eigvecs, int argc, char *argv[],
             time_struct *time_measure) {
  // Allocate memory
  double *orth_mtx = (double *)calloc(size * m, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

  // Warm up runs
  for (unsigned k = 0; k < 10; k++)
    lanczos_algo(row_ptrs, columns, vals, alpha, beta, w_vec, orth_vec,
                 orth_mtx, m, size);

  clock_t t = clock();
  lanczos_algo(row_ptrs, columns, vals, alpha, beta, w_vec, orth_vec, orth_mtx,
               m, size);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  // tqli(eigvecs, eigvals, size, alpha, beta, 0);

  free(orth_mtx), free(alpha), free(beta), free(orth_vec), free(w_vec);
}
