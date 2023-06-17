#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

void lanczos_algo(int *row_ptrs, int *columns, double *vals, double *alpha,
                  double *beta, double *w_vec, double *orth_vec,
                  double *orth_vec_pre, int m, int size) {
  for (int i = 0; i < m; i++) {
    if (i > 0) {
      beta[i] = nomp_vec_norm(w_vec, size);
      nomp_vec_copy(orth_vec, orth_vec_pre, size);
    } else
      beta[i] = 0;

    if (fabs(beta[i] - 0) > EPS) {
      nomp_vec_sclr_mul(w_vec, orth_vec, 1 / beta[i], size);
    } else {
      for (int i = 0; i < size; i++)
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
#pragma nomp update(to : orth_vec[0, size])
      double norm_val = nomp_vec_norm(orth_vec, size);
      nomp_vec_sclr_mul(orth_vec, orth_vec, 1 / norm_val, size);
    }
    nomp_spmv(row_ptrs, columns, vals, orth_vec, w_vec, size, size);

    alpha[i] = nomp_vec_dot(orth_vec, w_vec, size);

    if (i == 0) {
      nomp_calc_w_init(w_vec, alpha[i], orth_vec, size);
    } else {
      nomp_calc_w(w_vec, alpha[i], orth_vec, orth_vec_pre, beta[i], size);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count, int size,
             int m, double *eigvals, double *eigvecs, int argc, char *argv[]) {
  // Allocate host memory
  double *orth_vec_pre = (double *)calloc(size, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

#pragma nomp init(argc, argv)

#pragma nomp update(to                                                         \
                    : row_ptrs[0, size + 1], columns[0, val_count],            \
                      vals[0, val_count], orth_vec_pre[0, size],               \
                      w_vec[0, size])

  // Warm up runs
  for (int k = 0; k < 10; k++)
    lanczos_algo(row_ptrs, columns, vals, alpha, beta, w_vec, orth_vec,
                 orth_vec_pre, m, size);

  // Measure time
  clock_t t = clock();
  for (int k = 0; k < TRIALS; k++)
    lanczos_algo(row_ptrs, columns, vals, alpha, beta, w_vec, orth_vec,
                 orth_vec_pre, m, size);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC * TRIALS));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

#pragma nomp update(free                                                       \
                    : row_ptrs[0, size + 1], columns[0, val_count],            \
                      vals[0, val_count], orth_vec_pre[0, size],               \
                      w_vec[0, size], orth_vec[0, size])

#pragma nomp finalize

  free(orth_vec_pre), free(alpha), free(beta), free(orth_vec), free(w_vec);

  return;
}
