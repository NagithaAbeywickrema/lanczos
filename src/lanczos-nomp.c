#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

void lanczos_algo(unsigned *row_ptrs, unsigned *columns, double *vals,
                  double *alpha, double *beta, double *w_vec, double *orth_vec,
                  double *orth_mtx, const unsigned m, const unsigned size) {
  for (unsigned i = 0; i < m; i++) {
    beta[i] = nomp_vec_norm(w_vec, size);

    if (fabs(beta[i] - 0) > EPS) {
      nomp_vec_sclr_div(w_vec, orth_vec, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++)
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
#pragma nomp update(to : orth_vec[0, size])
      double norm_val = nomp_vec_norm(orth_vec, size);
      nomp_vec_sclr_div(orth_vec, orth_vec, norm_val, size);
    }

    nomp_mtx_col_copy(orth_vec, orth_mtx, i, size);

    // nomp_mtx_vec_mul(lap, orth_vec, w_vec, size, size);
    nomp_spmv(row_ptrs, columns, vals, orth_vec, w_vec, size, size);

    alpha[i] = nomp_vec_dot(orth_vec, w_vec, size);

    if (i == 0) {
      nomp_calc_w_init(w_vec, alpha[i], orth_mtx, i, size);
    } else {
      nomp_calc_w(w_vec, alpha[i], orth_mtx, beta[i], i, size);
    }
  }
}

void lanczos(unsigned *row_ptrs, unsigned *columns, double *vals,
             const unsigned val_count, const unsigned size, const unsigned m,
             double *eigvals, double *eigvecs, int argc, char *argv[]) {
  // Allocate host memory
  double *orth_mtx = (double *)calloc(size * m, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

#pragma nomp init(argc, argv)

#pragma nomp update(to                                                         \
                    : row_ptrs[0, size + 1], columns[0, val_count],            \
                      vals[0, val_count], orth_mtx[0, size * m],               \
                      w_vec[0, size])

  // Warm up runs
  for (unsigned k = 0; k < 10; k++)
    lanczos_algo(row_ptrs, columns, vals, alpha, beta, w_vec, orth_vec,
                 orth_mtx, m, size);

  // Measure time
  clock_t t = clock();
  for (unsigned k = 0; k < TRIALS; k++)
    lanczos_algo(row_ptrs, columns, vals, alpha, beta, w_vec, orth_vec,
                 orth_mtx, m, size);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC * TRIALS));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

#pragma nomp update(free                                                       \
                    : row_ptrs[0, size + 1], columns[0, val_count],            \
                      vals[0, val_count], orth_mtx[0, size * m],               \
                      w_vec[0, size], orth_vec[0, size])

#pragma nomp finalize

  free(orth_mtx), free(alpha), free(beta), free(orth_vec), free(w_vec);

  return;
}
