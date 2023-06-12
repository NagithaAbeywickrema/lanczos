#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

#define MAX 10
#define EPS 1e-12

void lanczos_algo(int *row_ptrs, int *columns, double *vals, double *alpha,
                  double *beta, double *w_vec, double *orth_vec,
                  double *orth_mtx, const int m, const int size) {
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

void lanczos_algo1(int *row_ptrs, int *columns, double *vals, double *alpha,
                  double *beta, double *w_vec, double *orth_vec,
                  double *orth_mtx, const int m, const int size,
             time_struct *time_measure) {
  for (unsigned i = 0; i < m; i++) {
    clock_t t = clock();
    beta[i] = nomp_vec_norm(w_vec, size);
  t = clock() - t;
    time_measure->vec_norm += (double)t / (CLOCKS_PER_SEC);
    if (fabs(beta[i] - 0) > 1e-8) {
      t = clock();
      nomp_vec_sclr_div(w_vec, orth_vec, beta[i], size);
      t = clock() - t;
      time_measure->vec_sclr_div += (double)t / (CLOCKS_PER_SEC);
    } else {
      for (unsigned i = 0; i < size; i++)
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
#pragma nomp update(to : orth_vec[0, size])
      double norm_val = nomp_vec_norm(orth_vec, size);
      nomp_vec_sclr_div(orth_vec, orth_vec, norm_val, size);
    }

   t = clock();
    nomp_mtx_col_copy(orth_vec, orth_mtx, i, size);
    t = clock() - t;
    time_measure->mtx_col_copy += (double)t / (CLOCKS_PER_SEC);
    // nomp_mtx_vec_mul(lap, orth_vec, w_vec, size, size);
    t = clock();
    nomp_spmv(row_ptrs, columns, vals, orth_vec, w_vec, size, size);
    t = clock() - t;
    time_measure->spmv += (double)t / (CLOCKS_PER_SEC);
    t = clock();
    alpha[i] = nomp_vec_dot(orth_vec, w_vec, size);
    t = clock() - t;
    time_measure->vec_dot += (double)t / (CLOCKS_PER_SEC);

    if (i == 0) {
      nomp_calc_w_init(w_vec, alpha[i], orth_mtx, i, size);
    } else {
      t = clock();
      nomp_calc_w(w_vec, alpha[i], orth_mtx, beta[i], i, size);
      t = clock() - t;
      time_measure->calc_w += (double)t / (CLOCKS_PER_SEC);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count,
             const unsigned size, const unsigned m, double *eigvals,
             double *eigvecs, int argc, char *argv[],
             time_struct *time_measure) {
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

  clock_t t = clock();
  lanczos_algo1(row_ptrs, columns, vals, alpha, beta, w_vec, orth_vec, orth_mtx,
               m, size,time_measure);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  // tqli(eigvecs, eigvals, size, alpha, beta, 0);

#pragma nomp update(free                                                       \
                    : row_ptrs[0, size + 1], columns[0, val_count],            \
                      vals[0, val_count], orth_mtx[0, size * m],               \
                      w_vec[0, size], orth_vec[0, size])

#pragma nomp finalize

  free(orth_mtx), free(alpha), free(beta), free(orth_vec), free(w_vec);

  return;
}
