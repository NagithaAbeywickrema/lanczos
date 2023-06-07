#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"
#include "print-helper.h"

#define MAX 10
#define EPS 1e-12

void lanczos_algo(int *d_row_ptrs, int *d_columns, double *d_vals,
                  double *alpha, double *beta, double *d_w_vec, double *w_vec,
                  double *d_orth_vec, double *orth_vec, double *d_orth_mtx,
                  const unsigned m, const unsigned size) {
  cudaMemcpy(d_w_vec, w_vec, (size) * sizeof(double), cudaMemcpyHostToDevice);

  for (unsigned i = 0; i < m; i++) {
    beta[i] = cuda_vec_norm(d_w_vec, size);

    if (fabs(beta[i] - 0) > 1e-8) {
      cuda_vec_sclr_div(d_w_vec, d_orth_vec, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++)
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      cudaMemcpy(d_orth_vec, orth_vec, (size) * sizeof(double),
                 cudaMemcpyHostToDevice);
      double norm_val = cuda_vec_norm(d_orth_vec, size);
      cuda_vec_sclr_div(d_orth_vec, d_orth_vec, norm_val, size);
    }

    cuda_mtx_col_copy(d_orth_vec, d_orth_mtx, i, size);

    // cuda_mtx_vec_mul(d_lap, d_orth_vec, d_w_vec, size, size);
    cuda_spmv(d_row_ptrs, d_columns, d_vals, d_orth_vec, d_w_vec, size, size);

    alpha[i] = cuda_vec_dot(d_orth_vec, d_w_vec, size);

    if (i == 0) {
      cuda_calc_w_init(d_w_vec, alpha[i], d_orth_mtx, i, size);
    } else {
      cuda_calc_w(d_w_vec, alpha[i], d_orth_mtx, beta[i], i, size);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count,
             const unsigned size, const unsigned m, double *eigvals,
             double *eigvecs, int argc, char *argv[]) {
  // Allocate host memory
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

  // Device memory
  double *d_vals, *d_orth_mtx, *d_orth_vec, *d_w_vec;
  int *d_row_ptrs, *d_columns;

  // Allocate device memory
  cudaMalloc((void **)&d_row_ptrs, (size + 1) * sizeof(int));
  cudaMalloc((void **)&d_columns, (val_count) * sizeof(int));
  cudaMalloc((void **)&d_vals, (val_count) * sizeof(double));
  cudaMalloc((void **)&d_orth_mtx, (size * m) * sizeof(double));
  cudaMalloc((void **)&d_orth_vec, (size) * sizeof(double));
  cudaMalloc((void **)&d_w_vec, (size) * sizeof(double));

  // H2D memory copy
  cudaMemcpy(d_row_ptrs, row_ptrs, (size + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_columns, columns, (val_count) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vals, vals, (val_count) * sizeof(double),
             cudaMemcpyHostToDevice);

  // Warm up runs
  for (unsigned k = 0; k < 10; k++)
    lanczos_algo(d_row_ptrs, d_columns, d_vals, alpha, beta, d_w_vec, w_vec,
                 d_orth_vec, orth_vec, d_orth_mtx, m, size);

  // Measure time
  clock_t t = clock();
  lanczos_algo(d_row_ptrs, d_columns, d_vals, alpha, beta, d_w_vec, w_vec,
               d_orth_vec, orth_vec, d_orth_mtx, m, size);
  t = clock() - t;

  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);
  print_eigen_vals(eigvals, size);

  // Free device memory
  cudaFree(d_row_ptrs), cudaFree(d_columns), cudaFree(d_vals),
      cudaFree(d_orth_mtx), cudaFree(d_orth_vec), cudaFree(d_w_vec);

  // Free host memory
  free(alpha), free(beta), free(orth_vec), free(w_vec);

  return;
}
