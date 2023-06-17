#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

#define BLOCK_SIZE 32

void lanczos_algo(int *d_row_ptrs, int *d_columns, double *d_vals,
                  double *alpha, double *beta, double *d_w_vec, double *w_vec,
                  double *d_orth_vec, double *orth_vec, double *d_orth_vec_pre,
                  int m, int size) {
  cudaMemcpy(d_w_vec, w_vec, (size) * sizeof(double), cudaMemcpyHostToDevice);
  int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int i = 0; i < m; i++) {
    if (i > 0) {
      beta[i] = cuda_vec_norm(d_w_vec, size, grid_size, BLOCK_SIZE);
      cuda_vec_copy(d_orth_vec, d_orth_vec_pre, size, grid_size, BLOCK_SIZE);
    } else
      beta[i] = 0;

    if (fabs(beta[i] - 0) > 1e-8) {
      cuda_vec_sclr_mul(d_w_vec, d_orth_vec, 1 / beta[i], size, grid_size,
                        BLOCK_SIZE);
    } else {
      for (int i = 0; i < size; i++)
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      cudaMemcpy(d_orth_vec, orth_vec, (size) * sizeof(double),
                 cudaMemcpyHostToDevice);
      double norm_val = cuda_vec_norm(d_orth_vec, size, grid_size, BLOCK_SIZE);
      cuda_vec_sclr_mul(d_orth_vec, d_orth_vec, 1 / norm_val, size, grid_size,
                        BLOCK_SIZE);
    }

    cuda_spmv(d_row_ptrs, d_columns, d_vals, d_orth_vec, d_w_vec, size, size,
              grid_size, BLOCK_SIZE);

    alpha[i] = cuda_vec_dot(d_orth_vec, d_w_vec, size, grid_size, BLOCK_SIZE);

    if (i == 0) {
      cuda_calc_w_init(d_w_vec, alpha[i], d_orth_vec, size, grid_size,
                       BLOCK_SIZE);
    } else {
      cuda_calc_w(d_w_vec, alpha[i], d_orth_vec, d_orth_vec_pre, beta[i], size,
                  grid_size, BLOCK_SIZE);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count, int size,
             int m, double *eigvals, double *eigvecs, int argc, char *argv[]) {
  // Allocate host memory
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

  // Device memory
  double *d_vals, *d_orth_vec_pre, *d_orth_vec, *d_w_vec;
  int *d_row_ptrs, *d_columns;

  // Allocate device memory
  cudaMalloc((void **)&d_row_ptrs, (size + 1) * sizeof(int));
  cudaMalloc((void **)&d_columns, (val_count) * sizeof(int));
  cudaMalloc((void **)&d_vals, (val_count) * sizeof(double));
  cudaMalloc((void **)&d_orth_vec_pre, (size) * sizeof(double));
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
  for (int k = 0; k < 10; k++)
    lanczos_algo(d_row_ptrs, d_columns, d_vals, alpha, beta, d_w_vec, w_vec,
                 d_orth_vec, orth_vec, d_orth_vec_pre, m, size);

  // Measure time
  clock_t t = clock();
  for (int k = 0; k < TRIALS; k++)
    lanczos_algo(d_row_ptrs, d_columns, d_vals, alpha, beta, d_w_vec, w_vec,
                 d_orth_vec, orth_vec, d_orth_vec_pre, m, size);
  t = clock() - t;

  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC * TRIALS));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

  // Free device memory
  cudaFree(d_row_ptrs), cudaFree(d_columns), cudaFree(d_vals),
      cudaFree(d_orth_vec_pre), cudaFree(d_orth_vec), cudaFree(d_w_vec);

  // Free host memory
  free(alpha), free(beta), free(orth_vec), free(w_vec);

  return;
}
