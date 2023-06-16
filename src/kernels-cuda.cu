#include "kernels.h"
#include <math.h>

#define BLOCK_SIZE 32

__global__ void cuda_vec_dot_knl(double *a_vec, double *b_vec, double *result,
                                 const unsigned size) {
  extern __shared__ double shared_data[];
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size)
    shared_data[threadIdx.x] = a_vec[tid] * b_vec[tid];
  else
    shared_data[threadIdx.x] = 0.0;

  __syncthreads();

  for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];

    __syncthreads();
  }

  if (threadIdx.x == 0)
    result[blockIdx.x] = shared_data[0];
}

__global__ void cuda_vec_sclr_div_knl(double *a_vec, double *out_vec,
                                      const double sclr, const unsigned size) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size)
    out_vec[tid] = a_vec[tid] / sclr;
}

__global__ void cuda_vec_sclr_mul_knl(double *a_vec, double *out_vec,
                                      const double sclr, const unsigned size) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size)
    out_vec[tid] = a_vec[tid] * sclr;
}

__global__ void cuda_d2d_mem_cpy_knl(double *a, double *b, unsigned int size) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size)
    b[tid] = a[tid];
}

__global__ void cuda_mtx_col_copy_knl(double *vec, double *mtx,
                                      const unsigned col_index,
                                      const unsigned size) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size)
    mtx[tid + size * col_index] = vec[tid];
}

__global__ void cuda_mtx_vec_mul_knl(double *a_mtx, double *b_vec,
                                     double *out_vec, const unsigned num_rows,
                                     const unsigned num_cols) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows) {
    double dot = 0;
    for (unsigned k = 0; k < num_cols; k++)
      dot += a_mtx[row * num_cols + k] * b_vec[k];
    out_vec[row] = dot;
  }
}

__global__ void cuda_spmv_knl(unsigned *a_row_ptrs, unsigned *a_columns,
                              double *a_vals, double *b_vec, double *out_vec,
                              const unsigned num_rows,
                              const unsigned num_cols) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows) {
    unsigned start = a_row_ptrs[row];
    unsigned end = a_row_ptrs[row + 1];
    double dot = 0;
    // Add each element in the row
    for (unsigned j = start; j < end; j++)
      dot += a_vals[j] * b_vec[a_columns[j]];
    out_vec[row] = dot;
  }
}

__global__ void cuda_calc_w_init_knl(double *w_vec, const double alpha,
                                     double *orth_mtx, const unsigned col_index,
                                     const unsigned size) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    w_vec[tid] = w_vec[tid] - alpha * orth_mtx[tid + size * col_index];
  }
}

__global__ void cuda_calc_w_knl(double *w_vec, const double alpha,
                                double *orth_mtx, const double beta,
                                const unsigned col_index, const unsigned size) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    w_vec[tid] = w_vec[tid] - alpha * orth_mtx[tid + size * col_index] -
                 beta * orth_mtx[tid + size * (col_index - 1)];
  }
}

double cuda_vec_dot(double *d_a_vec, double *d_b_vec, const unsigned size) {
  const unsigned grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const unsigned shared_data_size = BLOCK_SIZE * sizeof(double);

  double *d_result;

  cudaMalloc((void **)&d_result, grid_size * sizeof(double));

  cuda_vec_dot_knl<<<grid_size, BLOCK_SIZE, shared_data_size>>>(
      d_a_vec, d_b_vec, d_result, size);

  cudaDeviceSynchronize();

  double *interim_results = (double *)calloc(grid_size, sizeof(double));
  cudaMemcpy(interim_results, d_result, grid_size * sizeof(double),
             cudaMemcpyDeviceToHost);

  double result = 0.0;
  for (unsigned i = 0; i < grid_size; i++) {
    result += interim_results[i];
  }

  cudaFree(d_result), free(interim_results);

  return result;
}

double cuda_vec_norm(double *d_a_vec, const unsigned size) {
  double sum_of_prod = cuda_vec_dot(d_a_vec, d_a_vec, size);
  return sqrt(sum_of_prod);
}

void cuda_vec_sclr_div(double *d_a_vec, double *d_out_vec, const double sclr,
                       const unsigned size,const unsigned grid_size, const unsigned block_size) {

  cuda_vec_sclr_div_knl<<<grid_size, block_size>>>(d_a_vec, d_out_vec, sclr,
                                                   size);
}

void cuda_vec_sclr_mul(double *d_a_vec, double *d_out_vec, const double sclr,
                       const unsigned size,const unsigned grid_size, const unsigned block_size) {

  cuda_vec_sclr_mul_knl<<<grid_size, block_size>>>(d_a_vec, d_out_vec, sclr,
                                                   size);
}


void cuda_d2d_mem_cpy(double *a, double *b, unsigned int size,const unsigned grid_size, const unsigned block_size) {

  cuda_d2d_mem_cpy_knl<<<grid_size, block_size>>>(a, b, size);
}

void cuda_mtx_col_copy(double *d_vec, double *d_mtx, const unsigned col_index,
                       const unsigned size) {
  const unsigned grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cuda_mtx_col_copy_knl<<<grid_size, BLOCK_SIZE>>>(d_vec, d_mtx, col_index,
                                                   size);

  cudaDeviceSynchronize();
}

void cuda_mtx_vec_mul(double *d_a_mtx, double *d_b_vec, double *d_out_vec,
                      const unsigned num_rows, const unsigned num_cols) {
  const unsigned grid_size = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cuda_mtx_vec_mul_knl<<<grid_size, BLOCK_SIZE>>>(d_a_mtx, d_b_vec, d_out_vec,
                                                  num_rows, num_cols);

  cudaDeviceSynchronize();
}

void cuda_spmv(unsigned *d_a_row_ptrs, unsigned *d_a_columns, double *d_a_vals,
               double *d_b_vec, double *d_out_vec, const unsigned num_rows,
               const unsigned num_cols) {
  const unsigned grid_size = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cuda_spmv_knl<<<grid_size, BLOCK_SIZE>>>(d_a_row_ptrs, d_a_columns, d_a_vals,
                                           d_b_vec, d_out_vec, num_rows,
                                           num_cols);

  cudaDeviceSynchronize();
}

void cuda_calc_w_init(double *d_w_vec, const double alpha, double *d_orth_mtx,
                      const unsigned col_index, const unsigned size) {
  const unsigned grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cuda_calc_w_init_knl<<<grid_size, BLOCK_SIZE>>>(d_w_vec, alpha, d_orth_mtx,
                                                  col_index, size);

  cudaDeviceSynchronize();
}

void cuda_calc_w(double *d_w_vec, const double alpha, double *d_orth_mtx,
                 const double beta, const unsigned col_index,
                 const unsigned size,const unsigned grid_size, const unsigned block_size) {

  cuda_calc_w_knl<<<grid_size, block_size>>>(d_w_vec, alpha, d_orth_mtx, beta,
                                             col_index, size);

  cudaDeviceSynchronize();
}
