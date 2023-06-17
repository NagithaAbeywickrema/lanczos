#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"


#define BLOCKSIZE 32

void lanczos_algo(sycl::buffer<int> a_row_buf, sycl::buffer<int> a_columns_buf,
                  sycl::buffer<double> a_vals_buf, double *alpha, double *beta,
                  sycl::buffer<double> w_buf, sycl::buffer<double> orth_vec_buf,
                  double *orth_vec, sycl::buffer<double> orth_vec_pre_buf, int m,
                  int size, sycl::queue queue) {
  size_t global_size = ((size + BLOCKSIZE - 1) / BLOCKSIZE) * BLOCKSIZE;

  sycl::nd_range nd_range= sycl::nd_range(sycl::range(global_size), sycl::range(BLOCKSIZE));

  for (int i = 0; i < m; i++) {
    if (i > 0) {
      beta[i] = sycl_mtx_norm(w_buf, size, queue,nd_range);
      sycl_vec_copy(orth_vec_buf, orth_vec_pre_buf, size,queue,nd_range);
    } else
      beta[i] = 0;
    

    if (fabs(beta[i] - 0) > EPS) {
      sycl_mtx_sclr_mul(w_buf, 1/beta[i], orth_vec_buf, size, queue,nd_range);
    } else {
      for (int i = 0; i < size; i++) {
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      }
      sycl::buffer orth_vec_buf{orth_vec, sycl::range<1>(size)};
      double norm_val = sycl_mtx_norm(orth_vec_buf, size, queue,nd_range);
      sycl_mtx_sclr_mul(orth_vec_buf, 1/norm_val, orth_vec_buf, size, queue,nd_range);
    }

    // sycl_mtx_col_copy(orth_vec_buf, orth_mtx_buf, i, size, queue,nd_range);

    sycl_spmv(a_row_buf, a_columns_buf, a_vals_buf, orth_vec_buf, w_buf, size,
              size, queue,nd_range);

    alpha[i] = sycl_mtx_dot(orth_vec_buf, w_buf, size, queue,nd_range);

    if (i == 0) {
      sycl_calc_w_init(w_buf, alpha[i], orth_vec_buf, size, queue,nd_range);
    } else {
      sycl_calc_w(w_buf, alpha[i], orth_vec_buf,orth_vec_pre_buf, beta[i], size, queue,nd_range);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count, int size,
             int m, double *eigvals, double *eigvecs, int argc, char *argv[]) {
  auto sycl_platforms = sycl::platform().get_platforms();
  auto sycl_pdevices = sycl_platforms[2].get_devices();
  sycl::device device = sycl_pdevices[0];
  sycl::context context = sycl::context(device);
  sycl::queue queue = sycl::queue(context, device);

  // Allocate memory
  double *orth_vec_pre = (double *)calloc(size, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

  sycl::buffer a_row_buf{row_ptrs, sycl::range<1>(size + 1)};
  sycl::buffer a_columns_buf{columns, sycl::range<1>(val_count)};
  sycl::buffer a_vals_buf{vals, sycl::range<1>(val_count)};

  sycl::buffer orth_vec_pre_buf{orth_vec_pre, sycl::range<1>(size)};
  sycl::buffer orth_vec_buf{orth_vec, sycl::range<1>(size)};
  sycl::buffer w_buf{w_vec, sycl::range<1>(size)};

  // warm ups
  for (int k = 0; k < 10; k++)
    lanczos_algo(a_row_buf, a_columns_buf, a_vals_buf, alpha, beta, w_buf,
                 orth_vec_buf, orth_vec, orth_vec_pre_buf, m, size, queue);

  // Measure time
  clock_t t = clock();
  for (int k = 0; k < TRIALS; k++)
    lanczos_algo(a_row_buf, a_columns_buf, a_vals_buf, alpha, beta, w_buf,
                 orth_vec_buf, orth_vec, orth_vec_pre_buf, m, size, queue);
  t = clock() - t;

  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC * TRIALS));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

  w_buf.get_access<sycl::access::mode::read>();
  orth_vec_buf.get_access<sycl::access::mode::read>();
  orth_vec_pre_buf.get_access<sycl::access::mode::read>();

  free(orth_vec_pre), free(alpha), free(beta), free(orth_vec), free(w_vec);
}
