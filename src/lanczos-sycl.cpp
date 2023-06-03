#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

#define MAX 1000000

void lanczos(double *lap, const int size, const int M, double *eigvals,
             double *eigvecs, int argc, char *argv[]) {

  sycl::device device{sycl::gpu_selector_v};
  sycl::context context = sycl::context(device);
  sycl::queue queue = sycl::queue(context, device);

  // Allocate memory
  double *orth_mtx = (double *)calloc(size * M, sizeof(double));
  double *alpha = (double *)calloc(M, sizeof(double));
  double *beta = (double *)calloc(M, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

  sycl::buffer lap_buf{lap, sycl::range<1>(size * size)};
  sycl::buffer orth_mtx_buf{orth_mtx, sycl::range<1>(size * M)};
  sycl::buffer orth_vec_buf{orth_vec, sycl::range<1>(size)};
  sycl::buffer w_buf{w_vec, sycl::range<1>(size)};

  // Lanczos iteration
  clock_t t = clock();
  for (unsigned i = 0; i < M; i++) {
    beta[i] = sycl_mtx_norm(w_buf, size, queue);

    if (beta[i] != 0) {
      sycl_mtx_sclr_div(w_buf, beta[i], orth_vec_buf, size, queue);
    } else {
      for (unsigned i = 0; i < size; i++) {
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      }
      sycl::buffer orth_vec_buf{orth_vec, sycl::range<1>(size)};
      double norm_val = sycl_mtx_norm(orth_vec_buf, size, queue);
      sycl_mtx_sclr_div(orth_vec_buf, norm_val, orth_vec_buf, size, queue);
    }

    sycl_mtx_col_copy(orth_vec_buf, orth_mtx_buf, i, size, queue);

    sycl_mtx_vec_mul(lap_buf, orth_vec_buf, w_buf, size, size, queue);

    alpha[i] = sycl_mtx_dot(orth_vec_buf, w_buf, size, queue);
    sycl::buffer alpha_buf{alpha, sycl::range<1>(M)};
    sycl::buffer beta_buf{beta, sycl::range<1>(M)};
    if (i == 0) {
      sycl_calc_w_init(w_buf, alpha_buf, orth_mtx_buf, i, size, queue);
    } else {
      sycl_calc_w(w_buf, alpha_buf, orth_mtx_buf, beta_buf, i, size, queue);
    }
  }

  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

  w_buf.get_access<sycl::access::mode::read>();
  orth_vec_buf.get_access<sycl::access::mode::read>();

  free(orth_mtx), free(alpha), free(beta), free(orth_vec), free(w_vec);
}
