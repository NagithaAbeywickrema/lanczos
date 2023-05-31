#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

#define MAX 1000000
#define EPS 1e-12
#define MAX_ITER 100

void lanczos(double *lap, const int size, const int M, double *eigvals,
             double *eigvecs, int argc, char *argv[]) {

  sycl::device device{sycl::gpu_selector_v};
  sycl::context context = sycl::context(device);
  sycl::queue queue = sycl::queue(context, device);
  // print_matrix(lap, size, size);

  // Allocate memory
  double *V = (double *)calloc(size * M, sizeof(double));
  double *alpha = (double *)calloc(M, sizeof(double));
  double *beta = (double *)calloc(M, sizeof(double));
  double *v = (double *)calloc(size, sizeof(double));
  double *w = (double *)calloc(size, sizeof(double));

  sycl::buffer lap_buf{lap, sycl::range<1>(size * size)};
  sycl::buffer v_buf{V, sycl::range<1>(size * M)};
  sycl::buffer v_temp_buf{v, sycl::range<1>(size)};
  sycl::buffer T_buf{lap, sycl::range<1>(M * M)};
  sycl::buffer w_buf{w, sycl::range<1>(size)};

  // Lanczos iteration
  clock_t t = clock();
  for (unsigned i = 0; i < M; i++) {
    beta[i] = sycl_mtx_norm(w_buf, size, queue);

    if (beta[i] != 0) {
      sycl_mtx_sclr_div(w_buf, beta[i], v_temp_buf, size, queue);
    } else {
      for (unsigned i = 0; i < size; i++) {
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
      }
      sycl::buffer v_temp_buf{v, sycl::range<1>(size)};
      double norm_val = sycl_mtx_norm(v_temp_buf, size, queue);
      sycl_mtx_sclr_div(v_temp_buf, norm_val, v_temp_buf, size, queue);
    }

    sycl_mtx_col_copy(v_temp_buf, v_buf, i, size, queue);

    sycl_mtx_vec_mul(lap_buf, v_temp_buf, w_buf, size, size, queue);

    alpha[i] = sycl_mtx_dot(v_temp_buf, w_buf, size, queue);
    sycl::buffer alpha_buf{alpha, sycl::range<1>(M)};
    sycl::buffer beta_buf{beta, sycl::range<1>(M)};
    if (i == 0) {
      sycl_calc_w_init(w_buf, alpha_buf, v_buf, i, size, queue);
    } else {
      sycl_calc_w(w_buf, alpha_buf, v_buf, beta_buf, i, size, queue);
    }
  }

  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

  w_buf.get_access<sycl::access::mode::read>();
  v_temp_buf.get_access<sycl::access::mode::read>();

  free(V), free(alpha), free(beta), free(v), free(w);
}
