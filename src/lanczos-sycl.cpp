#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

#define MAX 1000000
#define EPS 1e-12

void lanczos_algo(sycl::buffer<int> a_row_buf, sycl::buffer<int> a_columns_buf,
                  sycl::buffer<double> a_vals_buf, double *alpha, double *beta,
                  sycl::buffer<double> w_buf, sycl::buffer<double> orth_vec_buf,
                  double *orth_vec, sycl::buffer<double> orth_mtx_buf,
                  const int m, const int size, sycl::queue queue) {
  for (unsigned i = 0; i < m; i++) {
    beta[i] = sycl_mtx_norm(w_buf, size, queue);

    if (fabs(beta[i] - 0) > EPS) {
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

    sycl_spmv(a_row_buf, a_columns_buf, a_vals_buf, orth_vec_buf, w_buf, size,
              size, queue);

    alpha[i] = sycl_mtx_dot(orth_vec_buf, w_buf, size, queue);

    if (i == 0) {
      sycl_calc_w_init(w_buf, alpha[i], orth_mtx_buf, i, size, queue);
    } else {
      sycl_calc_w(w_buf, alpha[i], orth_mtx_buf, beta[i], i, size, queue);
    }
  }
}

void lanczos_algo1(sycl::buffer<int> a_row_buf, sycl::buffer<int> a_columns_buf,
                   sycl::buffer<double> a_vals_buf, double *alpha, double *beta,
                   sycl::buffer<double> w_buf,
                   sycl::buffer<double> orth_vec_buf, double *orth_vec,
                   sycl::buffer<double> orth_mtx_buf, const int m,
                   const int size, sycl::queue queue,
                   time_struct *time_measure) {
  for (unsigned i = 0; i < m; i++) {
    clock_t t = clock();
    beta[i] = sycl_mtx_norm(w_buf, size, queue);
    t = clock() - t;
    time_measure->vec_norm += (double)t / (CLOCKS_PER_SEC);

    if (fabs(beta[i] - 0) > EPS) {
      t = clock();
      sycl_mtx_sclr_div(w_buf, beta[i], orth_vec_buf, size, queue);
      t = clock() - t;
      time_measure->vec_sclr_div += (double)t / (CLOCKS_PER_SEC);
    } else {
      for (unsigned i = 0; i < size; i++) {
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      }
      sycl::buffer orth_vec_buf{orth_vec, sycl::range<1>(size)};
      double norm_val = sycl_mtx_norm(orth_vec_buf, size, queue);
      sycl_mtx_sclr_div(orth_vec_buf, norm_val, orth_vec_buf, size, queue);
    }

    t = clock();
    sycl_mtx_col_copy(orth_vec_buf, orth_mtx_buf, i, size, queue);
    t = clock() - t;
    time_measure->mtx_col_copy += (double)t / (CLOCKS_PER_SEC);
    t = clock();

    sycl_spmv(a_row_buf, a_columns_buf, a_vals_buf, orth_vec_buf, w_buf, size,
              size, queue);
    t = clock() - t;
    time_measure->spmv += (double)t / (CLOCKS_PER_SEC);
    t = clock();

    alpha[i] = sycl_mtx_dot(orth_vec_buf, w_buf, size, queue);
    t = clock() - t;
    time_measure->vec_dot += (double)t / (CLOCKS_PER_SEC);

    if (i == 0) {
      sycl_calc_w_init(w_buf, alpha[i], orth_mtx_buf, i, size, queue);
    } else {
      t = clock();
      sycl_calc_w(w_buf, alpha[i], orth_mtx_buf, beta[i], i, size, queue);
      t = clock() - t;
      time_measure->calc_w += (double)t / (CLOCKS_PER_SEC);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count,
             const unsigned size, const unsigned m, double *eigvals,
             double *eigvecs, int argc, char *argv[],
             time_struct *time_measure) {
  auto sycl_platforms = sycl::platform().get_platforms();
  auto sycl_pdevices = sycl_platforms[2].get_devices();
  sycl::device device = sycl_pdevices[0];
  sycl::context context = sycl::context(device);
  sycl::queue queue = sycl::queue(context, device);

  // Allocate memory
  double *orth_mtx = (double *)calloc(size * m, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

  sycl::buffer a_row_buf{row_ptrs, sycl::range<1>(size + 1)};
  sycl::buffer a_columns_buf{columns, sycl::range<1>(val_count)};
  sycl::buffer a_vals_buf{vals, sycl::range<1>(val_count)};

  sycl::buffer orth_mtx_buf{orth_mtx, sycl::range<1>(size * m)};
  sycl::buffer orth_vec_buf{orth_vec, sycl::range<1>(size)};
  sycl::buffer w_buf{w_vec, sycl::range<1>(size)};

  // warm ups
  for (int k = 0; k < 10; k++)
    lanczos_algo(a_row_buf, a_columns_buf, a_vals_buf, alpha, beta, w_buf,
                 orth_vec_buf, orth_vec, orth_mtx_buf, m, size, queue);

  clock_t t = clock();
  lanczos_algo1(a_row_buf, a_columns_buf, a_vals_buf, alpha, beta, w_buf,
                orth_vec_buf, orth_vec, orth_mtx_buf, m, size, queue,
                time_measure);
  t = clock() - t;

  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  // tqli(eigvecs, eigvals, size, alpha, beta, 0);

  w_buf.get_access<sycl::access::mode::read>();
  orth_vec_buf.get_access<sycl::access::mode::read>();
  orth_mtx_buf.get_access<sycl::access::mode::read>();

  free(orth_mtx), free(alpha), free(beta), free(orth_vec), free(w_vec);
}
