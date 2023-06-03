#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"
#include "print-helper.h"

#define MAX 1000000
#define EPS 1e-12

void lanczos_algo(double *lap, double *alpha, double *beta, double *w,
                  double *v, double *V, const unsigned m, const unsigned size) {
  for (unsigned i = 0; i < m; i++) {
    beta[i] = vec_norm(w, size);

    if (fabs(beta[i] - 0) > 1e-8) {
      mtx_sclr_div(w, v, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++)
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
      double norm_val = vec_norm(v, size);
      mtx_sclr_div(v, v, norm_val, size);
    }

    mtx_col_copy(v, V, i, size);

    mtx_vec_mul(lap, v, w, size, size);

    alpha[i] = vec_dot(v, w, size);

    if (i == 0) {
      calc_w_init(w, alpha[i], V, i, size);
    } else {
      calc_w(w, alpha[i], V, beta[i], i, size);
    }
  }
}

void lanczos(double *lap, const unsigned size, const unsigned m,
             double *eigvals, double *eigvecs, int argc, char *argv[]) {
  // print_matrix(lap, size, size);

  // Allocate memory
  double *V = (double *)calloc(size * m, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *v = (double *)calloc(size, sizeof(double));
  double *w = (double *)calloc(size, sizeof(double));

  for (int k = 0; k < 10; k++)
    lanczos_algo(lap, alpha, beta, w, v, V, m, size);

  clock_t t = clock();
  lanczos_algo(lap, alpha, beta, w, v, V, m, size);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);
  // not sorted
  print_eigen_vals(eigvals, size);
}
