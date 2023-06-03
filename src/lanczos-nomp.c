#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"
#include "print-helper.h"

#define MAX 10
#define EPS 1e-12

void lanczos_algo(double *lap, double *alpha, double *beta, double *w,
                  double *v, double *V, const int m, const int size) {
  for (unsigned i = 0; i < m; i++) {
    beta[i] = nomp_vec_norm(w, size);

    if (fabs(beta[i] - 0) > 1e-8) {
      nomp_mtx_sclr_div(w, v, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++)
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
#pragma nomp update(to : v[0, size])
      double norm_val = nomp_vec_norm(v, size);
      nomp_mtx_sclr_div(v, v, norm_val, size);
    }

    nomp_mtx_col_copy(v, V, i, size);
    nomp_mtx_vec_mul(lap, v, w, size, size);

    alpha[i] = nomp_vec_dot(v, w, size);

    if (i == 0) {
      nomp_calc_w_int(w, alpha[i], V, i, size);
    } else {
      nomp_calc_w(w, alpha[i], V, beta[i], i, size);
    }
  }
}

void lanczos(double *lap, const unsigned size, const unsigned m,
             double *eigvals, double *eigvecs, int argc, char *argv[]) {
  double *V = (double *)calloc(size * m, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *v = (double *)calloc(size, sizeof(double));
  double *w = (double *)calloc(size, sizeof(double));
#pragma nomp init(argc, argv)

#pragma nomp update(to : lap[0, size * size], V[0, size * m], w[0, size])

  // warm ups
  for (int k = 0; k < 10; k++)
    lanczos_algo(lap, alpha, beta, w, v, V, m, size);

  clock_t t = clock();
  lanczos_algo(lap, alpha, beta, w, v, V, m, size);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

  print_eigen_vals(eigvals, size);

#pragma nomp update(free                                                       \
                    : lap[0, size * size], V[0, size * m], w[0, size],         \
                      v[0, size])

#pragma nomp finalize

  free(V), free(alpha), free(beta), free(v), free(w);

  return;
}
