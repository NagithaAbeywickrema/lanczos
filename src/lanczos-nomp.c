#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"
#include "print-helper.h"

#define MAX 10
#define EPS 1e-12

void lanczos(double *lap, const int size, const int M, double *eigvals,
             double *eigvecs, int argc, char *argv[]) {
  double *V = (double *)calloc(size * M, sizeof(double));
  double *alpha = (double *)calloc(M, sizeof(double));
  double *beta = (double *)calloc(M, sizeof(double));
  double *v = (double *)calloc(size, sizeof(double));
  double *w = (double *)calloc(size, sizeof(double));
  double prod;
#pragma nomp init(argc, argv)

#pragma nomp update(to : lap[0, size * size], V[0, size * M], w[0, size])

  clock_t t = clock();
  for (unsigned i = 0; i < M; i++) {
    prod = 0;

    beta[i] = nomp_mtx_norm(w, size);
    ;

    if (fabs(beta[i] - 0) > 1e-8) {
      nomp_mtx_sclr_div(w, v, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++)
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
#pragma nomp update(to : v[0, size])
      prod = 0;
      double norm_val = nomp_mtx_norm(v, size);
      nomp_mtx_sclr_div(v, v, norm_val, size);
    }

    nomp_mtx_col_copy(v, V, i, size);
    nomp_mtx_vec_mul(lap, v, w, size, size, 1);
    prod = 0;

    alpha[i] = nomp_mtx_dot(v, w, size);

    if (i == 0) {
      nomp_calc_w_int(w, alpha[i], V, i, size);
    } else {
      nomp_calc_w(w, alpha[i], V, beta[i], i, size);
    }
  }
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);

  print_eigen_vals(eigvals, size);

#pragma nomp update(free                                                       \
                    : lap[0, size * size], V[0, size * M],                     \
                      w[0, size] v[0, size])

#pragma nomp finalize

  free(lap), free(eigvals), free(eigvecs), free(V), free(alpha), free(beta),
      free(v), free(w);

  return;
}
