#include "lanczos.h"
#include "mtx.h"
#include "print-helper.h"

#define MAX 10
#define EPS 1e-12
#define MAX_ITER 100000

void calc_w_int(double *w, double *alpha, double *V, unsigned i,
                const int SIZE) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (int j = 0; j < SIZE; j++) {
    w[j] = w[j] - alpha[i] * V[j + SIZE * i];
  }
#pragma nomp sync
}

void calc_w(double *w, double *alpha, double *V, double *beta, unsigned i,
            const int SIZE) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (int j = 0; j < SIZE; j++) {
    w[j] = w[j] - alpha[i] * V[j + SIZE * i] - beta[i] * V[j + SIZE * (i - 1)];
  }
#pragma nomp sync
}

void qr_algorithm(double *T, const int SIZE, double *eigvals, double *eigvecs) {

  double *Q = (double *)calloc(SIZE * SIZE, sizeof(double));
  double *R = (double *)calloc(SIZE * SIZE, sizeof(double));

#pragma nomp update(to : Q[0, SIZE * SIZE], R[0, SIZE * SIZE])

  mtx_identity(eigvecs, SIZE);

  for (unsigned i = 0; i < MAX_ITER; i++) {
#pragma nomp for transform("transforms", "identity_mtx")
    for (unsigned j = 0; j < SIZE; j++) {
      for (unsigned t = 0; t < SIZE; t++) {
        Q[j + SIZE * t] = T[j + SIZE * t];
      }
      for (unsigned k = 0; k < j; k++) {
        double dot = 0;
        for (unsigned t = 0; t < SIZE; t++) {
          dot += Q[k + SIZE * t] * T[j + SIZE * t];
        }
        R[k * SIZE + j] = dot;
        for (unsigned t = 0; t < SIZE; t++) {
          Q[j + SIZE * t] -= R[k * SIZE + j] * Q[k + SIZE * t];
        }
      }

      double total = 0;
      for (unsigned t = 0; t < SIZE; t++) {
        total += Q[j + SIZE * t] * Q[j + SIZE * t];
      }
      R[j * SIZE + j] = sqrt(total); // norm(temp_q, SIZE);

      for (unsigned t = 0; t < SIZE; t++)
        Q[j + SIZE * t] = Q[j + SIZE * t] / R[j * SIZE + j];
    }

    mtx_mul(R, Q, T, SIZE, SIZE, SIZE);
    mtx_mul(eigvecs, Q, eigvecs, SIZE, SIZE, SIZE);

#pragma nomp update(from : T[0, SIZE * SIZE])
    double subdiag = 0;
    for (unsigned k = 0; k < SIZE; k++) {
      if (subdiag < abs(T[k * SIZE + k]))
        subdiag = abs(T[k * SIZE + k]);
    }
    if (subdiag < EPS)
      break;
  }

#pragma nomp for transform("transforms", "matrix_norm")
  for (unsigned j = 0; j < SIZE; j++) {
    eigvals[j] = T[j * SIZE + j];
  }
}

void lanczos(double *lap, const int SIZE, const int M, double *eigvals,
             double *eigvecs) {
  double *V = (double *)calloc(SIZE * M, sizeof(double));
  double *T = (double *)calloc(M * M, sizeof(double));
  double *alpha = (double *)calloc(M, sizeof(double));
  double *beta = (double *)calloc(M, sizeof(double));
  double *v = (double *)calloc(SIZE, sizeof(double));
  double *w = (double *)calloc(SIZE, sizeof(double));
  double *d_w = (double *)calloc(SIZE, sizeof(double));
  double *d_v = (double *)calloc(SIZE, sizeof(double));

#pragma nomp update(to                                                         \
                    : lap[0, SIZE * SIZE], V[0, SIZE * M], T[0, M * M],        \
                      w[0, SIZE], eigenvecs[0, M * SIZE], eigvals[0, M])

#pragma nomp update(alloc : d_w[0, SIZE], d_v[0, SIZE])

  for (unsigned i = 0; i < M; i++) {
    beta[i] = mtx_norm(w, d_w, SIZE);

    if (beta[i] != 0) {
      mtx_sclr_div(w, v, beta[i], SIZE);
    } else {
      for (unsigned i = 0; i < SIZE; i++)
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
#pragma nomp update(to : v[0, SIZE])
      double norm_val = mtx_norm(v, d_v, SIZE);
      mtx_sclr_div(v, v, norm_val, SIZE);
    }

    mtx_col_copy(v, V, i, SIZE);
    mtx_mul(lap, v, w, SIZE, SIZE, 1);
    alpha[i] = mtx_dot(v, w, d_w, SIZE);

#pragma nomp update(to : alpha[0, SIZE], beta[0, SIZE])

    if (i == 0) {
      calc_w_int(w, alpha, V, i, SIZE);
    } else {
      calc_w(w, alpha, V, beta, i, SIZE);
    }
  }

  mtx_tri(alpha, beta, T, M);

#pragma nomp update(from : T[0, M * M])

  qr_algorithm(T, SIZE, eigvals, eigvecs);

#pragma nomp update(from : eigvals[0, M])
  print_eigen_vals(eigvals, SIZE);

#pragma nomp update(free                                                       \
                    : lap[0, SIZE * SIZE], V[0, SIZE * M], T[0, M * M],        \
                      w[0, SIZE], eigenvecs[0, M * SIZE], eigvals[0, M],       \
                      alpha[0, SIZE], beta[0, SIZE], Q[0, SIZE * SIZE],        \
                      R[0, SIZE * SIZE], v[0, SIZE])

#pragma nomp finalize

  free(lap), free(eigvals), free(eigvecs), free(V), free(T), free(alpha),
      free(beta), free(v), free(w);

  return;
}
