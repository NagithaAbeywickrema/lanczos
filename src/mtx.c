#include "lanczos.h"

void nomp_mtx_norm(double *w, double *prod, const unsigned SIZE) {
#pragma nomp for reduce("prod", "+")
  for (unsigned i = 0; i < SIZE; i++)
    prod[0] += w[i] * w[i];
#pragma nomp sync
}

void nomp_mtx_sclr_div(double *v, double *w, double sclr, const unsigned SIZE) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (unsigned i = 0; i < SIZE; i++)
    w[i] = v[i] / sclr;
#pragma nomp sync
}

void nomp_mtx_col_copy(double *v, double *V, int i, unsigned SIZE) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (unsigned j = 0; j < SIZE; j++)
    V[j + SIZE * i] = v[j];
#pragma nomp sync
}

void nomp_mtx_mul(double *a, double *b, double *out, const int h_a,
                  const int w_a, const int w_b) {
#pragma nomp for transform("transforms", "matrix_mul")
  for (unsigned i = 0; i < h_a; i++) {
    for (unsigned j = 0; j < w_b; j++) {
      double dot = 0;
      for (unsigned k = 0; k < w_a; k++)
        dot += a[i * w_a + k] * b[k * w_b + j];
      out[i * w_b + j] = dot;
    }
  }
#pragma nomp sync
}

void nomp_mtx_dot(double *v, double *w, double *prod, const unsigned SIZE) {
#pragma nomp for reduce("prod", "+")
  for (unsigned i = 0; i < SIZE; i++)
    prod[0] += v[i] * w[i];
#pragma nomp sync
}

void nomp_mtx_tri(double *alpha, double *beta, double *T, const int M) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (unsigned i = 0; i < M; i++) {
    T[i * M + i] = alpha[i];
    if (i < M - 1) {
      T[i * M + i + 1] = beta[i + 1];
      T[(i + 1) * M + i] = beta[i + 1];
    }
  }
}

void nomp_mtx_identity(double *out, const int SIZE) {
#pragma nomp for transform("transforms", "identity_mtx")
  for (unsigned i = 0; i < SIZE; i++) {
    for (unsigned j = 0; j < SIZE; j++) {
      if (i == j)
        out[i * SIZE + j] = 1;
      else
        out[i * SIZE + j] = 0;
    }
  }
}
