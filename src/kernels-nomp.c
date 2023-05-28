#include <math.h>

void nomp_mtx_sclr_div(double *v, double *w, double sclr, const unsigned size) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (unsigned i = 0; i < size; i++)
    w[i] = v[i] / sclr;
#pragma nomp sync
}

void nomp_mtx_col_copy(double *v, double *V, int i, unsigned size) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (unsigned j = 0; j < size; j++)
    V[j + size * i] = v[j];
#pragma nomp sync
}

void nomp_mtx_vec_mul(double *a, double *b, double *out, const int h_a,
                      const int w_a) {
#pragma nomp for transform("transforms", "matrix_mul")
  for (unsigned i = 0; i < h_a; i++) {
    double dot = 0;
    for (unsigned k = 0; k < w_a; k++)
      dot += a[i * w_a + k] * b[k];
    out[i] = dot;
  }
#pragma nomp sync
}

double nomp_mtx_dot(double *v, double *w, const unsigned size) {
  double prod[1] = {0};
#pragma nomp for reduce("prod", "+")
  for (unsigned i = 0; i < size; i++)
    prod[0] += v[i] * w[i];
#pragma nomp sync
  return prod[0];
}

double nomp_mtx_norm(double *w, const unsigned size) {
  double prod = nomp_mtx_dot(w, w, size);
  return sqrt(prod);
}

void nomp_mtx_identity(double *out, const int size) {
#pragma nomp for transform("transforms", "identity_mtx")
  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      if (i == j)
        out[i * size + j] = 1;
      else
        out[i * size + j] = 0;
    }
  }
}

void nomp_calc_w_int(double *w, double alpha, double *V, unsigned i,
                     const int size) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (int j = 0; j < size; j++) {
    w[j] = w[j] - alpha * V[j + size * i];
  }
#pragma nomp sync
}

void nomp_calc_w(double *w, double alpha, double *V, double beta, unsigned i,
                 const int size) {
#pragma nomp for transform("transforms", "matrix_norm")
  for (int j = 0; j < size; j++) {
    w[j] = w[j] - alpha * V[j + size * i] - beta * V[j + size * (i - 1)];
  }
#pragma nomp sync
}
