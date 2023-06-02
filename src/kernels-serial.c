#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void create_identity_matrix(double *out, const int size) {
  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      if (i == j)
        out[i * size + j] = 1;
      else
        out[i * size + j] = 0;
    }
  }
}

void mtx_vec_mul(double *A, double *B, double *out, const int height_a,
                 const int width_a) {
  for (unsigned i = 0; i < height_a; i++) {
    double dot = 0;
    for (unsigned k = 0; k < width_a; k++)
      dot += A[i * width_a + k] * B[k];
    out[i] = dot;
  }
}

double vec_dot(double *v, double *w, const int size) {
  double dot = 0;
  for (unsigned i = 0; i < size; i++)
    dot += v[i] * w[i];
  return dot;
}
void mtx_sclr_div(double *in, double *out, double scalar, const int size) {
  for (unsigned i = 0; i < size; i++)
    out[i] = in[i] / scalar;
}

void calc_w_init(double *w, double alpha, double *V, unsigned i,
                 const int size) {
  for (int j = 0; j < size; j++) {
    w[j] = w[j] - alpha * V[j + size * i];
  }
}

void calc_w(double *w, double alpha, double *V, double beta, unsigned i,
            const int size) {
  for (int j = 0; j < size; j++) {
    w[j] = w[j] - alpha * V[j + size * i] - beta * V[j + size * (i - 1)];
  }
}

double vec_norm(double *w, const int size) {
  double total = 0;
  for (unsigned i = 0; i < size; i++)
    total += w[i] * w[i];
  return sqrt(total);
}

void mtx_col_copy(double *v, double *V, int j, const int size) {

  memcpy(V + size * j, v, sizeof(double) * size);
}
