#include <math.h>

double nomp_vec_dot(double *a_vec, double *b_vec, const unsigned size) {
  double prod[1] = {0};
#pragma nomp for reduce("prod", "+")
  for (unsigned i = 0; i < size; i++)
    prod[0] += a_vec[i] * b_vec[i];
#pragma nomp sync
  return prod[0];
}

double nomp_vec_norm(double *a_vec, const unsigned size) {
  double sum_of_prod = nomp_vec_dot(a_vec, a_vec, size);
  return sqrt(sum_of_prod);
}

void nomp_vec_sclr_div(double *a_vec, double *out_vec, const double sclr,
                       const unsigned size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (unsigned i = 0; i < size; i++)
    out_vec[i] = a_vec[i] / sclr;
#pragma nomp sync
}

void nomp_mtx_col_copy(double *vec, double *mtx, const unsigned col_index,
                       const unsigned size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (unsigned j = 0; j < size; j++)
    mtx[j + size * col_index] = vec[j];
#pragma nomp sync
}

void nomp_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                      const unsigned num_rows, const unsigned num_cols) {
#pragma nomp for transform("transforms", "mat_vec_mul")
  for (unsigned i = 0; i < num_rows; i++) {
    double dot = 0;
    for (unsigned k = 0; k < num_cols; k++)
      dot += a_mtx[i * num_cols + k] * b_vec[k];
    out_vec[i] = dot;
  }
#pragma nomp sync
}

void nomp_calc_w_init(double *w_vec, const double alpha, double *orth_mtx,
                      const unsigned col_index, const int size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (unsigned j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_mtx[j + size * col_index];
  }
#pragma nomp sync
}

void nomp_calc_w(double *w_vec, const double alpha, double *orth_mtx,
                 const double beta, const unsigned col_index,
                 const unsigned size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (unsigned j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_mtx[j + size * col_index] -
               beta * orth_mtx[j + size * (col_index - 1)];
  }
#pragma nomp sync
}
