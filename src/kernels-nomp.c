#include <math.h>

void nomp_d2d_mem_cpy(double *a, double *b, int N) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int i = 0; i < N; i++)
    b[i] = a[i];
}

double nomp_vec_dot(double *a_vec, double *b_vec, int size) {
  double prod[1] = {0};
#pragma nomp for reduce("prod", "+")
  for (int i = 0; i < size; i++)
    prod[0] += a_vec[i] * b_vec[i];
#pragma nomp sync
  return prod[0];
}

double nomp_vec_norm(double *a_vec, int size) {
  double sum_of_prod = nomp_vec_dot(a_vec, a_vec, size);
  return sqrt(sum_of_prod);
}

void nomp_vec_sclr_div(double *a_vec, double *out_vec, double sclr, int size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int i = 0; i < size; i++)
    out_vec[i] = a_vec[i] / sclr;
}

void nomp_vec_sclr_mul(double *a_vec, double *out_vec, double sclr, int size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int i = 0; i < size; i++)
    out_vec[i] = a_vec[i] * sclr;
}

void nomp_vec_add(double *a_vec, double *b_vec, double *out_vec, int size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int i = 0; i < size; i++)
    out_vec[i] = a_vec[i] + b_vec[i];
}

void nomp_mtx_col_copy(double *vec, double *mtx, int col_index, int size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int j = 0; j < size; j++)
    mtx[j + size * col_index] = vec[j];
}
void nomp_vec_copy(double *vec, double *out, int N) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int i = 0; i < N; i++)
    out[i] = vec[i];
}

void nomp_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                      int num_rows, int num_cols) {
#pragma nomp for transform("transforms", "mat_vec_mul")
  for (int i = 0; i < num_rows; i++) {
    double dot = 0;
    for (int k = 0; k < num_cols; k++)
      dot += a_mtx[i * num_cols + k] * b_vec[k];
    out_vec[i] = dot;
  }
}

void nomp_spmv(int *a_row_ptrs, int *a_columns, double *a_vals, double *b_vec,
               double *out_vec, int num_rows, int num_cols) {
#pragma nomp for transform("transforms", "spmv")
  for (int row = 0; row < num_rows; row++) {
    double dot = 0;
    int row_start = a_row_ptrs[row];
    int row_end = a_row_ptrs[row + 1];
    int length = row_end - row_start;
    for (int jj = 0; jj < length; jj++)
      dot += a_vals[row_start + jj] * b_vec[a_columns[row_start + jj]];
    out_vec[row] = dot;
  }
}

void nomp_calc_w_init(double *w_vec, double alpha, double *orth_vec, int size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_vec[j];
  }
}

void nomp_calc_w(double *w_vec, double alpha, double *orth_vec,
                 double *orth_vec_pre, double beta, int size) {
#pragma nomp for transform("transforms", "stream_data_flow_loop")
  for (int j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_vec[j] - beta * orth_vec_pre[j];
  }
}
