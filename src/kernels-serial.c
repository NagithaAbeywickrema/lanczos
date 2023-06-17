#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double serial_vec_dot(double *a_vec, double *b_vec, int size) {
  double dot = 0;
  for (int i = 0; i < size; i++)
    dot += a_vec[i] * b_vec[i];
  return dot;
}

double serial_vec_norm(double *a_vec, int size) {
  double sum_of_prod = serial_vec_dot(a_vec, a_vec, size);
  return sqrt(sum_of_prod);
}

void serial_vec_sclr_div(double *a_vec, double *out_vec, double sclr,
                         int size) {
  for (int i = 0; i < size; i++)
    out_vec[i] = a_vec[i] / sclr;
}

void serial_vec_sclr_mul(double *a_vec, double *out_vec, double sclr,
                         int size) {
  for (int i = 0; i < size; i++)
    out_vec[i] = a_vec[i] * sclr;
}

void serial_mtx_col_copy(double *vec, double *mtx, int col_index, int size) {

  memcpy(mtx + size * col_index, vec, sizeof(double) * size);
}
void serial_vec_copy(double *vec, double *out, int size) {
  memcpy(out, vec, sizeof(double) * size);
}

void serial_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                        int num_rows, int num_cols) {
  for (int i = 0; i < num_rows; i++) {
    double dot = 0;
    for (int k = 0; k < num_cols; k++)
      dot += a_mtx[i * num_cols + k] * b_vec[k];
    out_vec[i] = dot;
  }
}

void serial_spmv(int *a_row_ptrs, int *a_columns, double *a_vals, double *b_vec,
                 double *out_vec, int num_rows, int num_cols) {
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

void serial_calc_w_init(double *w_vec, double alpha, double *orth_vec, int size) {
  for (int j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_vec[j];
  }
}

void serial_calc_w(double *w_vec, double alpha,double *orth_vec, double *orth_vec_pre, double beta, int size) {
  for (int j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_vec[j] -
               beta * orth_vec_pre[j];
  }
}
