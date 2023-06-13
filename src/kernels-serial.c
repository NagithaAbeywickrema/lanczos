#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double serial_vec_dot(double *a_vec, double *b_vec, const unsigned size) {
  double dot = 0;
  for (unsigned i = 0; i < size; i++)
    dot += a_vec[i] * b_vec[i];
  return dot;
}

double serial_vec_norm(double *a_vec, const unsigned size) {
  double sum_of_prod = serial_vec_dot(a_vec, a_vec, size);
  return sqrt(sum_of_prod);
}

void serial_vec_sclr_div(double *a_vec, double *out_vec, const double sclr,
                         const unsigned size) {
  for (unsigned i = 0; i < size; i++)
    out_vec[i] = a_vec[i] / sclr;
}

void serial_mtx_col_copy(double *vec, double *mtx, const unsigned col_index,
                         const unsigned size) {

  memcpy(mtx + size * col_index, vec, sizeof(double) * size);
}

void serial_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                        const unsigned num_rows, const unsigned num_cols) {
  for (unsigned i = 0; i < num_rows; i++) {
    double dot = 0;
    for (unsigned k = 0; k < num_cols; k++)
      dot += a_mtx[i * num_cols + k] * b_vec[k];
    out_vec[i] = dot;
  }
}

void serial_spmv(unsigned *a_row_ptrs, unsigned *a_columns, double *a_vals,
                 double *b_vec, double *out_vec, const unsigned num_rows,
                 const unsigned num_cols) {
  for (unsigned row = 0; row < num_rows; row++) {
    double dot = 0;
    unsigned row_start = a_row_ptrs[row];
    unsigned row_end = a_row_ptrs[row + 1];
    unsigned length = row_end - row_start;
    for (unsigned jj = 0; jj < length; jj++)
      dot += a_vals[row_start + jj] * b_vec[a_columns[row_start + jj]];
    out_vec[row] = dot;
  }
}

void serial_calc_w_init(double *w_vec, const double alpha, double *orth_mtx,
                        const unsigned col_index, const unsigned size) {
  for (unsigned j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_mtx[j + size * col_index];
  }
}

void serial_calc_w(double *w_vec, const double alpha, double *orth_mtx,
                   const double beta, const unsigned col_index,
                   const unsigned size) {
  for (unsigned j = 0; j < size; j++) {
    w_vec[j] = w_vec[j] - alpha * orth_mtx[j + size * col_index] -
               beta * orth_mtx[j + size * (col_index - 1)];
  }
}
