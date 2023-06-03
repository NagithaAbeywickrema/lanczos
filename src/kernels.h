// Serially implemented functions of Lanczos routine.
double serial_vec_dot(double *a_vec, double *b_vec, const unsigned size);
double serial_vec_norm(double *a_vec, const unsigned size);
void serial_vec_sclr_div(double *a_vec, double *out_vec, const double sclr,
                         const unsigned size);
void serial_mtx_col_copy(double *vec, double *mtx, const unsigned col_index,
                         const unsigned size);
void serial_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                        const unsigned num_rows, const unsigned num_cols);
void serial_calc_w_init(double *w_vec, const double alpha, double *orth_mtx,
                        const unsigned col_index, const unsigned size);
void serial_calc_w(double *w_vec, const double alpha, double *orth_mtx,
                   const double beta, const unsigned col_index,
                   const unsigned size);

// Parallelized functions of Lanczos routine implemented using Nomp.
double nomp_vec_dot(double *a_vec, double *b_vec, const unsigned size);
double nomp_vec_norm(double *a_vec, const unsigned size);
void nomp_vec_sclr_div(double *a_vec, double *out_vec, const double sclr,
                       const unsigned size);
void nomp_mtx_col_copy(double *vec, double *mtx, const unsigned col_index,
                       const unsigned size);
void nomp_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                      const unsigned num_rows, const unsigned num_cols);
void nomp_calc_w_init(double *w_vec, const double alpha, double *orth_mtx,
                      const unsigned col_index, const unsigned size);
void nomp_calc_w(double *w_vec, const double alpha, double *orth_mtx,
                 const double beta, const unsigned col_index,
                 const unsigned size);

// Parallelized functions of Lanczos routine implemented using Cuda.
double cuda_vec_dot(double *d_a_vec, double *d_b_vec, const unsigned size);
double cuda_vec_norm(double *d_a_vec, const unsigned size);
void cuda_vec_sclr_div(double *d_a_vec, double *d_out_vec, const double sclr,
                       const unsigned size);
void cuda_mtx_col_copy(double *d_vec, double *d_mtx, const unsigned col_index,
                       const unsigned size);
void cuda_mtx_vec_mul(double *d_a_mtx, double *d_b_vec, double *d_out_vec,
                      const unsigned num_rows, const unsigned num_cols);
void cuda_calc_w_init(double *d_w_vec, const double alpha, double *d_orth_mtx,
                      const unsigned col_index, const unsigned size);
void cuda_calc_w(double *d_w_vec, const double alpha, double *d_orth_mtx,
                 const double beta, const unsigned col_index,
                 const unsigned size);
