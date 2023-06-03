// Serially implemented functions of Lanczos routine.
double vec_norm(double *w, const unsigned size);
void mtx_sclr_div(double *v, double *w, double sclr, const unsigned size);
void mtx_col_copy(double *v, double *V, int i, unsigned size);
void mtx_vec_mul(double *a, double *b, double *out, const int h_a,
                 const int w_a);
double vec_dot(double *v, double *w, const unsigned size);
void calc_w_init(double *w, double alpha, double *V, unsigned i,
                 const int size);
void calc_w(double *w, double alpha, double *V, double beta, unsigned i,
            const int size);

// Parallelized functions of Lanczos routine implemented using Nomp.
double nomp_vec_norm(double *w, const unsigned size);
void nomp_mtx_sclr_div(double *v, double *w, double sclr, const unsigned size);
void nomp_mtx_col_copy(double *v, double *V, int i, unsigned size);
void nomp_mtx_vec_mul(double *a, double *b, double *out, const int h_a,
                      const int w_a);
double nomp_vec_dot(double *v, double *w, const unsigned size);
void nomp_calc_w_int(double *w, double alpha, double *V, unsigned i,
                     const int size);
void nomp_calc_w(double *w, double alpha, double *V, double beta, unsigned i,
                 const int size);

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
