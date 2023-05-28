double mtx_norm(double *w, const unsigned size);
void mtx_sclr_div(double *v, double *w, double sclr, const unsigned size);
void mtx_col_copy(double *v, double *V, int i, unsigned size);
void mtx_vec_mul(double *a, double *b, double *out, const int h_a,
                 const int w_a);
double mtx_dot(double *v, double *w, const unsigned size);
void mtx_identity(double *out, const int size);
void calc_w_init(double *w, double alpha, double *V, unsigned i,
                 const int size);
void calc_w(double *w, double alpha, double *V, double beta, unsigned i,
            const int size);

double nomp_mtx_norm(double *w, const unsigned size);
void nomp_mtx_sclr_div(double *v, double *w, double sclr, const unsigned size);
void nomp_mtx_col_copy(double *v, double *V, int i, unsigned size);
void nomp_mtx_vec_mul(double *a, double *b, double *out, const int h_a,
                      const int w_a, const int w_b);
double nomp_mtx_dot(double *v, double *w, const unsigned size);
void nomp_mtx_identity(double *out, const int size);
void nomp_calc_w_int(double *w, double alpha, double *V, unsigned i,
                     const int size);
void nomp_calc_w(double *w, double alpha, double *V, double beta, unsigned i,
                 const int size);
