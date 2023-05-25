void nomp_mtx_norm(double *w, double *prod, const unsigned SIZE);
void nomp_mtx_sclr_div(double *v, double *w, double sclr, const unsigned SIZE);
void nomp_mtx_col_copy(double *v, double *V, int i, unsigned SIZE);
void nomp_mtx_mul(double *a, double *b, double *out, const int h_a,
                  const int w_a, const int w_b);
void nomp_mtx_dot(double *v, double *w, double *prod, const unsigned SIZE);
void nomp_mtx_tri(double *alpha, double *beta, double *T, const int M);
void nomp_mtx_identity(double *out, const int SIZE);
