double mtx_norm(double *w, double *d_w, const unsigned SIZE);
void mtx_sclr_div(double *v, double *w, double sclr, const unsigned SIZE);
void mtx_col_copy(double *v, double *V, int i, unsigned SIZE);
void mtx_mul(double *a, double *b, double *out, const int h_a, const int w_a,
             const int w_b);
double mtx_dot(double *v, double *w, double *d_w, const int SIZE);
void mtx_tri(double *alpha, double *beta, double *T, const int M);
void mtx_identity(double *out, const int SIZE);
