#if !defined(_MATRIX_UTIL_H_)
#define _MATRIX_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

void mm_to_csr(char *file_name, int **row_ptrs, int **columns, double **vals,
               int *m, int *n, int *val_count);

void create_lap(double *lap, int size, int nnzp);

void lap_to_csr(double *matrix, int rows, int cols, int **row_ptrs,
                int **columns, double **vals, int *nnz);

#ifdef __cplusplus
}
#endif

#endif // _MATRIX_UTIL_H_
