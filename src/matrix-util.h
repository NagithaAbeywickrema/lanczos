#if !defined(_MATRIX_UTIL_H_)
#define _MATRIX_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

void mm_to_csr(char *file_name, int **row_ptrs, int **columns, double **vals,
               int *m, int *n, int *val_count);

#ifdef __cplusplus
}
#endif

#endif // _MATRIX_UTIL_H_
