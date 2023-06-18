#include "lanczos.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void print_matrix(double *matrix, int size1, int size2);
void print_eigen_vals(double *eigen_vals, int size);

void print_kernel_time(time_struct *time_measure);

#ifdef __cplusplus
}
#endif
