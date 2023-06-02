#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void create_lap(double *lap, const int size);
void print_matrix(double *matrix, const int size1, const int size2);
void print_eigen_vals(double *eigen_vals, const int size);
void lanczos(double *lap, const int size, const int M, double *eigvals,
             double *eigvecs, int argc, char *argv[]);

#ifdef __cplusplus
}
#endif
