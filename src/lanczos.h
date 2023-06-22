#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX 1000000
#define EPS 1e-12
#define TRIALS 1000

#ifdef __cplusplus
extern "C" {
#endif

void lanczos(unsigned *row_ptrs, unsigned *columns, double *vals,
             const unsigned val_count, const unsigned size, const unsigned m,
             double *eigvals, double *eigvecs, int argc, char *argv[]);

#ifdef __cplusplus
}
#endif
