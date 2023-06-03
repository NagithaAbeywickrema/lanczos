#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void lanczos(double *lap, const unsigned size, const unsigned m,
             double *eigvals, double *eigvecs, int argc, char *argv[]);

#ifdef __cplusplus
}
#endif
