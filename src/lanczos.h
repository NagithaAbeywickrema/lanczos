#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void create_lap(double *lap, const int SIZE);
void lanczos_aux(double *V, double *T, double *alpha, double *beta, double *v,
                 const int SIZE, const int M);
void lanczos(double *lap, const int SIZE, const int M, double *V, double *T,
             double *alpha, double *beta, double *v, double *eigvals,
             double *eigvecs);
