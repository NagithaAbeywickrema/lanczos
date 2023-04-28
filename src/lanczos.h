#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void create_lap(double *lap, const int SIZE);
void lanczos(double *lap, const int SIZE, const int M, double *eigvals,
             double *eigvecs);
