#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void create_lap(double *lap, const int SIZE);
void print_matrix(double *matrix, const int SIZE1, const int SIZE2);
void print_eigen_vals(double *eigen_vals, const int SIZE);
void lanczos(double *lap, const int SIZE, const int M, double *eigvals,
             double *eigvecs);
