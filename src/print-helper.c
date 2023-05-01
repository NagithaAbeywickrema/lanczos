#include "print-helper.h"

void print_matrix(double *matrix, const int SIZE1, const int SIZE2) {
  printf("Matrix\n");
  for (unsigned i = 0; i < SIZE1; i++) {
    for (unsigned j = 0; j < SIZE2; j++) {
      printf("%f ", matrix[i * SIZE1 + j]);
    }
    printf("\n");
  }
}

void print_eigen_vals(double *eigen_vals, const int SIZE) {
  printf("eigen_vals\n");
  for (unsigned i = 0; i < SIZE; i++) {
    printf("%f\n", eigen_vals[i]);
  }
}
