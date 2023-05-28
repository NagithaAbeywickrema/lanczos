#include "print-helper.h"

void print_matrix(double *matrix, const int size1, const int size2) {
  printf("Matrix\n");
  for (unsigned i = 0; i < size1; i++) {
    for (unsigned j = 0; j < size2; j++) {
      printf("%f ", matrix[i * size1 + j]);
    }
    printf("\n");
  }
}

void print_eigen_vals(double *eigen_vals, const int size) {
  printf("eigen_vals\n");
  for (unsigned i = 0; i < size; i++) {
    printf("%f\n", eigen_vals[i]);
  }
}
