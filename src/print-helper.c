#include "print-helper.h"

void print_matrix(double *matrix,  int size1,  int size2) {
  printf("Matrix\n");
  for (int i = 0; i < size1; i++) {
    for (int j = 0; j < size2; j++) {
      printf("%f ", matrix[i * size1 + j]);
    }
    printf("\n");
  }
}

void print_eigen_vals(double *eigen_vals,  int size) {
  printf("eigen_vals\n");
  for (int i = 0; i < size; i++) {
    printf("%f\n", eigen_vals[i]);
  }
}
