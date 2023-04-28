#include "lanczos.h"

int main(int argc, char *argv[]) {
  const int SIZE = 5;
  const int M = SIZE; // TODO: replace M with SIZE?

  // Create Laplacian matrix
  double *lap = (double *)calloc(SIZE * SIZE, sizeof(double));
  create_lap(lap, SIZE);

  // Run Lanczos algorithm
  double *eigvals = (double *)calloc(M, sizeof(double));
  double *eigvecs = (double *)calloc(M * SIZE, sizeof(double));
  lanczos(lap, SIZE, M, eigvals, eigvecs);

  return 0;
}
