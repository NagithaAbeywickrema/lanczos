#include "lanczos.h"

void create_lap(double *lap, const int SIZE) {
  // Create random binary matrix
  double adj[SIZE * SIZE];
  for (unsigned i = 0; i < SIZE * SIZE; i++) {
    adj[i] = rand() % 2;
  }

  // Make matrix symmetric
  for (unsigned i = 0; i < SIZE; i++)
    for (unsigned j = i + 1; j < SIZE; j++)
      adj[i * SIZE + j] = adj[j * SIZE + i];

  // Create degree matrix
  double diag[SIZE * SIZE];
  for (unsigned i = 0; i < SIZE; i++) {
    double sum = 0;
    for (unsigned j = 0; j < SIZE; j++) {
      diag[i * SIZE + j] = 0;
      sum += adj[i * SIZE + j];
    }
    diag[i * SIZE + i] = sum;
  }

  // Create Laplacian matrix
  for (unsigned i = 0; i < SIZE * SIZE; i++)
    lap[i] = diag[i] - adj[i];
}

int main(int argc, char *argv[]) {
  const int SIZE = 30;
  const int M = SIZE; // TODO: replace M with SIZE?

  // Create Laplacian matrix
  double *lap = (double *)calloc(SIZE * SIZE, sizeof(double));
  create_lap(lap, SIZE);

  // Run Lanczos algorithm
  double *eigvals = (double *)calloc(M, sizeof(double));
  double *eigvecs = (double *)calloc(M * SIZE, sizeof(double));
  lanczos(lap, SIZE, M, eigvals, eigvecs, argc, argv);

  return 0;
}
