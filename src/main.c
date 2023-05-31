#include "lanczos.h"
#define SIZE 5
void create_lap(double *lap, const unsigned size) {
  // Create random binary matrix
  double adj[size * size];
  for (unsigned i = 0; i < size * size; i++) {
    adj[i] = rand() % 2;
  }

  // Make matrix symmetric
  for (unsigned i = 0; i < size; i++)
    for (unsigned j = i + 1; j < size; j++)
      adj[i * size + j] = adj[j * size + i];

  // Create degree matrix
  double diag[size * size];
  for (unsigned i = 0; i < size; i++) {
    double sum = 0;
    for (unsigned j = 0; j < size; j++) {
      diag[i * size + j] = 0;
      sum += adj[i * size + j];
    }
    diag[i * size + i] = sum;
  }

  // Create Laplacian matrix
  for (unsigned i = 0; i < SIZE * SIZE; i++)
    lap[i] = diag[i] - adj[i];
}

int main(int argc, char *argv[]) {
  const unsigned M = SIZE;

  // Create Laplacian matrix
  double *lap = (double *)calloc(SIZE * SIZE, sizeof(double));
  create_lap(lap, SIZE);

  // Run Lanczos algorithm
  double *eigvals = (double *)calloc(M, sizeof(double));
  double *eigvecs = (double *)calloc(M * SIZE, sizeof(double));

  lanczos(lap, SIZE, M, eigvals, eigvecs, argc, argv);
  // print_eigen_vals(eigvals, SIZE);
  free(lap), free(eigvals), free(eigvecs);

  free(lap), free(eigvals), free(eigvecs);

  return 0;
}
