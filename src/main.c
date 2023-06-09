#include "lanczos.h"
#define SIZE 500
void create_lap(double *lap, const unsigned size) {
  // Create random binary matrix
  double *adj = (double *)calloc(SIZE * SIZE, sizeof(double));
  for (unsigned i = 0; i < size * size; i++) {
    adj[i] = rand() % 2;
  }

  // Make matrix symmetric
  for (unsigned i = 0; i < size; i++)
    for (unsigned j = i + 1; j < size; j++)
      adj[i * size + j] = adj[j * size + i];

  // Create degree matrix
  double *diag = (double *)calloc(SIZE * SIZE, sizeof(double));
  for (unsigned i = 0; i < size; i++) {
    double sum = 0;
    for (unsigned j = 0; j < size; j++) {
      diag[i * size + j] = 0;
      sum += adj[i * size + j];
    }
    diag[i * size + i] = sum;
  }

  // Create Laplacian matrix
  for (unsigned i = 0; i < size * size; i++)
    lap[i] = diag[i] - adj[i];
  free(adj), free(diag);
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

  free(lap), free(eigvals), free(eigvecs);

  return 0;
}
