#include "lanczos.h"

int main(int argc, char *argv[]) {
  const int SIZE = 5;
  const int M = SIZE; // TODO: replace M with SIZE
  double *lap, *eigvals, *eigvecs, *V, *T, *alpha, *beta, *v,
      *w; // TODO: move decl to init

  // Create Laplacian matrix
  lap = (double *)calloc(SIZE * SIZE, sizeof(double));
  create_lap(lap, SIZE);

  // Initialize variables
  V = (double *)calloc(SIZE * M, sizeof(double));
  T = (double *)calloc(M * M, sizeof(double));
  alpha = (double *)calloc(M, sizeof(double));
  beta = (double *)calloc(M - 1, sizeof(double));
  v = (double *)calloc(SIZE, sizeof(double));
  w = (double *)calloc(SIZE, sizeof(double));
  lanczos_aux(V, T, alpha, beta, v, SIZE, M);

#pragma nomp init(argc, argv);
#pragma nomp update(alloc : w [0:SIZE], alpha [0:M], beta [0:M])
#pragma nomp update(to                                                         \
                    : lap[0, SIZE * SIZE], V[0, SIZE * M], T[0, M * M],        \
                      v [0:SIZE]);

  // Run Lanczos algorithm
  eigvals = (double *)calloc(M, sizeof(double));
  eigvecs = (double *)calloc(M * SIZE, sizeof(double));
  lanczos(lap, SIZE, M, V, T, alpha, beta, v, eigvals, eigvecs);

  return 0;
}
