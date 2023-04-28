#include "lanczos.h"

#define MAX 1000000

void create_lap(double *lap, const int SIZE) {
  // Create random binary matrix
  double adj[SIZE * SIZE];
  for (unsigned i = 0; i < SIZE * SIZE; i++) {
    adj[i] = ((double)rand() / (RAND_MAX) < 0.5) ? 0 : 1;
  }
  // Make matrix symmetric
  for (unsigned i = 0; i < SIZE; i++)
    for (unsigned j = i + 1; j < SIZE; j++)
      adj[i * SIZE + j] = adj[j * SIZE + i];

  // Create degree matrix
  double diag[SIZE * SIZE];
  for (unsigned i = 0; i < SIZE; i++) {
    int sum = 0;
    for (unsigned j = 0; j < SIZE; j++)
      sum += adj[i * SIZE + j];
    diag[i * SIZE + i] = sum;
  }

  // Create Laplacian matrix
  for (unsigned i = 0; i < SIZE * SIZE; i++)
    lap[i] = diag[i] - adj[i];
}

void lanczos_aux(double *V, double *T, double *alpha, double *beta, double *v,
                 const int SIZE, const int M) {
  for (unsigned i = 0; i < SIZE * M; i++)
    V[i] = 0;
  for (unsigned i = 0; i < M * M; i++)
    T[i] = 0;
  for (unsigned i = 0; i < M; i++)
    alpha[i] = 0;
  for (unsigned i = 0; i < M - 1; i++)
    beta[i] = 0;
  for (unsigned i = 0; i < SIZE; i++)
    v[i] = (double)rand() / (double)(RAND_MAX / MAX);
  double diviser = 0;
  for (unsigned i = 0; i < SIZE; i++)
    diviser += v[i] * v[i];
  diviser = sqrt(diviser);
  for (unsigned i = 0; i < SIZE; i++)
    v[i] = v[i] / diviser;
  for (unsigned i = 0; i < SIZE; i++)
    V[i] = v[i];
}

void matrix_mul(double *A, double *B, double *out, const int height_a,
                const int width_a, const int width_b) {

  for (unsigned i = 0; i < height_a; i++) {
    for (unsigned j = 0; j < width_b; j++) {
      double dot = 0;
      for (unsigned k = 0; k < width_a; k++)
        dot += A[i * width_a + k] * B[k * width_b + j];
      out[i * width_b + j] = dot;
    }
  }
}
double matrix_dot(double *v, double *w, const int SIZE) {}
void w_calc(double *w, double alpha, int j, double *V, double beta,
            const int SIZE) {}
double norm(double *w, const int SIZE) {}
void w_modify(double *w, double beta, double *v, const int SIZE) {}
void update_v(double *v, double *V, int j, const int SIZE) {}
void make_tri(double *alpha, double *beta, double *T, const int M) {}

void lanczos(double *lap, const int SIZE, const int M, double *V, double *T,
             double *alpha, double *beta, double *v, double *eigvals,
             double *eigvecs) {
  double *w;

  w = (double *)calloc(SIZE, sizeof(double));
  for (unsigned i = 0; i < M; i++) {
    // w = A @ v
    matrix_mul(lap, v, w, SIZE, SIZE, 1);

    // alpha[j] = np.dot(v, w)
    alpha[i] = matrix_dot(v, w, SIZE);

    if (i == M - 1)
      break;

    // w = w - alpha[j] * V[:, j] - beta[j] * V[:, j-1]
    w_calc(w, alpha[i], i, V, beta[i], SIZE);

    // beta[j] = np.linalg.norm(w)
    beta[i] = norm(w, SIZE);

    if (beta[i] == 0)
      break;

    // v = w / beta[j]
    w_modify(w, beta[i], v, SIZE);

    // V[:, j+1] = v
    update_v(v, V, i, SIZE);
  }

  // Make triadiagonal matrix
  make_tri(alpha, beta, T, M);
}
