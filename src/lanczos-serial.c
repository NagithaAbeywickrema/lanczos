#include "lanczos-aux.h"
#include "lanczos.h"

#define MAX 1000000
#define EPS 1e-12
#define MAX_ITER 100000

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

void create_identity_matrix(double *out, const int SIZE) {
  for (unsigned i = 0; i < SIZE; i++) {
    for (unsigned j = 0; j < SIZE; j++) {
      if (i == j)
        out[i * SIZE + j] = 1;
      else
        out[i * SIZE + j] = 0;
    }
  }
}

void mtx_mul(double *A, double *B, double *out, const int height_a,
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

void mtx_vec_mul(double *A, double *B, double *out, const int height_a,
                 const int width_a) {

  for (unsigned i = 0; i < height_a; i++) {
    double dot = 0;
    for (unsigned k = 0; k < width_a; k++)
      dot += A[i * width_a + k] * B[k];
    out[i] = dot;
  }
}

double mtx_dot(double *v, double *w, const int SIZE) {
  double dot = 0;
  for (unsigned i = 0; i < SIZE; i++)
    dot += v[i] * w[i];
  return dot;
}
void mtx_sclr_div(double *in, double scalar, double *out, const int SIZE) {
  for (unsigned i = 0; i < SIZE; i++)
    out[i] = in[i] / scalar;
}

void calc_w_init(double *w, double *alpha, double *V, unsigned i,
                 const int SIZE) {
  for (int j = 0; j < SIZE; j++) {
    w[j] = w[j] - alpha[i] * V[j + SIZE * i];
  }
}

void calc_w(double *w, double *alpha, double *V, double *beta, unsigned i,
            const int SIZE) {
  for (int j = 0; j < SIZE; j++) {
    w[j] = w[j] - alpha[i] * V[j + SIZE * i] - beta[i] * V[j + SIZE * (i - 1)];
  }
}

double mtx_norm(double *w, const int SIZE) {
  double total = mtx_dot(w, w, SIZE);
  return sqrt(total);
}

void mtx_col_copy(double *v, double *V, int j, const int SIZE) {

  memcpy(V + SIZE * j, v, sizeof(double) * SIZE);
}

void make_tri(double *alpha, double *beta, double *T, const int M) {
  for (unsigned i = 0; i < M; i++) {
    T[i * M + i] = alpha[i];
    if (i < M - 1) {
      T[i * M + i + 1] = beta[i + 1];
      T[(i + 1) * M + i] = beta[i + 1];
    }
  }
}

void print_eigen_vals(double *eigen_vals, const int SIZE) {
  printf("eigen_vals\n");
  for (unsigned i = 0; i < SIZE; i++) {
    printf("%f\n", eigen_vals[i]);
  }
}

void print_matrix(double *matrix, const int SIZE1, const int SIZE2) {
  printf("Matrix\n");
  for (unsigned i = 0; i < SIZE1; i++) {
    for (unsigned j = 0; j < SIZE2; j++) {
      printf("%f ", matrix[i * SIZE1 + j]);
    }
    printf("\n");
  }
}

void lanczos(double *lap, const int SIZE, const int M, double *eigvals,
             double *eigvecs) {
  print_matrix(lap, SIZE, SIZE);

  // Allocate memory
  double *V = (double *)calloc(SIZE * M, sizeof(double));
  double *T = (double *)calloc(M * M, sizeof(double));
  double *alpha = (double *)calloc(M, sizeof(double));
  double *beta = (double *)calloc(M, sizeof(double));
  double *v = (double *)calloc(SIZE, sizeof(double));
  double *w = (double *)calloc(SIZE, sizeof(double));

  // Initialize host memory
  for (unsigned i = 0; i < SIZE; i++)
    w[i] = 0;

  // Lanczos iteration
  for (unsigned i = 0; i < M; i++) {
    beta[i] = mtx_norm(w, SIZE);

    if (beta[i] != 0) {
      mtx_sclr_div(w, beta[i], v, SIZE);
    } else {
      for (unsigned i = 0; i < SIZE; i++)
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
      double norm_val = mtx_norm(v, SIZE);
      mtx_sclr_div(v, norm_val, v, SIZE);
    }

    mtx_col_copy(v, V, i, SIZE);

    mtx_vec_mul(lap, v, w, SIZE, SIZE);

    alpha[i] = mtx_dot(v, w, SIZE);

    if (i == 0) {
      calc_w_init(w, alpha, V, i, SIZE);
    } else {
      calc_w(w, alpha, V, beta, i, SIZE);
    }
  }

  tqli(eigvecs, eigvals, SIZE, alpha, beta, 0);
  // not sorted
  print_eigen_vals(eigvals, SIZE);
}
