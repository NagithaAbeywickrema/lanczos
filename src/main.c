#include "lanczos.h"
#include "print-helper.h"
#include "matrix-util.h"

#define SIZE 10

void create_lap(double *lap, const unsigned size) {
  // Create random binary matrix
  double *adj = (double *)calloc(size * size, sizeof(double));
  for (unsigned i = 0; i < size * size; i++) {
    adj[i] = rand() % 2;
  }

  // Make matrix symmetric
  for (unsigned i = 0; i < size; i++)
    for (unsigned j = i + 1; j < size; j++)
      adj[i * size + j] = adj[j * size + i];

  // Create degree matrix
  double *diag = (double *)calloc(size * size, sizeof(double));
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

void lap_to_csr(double *matrix, int rows, int cols, int **row_ptrs,
                int **columns, double **vals, int *nnz) {
  int i, j = 0;
  (*nnz) = 0;

  // Count the number of non-zero elements
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      if (matrix[i * cols + j] != 0)
        (*nnz)++;
    }
  }

  // Allocate memory for the CSR arrays
  *row_ptrs = (int *)malloc((rows + 1) * sizeof(int));
  *columns = (int *)malloc((*nnz) * sizeof(int));
  *vals = (double *)malloc((*nnz) * sizeof(double));

  int k = 0; // Index for the vals and columns arrays
  (*row_ptrs)[0] = 0;

  // Convert matrix to CSR format
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      if (matrix[i * cols + j] != 0) {
        (*vals)[k] = matrix[i * cols + j];
        (*columns)[k] = j;
        k++;
      }
    }
    (*row_ptrs)[i + 1] = k;
  }
}

int main(int argc, char *argv[]) {
  unsigned M = SIZE;
  unsigned N = SIZE;
  char *file_name = "../src/dictionary28.mtx";

  // Create Laplacian matrix
  int *row_ptrs, *columns, val_count;
  double *vals;
  double *lap = (double *)calloc(SIZE * SIZE, sizeof(double));
  // create_lap(lap, SIZE);
  // lap_to_csr(lap, SIZE, SIZE, &row_ptrs, &columns, &vals, &val_count);

  mm_to_csr(file_name, &row_ptrs, &columns, &vals, &N, &M, &val_count);
  // Run Lanczos algorithm
  double *eigvals = (double *)calloc(M, sizeof(double));
  double *eigvecs = (double *)calloc(M * N, sizeof(double));


  lanczos(row_ptrs, columns, vals, val_count, N, M, eigvals, eigvecs, argc,
          argv);
  print_eigen_vals(eigvals, SIZE);

  free(lap), free(eigvals), free(eigvecs), free(row_ptrs), free(columns),
      free(vals);

  return 0;
}
