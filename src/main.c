#include "lanczos.h"
#include "matrix-util.h"
#include "print-helper.h"

void create_lap(double *lap, int size) {
  // Create random binary matrix
  double *adj = (double *)calloc(size * size, sizeof(double));
  for (int i = 0; i < size * size; i++) {
    adj[i] = rand() % 2;
  }

  // Make matrix symmetric
  for (int i = 0; i < size; i++)
    for (int j = i + 1; j < size; j++)
      adj[i * size + j] = adj[j * size + i];

  // Create degree matrix
  double *diag = (double *)calloc(size * size, sizeof(double));
  for (int i = 0; i < size; i++) {
    double sum = 0;
    for (int j = 0; j < size; j++) {
      diag[i * size + j] = 0;
      sum += adj[i * size + j];
    }
    diag[i * size + i] = sum;
  }

  // Create Laplacian matrix
  for (int i = 0; i < size * size; i++)
    lap[i] = diag[i] - adj[i];
  free(adj), free(diag);
}

void lap_to_csr(double *matrix, int rows, int cols, int **row_ptrs,
                int **columns, double **vals, int *nnz) {
  // Count the number of non-zero elements
  (*nnz) = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (matrix[i * cols + j] != 0)
        (*nnz)++;
    }
  }

  // Allocate memory for the CSR arrays
  *row_ptrs = (int *)malloc((rows + 1) * sizeof(int));
  *columns = (int *)malloc((*nnz) * sizeof(int));
  *vals = (double *)malloc((*nnz) * sizeof(double));

  // Convert matrix to CSR format
  int k = 0; // Index for the vals and columns arrays
  (*row_ptrs)[0] = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
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
  char *file_name =
      (argc > 1 ? argv[1] : "../data/sparse-matrices/small-test.mtx");
  int size = 10;             // (argc > 2 ? atoi(argv[2]) : 10);
  int do_read_from_file = 0; //(argc > 3 ? atoi(argv[3]) : 1);

  // Create Laplacian matrix
  int *row_ptrs, *columns, val_count;
  double *vals;
  double *lap = (double *)calloc(size * size, sizeof(double));
  if (do_read_from_file > 0) {
    mm_to_csr(file_name, &row_ptrs, &columns, &vals, &size, &size, &val_count);
  } else {
    create_lap(lap, size);
    lap_to_csr(lap, size, size, &row_ptrs, &columns, &vals, &val_count);
  }
  int m = size;

  // Run Lanczos algorithm
  double *eigvals = (double *)calloc(m, sizeof(double));
  double *eigvecs = (double *)calloc(m * size, sizeof(double));

#if defined(ENABLE_BENCH)
  lanczos_bench(argc, argv);
#else
  lanczos(row_ptrs, columns, vals, val_count, size, m, eigvals, eigvecs, argc,
          argv);
  print_eigen_vals(eigvals, size);
#endif

  free(lap), free(eigvals), free(eigvecs), free(row_ptrs), free(columns),
      free(vals);

  return 0;
}
