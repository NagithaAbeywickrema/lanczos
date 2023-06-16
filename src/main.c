#include "lanczos.h"
#include "matrix-util.h"
#include "print-helper.h"

#if defined(ENABLE_BENCH)
#include "../bench/bench.h"
#endif

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
    create_lap(lap, size, 10);
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
