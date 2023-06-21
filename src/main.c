#include "lanczos.h"
#include "matrix-util.h"
#include "print-helper.h"
#if defined(ENABLE_BENCH)
#include "../bench/bench.h"
#endif

int main(int argc, char *argv[]) {
  char *file_name = "../data/sparse-matrices/delaunay_n13.mtx";
  // (argc > 1 ? argv[1] : "../data/sparse-matrices/dictionary28.mtx");
  int size = 1e4;            // (argc > 2 ? atoi(argv[2]) : 10);
  int do_read_from_file = 1; //(argc > 3 ? atoi(argv[3]) : 1);

  // Create Laplacian matrix
  int *row_ptrs, *columns, val_count;
  double *vals;
  double *lap = (double *)calloc(size * size, sizeof(double));
  if (do_read_from_file > 0) {
    mm_to_csr(file_name, &row_ptrs, &columns, &vals, &size, &size, &val_count);
  } else {
    create_lap(lap, size, 100000);
    lap_to_csr(lap, size, size, &row_ptrs, &columns, &vals, &val_count);
  }
  int m = size;

  // Run Lanczos algorithm
  double *eigvals = (double *)calloc(m, sizeof(double));
  double *eigvecs = (double *)calloc(m * size, sizeof(double));

#if defined(ENABLE_BENCH)
  lanczos_bench(argc, argv);
#else
  time_data vec_norm = {0, 0}, vec_dot = {0, 0}, vec_sclr_mul = {0, 0},
            vec_copy = {0, 0}, spmv = {0, 0}, calc_w = {0, 0},
            calc_w_init = {0, 0};
  time_struct time_measure = {&vec_norm, &vec_dot, &vec_sclr_mul, &vec_copy,
                              &spmv,     &calc_w,  &calc_w_init};

  lanczos(row_ptrs, columns, vals, val_count, size, m, eigvals, eigvecs,
          &time_measure, argc, argv);
  // print_eigen_vals(eigvals, size);
  print_kernel_time(&time_measure);
#endif

  free(lap), free(eigvals), free(eigvecs), free(row_ptrs), free(columns),
      free(vals);

  return 0;
}
