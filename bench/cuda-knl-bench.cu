#include "bench.h"

void vec_sclr_mul_bench() {
  FILE *fp = open_file("vec-sclr-mul-cuda");
  for (int i = 1e4; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);

    double *d_a, *d_b;

    // Allocate device memory
    cudaMalloc((void **)&d_a, (i) * sizeof(double));
    cudaMalloc((void **)&d_b, (i) * sizeof(double));

    cudaMemcpy(d_a, h_a, (i) * sizeof(double), cudaMemcpyHostToDevice);
    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Warmup
    for (int j = 0; j < WARMUP; j++)
      cuda_vec_sclr_mul(d_a, d_b, 1 / 10, i, grid_size, BLOCK_SIZE);

    cudaDeviceSynchronize();

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      cuda_vec_sclr_mul(d_a, d_b, 1 / 10, i, grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();
    t = clock() - t;

    double *w_vec = (double *)calloc(i, sizeof(double));
    double *out = (double *)calloc(i, sizeof(double));
    serial_vec_sclr_mul(h_a, w_vec, 1 / 10, i);

    cudaMemcpy(out, d_b, i * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < i; k++) {
      assert(fabs(w_vec[k] - out[k]) < EPS);
    }

    cudaFree(d_a), cudaFree(d_b);
    fprintf(fp, "%s,%s,%u,%u,%e\n", "vec-sclr-mul", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));
    tfree(&h_a);
  }
  fclose(fp);
}

void vec_sclr_div_bench() {
  FILE *fp = open_file("vec-sclr-div-cuda");
  for (int i = 1e4; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);

    double *d_a, *d_b;

    // Allocate device memory
    cudaMalloc((void **)&d_a, (i) * sizeof(double));
    cudaMalloc((void **)&d_b, (i) * sizeof(double));

    cudaMemcpy(d_a, h_a, (i) * sizeof(double), cudaMemcpyHostToDevice);
    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Warmup
    for (int j = 0; j < WARMUP; j++)
      cuda_vec_sclr_div(d_a, d_b, 10, i, grid_size, BLOCK_SIZE);

    cudaDeviceSynchronize();

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      cuda_vec_sclr_div(d_a, d_b, 10, i, grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();
    t = clock() - t;

    double *w_vec = (double *)calloc(i, sizeof(double));
    double *out = (double *)calloc(i, sizeof(double));
    serial_vec_sclr_div(h_a, w_vec, 10, i);

    cudaMemcpy(out, d_b, i * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < i; k++) {
      assert(fabs(w_vec[k] - out[k]) < EPS);
    }

    cudaFree(d_a), cudaFree(d_b);
    fprintf(fp, "%s,%s,%u,%u,%e\n", "vec-sclr-div", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));
    tfree(&h_a);
  }
  fclose(fp);
}

void calc_w_bench() {
  FILE *fp = open_file("calc_w_cuda");
  for (int i = 1e2; i < 3.7e4; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = create_host_vec(i * i);

    double *d_a, *d_b;

    // Allocate device memory
    cudaMalloc((void **)&d_a, (i) * sizeof(double));
    cudaMalloc((void **)&d_b, (i * i) * sizeof(double));

    cudaMemcpy(d_a, h_a, (i) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (i * i) * sizeof(double), cudaMemcpyHostToDevice);
    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Warmup
    for (int j = 0; j < WARMUP; j++)
      cuda_calc_w(d_a, 10, d_b, 20, i - 1, i, grid_size, BLOCK_SIZE);

    cudaDeviceSynchronize();

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      cuda_calc_w(d_a, 10, d_b, 20, i - 1, i, grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();
    t = clock() - t;

    double *out = (double *)calloc(i, sizeof(double));
    serial_calc_w(h_a, 10, h_b, 20, i - 1, i);

    cudaMemcpy(out, d_b, i * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < i; k++) {
      assert(fabs(h_b[k] - out[k]) < EPS);
    }

    cudaFree(d_a), cudaFree(d_b);
    fprintf(fp, "%s,%s,%u,%u,%e\n", "calc_w", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));
    tfree(&h_a);
    tfree(&h_b);
  }
  fclose(fp);
}

void vec_norm_bench() {
  FILE *fp = open_file("vec-norm");
  for (int i = 1e4; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *d_a;

    double out;
    cudaMalloc((void **)&d_a, i * sizeof(double));
    cudaMemcpy(d_a, h_a, i * sizeof(double), cudaMemcpyHostToDevice);

    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup runs
    for (int j = 0; j < WARMUP; j++)
      cuda_vec_norm(d_a, i, grid_size, BLOCK_SIZE);

    // Measure time
    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      out = cuda_vec_norm(d_a, i, grid_size, BLOCK_SIZE);
    t = clock() - t;

    double norm_val = serial_vec_norm(h_a, i);

    assert(fabs(norm_val - out) < EPS);

    fprintf(fp, "%s,%s,%u,%u,%e\n", "vec-norm", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));

    cudaFree(d_a);
    tfree(&h_a);
  }
  fclose(fp);
}

void vec_dot_bench() {
  FILE *fp = open_file("vec-dot");
  for (int i = 1e4; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = create_host_vec(i);
    double *d_a, *d_b;
    double out;
    cudaMalloc((void **)&d_a, i * sizeof(double));
    cudaMemcpy(d_a, h_a, i * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_b, i * sizeof(double));
    cudaMemcpy(d_b, h_b, i * sizeof(double), cudaMemcpyHostToDevice);

    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup runs
    for (int j = 0; j < WARMUP; j++)
      cuda_vec_dot(d_a, d_b, i, grid_size, BLOCK_SIZE);

    // Measure time
    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      out = cuda_vec_dot(d_a, d_b, i, grid_size, BLOCK_SIZE);
    t = clock() - t;

    double dot = serial_vec_dot(h_a, h_b, i);

    assert(fabs(dot - out) < EPS);

    fprintf(fp, "%s,%s,%u,%u,%e\n", "vec-dot", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));

    cudaFree(d_a);
    cudaFree(d_b);
    tfree(&h_a);
    tfree(&h_b);
  }
  fclose(fp);
}

void mtx_col_copy_bench() {
  FILE *fp = open_file("mtx-col-copy");
  for (int i = 1e2; i < 3.7e4; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *d_a, *d_b;
    cudaMalloc((void **)&d_a, i * sizeof(double));
    cudaMalloc((void **)&d_b, i * i * sizeof(double));
    cudaMemcpy(d_a, h_a, i * sizeof(double), cudaMemcpyHostToDevice);

    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup runs
    for (int j = 0; j < WARMUP; j++)
      cuda_mtx_col_copy(d_a, d_b, 0, i, grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();
    // Measure time
    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      cuda_mtx_col_copy(d_a, d_b, 0, i, grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();
    t = clock() - t;

    double *mtx = (double *)calloc(i * i, sizeof(double));
    double *out = (double *)calloc(i, sizeof(double));
    serial_mtx_col_copy(h_a, mtx, 0, i);

    cudaMemcpy(out, d_b, i * i * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < i * i; k++) {
      assert(fabs(mtx[k] - out[k]) < EPS);
    }

    fprintf(fp, "%s,%s,%u,%u,%e\n", "mtx-col-copy", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));

    cudaFree(d_a), cudaFree(d_b);
    tfree(&h_a);
  }
  fclose(fp);
}

void create_roofline() {
  FILE *fp = open_file("roofline_data");
  for (int i = 1e4; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);

    double *d_a, *d_b;

    // Allocate device memory
    cudaMalloc((void **)&d_a, (i) * sizeof(double));
    cudaMalloc((void **)&d_b, (i) * sizeof(double));

    cudaMemcpy(d_a, h_a, (i) * sizeof(double), cudaMemcpyHostToDevice);

    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup d2d
    for (int j = 0; j < WARMUP; j++)
      cuda_d2d_mem_cpy(d_a, d_b, i, grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      cuda_d2d_mem_cpy(d_a, d_b, i, grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();
    t = clock() - t;

    double *out = (double *)calloc(i, sizeof(double));

    cudaMemcpy(out, d_b, i * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < i; k++) {
      assert(fabs(h_a[k] - out[k]) < EPS);
    }

    cudaFree(d_a), cudaFree(d_b);
    fprintf(fp, "%s,%s,%u,%u,%e\n", "roofline", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));
    tfree(&h_a);
  }
  fclose(fp);
}

void spmv_bench() {
  FILE *fp = open_file("spmv_data");
  for (int i = 1e2; i < 1e4; i = inc(i)) {
    double *lap, *vals, *h_orth_vec;
    int *row_ptrs, *columns, val_count;
    lap = (double *)calloc(i * i, sizeof(double));
    create_lap(lap, i, 100);
    lap_to_csr(lap, i, i, &row_ptrs, &columns, &vals, &val_count);
    h_orth_vec = create_host_vec(i);
    double *d_vals, *d_orth_mtx, *d_orth_vec, *d_w_vec;
    int *d_row_ptrs, *d_columns;

    // Allocate device memory
    cudaMalloc((void **)&d_row_ptrs, (i + 1) * sizeof(int));
    cudaMalloc((void **)&d_columns, (val_count) * sizeof(int));
    cudaMalloc((void **)&d_vals, (val_count) * sizeof(double));
    cudaMalloc((void **)&d_orth_vec, (i) * sizeof(double));
    cudaMalloc((void **)&d_w_vec, (i) * sizeof(double));

    // H2D memory copy
    cudaMemcpy(d_row_ptrs, row_ptrs, (i + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, columns, (val_count) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals, (val_count) * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_orth_vec, h_orth_vec, (i) * sizeof(double),
               cudaMemcpyHostToDevice);

    int grid_size = (i + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup d2d
    for (int j = 0; j < WARMUP; j++)
      cuda_spmv(d_row_ptrs, d_columns, d_vals, d_orth_vec, d_w_vec, i, i,
                grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      cuda_spmv(d_row_ptrs, d_columns, d_vals, d_orth_vec, d_w_vec, i, i,
                grid_size, BLOCK_SIZE);
    cudaDeviceSynchronize();
    t = clock() - t;

    double *w_vec = (double *)calloc(i, sizeof(double));
    double *out = (double *)calloc(i, sizeof(double));
    serial_spmv(row_ptrs, columns, vals, h_orth_vec, w_vec, i, i);

    cudaMemcpy(out, d_w_vec, i * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < i; k++) {
      assert(fabs(w_vec[k] - out[k]) < EPS);
    }
    // Free device memory
    cudaFree(d_row_ptrs), cudaFree(d_columns), cudaFree(d_vals),
        cudaFree(d_orth_vec), cudaFree(d_w_vec);

    fprintf(fp, "%s,%s,%u,%u,%e,%u\n", "spmv", "cuda", 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS), val_count);

    tfree(&lap);
    tfree(&vals);
    tfree(&row_ptrs);
    tfree(&columns);
    tfree(&h_orth_vec);
  }
  fclose(fp);
}

void lanczos_bench(int argc, char *argv[]) {
  vec_sclr_mul_bench();
  vec_sclr_div_bench();
  calc_w_bench();
  vec_norm_bench();
  vec_dot_bench();
  mtx_col_copy_bench();
  create_roofline();
  spmv_bench();
}
