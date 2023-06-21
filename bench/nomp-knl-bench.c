#include "bench.h"
#include <assert.h>

void vec_norm_bench(char *backend) {
  FILE *fp = open_file("lanczos_vec_norm_data");
  for (int i = 1e3; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double s_norm = serial_vec_norm(h_a, i);
    double n_norm;
#pragma nomp update(to : h_a[0, i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_vec_norm(h_a, i);

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      n_norm = nomp_vec_norm(h_a, i);
    t = clock() - t;

    // assert((fabs(s_norm - n_norm) / i) < EPS);
    fprintf(fp, "%s,%s_%s,%u,%u,%e\n", "vec-norm", "nomp", 32, backend, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));
#pragma nomp update(free : h_a[0, i])
    tfree(&h_a);
  }
  fclose(fp);
}

void vec_dot_bench(char *backend) {
  FILE *fp = open_file("lanczos_vec_dot_without_d2d_data-prof");
  for (int i = 1e3; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = create_host_vec(i);
    double s_dot = serial_vec_dot(h_a, h_b, i);
    double n_dot;
#pragma nomp update(to : h_a[0, i], h_b[0, i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_vec_dot(h_a, h_b, i);
#pragma nomp sync
    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      nomp_vec_dot(h_a, h_b, i);
#pragma nomp sync
    t = clock() - t;
    // assert((fabs(s_dot - n_dot) / i) < EPS);

    fprintf(fp, "%s,%s_%s,%u,%u,%e\n", "vec-dot-without-d2h", "nomp", backend,
            32, i, (double)t / (CLOCKS_PER_SEC * TRAILS));
#pragma nomp update(free : h_a[0, i], h_b[0, i])
    tfree(&h_a), tfree(&h_b);
  }
  fclose(fp);
}

void vec_sclr_mul_bench(char *backend) {
  FILE *fp = open_file("lanczos_vec_sclr_mul_data");
  for (int i = 1e3; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = (double *)malloc(sizeof(double) * i);
    double *h_c = (double *)malloc(sizeof(double) * i);
    serial_vec_sclr_mul(h_a, h_c, 1 / 10, i);
#pragma nomp update(to : h_a[0, i])
#pragma nomp update(to : h_b[0, i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_vec_sclr_mul(h_a, h_b, 1 / 10, i);
#pragma nomp sync

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      nomp_vec_sclr_mul(h_a, h_b, 1 / 10, i);
#pragma nomp sync
    t = clock() - t;

#pragma nomp update(from : h_b[0, i])
    for (int k = 0; k < i; k++)
      assert(fabs(h_b[k] - h_c[k]) < EPS);

    fprintf(fp, "%s,%s_%s,%u,%u,%e\n", "vec-sclr-mul", "nomp", backend, 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));

#pragma nomp update(free : h_a[0, i], h_b[0, i])
    tfree(&h_a);
    free(h_b), free(h_c);
  }
  fclose(fp);
}

void vec_add_bench(char *backend) {
  FILE *fp = open_file("lanczos_vec_add_data");
  for (int i = 1e3; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = create_host_vec(i);
    double *h_c = (double *)malloc(sizeof(double) * i);
    double *h_d = (double *)malloc(sizeof(double) * i);
    serial_vec_add(h_a, h_b, h_d, i);

#pragma nomp update(to : h_a[0, i], h_b[0, i])
#pragma nomp update(alloc : h_c[0, i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_vec_add(h_a, h_b, h_c, i);
#pragma nomp sync

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      nomp_vec_add(h_a, h_b, h_c, i);
#pragma nomp sync
    t = clock() - t;

#pragma nomp update(from : h_c[0, i])
    for (int k = 0; k < i; k++)
      assert(fabs(h_c[k] - h_d[k]) < EPS);

    fprintf(fp, "%s,%s_%s,%u,%u,%e\n", "vec-add", "nomp", backend, 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));

#pragma nomp update(free : h_a[0, i], h_b[0, i])
    tfree(&h_a), tfree(&h_b);
    free(h_c), free(h_d);
  }
  fclose(fp);
}

void vec_sclr_div_bench(char *backend) {
  FILE *fp = open_file("lanczos_vec_sclr_div_data-new");
  for (int i = 1e3; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = (double *)malloc(sizeof(double) * i);
    double *h_c = (double *)malloc(sizeof(double) * i);
    serial_vec_sclr_div(h_a, h_c, 10, i);

#pragma nomp update(to : h_a[0, i])
#pragma nomp update(to : h_b[0, i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_vec_sclr_div(h_a, h_b, 10, i);
#pragma nomp sync

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      nomp_vec_sclr_div(h_a, h_b, 10, i);
#pragma nomp sync
    t = clock() - t;

#pragma nomp update(from : h_b[0, i])
    for (int k = 0; k < i; k++)
      assert(fabs(h_b[k] - h_c[k]) < EPS);

    fprintf(fp, "%s,%s_%s,%u,%u,%e\n", "vec-sclr-div", "nomp", backend, 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));

#pragma nomp update(free : h_a[0, i], h_b[0, i])
    tfree(&h_a);
    free(h_b), free(h_c);
  }
  fclose(fp);
}

void mtx_col_copy_bench(char *backend) {
  FILE *fp = open_file("lanczos_mtx_col_copy_data");
  for (int i = 100; i < 3e4; i = inc(i)) {
    double *h_a = (double *)calloc(i * i, sizeof(double));
    double *h_b = create_host_vec(i);
    double *h_c = (double *)calloc(i * i, sizeof(double));
    serial_mtx_col_copy(h_b, h_c, i - 1, i);

#pragma nomp update(to : h_b[0, i])
#pragma nomp update(alloc : h_a[0, i * i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_mtx_col_copy(h_b, h_a, i - 1, i); // col_index

    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      nomp_mtx_col_copy(h_b, h_a, i - 1, i);
    t = clock() - t;

#pragma nomp update(from : h_a[0, i * i])
    for (int k = 0; k < i; k++)
      assert(fabs(h_a[k + i * (i - 1)] - h_c[k + i * (i - 1)]) < EPS);

    fprintf(fp, "%s,%s_%s,%u,%e\n", "mtx-col-copy", "nomp", backend, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));
#pragma nomp update(free : h_a[0, i * i], h_b[0, i])
    tfree(&h_b);
    free(h_a), free(h_c);
  }
  fclose(fp);
}

void calc_w_bench(char *backend) {
  FILE *fp = open_file("lanczos_calc_w_data");
  for (int i = 1e3; i < 1e7; i = inc(i)) {
    double *h_a = create_host_vec(i);
    double *h_b = create_host_vec(i);
    double *h_b_pre = create_host_vec(i);
    double *h_c = h_a;
    serial_calc_w(h_c, 2, h_b, h_b_pre, 2, i);

#pragma nomp update(to : h_a[0, i], h_b[0, i], h_b_pre[0, i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_calc_w(h_a, 10, h_b, h_b_pre, 20, i);
#pragma nomp sync
    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      nomp_calc_w(h_a, 10, h_b, h_b_pre, 20, i);
#pragma nomp sync
    t = clock() - t;

    // #pragma nomp update(from : h_a[0, i])
    // for (int k = 0; k < i; k++)
    //   assert(fabs(h_a[k] - h_c[k]) < EPS);

    fprintf(fp, "%s,%s_%s,%u,%u,%e\n", "calc-w", "nomp", backend, 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS));
#pragma nomp update(free : h_a[0, i], h_b[0, i], h_b_pre[0, i])
    tfree(&h_a), tfree(&h_b), tfree(&h_b_pre);
  }
  fclose(fp);
}

void spmv_bench(char *backend) {
  FILE *fp = open_file("lanczos_spmv_data");
  for (int i = 2e2; i < 3e4; i = inc(i)) {
    double *lap, *vals, *h_orth_vec;
    int *row_ptrs, *columns, val_count;
    lap = (double *)calloc(i * i, sizeof(double));

    create_lap(lap, i, 1000);
    lap_to_csr(lap, i, i, &row_ptrs, &columns, &vals, &val_count);
    h_orth_vec = create_host_vec(i);

    double *w_vec = (double *)calloc(i, sizeof(double));
    double *sw_vec = (double *)calloc(i, sizeof(double));

    serial_spmv(row_ptrs, columns, vals, h_orth_vec, sw_vec, i, i);

#pragma nomp update(to : row_ptrs[0, i + 1], columns[0, val_count],            \
                        vals[0, val_count], h_orth_vec[0, i], w_vec[0, i])

    // Warmup
    for (int j = 0; j < WARMUP; j++)
      nomp_spmv(row_ptrs, columns, vals, h_orth_vec, w_vec, i, i);
#pragma nomp sync
    clock_t t = clock();
    for (int j = 0; j < TRAILS; j++)
      nomp_spmv(row_ptrs, columns, vals, h_orth_vec, w_vec, i, i);
#pragma nomp sync
    t = clock() - t;

#pragma nomp update(from : w_vec[0, i])
    for (int k = 0; k < i; k++)
      assert(fabs(w_vec[k] - sw_vec[k]) < EPS);

    fprintf(fp, "%s,%s_%s,%u,%u,%e,%u\n", "spmv", "nomp", backend, 32, i,
            (double)t / (CLOCKS_PER_SEC * TRAILS), val_count);
#pragma nomp update(free : row_ptrs[0, i + 1], columns[0, val_count],          \
                        vals[0, val_count], h_orth_vec[0, i], w_vec[0, i])
    tfree(&lap);
    tfree(&vals);
    tfree(&row_ptrs);
    tfree(&columns);
    tfree(&h_orth_vec);
  }
  fclose(fp);
}

void lanczos_bench(int argc, char *argv[]) {
#pragma nomp init(argc, argv)
  unsigned i = 0;
  char *backend = (char *)malloc(BUFSIZ);
  while (i < argc) {
    if (!strncmp("--nomp", argv[i], 6)) {
      if (!strncmp("--nomp-backend", argv[i], BUFSIZ)) {
        strncpy(backend, argv[i + 1], BUFSIZ);
        i++;
      }
      i++;
    }
    i++;
  }
  // vec_add_bench(backend);
  // vec_sclr_mul_bench(backend);
  // vec_sclr_div_bench(backend);
  calc_w_bench(backend);
  vec_norm_bench(backend);
  vec_dot_bench(backend);
  // spmv_bench(backend);
#pragma nomp finalize
  free(backend);
}
