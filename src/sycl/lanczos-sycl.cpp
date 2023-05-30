#include "lanczos.h"

#define MAX 1000000
#define EPS 1e-12
#define MAX_ITER 100

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
             const int width_a, const int width_b, sycl::queue queue) {
  sycl::buffer a_buf{A, sycl::range<1>(height_a * width_a)};
  sycl::buffer b_buf{B, sycl::range<1>(width_a * width_b)};
  sycl::buffer out_buf{out, sycl::range<1>(height_a * width_b)};

  queue.submit([&](sycl::handler &h) {
    auto d_a = a_buf.get_access<sycl::access::mode::read>(h);
    auto d_b = b_buf.get_access<sycl::access::mode::read>(h);
    size_t LOCAL_SIZE = 16;

    auto d_c = out_buf.get_access<sycl::access::mode::read_write>(h);
    unsigned grid_rows =
        ((height_a + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
    unsigned grid_cols = ((width_b + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
    sycl::range local_size = sycl::range(LOCAL_SIZE, LOCAL_SIZE);
    sycl::range global_size = sycl::range(grid_cols, grid_rows);

    h.parallel_for(sycl::nd_range(global_size, local_size), [=](auto item) {
      int col = item.get_global_id(0);
      int row = item.get_global_id(1);
      double dot = 0;
      if (col < width_b && row < height_a) {
        for (int i = 0; i < width_a; i++)
          dot += d_a[width_a * row + i] * d_b[width_b * i + col];
        d_c[width_b * row + col] = dot;
      }
    });
  });
  queue.wait();
  out_buf.get_access<sycl::access::mode::read>();
}

void mtx_vec_mul(double *A, double *B, double *out, const int height_a,
                 const int width_a, sycl::queue queue) {
  sycl::buffer a_buf{A, sycl::range<1>(height_a * width_a)};
  sycl::buffer b_buf{B, sycl::range<1>(width_a)};
  sycl::buffer out_buf{out, sycl::range<1>(height_a)};

  queue.submit([&](sycl::handler &h) {
    auto d_a = a_buf.get_access<sycl::access::mode::read>(h);
    auto d_b = b_buf.get_access<sycl::access::mode::read>(h);
    size_t LOCAL_SIZE = 16;

    auto d_c = out_buf.get_access<sycl::access::mode::read_write>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size =
        ((height_a + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          int id = item.get_global_id(0);
          double dot = 0;
          if (id < height_a) {
            for (int i = 0; i < width_a; i++)
              dot += d_a[width_a * id + i] * d_b[i];
            d_c[id] = dot;
          }
        });
  });
  queue.wait();
  out_buf.get_access<sycl::access::mode::read>();
}

double mtx_dot(double *v, double *w, const int SIZE, sycl::queue queue) {
  double sumResult = 0;
  sycl::buffer v_buf{v, sycl::range<1>(SIZE)};
  sycl::buffer w_buf{w, sycl::range<1>(SIZE)};
  sycl::buffer<double> sumBuf{&sumResult, 1};
  queue.submit([&](sycl::handler &h) {
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    auto d_w = w_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((SIZE + local_size - 1) / local_size) * local_size;

    auto sumReduction = sycl::reduction(sumBuf, h, sycl::plus<>());

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        sumReduction, [=](auto item, auto &sum) {
          unsigned id = item.get_global_id(0);
          if (id < SIZE)
            sum += d_v[id] * d_w[id];
        });
  });
  queue.wait();
  sumBuf.get_access<sycl::access::mode::read>();
  return sumResult;
}
void mtx_sclr_div(double *in, double scalar, double *out, const int SIZE,
                  sycl::queue queue) {

  sycl::buffer in_buf{in, sycl::range<1>(SIZE)};
  sycl::buffer out_buf{out, sycl::range<1>(SIZE)};

  queue.submit([&](sycl::handler &h) {
    auto d_in = in_buf.get_access<sycl::access::mode::read>(h);
    auto d_out = out_buf.get_access<sycl::access::mode::write>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((SIZE + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          if (id < SIZE)
            d_out[id] = d_in[id] / scalar;
        });
  });
  queue.wait();
  out_buf.get_access<sycl::access::mode::read>();
}

void calc_w_init(double *w, double *alpha, double *V, unsigned i,
                 const int SIZE, sycl::queue queue) {

  sycl::buffer w_buf{w, sycl::range<1>(SIZE)};
  sycl::buffer alpha_buf{alpha, sycl::range<1>(SIZE)};
  sycl::buffer v_buf{V, sycl::range<1>(SIZE * SIZE)};

  queue.submit([&](sycl::handler &h) {
    auto d_w = w_buf.get_access<sycl::access::mode::read_write>(h);
    auto d_alpha = alpha_buf.get_access<sycl::access::mode::read>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((SIZE + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          if (id < SIZE)
            d_w[id] = d_w[id] - d_alpha[i] * d_v[id + SIZE * i];
        });
  });
  queue.wait();
  w_buf.get_access<sycl::access::mode::read>();
}

void calc_w(double *w, double *alpha, double *V, double *beta, unsigned i,
            const int SIZE, sycl::queue queue) {
  sycl::buffer w_buf{w, sycl::range<1>(SIZE)};
  sycl::buffer alpha_buf{alpha, sycl::range<1>(SIZE)};
  sycl::buffer beta_buf{beta, sycl::range<1>(SIZE)};
  sycl::buffer v_buf{V, sycl::range<1>(SIZE * SIZE)};

  queue.submit([&](sycl::handler &h) {
    auto d_w = w_buf.get_access<sycl::access::mode::read_write>(h);
    auto d_alpha = alpha_buf.get_access<sycl::access::mode::read>(h);
    auto d_beta = beta_buf.get_access<sycl::access::mode::read>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((SIZE + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          if (id < SIZE)
            d_w[id] = d_w[id] - d_alpha[i] * d_v[id + SIZE * i] -
                      d_beta[i] * d_v[id + SIZE * (i - 1)];
        });
  });
  queue.wait();
  w_buf.get_access<sycl::access::mode::read>();
}

double mtx_norm(double *w, const int SIZE, sycl::queue queue) {

  return sqrt(mtx_dot(w, w, SIZE, queue));
}

void mtx_col_copy(double *v, double *V, int j, const int SIZE) {

  memcpy(V + SIZE * j, v, sizeof(double) * SIZE);
}

void make_tri(double *alpha, double *beta, double *T, const int M,
              sycl::queue queue) {
  sycl::buffer t_buf{T, sycl::range<1>(M * M)};
  sycl::buffer alpha_buf{alpha, sycl::range<1>(M)};
  sycl::buffer beta_buf{beta, sycl::range<1>(M)};

  queue.submit([&](sycl::handler &h) {
    auto d_t = t_buf.get_access<sycl::access::mode::read_write>(h);
    auto d_alpha = alpha_buf.get_access<sycl::access::mode::read>(h);
    auto d_beta = beta_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((M + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          d_t[id * M + id] = d_alpha[id];
          if (id < M - 1) {
            d_t[id * M + id + 1] = d_beta[id + 1];
            d_t[(id + 1) * M + id] = d_beta[id + 1];
          }
        });
  });
  queue.wait();
  t_buf.get_access<sycl::access::mode::read>();
}

void qr_algorithm(double *T, const int SIZE, double *eigvals, double *eigvecs,
                  sycl::queue queue) {

  double *Q = (double *)calloc(SIZE * SIZE, sizeof(double));
  double *R = (double *)calloc(SIZE * SIZE, sizeof(double));
  create_identity_matrix(eigvecs, SIZE);

  for (unsigned i = 0; i < MAX_ITER; i++) {
    for (unsigned j = 0; j < SIZE; j++) {
      for (unsigned t = 0; t < SIZE; t++) {
        Q[j + SIZE * t] = T[j + SIZE * t];
      }
      for (unsigned k = 0; k < j; k++) {
        double dot = 0;
        for (unsigned t = 0; t < SIZE; t++) {
          dot += Q[k + SIZE * t] * T[j + SIZE * t];
        }
        R[k * SIZE + j] = dot;
        for (unsigned t = 0; t < SIZE; t++) {
          Q[j + SIZE * t] -= R[k * SIZE + j] * Q[k + SIZE * t];
        }
      }

      double total = 0;
      for (unsigned t = 0; t < SIZE; t++) {
        total += Q[j + SIZE * t] * Q[j + SIZE * t];
      }
      R[j * SIZE + j] = sqrt(total); // norm(temp_q, SIZE);

      for (unsigned t = 0; t < SIZE; t++)
        Q[j + SIZE * t] = Q[j + SIZE * t] / R[j * SIZE + j];
    }

    mtx_mul(R, Q, T, SIZE, SIZE, SIZE, queue);
    mtx_mul(eigvecs, Q, eigvecs, SIZE, SIZE, SIZE, queue);

    double subdiag = 0;
    for (unsigned k = 0; k < SIZE; k++) {
      if (subdiag < abs(T[k * SIZE + k]))
        subdiag = abs(T[k * SIZE + k]);
    }
    if (subdiag < EPS)
      break;
  }
  for (unsigned j = 0; j < SIZE; j++) {
    eigvals[j] = T[j * SIZE + j];
  }
}

void lanczos(double *lap, const int SIZE, const int M, double *eigvals,
             double *eigvecs, sycl::queue queue) {
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
    beta[i] = mtx_norm(w, SIZE, queue);

    if (beta[i] != 0) {
      mtx_sclr_div(w, beta[i], v, SIZE, queue);
    } else {
      for (unsigned i = 0; i < SIZE; i++)
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
      double norm_val = mtx_norm(v, SIZE, queue);
      mtx_sclr_div(v, norm_val, v, SIZE, queue);
    }

    mtx_col_copy(v, V, i, SIZE);

    mtx_vec_mul(lap, v, w, SIZE, SIZE, queue);

    alpha[i] = mtx_dot(v, w, SIZE, queue);

    if (i == 0) {
      calc_w_init(w, alpha, V, i, SIZE, queue);
    } else {
      calc_w(w, alpha, V, beta, i, SIZE, queue);
    }
  }

  // Make triadiagonal matrix
  make_tri(alpha, beta, T, M, queue);
  qr_algorithm(T, SIZE, eigvals, eigvecs, queue);
  print_eigen_vals(eigvals, SIZE);
}
