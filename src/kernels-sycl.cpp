#include "kernels.h"

void sycl_mtx_identity(double *out, const int size) {
  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < size; j++) {
      if (i == j)
        out[i * size + j] = 1;
      else
        out[i * size + j] = 0;
    }
  }
}

void sycl_mtx_vec_mul(sycl::buffer<double> a_buf, sycl::buffer<double> b_buf,
                      sycl::buffer<double> out_buf, const int height_a,
                      const int width_a, sycl::queue queue) {

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
}

double sycl_mtx_dot(sycl::buffer<double> v_buf, sycl::buffer<double> w_buf,
                    const int size, sycl::queue queue) {
  double sumResult = 0;
  sycl::buffer<double> sumBuf{&sumResult, 1};
  queue.submit([&](sycl::handler &h) {
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    auto d_w = w_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((size + local_size - 1) / local_size) * local_size;

    auto sumReduction = sycl::reduction(sumBuf, h, sycl::plus<>());

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        sumReduction, [=](auto item, auto &sum) {
          unsigned id = item.get_global_id(0);
          if (id < size)
            sum += d_v[id] * d_w[id];
        });
  });
  queue.wait();
  sumBuf.get_access<sycl::access::mode::read>();
  return sumResult;
}
void sycl_mtx_sclr_div(sycl::buffer<double> in_buf, double scalar,
                       sycl::buffer<double> out_buf, const int size,
                       sycl::queue queue) {

  queue.submit([&](sycl::handler &h) {
    auto d_in = in_buf.get_access<sycl::access::mode::read>(h);
    auto d_out = out_buf.get_access<sycl::access::mode::write>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((size + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          if (id < size)
            d_out[id] = d_in[id] / scalar;
        });
  });
  queue.wait();
}

void sycl_calc_w_init(sycl::buffer<double> w_buf,
                      sycl::buffer<double> alpha_buf,
                      sycl::buffer<double> v_buf, unsigned i, const int size,
                      sycl::queue queue) {

  queue.submit([&](sycl::handler &h) {
    auto d_w = w_buf.get_access<sycl::access::mode::read_write>(h);
    auto d_alpha = alpha_buf.get_access<sycl::access::mode::read>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((size + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          if (id < size)
            d_w[id] = d_w[id] - d_alpha[i] * d_v[id + size * i];
        });
  });
  queue.wait();
}

void sycl_calc_w(sycl::buffer<double> w_buf, sycl::buffer<double> alpha_buf,
                 sycl::buffer<double> v_buf, sycl::buffer<double> beta_buf,
                 unsigned i, const int size, sycl::queue queue) {

  queue.submit([&](sycl::handler &h) {
    auto d_w = w_buf.get_access<sycl::access::mode::read_write>(h);
    auto d_alpha = alpha_buf.get_access<sycl::access::mode::read>(h);
    auto d_beta = beta_buf.get_access<sycl::access::mode::read>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((size + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          if (id < size)
            d_w[id] = d_w[id] - d_alpha[i] * d_v[id + size * i] -
                      d_beta[i] * d_v[id + size * (i - 1)];
        });
  });
  queue.wait();
}

double sycl_mtx_norm(sycl::buffer<double> w, const int size,
                     sycl::queue queue) {

  return sqrt(sycl_mtx_dot(w, w, size, queue));
}

void sycl_mtx_col_copy(sycl::buffer<double> v_temp_buf,
                       sycl::buffer<double> v_buf, int j, const int size,
                       sycl::queue queue) {
  queue.submit([&](sycl::handler &h) {
    auto d_temp_v = v_temp_buf.get_access<sycl::access::mode::read>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::write>(h);
    // Number of work items in each local work group
    size_t local_size = 256;

    // Number of total work items - local_size must be devisor
    size_t global_size = ((size + local_size - 1) / local_size) * local_size;

    h.parallel_for(
        sycl::nd_range(sycl::range(global_size), sycl::range(local_size)),
        [=](auto item) {
          unsigned id = item.get_global_id(0);
          if (id < size)
            d_v[size * j + id] = d_temp_v[id];
        });
  });
  queue.wait();
}