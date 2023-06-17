#include "kernels.h"

void sycl_mtx_vec_mul(sycl::buffer<double> a_buf, sycl::buffer<double> b_buf,
                      sycl::buffer<double> out_buf, int height_a, int width_a,
                      sycl::queue queue, sycl::nd_range<1> nd_range) {

  queue.submit([&](sycl::handler &h) {
    auto d_a = a_buf.get_access<sycl::access::mode::read>(h);
    auto d_b = b_buf.get_access<sycl::access::mode::read>(h);

    auto d_c = out_buf.get_access<sycl::access::mode::read_write>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor
    size_t global_size = ((height_a + BLOCKSIZE - 1) / BLOCKSIZE) * BLOCKSIZE;

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      double dot = 0;
      if (id < height_a) {
        for (int i = 0; i < width_a; i++)
          dot += d_a[width_a * id + i] * d_b[i];
        d_c[id] = dot;
      }
    });
  });
}

void sycl_spmv(sycl::buffer<int> a_row_buf, sycl::buffer<int> a_columns_buf,
               sycl::buffer<double> a_vals_buf, sycl::buffer<double> b_buf,
               sycl::buffer<double> out_buf, int height_a, int width_a,
               sycl::queue queue, sycl::nd_range<1> nd_range) {

  queue.submit([&](sycl::handler &h) {
    auto a_row_ptrs = a_row_buf.get_access<sycl::access::mode::read>(h);
    auto a_columns = a_columns_buf.get_access<sycl::access::mode::read>(h);
    auto a_vals = a_vals_buf.get_access<sycl::access::mode::read>(h);

    auto d_b = b_buf.get_access<sycl::access::mode::read>(h);

    auto d_c = out_buf.get_access<sycl::access::mode::read_write>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor
    size_t global_size = ((height_a + BLOCKSIZE - 1) / BLOCKSIZE) * BLOCKSIZE;

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      if (id < height_a) {
        int start = a_row_ptrs[id];
        int end = a_row_ptrs[id + 1];
        double dot = 0;
        // Add each element in the id
        for (int j = start; j < end; j++)
          dot += a_vals[j] * d_b[a_columns[j]];
        d_c[id] = dot;
      }
    });
  });
}

double sycl_mtx_dot(sycl::buffer<double> a_vec_buf,
                    sycl::buffer<double> b_vec_buf, int size, sycl::queue queue,
                    sycl::nd_range<1> nd_range) {
  // Number of work items in each local work group

  // Number of total work items - BLOCKSIZE must be devisor
  size_t group_size = ((size + BLOCKSIZE - 1) / BLOCKSIZE);
  size_t global_size = group_size * BLOCKSIZE;

  double *interim_results = (double *)calloc(group_size, sizeof(double));
  sycl::buffer result_buf{interim_results, sycl::range<1>(group_size)};

  queue.submit([&](sycl::handler &h) {
    auto a_vec_acc = a_vec_buf.get_access<sycl::access::mode::read>(h);
    auto b_vec_acc = b_vec_buf.get_access<sycl::access::mode::read>(h);
    auto result_acc = result_buf.get_access<sycl::access::mode::write>(h);

    sycl::local_accessor<float, 1> shared_data(sycl::range<1>(BLOCKSIZE), h);

    h.parallel_for(nd_range, [=](auto item) {
      int tid = item.get_global_id(0);

      if (tid < size)
        shared_data[item.get_local_id(0)] = a_vec_acc[tid] * b_vec_acc[tid];
      else
        shared_data[item.get_local_id(0)] = 0.0;

      item.barrier(sycl::access::fence_space::local_space);

      for (int stride = item.get_local_range()[0] >> 1; stride > 0;
           stride >>= 1) {
        if (item.get_local_id(0) < stride)
          shared_data[item.get_local_id(0)] +=
              shared_data[item.get_local_id(0) + stride];

        item.barrier(sycl::access::fence_space::local_space);
      }

      if (item.get_local_id(0) == 0)
        result_acc[item.get_group(0)] = shared_data[0];
    });
  });

  queue.wait();
  result_buf.get_access<sycl::access::mode::read>();

  double result = 0.0;
  for (int i = 0; i < group_size; i++) {
    result += interim_results[i];
  }

  free(interim_results);

  return result;
}
void sycl_mtx_sclr_div(sycl::buffer<double> in_buf, double scalar,
                       sycl::buffer<double> out_buf, int size,
                       sycl::queue queue, sycl::nd_range<1> nd_range) {

  queue.submit([&](sycl::handler &h) {
    auto d_in = in_buf.get_access<sycl::access::mode::read>(h);
    auto d_out = out_buf.get_access<sycl::access::mode::write>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      if (id < size)
        d_out[id] = d_in[id] / scalar;
    });
  });
}

void sycl_mtx_sclr_mul(sycl::buffer<double> in_buf, double scalar,
                       sycl::buffer<double> out_buf, int size,
                       sycl::queue queue, sycl::nd_range<1> nd_range) {

  queue.submit([&](sycl::handler &h) {
    auto d_in = in_buf.get_access<sycl::access::mode::read>(h);
    auto d_out = out_buf.get_access<sycl::access::mode::write>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      if (id < size)
        d_out[id] = d_in[id] * scalar;
    });
  });
}

void sycl_calc_w_init(sycl::buffer<double> w_buf, double alpha,
                      sycl::buffer<double> v_buf, int size, sycl::queue queue,
                      sycl::nd_range<1> nd_range) {

  queue.submit([&](sycl::handler &h) {
    auto d_w = w_buf.get_access<sycl::access::mode::read_write>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      if (id < size)
        d_w[id] = d_w[id] - alpha * d_v[id];
    });
  });
}

void sycl_calc_w(sycl::buffer<double> w_buf, double alpha,
                 sycl::buffer<double> v_buf, sycl::buffer<double> v_pre_buf,
                 double beta, int size, sycl::queue queue,
                 sycl::nd_range<1> nd_range) {

  queue.submit([&](sycl::handler &h) {
    auto d_w = w_buf.get_access<sycl::access::mode::read_write>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    auto d_pre_v = v_pre_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      if (id < size)
        d_w[id] = d_w[id] - alpha * d_v[id] - beta * d_pre_v[id];
    });
  });
}

double sycl_mtx_norm(sycl::buffer<double> w, int size, sycl::queue queue,
                     sycl::nd_range<1> nd_range) {

  return sqrt(sycl_mtx_dot(w, w, size, queue, nd_range));
}

void sycl_mtx_col_copy(sycl::buffer<double> v_temp_buf,
                       sycl::buffer<double> v_buf, int j, int size,
                       sycl::queue queue, sycl::nd_range<1> nd_range) {
  queue.submit([&](sycl::handler &h) {
    auto d_temp_v = v_temp_buf.get_access<sycl::access::mode::read>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::write>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      if (id < size)
        d_v[size * j + id] = d_temp_v[id];
    });
  });
}
void sycl_vec_copy(sycl::buffer<double> v_buf, sycl::buffer<double> out_buf,
                   int size, sycl::queue queue, sycl::nd_range<1> nd_range) {

  queue.submit([&](sycl::handler &h) {
    auto d_out = out_buf.get_access<sycl::access::mode::write>(h);
    auto d_v = v_buf.get_access<sycl::access::mode::read>(h);
    // Number of work items in each local work group

    // Number of total work items - BLOCKSIZE must be devisor

    h.parallel_for(nd_range, [=](auto item) {
      int id = item.get_global_id(0);
      if (id < size)
        d_out[id] = d_v[id];
    });
  });
}