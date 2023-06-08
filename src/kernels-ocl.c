#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

#include <math.h>
#include <stdio.h>

double ocl_vec_norm(cl_context ctx, cl_command_queue queue, cl_program prg,
                    cl_mem d_a_vec, const unsigned size) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32, global_size;
  unsigned num_blocks = (size + local_size - 1) / local_size;
  global_size = num_blocks * local_size;

  knl = clCreateKernel(prg, "vec_dot", &err);
  cl_mem d_v_r = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                num_blocks * sizeof(double), NULL, NULL);
  double *v_r = (double *)calloc(num_blocks, sizeof(double));

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_a_vec);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_a_vec);
  err |= clSetKernelArg(knl, 2, sizeof(cl_mem), &d_v_r);
  err |= clSetKernelArg(knl, 3, sizeof(unsigned), &size);
  err |= clSetKernelArg(knl, 4, sizeof(double) * local_size, NULL);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);

  err = clEnqueueReadBuffer(queue, d_v_r, CL_TRUE, 0,
                            sizeof(double) * num_blocks, v_r, 0, NULL, NULL);

  clReleaseMemObject(d_v_r);
  clReleaseKernel(knl);

  if (err != 0) {
    printf("error in ocl_vec_norm: %d \n", err);
    return 0;
  } else {
    double total = 0;
    for (unsigned i = 0; i < num_blocks; i++) {
      total += v_r[i];
    }
    free(v_r);
    return sqrt(total);
  }
}

void ocl_mtx_sclr_div(cl_context ctx, cl_command_queue queue, cl_program prg,
                      cl_mem d_a_vec, cl_mem d_out_vec, const double sclr,
                      const unsigned size) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32,
         global_size = ((size + local_size - 1) / local_size) * local_size;

  knl = clCreateKernel(prg, "mtx_sclr_div", &err);

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_out_vec);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_a_vec);
  err |= clSetKernelArg(knl, 2, sizeof(double), &sclr);
  err |= clSetKernelArg(knl, 3, sizeof(unsigned), &size);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);
  clReleaseKernel(knl);
}

void ocl_mtx_col_copy(cl_context ctx, cl_command_queue queue, cl_program prg,
                      cl_mem d_vec, cl_mem d_mtx, const unsigned col_index,
                      const unsigned size) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32,
         global_size = ((size + local_size - 1) / local_size) * local_size;

  knl = clCreateKernel(prg, "mtx_col_copy", &err);

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_vec);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_mtx);
  err |= clSetKernelArg(knl, 2, sizeof(double), &col_index);
  err |= clSetKernelArg(knl, 3, sizeof(unsigned), &size);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);
  clReleaseKernel(knl);
}

void ocl_mtx_vec_mul(cl_context ctx, cl_command_queue queue, cl_program prg,
                     cl_mem d_a_mtx, cl_mem d_b_vec, cl_mem d_out_vec,
                     const unsigned num_rows, const unsigned num_cols) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32,
         global_size = ((num_rows + local_size - 1) / local_size) * local_size;

  knl = clCreateKernel(prg, "mtx_vec_mul", &err);

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_a_mtx);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_b_vec);
  err |= clSetKernelArg(knl, 2, sizeof(cl_mem), &d_out_vec);
  err |= clSetKernelArg(knl, 3, sizeof(unsigned), &num_rows);
  err |= clSetKernelArg(knl, 4, sizeof(unsigned), &num_cols);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);
  clReleaseKernel(knl);
}

void ocl_spmv(cl_context ctx, cl_command_queue queue, cl_program prg,
              cl_mem d_a_row_ptrs, cl_mem d_a_columns, cl_mem d_a_vals,
              cl_mem d_b_vec, cl_mem d_out_vec, const unsigned num_rows,
              const unsigned num_cols) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32,
         global_size = ((num_rows + local_size - 1) / local_size) * local_size;

  knl = clCreateKernel(prg, "spmv", &err);

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_a_row_ptrs);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_a_columns);
  err |= clSetKernelArg(knl, 2, sizeof(cl_mem), &d_a_vals);
  err |= clSetKernelArg(knl, 3, sizeof(cl_mem), &d_b_vec);
  err |= clSetKernelArg(knl, 4, sizeof(cl_mem), &d_out_vec);
  err |= clSetKernelArg(knl, 5, sizeof(unsigned), &num_rows);
  err |= clSetKernelArg(knl, 6, sizeof(unsigned), &num_cols);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);
  clReleaseKernel(knl);
}

double ocl_vec_dot(cl_context ctx, cl_command_queue queue, cl_program prg,
                   cl_mem d_a_vec, cl_mem d_b_vec, const unsigned size) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32, global_size;
  unsigned num_blocks = (size + local_size - 1) / local_size;
  global_size = num_blocks * local_size;

  knl = clCreateKernel(prg, "vec_dot", &err);
  cl_mem d_v_r = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                num_blocks * sizeof(double), NULL, NULL);
  double *v_r = (double *)malloc(num_blocks * sizeof(double));

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_a_vec);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_b_vec);
  err |= clSetKernelArg(knl, 2, sizeof(cl_mem), &d_v_r);
  err |= clSetKernelArg(knl, 3, sizeof(unsigned), &size);
  err |= clSetKernelArg(knl, 4, sizeof(double) * local_size, NULL);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);

  clEnqueueReadBuffer(queue, d_v_r, CL_TRUE, 0, sizeof(double) * num_blocks,
                      v_r, 0, NULL, NULL);
  clReleaseMemObject(d_v_r);
  clReleaseKernel(knl);

  if (err != 0) {
    printf("error in ocl_vec_dot \n");
    return 0;
  } else {
    double total = 0;
    for (unsigned i = 0; i < num_blocks; i++)
      total += v_r[i];
    free(v_r);
    return total;
  }
}

void ocl_calc_w_init(cl_context ctx, cl_command_queue queue, cl_program prg,
                     cl_mem d_w_vec, const double alpha, cl_mem d_orth_mtx,
                     const unsigned col_index, const unsigned size) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32,
         global_size = ((size + local_size - 1) / local_size) * local_size;

  knl = clCreateKernel(prg, "calc_w_init", &err);

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_w_vec);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_orth_mtx);
  err |= clSetKernelArg(knl, 2, sizeof(double), &alpha);
  err |= clSetKernelArg(knl, 3, sizeof(unsigned), &col_index);
  err |= clSetKernelArg(knl, 4, sizeof(unsigned), &size);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);
  clReleaseKernel(knl);
}

void ocl_calc_w(cl_context ctx, cl_command_queue queue, cl_program prg,
                cl_mem d_w_vec, const double alpha, cl_mem d_orth_mtx,
                const double beta, const unsigned col_index,
                const unsigned size) {
  cl_int err;
  cl_kernel knl;

  size_t local_size = 32,
         global_size = ((size + local_size - 1) / local_size) * local_size;

  knl = clCreateKernel(prg, "calc_w", &err);

  err = clSetKernelArg(knl, 0, sizeof(cl_mem), &d_w_vec);
  err |= clSetKernelArg(knl, 1, sizeof(cl_mem), &d_orth_mtx);
  err |= clSetKernelArg(knl, 2, sizeof(double), &alpha);
  err |= clSetKernelArg(knl, 3, sizeof(double), &beta);
  err |= clSetKernelArg(knl, 4, sizeof(unsigned), &col_index);
  err |= clSetKernelArg(knl, 5, sizeof(unsigned), &size);

  err = clEnqueueNDRangeKernel(queue, knl, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  clFinish(queue);
  clReleaseKernel(knl);
}
