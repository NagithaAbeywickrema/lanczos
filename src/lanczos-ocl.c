#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

#define MAX 10
#define EPS 1e-12
#define MAX_SOURCE_SIZE (0x100000)

void lanczos_algo(cl_context ctx, cl_command_queue queue, cl_program prg,
                  double *alpha, double *beta, double *orth_vec,
                  cl_mem d_row_ptrs, cl_mem d_columns, cl_mem d_vals,
                  cl_mem d_w_vec, cl_mem d_orth_vec, cl_mem d_orth_mtx,
                  const unsigned m, const unsigned size) {
  for (unsigned i = 0; i < m; i++) {
    beta[i] = ocl_vec_norm(ctx, queue, prg, d_w_vec, size);
    if (fabs(beta[i] - 0) > EPS) {
      ocl_mtx_sclr_div(ctx, queue, prg, d_orth_vec, d_w_vec, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++) {
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      }
      cl_int err =
          clEnqueueWriteBuffer(queue, d_orth_vec, CL_TRUE, 0,
                               size * sizeof(double), orth_vec, 0, NULL, NULL);
      double norm_val = ocl_vec_norm(ctx, queue, prg, d_orth_vec, size);
      ocl_mtx_sclr_div(ctx, queue, prg, d_orth_vec, d_orth_vec, norm_val, size);
    }

    ocl_mtx_col_copy(ctx, queue, prg, d_orth_vec, d_orth_mtx, i, size);
    // ocl_mtx_vec_mul(ctx, queue, prg, d_lap, d_orth_vec, d_w_vec, size, size);
    ocl_spmv(ctx, queue, prg, d_row_ptrs, d_columns, d_vals, d_orth_vec,
             d_w_vec, size, size);
    alpha[i] = ocl_vec_dot(ctx, queue, prg, d_orth_vec, d_w_vec, size);
    if (i == 0) {
      ocl_calc_w_init(ctx, queue, prg, d_w_vec, alpha[i], d_orth_mtx, i, size);
    } else {
      ocl_calc_w(ctx, queue, prg, d_w_vec, alpha[i], d_orth_mtx, beta[i], i,
                 size);
    }
  }
}

void lanczos_algo1(cl_context ctx, cl_command_queue queue, cl_program prg,
                   double *alpha, double *beta, double *orth_vec,
                   cl_mem d_row_ptrs, cl_mem d_columns, cl_mem d_vals,
                   cl_mem d_w_vec, cl_mem d_orth_vec, cl_mem d_orth_mtx,
                   const unsigned m, const unsigned size,
                   time_struct *time_measure) {

  for (unsigned i = 0; i < m; i++) {
    clock_t t = clock();
    beta[i] = ocl_vec_norm(ctx, queue, prg, d_w_vec, size);
    t = clock() - t;
    time_measure->vec_norm += (double)t / (CLOCKS_PER_SEC);
    if (fabs(beta[i] - 0) > EPS) {
      t = clock();
      ocl_mtx_sclr_div(ctx, queue, prg, d_orth_vec, d_w_vec, beta[i], size);
      t = clock() - t;
      time_measure->vec_sclr_div += (double)t / (CLOCKS_PER_SEC);
    } else {
      for (unsigned i = 0; i < size; i++) {
        orth_vec[i] = (double)rand() / (double)(RAND_MAX / MAX);
      }
      cl_int err =
          clEnqueueWriteBuffer(queue, d_orth_vec, CL_TRUE, 0,
                               size * sizeof(double), orth_vec, 0, NULL, NULL);
      double norm_val = ocl_vec_norm(ctx, queue, prg, d_orth_vec, size);
      ocl_mtx_sclr_div(ctx, queue, prg, d_orth_vec, d_orth_vec, norm_val, size);
    }
    t = clock();
    ocl_mtx_col_copy(ctx, queue, prg, d_orth_vec, d_orth_mtx, i, size);
    t = clock() - t;
    time_measure->mtx_col_copy += (double)t / (CLOCKS_PER_SEC);
    // ocl_mtx_vec_mul(ctx, queue, prg, d_lap, d_orth_vec, d_w_vec, size, size);
    t = clock();
    ocl_spmv(ctx, queue, prg, d_row_ptrs, d_columns, d_vals, d_orth_vec,
             d_w_vec, size, size);
    t = clock() - t;
    time_measure->spmv += (double)t / (CLOCKS_PER_SEC);
    t = clock();
    alpha[i] = ocl_vec_dot(ctx, queue, prg, d_orth_vec, d_w_vec, size);
    t = clock() - t;
    time_measure->vec_dot += (double)t / (CLOCKS_PER_SEC);
    if (i == 0) {
      ocl_calc_w_init(ctx, queue, prg, d_w_vec, alpha[i], d_orth_mtx, i, size);
    } else {
      t = clock();

      ocl_calc_w(ctx, queue, prg, d_w_vec, alpha[i], d_orth_mtx, beta[i], i,
                 size);
      t = clock() - t;
      time_measure->calc_w += (double)t / (CLOCKS_PER_SEC);
    }
  }
}

void lanczos(int *row_ptrs, int *columns, double *vals, int val_count,
             const unsigned size, const unsigned m, double *eigvals,
             double *eigvecs, int argc, char *argv[],
             time_struct *time_measure) {

  double *orth_mtx = (double *)calloc(size * m, sizeof(double));
  double *alpha = (double *)calloc(m, sizeof(double));
  double *beta = (double *)calloc(m, sizeof(double));
  double *orth_vec = (double *)calloc(size, sizeof(double));
  double *w_vec = (double *)calloc(size, sizeof(double));

  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  cl_int err;

  cl_uint num_platforms;
  const int platform_id = 2;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (platform_id < 0 | platform_id >= num_platforms) {
    printf("Given platform id is invalid \n");
  }

  cl_platform_id *cl_platforms = calloc(sizeof(cl_platform_id), num_platforms);
  err = clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms);
  cl_platform_id platform = cl_platforms[platform_id];
  free(cl_platforms);

  cl_uint num_devices;
  const int device = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if (device < 0 || device >= num_devices) {
    printf("Invalid device \n");
  }

  cl_device_id *cl_devices = calloc(sizeof(cl_device_id), num_devices);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices);
  cl_device_id device_id = cl_devices[device];
  free(cl_devices);

  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // Create a command queue
  queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

  FILE *kernelFile;
  char *kernelSource;
  size_t kernelSize;

  kernelFile = fopen("../src/kernels.cl", "r");

  if (kernelFile == NULL) {
    fprintf(stderr, "No file named kernels.cl was found\n");
  }
  kernelSource = (char *)malloc(MAX_SOURCE_SIZE);
  kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
  fclose(kernelFile);

  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,
                                      (const size_t *)&kernelSize, &err);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // Allocate memory
  cl_mem d_row_ptrs, d_columns, d_vals, d_orth_mtx, d_w_vec, d_orth_vec;

  size_t bytes = size * size * sizeof(double);
  d_row_ptrs = clCreateBuffer(context, CL_MEM_READ_ONLY,
                              (size + 1) * sizeof(int), NULL, NULL);
  d_columns = clCreateBuffer(context, CL_MEM_READ_ONLY, val_count * sizeof(int),
                             NULL, NULL);
  d_vals = clCreateBuffer(context, CL_MEM_READ_ONLY, val_count * sizeof(double),
                          NULL, NULL);
  d_orth_mtx = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
  d_w_vec = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(double),
                           NULL, NULL);
  d_orth_vec = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(double),
                              NULL, NULL);

  err = clEnqueueWriteBuffer(queue, d_w_vec, CL_TRUE, 0, size * sizeof(double),
                             w_vec, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_row_ptrs, CL_TRUE, 0,
                             (size + 1) * sizeof(int), row_ptrs, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_columns, CL_TRUE, 0,
                             val_count * sizeof(int), columns, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_vals, CL_TRUE, 0,
                             val_count * sizeof(double), vals, 0, NULL, NULL);

  // Warm up runs
  for (int t = 0; t < 10; t++)
    lanczos_algo(context, queue, program, alpha, beta, orth_vec, d_row_ptrs,
                 d_columns, d_vals, d_w_vec, d_orth_vec, d_orth_mtx, m, size);

  clock_t t = clock();
  lanczos_algo1(context, queue, program, alpha, beta, orth_vec, d_row_ptrs,
                d_columns, d_vals, d_w_vec, d_orth_vec, d_orth_mtx, m, size,
                time_measure);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  // tqli(eigvecs, eigvals, size, alpha, beta, 0);

  // release OpenCL resources
  clReleaseMemObject(d_row_ptrs);
  clReleaseMemObject(d_columns);
  clReleaseMemObject(d_vals);
  clReleaseMemObject(d_orth_mtx);
  clReleaseMemObject(d_w_vec);
  clReleaseMemObject(d_orth_vec);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  free(orth_mtx), free(alpha), free(beta), free(orth_vec), free(w_vec);
}
