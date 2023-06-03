#include "kernels.h"
#include "lanczos-aux.h"
#include "lanczos.h"
#include "print-helper.h"

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
                  double *alpha, double *beta, double *v, cl_mem d_lap,
                  cl_mem d_w, cl_mem d_v, cl_mem d_V, const int M,
                  const int size) {
  for (unsigned i = 0; i < M; i++) {
    beta[i] = ocl_vec_norm(ctx, queue, prg, d_w, size);
    if (fabs(beta[i] - 0) > EPS) {
      ocl_mtx_sclr_div(ctx, queue, prg, d_v, d_w, beta[i], size);
    } else {
      for (unsigned i = 0; i < size; i++) {
        v[i] = (double)rand() / (double)(RAND_MAX / MAX);
      }
      cl_int err = clEnqueueWriteBuffer(
          queue, d_v, CL_TRUE, 0, size * sizeof(double), v, 0, NULL, NULL);
      double norm_val = ocl_vec_norm(ctx, queue, prg, d_v, size);
      ocl_mtx_sclr_div(ctx, queue, prg, d_v, d_v, norm_val, size);
    }

    ocl_mtx_col_copy(ctx, queue, prg, d_v, d_V, i, size);
    ocl_mtx_vec_mul(ctx, queue, prg, d_lap, d_v, d_w, size, size);
    alpha[i] = ocl_vec_dot(ctx, queue, prg, d_v, d_w, size);
    if (i == 0) {
      ocl_calc_w_init(ctx, queue, prg, d_w, alpha[i], d_V, i, size);
    } else {
      ocl_calc_w(ctx, queue, prg, d_w, alpha[i], d_V, beta[i], i, size);
    }
  }
}

void lanczos(double *lap, const int size, const int M, double *eigvals,
             double *eigvecs, int argc, char *argv[]) {

  double *V = (double *)calloc(size * M, sizeof(double));
  double *alpha = (double *)calloc(M, sizeof(double));
  double *beta = (double *)calloc(M, sizeof(double));
  double *v = (double *)calloc(size, sizeof(double));
  double *w = (double *)calloc(size, sizeof(double));

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
  cl_mem d_lap, d_V, d_w, d_v;

  size_t bytes = size * size * sizeof(double);
  d_lap = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  d_V = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
  d_w = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(double), NULL,
                       NULL);
  d_v = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(double), NULL,
                       NULL);

  err = clEnqueueWriteBuffer(queue, d_w, CL_TRUE, 0, size * sizeof(double), w,
                             0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_lap, CL_TRUE, 0,
                             size * size * sizeof(double), lap, 0, NULL, NULL);

  // warm ups
  for (int t = 0; t < 10; t++)
    lanczos_algo(context, queue, program, alpha, beta, v, d_lap, d_w, d_v, d_V,
                 M, size);

  clock_t t = clock();
  lanczos_algo(context, queue, program, alpha, beta, v, d_lap, d_w, d_v, d_V, M,
               size);
  t = clock() - t;
  printf("size: %d, time: %e \n", size, (double)t / (CLOCKS_PER_SEC));

  tqli(eigvecs, eigvals, size, alpha, beta, 0);
  print_eigen_vals(eigvals, size);

  // release OpenCL resources
  clReleaseMemObject(d_lap);
  clReleaseMemObject(d_V);
  clReleaseMemObject(d_w);
  clReleaseMemObject(d_v);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  free(V), free(alpha), free(beta), free(v), free(w);
}
