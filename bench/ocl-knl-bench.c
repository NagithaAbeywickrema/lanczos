#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

#include "../src/kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX 10
#define EPS 1e-12
#define MAX_SOURCE_SIZE (0x100000)

#define tcalloc(T, n) (T *)calloc(n, sizeof(T))
#define tfree(p) free_((void **)p)
void free_(void **p) { free(*p), *p = NULL; }

double *create_host_vec(const unsigned size) {
  double *x = tcalloc(double, size);
  for (unsigned i = 0; i < size; i++)
    x[i] = (rand() + 1.0) / RAND_MAX;

  return x;
}

unsigned inc(const unsigned i) {
  return i + 1;
  // return (unsigned)(1.01 * i);
  // if (i < 1000)
  // else
  //   return (unsigned)(1.03 * i);
}

FILE *open_file(const char *suffix) {
  char fname[2 * BUFSIZ];
  strncpy(fname, "lanczos", BUFSIZ);
  strncat(fname, "_", 2);
  strncat(fname, suffix, BUFSIZ);
  strncat(fname, ".txt", 5);

  FILE *fp = fopen(fname, "a");
  if (!fp)
    printf("Not found \n");
  return fp;
}

void vec_norm_bench(cl_context ctx, cl_command_queue queue, cl_program prg) {
  FILE *fp = open_file("vec-norm");
  for (unsigned i = 1e4; i < 1e6; i = inc(i)) {
    cl_int err;
    cl_kernel knl;
    double *h_a = create_host_vec(i);
    cl_mem d_a =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, i * sizeof(double), NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, i * sizeof(double), h_a,
                               0, NULL, NULL);

    // Warmup
    for (int j = 0; j < 1000; j++)
      ocl_vec_norm(ctx, queue, prg, d_a, i);

    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      ocl_vec_norm(ctx, queue, prg, d_a, i);
    t = clock() - t;

    fprintf(fp, "%s,%s,%u,%e\n", "vec-norm", "ocl", i,
            (double)t / (CLOCKS_PER_SEC * 1000));
    clReleaseMemObject(d_a);
    tfree(&h_a);
  }
  fclose(fp);
}

void vec_sclr_div_bench(cl_context ctx, cl_command_queue queue,
                        cl_program prg) {
  FILE *fp = open_file("vec-sclr-div-ocl");
  for (unsigned i = 1; i < 5; i = inc(i)) {
    cl_int err;
    cl_kernel knl;
    double *h_a = create_host_vec(i);
    double *h_b = (double *)malloc(sizeof(double) * i);
    double *h_c = (double *)malloc(sizeof(double) * i);

    serial_vec_sclr_div(h_a, h_c, 10, i);
    cl_mem d_a =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, i * sizeof(double), NULL, NULL);
    cl_mem d_b =
        clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, i * sizeof(double), NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, i * sizeof(double), h_a,
                               0, NULL, NULL);

    // Warmup
    for (int j = 0; j < 1000; j++)
      ocl_mtx_sclr_div(ctx, queue, prg, d_a, d_b, 1 / 10, i);

    for (int j = 0; j < 1000; j++)
      ocl_d2d_mem_cpy(ctx, queue, prg, d_a, d_b, i);

    clock_t t1 = clock();
    for (int j = 0; j < 1000; j++)
      ocl_d2d_mem_cpy(ctx, queue, prg, d_a, d_b, i);
    t1 = clock() - t1;

    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      ocl_mtx_sclr_div(ctx, queue, prg, d_a, d_b, 1 / 10, i);
    t = clock() - t;

    err = clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0, i * sizeof(double), h_b, 0, NULL, NULL);
    for(int k=0; k< i; k++){
      printf("h_b[%d]: %f \n", k, h_b[k]);
    }

    fprintf(fp, "%s,%s,%u,%u,%e,%e\n", "vec-sclr-div", "ocl", 32, i,
            (double)t1 / (CLOCKS_PER_SEC * 1000),
            (double)t / (CLOCKS_PER_SEC * 1000));
    clReleaseMemObject(d_a), clReleaseMemObject(d_b);
    tfree(&h_a);
  }
  fclose(fp);
}

void mtx_col_copy_bench(cl_context ctx, cl_command_queue queue,
                        cl_program prg) {
  FILE *fp = open_file("mtx-col-copy");
  for (unsigned i = 100; i < 1e4; i = inc(i)) {
    cl_int err;
    cl_kernel knl;
    double *h_a = create_host_vec(i);
    cl_mem d_a =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, i * sizeof(double), NULL, NULL);
    cl_mem d_b = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, i * i * sizeof(double),
                                NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, i * sizeof(double), h_a,
                               0, NULL, NULL);

    // Warmup
    for (int j = 0; j < 1000; j++)
      ocl_mtx_col_copy(ctx, queue, prg, d_a, d_b, i - 1, i);

    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      ocl_mtx_col_copy(ctx, queue, prg, d_a, d_b, i - 1, i);
    t = clock() - t;

    fprintf(fp, "%s,%s,%u,%e\n", "mtx-col-copy", "ocl", i,
            (double)t / (CLOCKS_PER_SEC * 1000));
    clReleaseMemObject(d_a), clReleaseMemObject(d_b);
    tfree(&h_a);
  }
  fclose(fp);
}

void calc_w_bench(cl_context ctx, cl_command_queue queue, cl_program prg) {
  FILE *fp = open_file("calc-w");
  for (unsigned i = 100; i < 1e4; i = inc(i)) {
    cl_int err;
    cl_kernel knl;
    double *h_a = create_host_vec(i);
    double *h_b = create_host_vec(i * i);
    cl_mem d_a =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, i * sizeof(double), NULL, NULL);
    cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, i * i * sizeof(double),
                                NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, i * sizeof(double), h_a,
                               0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, i * i * sizeof(double),
                               h_b, 0, NULL, NULL);

    // Warmup
    for (int j = 0; j < 1000; j++)
      ocl_calc_w(ctx, queue, prg, d_a, 2, d_b, 2, i - 1, i);

    clock_t t = clock();
    for (int j = 0; j < 1000; j++)
      ocl_calc_w(ctx, queue, prg, d_a, 2, d_b, 2, i - 1, i);
    t = clock() - t;

    fprintf(fp, "%s,%s,%u,%e\n", "calc-w", "ocl", i,
            (double)t / (CLOCKS_PER_SEC * 1000));
    clReleaseMemObject(d_a), clReleaseMemObject(d_b);
    tfree(&h_a), tfree(&h_b);
  }
  fclose(fp);
}

void lanczos_bench(int argc, char *argv[]) {
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

  kernelFile = fopen("/home/pubudu/nomp/lanczos/src/kernels.cl", "r");

  if (kernelFile == NULL) {
    fprintf(stderr, "No file named kernels.cl was found\n");
  }
  kernelSource = (char *)malloc(MAX_SOURCE_SIZE);
  kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
  fclose(kernelFile);

  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,
                                      (const size_t *)&kernelSize, &err);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // bench
  // vec_norm_bench(context, queue, program);
  vec_sclr_div_bench(context, queue, program);
  // mtx_col_copy_bench(context, queue, program);
  // calc_w_bench(context, queue, program);

  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}