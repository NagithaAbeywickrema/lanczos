#if defined(ENABLE_SYCL)
#include <CL/sycl.hpp>
#endif
#ifdef __cplusplus
extern "C" {
#endif

// Serially implemented functions of Lanczos routine.
double serial_vec_dot(double *a_vec, double *b_vec, int size);
double serial_vec_norm(double *a_vec, int size);
void serial_vec_sclr_div(double *a_vec, double *out_vec, double sclr, int size);
void serial_mtx_col_copy(double *vec, double *mtx, int col_index, int size);
void serial_vec_copy(double *vec, double *out, int size);
void serial_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                        int num_rows, int num_cols);
void serial_spmv(int *a_row_ptrs, int *a_columns, double *a_vals, double *b_vec,
                 double *out_vec, int num_rows, int num_cols);
void serial_calc_w_init(double *w_vec, double alpha, double *orth_vec,
                        int size);
void serial_calc_w(double *w_vec, double alpha, double *orth_vec,
                   double *orth_vec_pre, double beta, int size);
void serial_vec_sclr_mul(double *a_vec, double *out_vec, double sclr, int size);

// Parallelized functions of Lanczos routine implemented using Nomp.
double nomp_vec_dot(double *a_vec, double *b_vec, int size);
double nomp_vec_norm(double *a_vec, int size);
void nomp_vec_sclr_div(double *a_vec, double *out_vec, double sclr, int size);
void nomp_vec_sclr_mul(double *a_vec, double *out_vec, double sclr, int size);
void nomp_mtx_col_copy(double *vec, double *mtx, int col_index, int size);
void nomp_vec_copy(double *vec, double *out, int size);
void nomp_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                      int num_rows, int num_cols);
void nomp_spmv(int *a_row_ptrs, int *a_columns, double *a_vals, double *b_vec,
               double *out_vec, int num_rows, int num_cols);
void nomp_calc_w_init(double *w_vec, double alpha, double *orth_vec, int size);
void nomp_calc_w(double *w_vec, double alpha, double *orth_vec,
                 double *orth_vec_pre, double beta, int size);
void nomp_d2d_mem_cpy(double *a, double *b, int N);
void nomp_vec_add(double *a_vec, double *b_vec, double *out_vec, int size);

// Parallelized functions of Lanczos routine implemented using Cuda.
double cuda_vec_dot(double *d_a_vec, double *d_b_vec, int size, int grid_size,
                    int block_size);
double cuda_vec_norm(double *d_a_vec, int size, int grid_size, int block_size);
void cuda_vec_sclr_mul(double *d_a_vec, double *d_out_vec, double sclr,
                       int size, int grid_size, int block_size);
void cuda_vec_sclr_div(double *d_a_vec, double *d_out_vec, double sclr,
                       int size, int grid_size, int block_size);
void cuda_mtx_col_copy(double *d_vec, double *d_mtx, int col_index, int size,
                       int grid_size, int block_size);
void cuda_vec_copy(double *d_vec, double *d_vec_pre, int size, int grid_size,
                   int block_size);
void cuda_mtx_vec_mul(double *d_a_mtx, double *d_b_vec, double *d_out_vec,
                      int num_rows, int num_cols);
void cuda_spmv(int *d_a_row_ptrs, int *d_a_columns, double *d_a_vals,
               double *d_b_vec, double *d_out_vec, int num_rows, int num_cols,
               int grid_size, int block_size);
void cuda_calc_w_init(double *d_w_vec, double alpha, double *d_orth_vec,
                      int size, int grid_size, int block_size);
void cuda_calc_w(double *d_w_vec, double alpha, double *d_orth_vec,
                 double *d_orth_vec_pre, double beta, int size, int grid_size,
                 int block_size);
void cuda_d2d_mem_cpy(double *a, double *b, int size, int grid_size,
                      int block_size);
void cuda_vec_dot_without_d2h(double *d_a_vec, double *d_b_vec, int size,
                              int shared_data_size, double *d_result,
                              int grid_size, int block_size);

#if defined(ENABLE_SYCL)

double sycl_mtx_norm(sycl::buffer<double> w, int size, sycl::queue queue,
                     sycl::nd_range<1> nd_range);
void sycl_mtx_sclr_div(sycl::buffer<double> in_buf, double scalar,
                       sycl::buffer<double> out_buf, int size,
                       sycl::queue queue, sycl::nd_range<1> nd_range);
void sycl_mtx_sclr_mul(sycl::buffer<double> in_buf, double scalar,
                       sycl::buffer<double> out_buf, int size,
                       sycl::queue queue, sycl::nd_range<1> nd_range);
void sycl_mtx_col_copy(sycl::buffer<double> v_temp_buf,
                       sycl::buffer<double> v_buf, int j, int size,
                       sycl::queue queue, sycl::nd_range<1> nd_range);
void sycl_vec_copy(sycl::buffer<double> v_buf, sycl::buffer<double> out_buf,
                   int size, sycl::queue queue, sycl::nd_range<1> nd_range);
void sycl_mtx_vec_mul(sycl::buffer<double> a_buf, sycl::buffer<double> b_buf,
                      sycl::buffer<double> out_buf, int height_a, int width_a,
                      sycl::queue queue, sycl::nd_range<1> nd_range);
double sycl_mtx_dot(sycl::buffer<double> v_buf, sycl::buffer<double> w_buf,
                    int size, sycl::queue queue, sycl::nd_range<1> nd_range);
void sycl_calc_w_init(sycl::buffer<double> w_buf, double alpha,
                      sycl::buffer<double> v_buf, int size, sycl::queue queue,
                      sycl::nd_range<1> nd_range);
void sycl_calc_w(sycl::buffer<double> w_buf, double alpha,
                 sycl::buffer<double> v_buf, sycl::buffer<double> v_pre_buf,
                 double beta, int size, sycl::queue queue,
                 sycl::nd_range<1> nd_range);
void sycl_spmv(sycl::buffer<int> a_row_buf, sycl::buffer<int> a_columns_buf,
               sycl::buffer<double> a_vals_buf, sycl::buffer<double> b_buf,
               sycl::buffer<double> out_buf, int height_a, int width_a,
               sycl::queue queue, sycl::nd_range<1> nd_range);
#endif

// Parallelized functions of Lanczos routine implemented using Opencl.
#if defined(ENABLE_OPENCL)

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

double ocl_vec_dot(cl_context ctx, cl_command_queue queue, cl_program prg,
                   cl_mem d_a_vec, cl_mem d_b_vec, int size);
double ocl_vec_norm(cl_context ctx, cl_command_queue queue, cl_program prg,
                    cl_mem d_a_vec, int size);
void ocl_vec_sclr_div(cl_context ctx, cl_command_queue queue, cl_program prg,
                      cl_mem d_a_vec, cl_mem d_out_vec, double sclr, int size);
void ocl_mtx_col_copy(cl_context ctx, cl_command_queue queue, cl_program prg,
                      cl_mem d_vec, cl_mem d_mtx, int col_index, int size);
void ocl_vec_copy(cl_context ctx, cl_command_queue queue, cl_program prg,
                  cl_mem d_vec, cl_mem d_vec_pre, int size);
void ocl_mtx_vec_mul(cl_context ctx, cl_command_queue queue, cl_program prg,
                     cl_mem d_a_mtx, cl_mem d_b_vec, cl_mem d_out_vec,
                     int num_rows, int num_cols);
void ocl_spmv(cl_context ctx, cl_command_queue queue, cl_program prg,
              cl_mem d_a_row_ptrs, cl_mem d_a_columns, cl_mem d_a_vals,
              cl_mem d_b_vec, cl_mem d_out_vec, int num_rows, int num_cols);
void ocl_calc_w_init(cl_context ctx, cl_command_queue queue, cl_program prg,
                     cl_mem d_w_vec, double alpha, cl_mem d_orth_vec, int size);
void ocl_calc_w(cl_context ctx, cl_command_queue queue, cl_program prg,
                cl_mem d_w_vec, double alpha, cl_mem d_orth_vec,
                cl_mem d_orth_vec_pre, double beta, int size);
void ocl_d2d_mem_cpy(cl_context ctx, cl_command_queue queue, cl_program prg,
                     cl_mem d_a_vec, cl_mem d_out_vec, int size);
#endif

#ifdef __cplusplus
}
#endif
