#if defined(ENABLE_SYCL)
#include <CL/sycl.hpp>
#endif
#ifdef __cplusplus
extern "C" {
#endif

// Serially implemented functions of Lanczos routine.
double serial_vec_dot(double *a_vec, double *b_vec, const unsigned size);
double serial_vec_norm(double *a_vec, const unsigned size);
void serial_vec_sclr_div(double *a_vec, double *out_vec, const double sclr,
                         const unsigned size);
void serial_mtx_col_copy(double *vec, double *mtx, const unsigned col_index,
                         const unsigned size);
void serial_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                        const unsigned num_rows, const unsigned num_cols);
void serial_calc_w_init(double *w_vec, const double alpha, double *orth_mtx,
                        const unsigned col_index, const unsigned size);
void serial_calc_w(double *w_vec, const double alpha, double *orth_mtx,
                   const double beta, const unsigned col_index,
                   const unsigned size);

// Parallelized functions of Lanczos routine implemented using Nomp.
double nomp_vec_dot(double *a_vec, double *b_vec, const unsigned size);
double nomp_vec_norm(double *a_vec, const unsigned size);
void nomp_vec_sclr_div(double *a_vec, double *out_vec, const double sclr,
                       const unsigned size);
void nomp_mtx_col_copy(double *vec, double *mtx, const unsigned col_index,
                       const unsigned size);
void nomp_mtx_vec_mul(double *a_mtx, double *b_vec, double *out_vec,
                      const unsigned num_rows, const unsigned num_cols);
void nomp_calc_w_init(double *w_vec, const double alpha, double *orth_mtx,
                      const unsigned col_index, const unsigned size);
void nomp_calc_w(double *w_vec, const double alpha, double *orth_mtx,
                 const double beta, const unsigned col_index,
                 const unsigned size);

// Parallelized functions of Lanczos routine implemented using Cuda.
double cuda_vec_dot(double *d_a_vec, double *d_b_vec, const unsigned size);
double cuda_vec_norm(double *d_a_vec, const unsigned size);
void cuda_vec_sclr_div(double *d_a_vec, double *d_out_vec, const double sclr,
                       const unsigned size);
void cuda_mtx_col_copy(double *d_vec, double *d_mtx, const unsigned col_index,
                       const unsigned size);
void cuda_mtx_vec_mul(double *d_a_mtx, double *d_b_vec, double *d_out_vec,
                      const unsigned num_rows, const unsigned num_cols);
void cuda_calc_w_init(double *d_w_vec, const double alpha, double *d_orth_mtx,
                      const unsigned col_index, const unsigned size);
void cuda_calc_w(double *d_w_vec, const double alpha, double *d_orth_mtx,
                 const double beta, const unsigned col_index,
                 const unsigned size);

#if defined(ENABLE_SYCL)

double sycl_mtx_norm(sycl::buffer<double> w, const int size, sycl::queue queue);
void sycl_mtx_sclr_div(sycl::buffer<double> in_buf, double scalar,
                       sycl::buffer<double> out_buf, const int size,
                       sycl::queue queue);
void sycl_mtx_col_copy(sycl::buffer<double> v_temp_buf,
                       sycl::buffer<double> v_buf, int j, const int size,
                       sycl::queue queue);
void sycl_mtx_vec_mul(sycl::buffer<double> a_buf, sycl::buffer<double> b_buf,
                      sycl::buffer<double> out_buf, const int height_a,
                      const int width_a, sycl::queue queue);
double sycl_mtx_dot(sycl::buffer<double> v_buf, sycl::buffer<double> w_buf,
                    const int size, sycl::queue queue);
void sycl_calc_w_init(sycl::buffer<double> w_buf, double alpha,
                      sycl::buffer<double> v_buf, unsigned i, const int size,
                      sycl::queue queue);
void sycl_calc_w(sycl::buffer<double> w_buf, double alpha,
                 sycl::buffer<double> v_buf, double beta, unsigned i,
                 const int size, sycl::queue queue);
#endif

// Opencl implemented parallelized functions
double ocl_vec_norm(cl_context ctx, cl_command_queue queue, cl_program prg,
                    cl_mem d_v, double *v_r, const int size);
void ocl_mtx_sclr_div(cl_context ctx, cl_command_queue queue, cl_program prg,
                      cl_mem d_v, cl_mem d_w, double sclr, const unsigned size);
void ocl_mtx_col_copy(cl_context ctx, cl_command_queue queue, cl_program prg,
                      cl_mem d_v, cl_mem d_V, int i, unsigned size);
void ocl_mtx_vec_mul(cl_context ctx, cl_command_queue queue, cl_program prg,
                     cl_mem d_lap, cl_mem d_v, cl_mem d_w, const int h_a,
                     const int w_a);
double ocl_vec_dot(cl_context ctx, cl_command_queue queue, cl_program prg,
                   cl_mem d_v, cl_mem d_w, double *v_r, const int SIZE);
void ocl_calc_w_init(cl_context ctx, cl_command_queue queue, cl_program prg,
                     cl_mem d_w, double alpha, cl_mem d_V, unsigned i,
                     const int size);
void ocl_calc_w(cl_context ctx, cl_command_queue queue, cl_program prg,
                cl_mem d_w, double alpha, cl_mem d_V, double beta, unsigned i,
                const int size);

#ifdef __cplusplus
}
#endif
