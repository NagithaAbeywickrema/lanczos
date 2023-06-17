__kernel void mtx_sclr_div(__global double *v, __global double *w, double sclr,
                           int n) {
  int id = get_global_id(0);

  if (id < n)
    w[id] = v[id] * sclr;
};

__kernel void d2d_mem_cpy(__global double *v, __global double *w, int n) {
  int id = get_global_id(0);

  if (id < n)
    w[id] = v[id];
};

__kernel void mtx_col_copy(__global double *v, __global double *V, int i,
                           int n) {
  int id = get_global_id(0);

  if (id < n)
    V[id + n * i] = v[id];
};

__kernel void vec_copy(__global double *v, __global double *v_pre, int n) {
  int id = get_global_id(0);

  if (id < n)
    v_pre[id] = v[id];
};

__kernel void mtx_vec_mul(__global double *a, __global double *b,
                          __global double *c, int h_a, int w_a) {
  int id = get_global_id(0);
  double dot = 0;
  if (id < h_a) {
    for (int i = 0; i < w_a; i++)
      dot += a[w_a * id + i] * b[i];
    c[id] = dot;
  }
};

__kernel void spmv(__global int *a_row_ptrs, __global int *a_columns,
                   __global double *a_vals, __global double *b,
                   __global double *c, int h_a, int w_a) {
  int id = get_global_id(0);
  if (id < h_a) {
    int start = a_row_ptrs[id];
    int end = a_row_ptrs[id + 1];
    double dot = 0;
    // Add each element in the id
    for (int j = start; j < end; j++)
      dot += a_vals[j] * b[a_columns[j]];
    c[id] = dot;
  }
};

__kernel void vec_dot(__global double *v, __global double *w,
                      __global double *v_r, int n, __local double *smemory) {
  int tid = get_local_id(0);
  int i = get_group_id(0) * get_local_size(0) + get_local_id(0);

  if (i < n)
    smemory[tid] = v[i] * w[i];
  else
    smemory[tid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  int stride = get_local_size(0) >> 1;
  for (; stride > 0; stride >>= 1) {
    if (tid < stride)
      smemory[tid] += smemory[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0)
    v_r[get_group_id(0)] = smemory[0];
};

__kernel void calc_w_init(__global double *w, __global double *orth_vec,
                          double alpha, int n) {
  int id = get_global_id(0);

  if (id < n)
    w[id] = w[id] - alpha * orth_vec[id];
};

__kernel void calc_w(__global double *w, __global double *v,
                     __global double *v_pre, double alpha, double beta, int n) {
  int id = get_global_id(0);

  if (id < n)
    w[id] = w[id] - alpha * v[id] - beta * v_pre[id];
};
