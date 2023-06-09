__kernel void mtx_sclr_div(__global double *v, __global double *w, double sclr,
                           const unsigned n) {
  int id = get_global_id(0);

  if (id < n)
    w[id] = v[id] / sclr;
};

__kernel void mtx_col_copy(__global double *v, __global double *V, int i,
                           const unsigned n) {
  int id = get_global_id(0);

  if (id < n)
    V[id + n * i] = v[id];
};

__kernel void mtx_vec_mul(__global double *a, __global double *b,
                          __global double *c, const unsigned h_a,
                          const unsigned w_a) {
  int id = get_global_id(0);
  double dot = 0;
  if (id < h_a) {
    for (int i = 0; i < w_a; i++)
      dot += a[w_a * id + i] * b[i];
    c[id] = dot;
  }
};

__kernel void vec_dot(__global double *v, __global double *w,
                      __global double *v_r, const unsigned n,
                      __local double *smemory) {
  unsigned int tid = get_local_id(0);
  unsigned int i = get_group_id(0) * get_local_size(0) + get_local_id(0);

  if (i < n)
    smemory[tid] = v[i] * w[i];
  else
    smemory[tid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned int stride = get_local_size(0) >> 1;
  for (; stride > 0; stride >>= 1) {
    if (tid < stride)
      smemory[tid] += smemory[tid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0)
    v_r[get_group_id(0)] = smemory[0];
};

__kernel void calc_w_init(__global double *w, __global double *V, double alpha,
                          int i, const unsigned n) {
  int id = get_global_id(0);

  if (id < n)
    w[id] = w[id] - alpha * V[id + n * i];
};

__kernel void calc_w(__global double *w, __global double *V, double alpha,
                     double beta, int i, const unsigned n) {
  int id = get_global_id(0);

  if (id < n)
    w[id] = w[id] - alpha * V[id + n * i] - beta * V[id + n * (i - 1)];
};
