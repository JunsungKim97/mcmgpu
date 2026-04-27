inline int idx4(int e, int i, int j, int k, int n) {
  return ((e * n + k) * n + j) * n + i;
}

// Nekbone-like tensor-product operator apply:
// y = D*x + D^T*x + lambda*x (separable 1D derivative stencil on each axis)
__kernel void nekbone10_apply(__global const float* x,
                              __global const float* d,
                              __global float* y,
                              int n,
                              int elements,
                              float lambda) {
  int i = get_global_id(0);
  int jk = get_global_id(1);
  int e = get_global_id(2);

  int j = jk % n;
  int k = jk / n;

  if (i >= n || j >= n || k >= n || e >= elements) return;

  float sumx = 0.0f;
  float sumy = 0.0f;
  float sumz = 0.0f;

  for (int p = 0; p < n; p++) {
    float dp_i = d[i * n + p];
    float dp_j = d[j * n + p];
    float dp_k = d[k * n + p];

    sumx += dp_i * x[idx4(e, p, j, k, n)];
    sumy += dp_j * x[idx4(e, i, p, k, n)];
    sumz += dp_k * x[idx4(e, i, j, p, n)];
  }

  int out = idx4(e, i, j, k, n);
  y[out] = sumx + sumy + sumz + lambda * x[out];
}
