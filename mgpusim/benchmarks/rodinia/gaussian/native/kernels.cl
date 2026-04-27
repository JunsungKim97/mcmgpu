inline int idx2(int r, int c, int n) { return r * n + c; }

// Normalize pivot row at step t:
// A[t,j] /= A[t,t], b[t] /= A[t,t], A[t,t] = 1
__kernel void gaussian_normalize(__global float* a,
                                 __global float* b,
                                 int n,
                                 int t) {
  int j = get_global_id(0) + t;
  if (j >= n) return;

  float piv = a[idx2(t, t, n)];
  a[idx2(t, j, n)] = a[idx2(t, j, n)] / piv;
  if (j == t) {
    b[t] = b[t] / piv;
  }
}

// Eliminate rows i>t:
// A[i,j] -= A[i,t]*A[t,j], b[i] -= A[i,t]*b[t], A[i,t]=0
__kernel void gaussian_eliminate(__global float* a,
                                 __global float* b,
                                 int n,
                                 int t) {
  int j = get_global_id(0) + t + 1;
  int i = get_global_id(1) + t + 1;
  if (i >= n || j >= n) return;

  float f = a[idx2(i, t, n)];
  a[idx2(i, j, n)] = a[idx2(i, j, n)] - f * a[idx2(t, j, n)];
  if (j == t + 1) {
    b[i] = b[i] - f * b[t];
    a[idx2(i, t, n)] = 0.0f;
  }
}
