inline int idx2(int r, int c, int n) { return r * n + c; }

// Update trailing submatrix for LU without pivoting:
// A(i,j) = A(i,j) - A(i,k) * A(k,j), for i,j > k
__kernel void lud_trailing_update(__global float* a, int n, int k) {
  int j = get_global_id(0) + k + 1;
  int i = get_global_id(1) + k + 1;

  if (i >= n || j >= n) return;
  a[idx2(i, j, n)] -= a[idx2(i, k, n)] * a[idx2(k, j, n)];
}
