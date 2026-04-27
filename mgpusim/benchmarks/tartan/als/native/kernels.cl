// ALS updates on sparse ratings in CSR form.
//
// This kernel performs an element-wise ALS-style update:
// x[u,f] = sum_{i in N(u)} r_ui * y[i,f] / (lambda + sum_{i in N(u)} y[i,f]^2)
// and symmetrically for item factors.
//
// The full normal-equation solve is approximated here by per-factor closed-form
// updates, which still preserves alternating user/item optimization structure.

__kernel void als_update_users(__global const int* user_ptr,
                               __global const int* user_item_idx,
                               __global const float* user_rating,
                               __global const float* item_factors,
                               __global float* user_factors,
                               int num_users,
                               int rank,
                               float lambda_reg) {
  int u = get_global_id(0);
  if (u >= num_users) return;

  int start = user_ptr[u];
  int end = user_ptr[u + 1];

  for (int f = 0; f < rank; f++) {
    float num = 0.0f;
    float den = lambda_reg;
    for (int p = start; p < end; p++) {
      int it = user_item_idx[p];
      float y = item_factors[it * rank + f];
      float r = user_rating[p];
      num += r * y;
      den += y * y;
    }
    user_factors[u * rank + f] = (den > 0.0f) ? (num * native_recip(den)) : 0.0f;
  }
}

__kernel void als_update_items(__global const int* item_ptr,
                               __global const int* item_user_idx,
                               __global const float* item_rating,
                               __global const float* user_factors,
                               __global float* item_factors,
                               int num_items,
                               int rank,
                               float lambda_reg) {
  int it = get_global_id(0);
  if (it >= num_items) return;

  int start = item_ptr[it];
  int end = item_ptr[it + 1];

  for (int f = 0; f < rank; f++) {
    float num = 0.0f;
    float den = lambda_reg;
    for (int p = start; p < end; p++) {
      int u = item_user_idx[p];
      float x = user_factors[u * rank + f];
      float r = item_rating[p];
      num += r * x;
      den += x * x;
    }
    item_factors[it * rank + f] = (den > 0.0f) ? (num * native_recip(den)) : 0.0f;
  }
}
