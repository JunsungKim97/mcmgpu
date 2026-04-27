#define WIDTH 16
#define HEIGHT 16
#define ETA 0.3f
#define MOMENTUM 0.3f

__kernel void bpnn_layerforward_ocl(__global float* input_cuda,
                                    __global float* output_hidden_cuda,
                                    __global float* input_hidden_cuda,
                                    __global float* hidden_partial_sum,
                                    int in,
                                    int hid) {
  int hidden_idx = (int)get_global_id(0);  // 0..hid-1
  int blk = (int)get_global_id(1);
  int num_blocks = in / HEIGHT;

  if (hidden_idx < hid && blk < num_blocks) {
    float sum = 0.0f;
    int j = hidden_idx + 1;
    for (int k = 0; k < HEIGHT; k++) {
      int in_idx = blk * HEIGHT + k + 1;
      int w_idx = in_idx * (hid + 1) + j;
      sum += input_hidden_cuda[w_idx] * input_cuda[in_idx];
    }
    hidden_partial_sum[blk * hid + hidden_idx] = sum;
  }
}

__kernel void bpnn_adjust_weights_ocl(__global float* delta,
                                      int hid,
                                      __global float* ly,
                                      int in,
                                      __global float* w,
                                      __global float* oldw) {
  int gid0 = (int)get_global_id(0);
  int gid1 = (int)get_global_id(1);
  int hidden_idx = gid0 + 1;  // 1..hid
  int in_idx = gid1 + 1;      // 1..in

  if (hidden_idx <= hid && in_idx <= in) {
    int w_idx = in_idx * (hid + 1) + hidden_idx;
    float new_dw = (ETA * delta[hidden_idx] * ly[in_idx]) + (MOMENTUM * oldw[w_idx]);
    w[w_idx] += new_dw;
    oldw[w_idx] = new_dw;
  }

  if (gid1 == 0 && gid0 < WIDTH) {
    int idx = gid0 + 1;
    float new_dw = (ETA * delta[idx]) + (MOMENTUM * oldw[idx]);
    w[idx] += new_dw;
    oldw[idx] = new_dw;
  }
}
