#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))

__kernel void hotspot(int iteration,
                      __global float* power,
                      __global float* temp_src,
                      __global float* temp_dst,
                      int grid_cols,
                      int grid_rows,
                      float cap,
                      float rx,
                      float ry,
                      float rz,
                      float step) {
  const float amb_temp = 80.0f;
  const float step_div_cap = step / cap;

  int x = get_global_id(0);
  int y = get_global_id(1);

  if (!IN_RANGE(x, 0, grid_cols - 1) || !IN_RANGE(y, 0, grid_rows - 1)) {
    return;
  }

  int idx = y * grid_cols + x;
  int w = (x == 0) ? idx : idx - 1;
  int e = (x == grid_cols - 1) ? idx : idx + 1;
  int n = (y == 0) ? idx : idx - grid_cols;
  int s = (y == grid_rows - 1) ? idx : idx + grid_cols;

  float temp = temp_src[idx];
  float delta = power[idx]
      + (temp_src[w] + temp_src[e] - 2.0f * temp) / rx
      + (temp_src[n] + temp_src[s] - 2.0f * temp) / ry
      + (amb_temp - temp) / rz;

  temp_dst[idx] = temp + step_div_cap * delta;
}
