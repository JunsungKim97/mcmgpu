__kernel void diffusion_step(__global const float* in,
                             __global float* out,
                             int nx,
                             int ny,
                             int nz,
                             float alpha) {
  int x = get_global_id(0);
  int yz = get_global_id(1);
  int y = yz % ny;
  int z = yz / ny;

  if (x >= nx || yz >= ny * nz) {
    return;
  }

  int idx = z * nx * ny + y * nx + x;

  // Keep the domain boundary unchanged.
  if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
    out[idx] = in[idx];
    return;
  }

  int xm = idx - 1;
  int xp = idx + 1;
  int ym = idx - nx;
  int yp = idx + nx;
  int zm = idx - nx * ny;
  int zp = idx + nx * ny;

  float center = in[idx];
  float lap = in[xm] + in[xp] + in[ym] + in[yp] + in[zm] + in[zp] - 6.0f * center;
  out[idx] = center + alpha * lap;
}
