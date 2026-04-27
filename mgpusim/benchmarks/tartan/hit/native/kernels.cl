// hit_viscous_periodic: one explicit viscous sub-step on a scalar field in a
// 3D periodic box (u <- u + dt * nu * Laplacian(u)), matching the periodic
// setting of Tartan HIT (Homogeneous Isotropic Turbulence) without the full
// spectral / FFT pipeline.

__kernel void hit_viscous_periodic(__global const float* in,
                                   __global float* out,
                                   int nx,
                                   int ny,
                                   int nz,
                                   float dt,
                                   float nu) {
  int x = get_global_id(0);
  int yz = get_global_id(1);
  int y = yz % ny;
  int z = yz / ny;

  if (x >= nx || yz >= ny * nz) {
    return;
  }

  int xm = (x + nx - 1) % nx;
  int xp = (x + 1) % nx;
  int ym = (y + ny - 1) % ny;
  int yp = (y + 1) % ny;
  int zm = (z + nz - 1) % nz;
  int zp = (z + 1) % nz;

  int idx = z * nx * ny + y * nx + x;
  float c = in[idx];
  float lap = in[z * nx * ny + y * nx + xm] + in[z * nx * ny + y * nx + xp] +
              in[z * nx * ny + ym * nx + x] + in[z * nx * ny + yp * nx + x] +
              in[zm * nx * ny + y * nx + x] + in[zp * nx * ny + y * nx + x] -
              6.0f * c;
  out[idx] = c + dt * nu * lap;
}
