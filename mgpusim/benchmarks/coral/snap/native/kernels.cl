// SNAP-like 3D transport sweep step.
//
// For each cell:
//   psi_out = (source + ax*psi_xm + ay*psi_ym + az*psi_zm) / (sigma_t + ax + ay + az)
//
// This captures key SNAP sweep dependencies and memory access pattern.

inline int idx3(int x, int y, int z, int nx, int ny) {
  return z * nx * ny + y * nx + x;
}

__kernel void snap_sweep_step(__global const float* psi_in,
                              __global const float* source,
                              __global float* psi_out,
                              int nx,
                              int ny,
                              int nz,
                              float ax,
                              float ay,
                              float az,
                              float sigma_t) {
  int x = get_global_id(0);
  int yz = get_global_id(1);
  int y = yz % ny;
  int z = yz / ny;
  if (x >= nx || y >= ny || z >= nz) return;

  int i = idx3(x, y, z, nx, ny);
  float px = (x > 0) ? psi_in[idx3(x - 1, y, z, nx, ny)] : 0.0f;
  float py = (y > 0) ? psi_in[idx3(x, y - 1, z, nx, ny)] : 0.0f;
  float pz = (z > 0) ? psi_in[idx3(x, y, z - 1, nx, ny)] : 0.0f;

  float num = source[i] + ax * px + ay * py + az * pz;
  float den = sigma_t + ax + ay + az;
  psi_out[i] = num / den;
}
