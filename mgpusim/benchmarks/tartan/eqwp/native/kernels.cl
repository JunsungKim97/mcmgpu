// EQWP stress update (4th-order central FD + Hooke's law increment), derived from
// Tartan b2reqwp eqwp_fd4_stress in eqwp.cu — constants match upstream #defines.
// Periodic boundaries simplify the mgpusim port (no B2R halo exchange).

#define EQWP_DT 0.5f
#define EQWP_DX 10.f
#define EQWP_DY 10.f
#define EQWP_DZ 10.f
#define EQWP_LAME1 32.00f
#define EQWP_LAME2 174.87f
#define EQWP_COEF_A (-0.083333333f)
#define EQWP_COEF_B (0.666666667f)

inline int wrap_idx(int a, int n) {
  int m = a % n;
  if (m < 0) m += n;
  return m;
}

inline float readv(__global const float* v, int x, int y, int z, int nx, int ny, int nz) {
  int xx = wrap_idx(x, nx);
  int yy = wrap_idx(y, ny);
  int zz = wrap_idx(z, nz);
  return v[zz * nx * ny + yy * nx + xx];
}

__kernel void eqwp_stress_fd4(__global const float* vx,
                              __global const float* vy,
                              __global const float* vz,
                              __global float* sigma_xx,
                              __global float* sigma_xy,
                              __global float* sigma_xz,
                              __global float* sigma_yy,
                              __global float* sigma_yz,
                              __global float* sigma_zz,
                              int nx,
                              int ny,
                              int nz) {
  int x = get_global_id(0);
  int yz = get_global_id(1);
  int y = yz % ny;
  int z = yz / ny;
  if (x >= nx || yz >= ny * nz) {
    return;
  }

  int idx = z * nx * ny + y * nx + x;

  float vx_bh2 = readv(vx, x, y, z - 2, nx, ny, nz);
  float vx_bh1 = readv(vx, x, y, z - 1, nx, ny, nz);
  float vx_if1 = readv(vx, x, y, z + 1, nx, ny, nz);
  float vx_if2 = readv(vx, x, y, z + 2, nx, ny, nz);

  float vy_bh2 = readv(vy, x, y, z - 2, nx, ny, nz);
  float vy_bh1 = readv(vy, x, y, z - 1, nx, ny, nz);
  float vy_if1 = readv(vy, x, y, z + 1, nx, ny, nz);
  float vy_if2 = readv(vy, x, y, z + 2, nx, ny, nz);

  float vz_bh2 = readv(vz, x, y, z - 2, nx, ny, nz);
  float vz_bh1 = readv(vz, x, y, z - 1, nx, ny, nz);
  float vz_if1 = readv(vz, x, y, z + 1, nx, ny, nz);
  float vz_if2 = readv(vz, x, y, z + 2, nx, ny, nz);

  float dvxx = (EQWP_COEF_A * (readv(vx, x + 2, y, z, nx, ny, nz) - readv(vx, x - 2, y, z, nx, ny, nz)) +
                EQWP_COEF_B * (readv(vx, x + 1, y, z, nx, ny, nz) - readv(vx, x - 1, y, z, nx, ny, nz))) /
               EQWP_DX;
  float dvyy = (EQWP_COEF_A * (readv(vy, x, y + 2, z, nx, ny, nz) - readv(vy, x, y - 2, z, nx, ny, nz)) +
                EQWP_COEF_B * (readv(vy, x, y + 1, z, nx, ny, nz) - readv(vy, x, y - 1, z, nx, ny, nz))) /
               EQWP_DY;
  float dvzz = (EQWP_COEF_A * (vz_if2 - vz_bh2) + EQWP_COEF_B * (vz_if1 - vz_bh1)) / EQWP_DZ;

  float dvxy = (EQWP_COEF_A * (readv(vx, x, y + 2, z, nx, ny, nz) - readv(vx, x, y - 2, z, nx, ny, nz)) +
                EQWP_COEF_B * (readv(vx, x, y + 1, z, nx, ny, nz) - readv(vx, x, y - 1, z, nx, ny, nz))) /
               EQWP_DY;
  float dvyx = (EQWP_COEF_A * (readv(vy, x + 2, y, z, nx, ny, nz) - readv(vy, x - 2, y, z, nx, ny, nz)) +
                EQWP_COEF_B * (readv(vy, x + 1, y, z, nx, ny, nz) - readv(vy, x - 1, y, z, nx, ny, nz))) /
               EQWP_DX;
  float dvxz = (EQWP_COEF_A * (vx_if2 - vx_bh2) + EQWP_COEF_B * (vx_if1 - vx_bh1)) / EQWP_DZ;
  float dvzx = (EQWP_COEF_A * (readv(vz, x + 2, y, z, nx, ny, nz) - readv(vz, x - 2, y, z, nx, ny, nz)) +
                EQWP_COEF_B * (readv(vz, x + 1, y, z, nx, ny, nz) - readv(vz, x - 1, y, z, nx, ny, nz))) /
               EQWP_DX;
  float dvyz = (EQWP_COEF_A * (vy_if2 - vy_bh2) + EQWP_COEF_B * (vy_if1 - vy_bh1)) / EQWP_DZ;
  float dvzy = (EQWP_COEF_A * (readv(vz, x, y + 2, z, nx, ny, nz) - readv(vz, x, y - 2, z, nx, ny, nz)) +
                EQWP_COEF_B * (readv(vz, x, y + 1, z, nx, ny, nz) - readv(vz, x, y - 1, z, nx, ny, nz))) /
               EQWP_DY;

  float dsxx = (EQWP_LAME1 + 2.f * EQWP_LAME2) * dvxx + EQWP_LAME1 * (dvyy + dvzz);
  float dsyy = (EQWP_LAME1 + 2.f * EQWP_LAME2) * dvyy + EQWP_LAME1 * (dvxx + dvzz);
  float dszz = (EQWP_LAME1 + 2.f * EQWP_LAME2) * dvzz + EQWP_LAME1 * (dvxx + dvyy);
  float dsxy = EQWP_LAME2 * (dvxy + dvyx);
  float dsxz = EQWP_LAME2 * (dvxz + dvzx);
  float dsyz = EQWP_LAME2 * (dvyz + dvzy);

  sigma_xx[idx] += EQWP_DT * dsxx;
  sigma_xy[idx] += EQWP_DT * dsxy;
  sigma_xz[idx] += EQWP_DT * dsxz;
  sigma_yy[idx] += EQWP_DT * dsyy;
  sigma_yz[idx] += EQWP_DT * dsyz;
  sigma_zz[idx] += EQWP_DT * dszz;
}
