// lulesh_lagrange_nodal: merges CalcAccelerationForNodes_kernel and
// CalcPositionAndVelocityForNodes_kernel from Tartan scale-out lulesh.cu
// (LULESH Lagrange nodal path). Forces and nodal mass are read-only inputs;
// MPI / symm BC / volume force kernels are omitted for this mgpusim microbench.
//
// Reference: https://github.com/uuudown/Tartan/tree/master/scale-out/scale-out/lulesh

__kernel void lulesh_lagrange_nodal(__global const float* fx,
                                     __global const float* fy,
                                     __global const float* fz,
                                     __global const float* nodal_mass,
                                     __global float* x,
                                     __global float* y,
                                     __global float* z,
                                     __global float* xd,
                                     __global float* yd,
                                     __global float* zd,
                                     int num_node,
                                     float deltatime,
                                     float u_cut) {
  int i = get_global_id(0);
  if (i >= num_node) {
    return;
  }

  float invm = 1.0f / nodal_mass[i];
  float xdd = fx[i] * invm;
  float ydd = fy[i] * invm;
  float zdd = fz[i] * invm;

  float dt = deltatime;
  float xdtmp = xd[i] + xdd * dt;
  float ydtmp = yd[i] + ydd * dt;
  float zdtmp = zd[i] + zdd * dt;

  if (fabs(xdtmp) < u_cut) xdtmp = 0.0f;
  if (fabs(ydtmp) < u_cut) ydtmp = 0.0f;
  if (fabs(zdtmp) < u_cut) zdtmp = 0.0f;

  x[i] += xdtmp * dt;
  y[i] += ydtmp * dt;
  z[i] += zdtmp * dt;

  xd[i] = xdtmp;
  yd[i] = ydtmp;
  zd[i] = zdtmp;
}
