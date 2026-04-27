inline int idx2(int r, int c, int cols) { return r * cols + c; }

// Compute diffusion coefficient (simplified SRAD form).
__kernel void srad_compute_coeff(__global const float* img,
                                 __global float* coeff,
                                 int rows,
                                 int cols,
                                 float q0sqr) {
  int c = get_global_id(0);
  int r = get_global_id(1);
  if (r >= rows || c >= cols) return;

  int n = (r == 0) ? 0 : r - 1;
  int s = (r == rows - 1) ? rows - 1 : r + 1;
  int w = (c == 0) ? 0 : c - 1;
  int e = (c == cols - 1) ? cols - 1 : c + 1;

  float jc = img[idx2(r, c, cols)];
  float dN = img[idx2(n, c, cols)] - jc;
  float dS = img[idx2(s, c, cols)] - jc;
  float dW = img[idx2(r, w, cols)] - jc;
  float dE = img[idx2(r, e, cols)] - jc;

  float g2 = (dN * dN + dS * dS + dW * dW + dE * dE) * native_recip(jc * jc + 1e-12f);
  float l = (dN + dS + dW + dE) * native_recip(jc + 1e-12f);
  float num = 0.5f * g2 - (1.0f / 16.0f) * (l * l);
  float den = 1.0f + 0.25f * l;
  float qsqr = num * native_recip(den * den + 1e-12f);

  float den2 = (qsqr - q0sqr) * native_recip(q0sqr * (1.0f + q0sqr) + 1e-12f);
  float cval = native_recip(1.0f + den2);
  if (cval < 0.0f) cval = 0.0f;
  if (cval > 1.0f) cval = 1.0f;
  coeff[idx2(r, c, cols)] = cval;
}

// Update image using computed coefficients.
__kernel void srad_update(__global float* img,
                          __global const float* coeff,
                          int rows,
                          int cols,
                          float lambda) {
  int c = get_global_id(0);
  int r = get_global_id(1);
  if (r >= rows || c >= cols) return;

  int n = (r == 0) ? 0 : r - 1;
  int s = (r == rows - 1) ? rows - 1 : r + 1;
  int w = (c == 0) ? 0 : c - 1;
  int e = (c == cols - 1) ? cols - 1 : c + 1;

  float jc = img[idx2(r, c, cols)];
  float dN = img[idx2(n, c, cols)] - jc;
  float dS = img[idx2(s, c, cols)] - jc;
  float dW = img[idx2(r, w, cols)] - jc;
  float dE = img[idx2(r, e, cols)] - jc;

  float cN = coeff[idx2(n, c, cols)];
  float cS = coeff[idx2(s, c, cols)];
  float cW = coeff[idx2(r, w, cols)];
  float cE = coeff[idx2(r, e, cols)];

  float div = cN * dN + cS * dS + cW * dW + cE * dE;
  img[idx2(r, c, cols)] = jc + 0.25f * lambda * div;
}
