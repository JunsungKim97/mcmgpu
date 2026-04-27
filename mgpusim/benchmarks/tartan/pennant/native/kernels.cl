// PENNANT-like unstructured mesh zone update.
//
// For each zone z:
//   avg_v = average(point_vel[p] for p in zone_points[z])
//   zone_energy[z] += dt * (zone_pressure[z] * avg_v - 0.1 * zone_energy[z])
//   zone_pressure[z] = gamma * zone_energy[z]
//
// This models indirect connectivity access (zone -> variable list of points)
// and per-zone physics state updates, which are core PENNANT-like patterns.

__kernel void pennant_zone_update(__global const int* zone_point_offsets,
                                  __global const int* zone_point_indices,
                                  __global const float* point_velocity,
                                  __global float* zone_energy,
                                  __global float* zone_pressure,
                                  int num_zones,
                                  float dt,
                                  float gamma) {
  int z = get_global_id(0);
  if (z >= num_zones) return;

  int begin = zone_point_offsets[z];
  int end = zone_point_offsets[z + 1];
  int count = end - begin;
  if (count <= 0) return;

  float v_sum = 0.0f;
  for (int p = begin; p < end; p++) {
    int pt = zone_point_indices[p];
    v_sum += point_velocity[pt];
  }
  float avg_v = v_sum / (float)count;

  float e = zone_energy[z];
  float p = zone_pressure[z];
  e = e + dt * (p * avg_v - 0.1f * e);
  p = gamma * e;

  zone_energy[z] = e;
  zone_pressure[z] = p;
}
