// Boruvka-like candidate edge selection.
// For each node u, choose the minimum-weight outgoing edge to a different
// component label.

#define INF_W 1.0e30f

__kernel void mst_find_min_edge(__global const int* row_offsets,
                                __global const int* col_indices,
                                __global const float* edge_weights,
                                __global const int* comp_label,
                                __global int* best_dst,
                                __global float* best_w,
                                int num_nodes) {
  int u = get_global_id(0);
  if (u >= num_nodes) return;

  int cu = comp_label[u];
  int start = row_offsets[u];
  int end = row_offsets[u + 1];

  int bestV = -1;
  float best = INF_W;
  for (int e = start; e < end; e++) {
    int v = col_indices[e];
    if (comp_label[v] == cu) continue;
    float w = edge_weights[e];
    if (w < best) {
      best = w;
      bestV = v;
    }
  }

  best_dst[u] = bestV;
  best_w[u] = best;
}
