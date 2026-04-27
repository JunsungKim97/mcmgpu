// Synchronous PageRank update on incoming-edge CSR.
// For node v:
//   pr_next[v] = base + damping * sum_{u in In(v)} pr[u] / out_deg[u]
//
// Graph construction ensures out_deg[u] >= 1 for all u, so dangling handling
// is not needed in this synthetic port.

__kernel void pagerank_update_sync(__global const int* in_ptr,
                                   __global const int* in_src,
                                   __global const int* out_deg,
                                   __global const float* pr_in,
                                   __global float* pr_out,
                                   int num_nodes,
                                   float damping,
                                   float base) {
  int v = get_global_id(0);
  if (v >= num_nodes) return;

  int begin = in_ptr[v];
  int end = in_ptr[v + 1];

  float acc = 0.0f;
  for (int p = begin; p < end; p++) {
    int u = in_src[p];
    int deg = out_deg[u];
    acc += pr_in[u] * native_recip((float)deg);
  }

  pr_out[v] = base + damping * acc;
}
