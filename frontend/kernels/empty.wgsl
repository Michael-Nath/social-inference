@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>, // Identifies the workgroup (and thus the row)
  @builtin(local_invocation_id) local_id: vec3<u32>   // Identifies the thread within the workgroup
) {}