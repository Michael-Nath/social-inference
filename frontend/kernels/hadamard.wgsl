struct Params {
  num_elements: u32,
  // Add padding if other members are added to ensure alignment
};

@group(0) @binding(0) var<storage, read> input_a: array<f32>; // Input tensor A
@group(0) @binding(1) var<storage, read> input_b: array<f32>; // Input tensor B
@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>; // Output tensor
@group(0) @binding(3) var<uniform> params: Params; // Uniform buffer for parameters

const WORKGROUP_SIZE_X: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;

  if (idx >= params.num_elements) {
    return;
  }

  output_tensor[idx] = input_a[idx] * input_b[idx];
}
