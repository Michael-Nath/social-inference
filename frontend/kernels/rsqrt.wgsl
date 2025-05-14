struct Params {
  num_elements: u32,
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>; 
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>; 
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE_X: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;

  if (idx >= params.num_elements) {
    return;
  }
  // rsqrt(x) is 1.0 / sqrt(x)
  // WGSL sqrt() handles negative inputs by returning NaN, which is fine.
  // Division by zero (if sqrt(x) is 0) will result in Inf.
  output_tensor[idx] = 1.0 / sqrt(input_tensor[idx]);
}
