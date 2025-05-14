struct Params {
  num_elements: u32,
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>; 
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>; 
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE_X: u32 = 256u;

// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;

  if (idx >= params.num_elements) {
    return;
  }
  let x = input_tensor[idx];
  output_tensor[idx] = x * sigmoid(x);
}
