// Simple element-wise addition kernel
// Assumes inputs and output have the same dimensions

struct Dimensions {
  size: u32,
}

@group(0) @binding(0) var<storage, read> inputA : array<f32>;
@group(0) @binding(1) var<storage, read> inputB : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(256) 
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    if (index >= dimensions.size) { return; }
    result[index] = inputA[index] + inputB[index];
} 