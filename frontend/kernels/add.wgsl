// Simple element-wise addition kernel
// Assumes inputs and output have the same dimensions

@group(0) @binding(0) var<storage, read> inputA : array<f32>;
@group(0) @binding(1) var<storage, read> inputB : array<f32>;
@group(0) @binding(2) var<storage, write> result : array<f32>;

// We can use a simpler workgroup size for element-wise ops
// Alternatively, dispatch size can be based on total elements
@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;

    // TODO: Add boundary checks if dispatch size might exceed array length
    // let arrayLength = arrayLength(&result); // Needs size uniform/storage buffer
    // if (index >= arrayLength) { return; }

    result[index] = inputA[index] + inputB[index];
} 