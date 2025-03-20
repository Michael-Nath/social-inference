@group(0) @binding(0) var<uniform> uniforms: vec4<f32>;
@group(0) @binding(1) var<storage, read> inputVectorX: array<f32>;
@group(0) @binding(2) var<storage, read> inputVectorY: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputVector: array<f32>;

// The compute shader that will execute the SAXPY operation in parallel
@compute @workgroup_size(256)
fn saxpyKernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Get the current thread's index
    let idx = global_id.x;
    
    // Make sure we don't go out of bounds
    if (idx < arrayLength(&inputVectorX)) {
      // Perform the SAXPY operation: result = a*x + y
      outputVector[idx] = uniforms.x * inputVectorX[idx] + inputVectorY[idx];
  }
}