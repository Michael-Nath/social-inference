@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>; // num_elements, padding

const WORKGROUP_SIZE_X = 256u;

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_elements = params[0];
    if (global_id.x >= num_elements) {
        return;
    }

    // Standard behavior for division by zero (e.g., inf, -inf, NaN) will apply.
    // Consider adding specific handling if required (e.g., outputting a specific sentinel value).
    output[global_id.x] = a[global_id.x] / b[global_id.x];
}
