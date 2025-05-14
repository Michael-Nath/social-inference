const MAX_DIMS: u32 = 8u; // Should match MAX_DIMS_SUPPORTED/MAX_DIMS_SOFTMAX in worker.js
const MAX_DIMS_VEC4: u32 = MAX_DIMS / 4u;

struct Params {
  // vec4 arrays for 16-byte alignment for uniform buffers
  input_shape_vecs: array<vec4<u32>, MAX_DIMS_VEC4>,
  input_strides_vecs: array<vec4<u32>, MAX_DIMS_VEC4>,
  rank: u32,
  dim0_to_swap: u32,
  dim1_to_swap: u32,
  num_elements: u32,
  // Padding to ensure struct size is a multiple of 16.
  // Current members: 32 (shape) + 32 (strides) + 4 (rank) + 4 (d0) + 4 (d1) + 4 (num_elements) = 80 bytes.
  // 80 is a multiple of 16, so no explicit padding variables needed here IF tightly packed.
  // However, WGSL struct layout rules can be tricky. Let's assume this is fine.
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>; 
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_shape_val(idx: u32) -> u32 {
  return params.input_shape_vecs[idx / 4u][idx % 4u];
}

fn get_stride_val(idx: u32) -> u32 {
  return params.input_strides_vecs[idx / 4u][idx % 4u];
}

const WORKGROUP_SIZE_X: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_flat_idx = global_id.x;

  if (output_flat_idx >= params.num_elements) {
    return;
  }

  if (params.rank == 0u) { // Handle 0D (scalar) case
    output_tensor[output_flat_idx] = input_tensor[output_flat_idx];
    return;
  }

  // Calculate output shape and strides (by transposing input shape/strides)
  var output_shape: array<u32, MAX_DIMS>;
  var output_strides: array<u32, MAX_DIMS>;
  var temp_output_shape_for_strides: array<u32, MAX_DIMS>; // for calculateStrides pattern

  for (var i = 0u; i < params.rank; i = i + 1u) {
    output_shape[i] = get_shape_val(i);
    temp_output_shape_for_strides[i] = get_shape_val(i);
  }
  // Transpose the shape for output_shape and temp_output_shape_for_strides
  var temp_dim_val = output_shape[params.dim0_to_swap];
  output_shape[params.dim0_to_swap] = output_shape[params.dim1_to_swap];
  output_shape[params.dim1_to_swap] = temp_dim_val;

  temp_dim_val = temp_output_shape_for_strides[params.dim0_to_swap];
  temp_output_shape_for_strides[params.dim0_to_swap] = temp_output_shape_for_strides[params.dim1_to_swap];
  temp_output_shape_for_strides[params.dim1_to_swap] = temp_dim_val;

  // Calculate output strides (standard C-order strides for the transposed shape)
  if (params.rank > 0u) {
      output_strides[params.rank - 1u] = 1u;
      for (var i = params.rank - 2u; i < params.rank; i = i - 1u) { // Loop condition i < params.rank is effectively i >= 0 for unsigned
          output_strides[i] = output_strides[i + 1u] * temp_output_shape_for_strides[i + 1u];
      }
  }
  
  // Convert flat output index to N-D output coordinates
  var output_coord: array<u32, MAX_DIMS>;
  var remainder = output_flat_idx;
  for (var i = 0u; i < params.rank; i = i + 1u) {
    if (output_strides[i] == 0u && remainder != 0u && output_shape[i] != 0u) {
        // This case should ideally not be hit if strides/shapes are consistent
        // Can happen if a dim size is 0, leading to stride 0.
        // If output_shape[i] is non-zero but stride is zero, it means higher dims were zero.
        // For safety, if a stride is zero, the coord for that dim must be zero if remainder is still non-zero.
        // However, with valid shapes, this is more about handling product of dims being 0.
        // If num_elements is 0, this loop body won't run due to the initial check.
        // This is complex. Let's assume valid strides from valid shapes for now.
    }
    if (output_shape[i] > 0u) { // Avoid division by zero for empty dimensions
        output_coord[i] = remainder / output_strides[i];
        remainder = remainder % output_strides[i];
    } else {
        output_coord[i] = 0u; // Coordinate is 0 for an empty dimension
    }
  }

  // Transform N-D output coordinates to N-D input coordinates by swapping d0 and d1
  var input_coord: array<u32, MAX_DIMS>; // = output_coord;
  for(var i=0u; i < params.rank; i=i+1u) { input_coord[i] = output_coord[i]; }

  temp_dim_val = input_coord[params.dim0_to_swap];
  input_coord[params.dim0_to_swap] = input_coord[params.dim1_to_swap];
  input_coord[params.dim1_to_swap] = temp_dim_val;

  // Convert N-D input coordinates back to a flat input index
  var input_flat_idx = 0u;
  for (var i = 0u; i < params.rank; i = i + 1u) {
    input_flat_idx = input_flat_idx + input_coord[i] * get_stride_val(i);
  }

  output_tensor[output_flat_idx] = input_tensor[input_flat_idx];
}
