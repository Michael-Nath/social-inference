const MAX_DIMS: u32 = 8u;
const MAX_DIMS_VEC4: u32 = MAX_DIMS / 4u;

struct Params {
  // Input tensor metadata
  input_shape_vecs: array<vec4<u32>, MAX_DIMS_VEC4>,
  input_strides_vecs: array<vec4<u32>, MAX_DIMS_VEC4>,
  input_rank: u32,
  
  // Output tensor metadata (rank is input_rank - 1)
  // Output strides are calculated based on output shape in the shader if needed for direct output indexing,
  // but since we map output based on input slice, explicit output strides might not be needed in params.
  // Let's include output_rank for clarity, and num_output_elements.
  output_rank: u32,
  num_output_elements: u32, // Total elements in the output tensor (num_slices)

  // Reduction specific
  reduce_dim: u32,          // The dimension in the *input* tensor being reduced
  reduce_dim_size: u32,     // The size of the dimension being reduced (input_shape[reduce_dim])
  
  // Padding to ensure 16-byte alignment for the whole struct
  // Current: 32(in_shape)+32(in_strides)+4(in_rank) + 4(out_rank)+4(num_out_el) + 4(red_dim)+4(red_dim_size) = 84 bytes
  // Next multiple of 16 is 96. Padding needed = 96 - 84 = 12 bytes (3 u32s)
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>; 
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>; 
@group(0) @binding(2) var<uniform> params: Params;

// Should be a power of two, e.g., 256.
const WORKGROUP_SIZE_X: u32 = 256u; 
var<workgroup> s_reduce_buffer: array<f32, WORKGROUP_SIZE_X>;

fn get_input_shape_val(idx: u32) -> u32 {
  return params.input_shape_vecs[idx / 4u][idx % 4u];
}
fn get_input_stride_val(idx: u32) -> u32 {
  return params.input_strides_vecs[idx / 4u][idx % 4u];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let tid = local_id.x; // Thread ID within the workgroup (0 to WORKGROUP_SIZE_X - 1)
  let slice_idx = workgroup_id.y; // Index of the output slice this workgroup is responsible for

  // Boundary check for slices (if num_output_elements is correctly params.num_total_slices)
  if (slice_idx >= params.num_output_elements) {
    return;
  }

  // --- Determine the multi-dimensional coordinate for the current output slice ---
  // This output_coord will have (input_rank - 1) dimensions.
  var output_coord: array<u32, MAX_DIMS>; 
  var current_output_flat_idx = slice_idx;
  var current_dim_idx_out = 0u;

  for (var d_in = 0u; d_in < params.input_rank; d_in = d_in + 1u) {
    if (d_in == params.reduce_dim) {
      continue; // Skip the reduction dimension for output coordinates
    }
    let dim_size = get_input_shape_val(d_in); // Output dim size matches input dim size for non-reduced dims
    // Calculate strides for this hypothetical output tensor on the fly (or pass output_strides)
    // Let's reconstruct output strides implicitly for now.
    // To do this properly, we need the output shape to calculate output strides.
    // For now, we assume the slice_idx directly maps to the flattened index of the output tensor.
    // The shader will write to output_tensor[slice_idx] after reduction.
    // The complex part is reading the correct INPUT elements for this slice_idx.

    // We need to map slice_idx to an N-D coordinate in the input space,
    // but with params.reduce_dim effectively skipped or set to 0 for the starting point of the slice.
    current_dim_idx_out = current_dim_idx_out + 1u;
  }
  // At this point, slice_idx corresponds to a unique combination of non-reduced dimension indices.

  // --- Summation along the reduce_dim for the current slice ---
  var thread_sum: f32 = 0.0;
  // Each thread in the workgroup sums a portion of the elements along params.reduce_dim
  for (var i = tid; i < params.reduce_dim_size; i = i + WORKGROUP_SIZE_X) {
    // Construct the full N-D coordinate in the input tensor
    var input_coord_full: array<u32, MAX_DIMS>;
    var temp_slice_idx_for_input_coord = slice_idx;
    
    var current_read_dim_out_tracker = 0u;
    for (var d_input = 0u; d_input < params.input_rank; d_input = d_input + 1u) {
        if (d_input == params.reduce_dim) {
            input_coord_full[d_input] = i; // This is the element along the reduction dimension
        } else {
            // Calculate coordinate for this non-reduced dimension based on slice_idx
            // This requires knowing the effective "shape" of the slices
            var stride_for_this_slice_dim = 1u;
            for(var k = current_read_dim_out_tracker + 1u; k < params.output_rank; k=k+1u) {
                var original_input_dim_k_idx = k;
                if (k >= params.reduce_dim && params.input_rank > params.output_rank) { // Adjust index if reduce_dim was before it
                    original_input_dim_k_idx = k + 1u; 
                }
                 if (original_input_dim_k_idx < params.input_rank) {
                    stride_for_this_slice_dim = stride_for_this_slice_dim * get_input_shape_val(original_input_dim_k_idx);
                 }
            }
            if (get_input_shape_val(d_input) == 0u) { // Avoid div by zero for empty non-reduced dim
                 input_coord_full[d_input] = 0u;
            } else {
                 // This logic is tricky without precomputed output strides / output shape in params
                 // For now, let's simplify assuming slice_idx gives us the right output slot
                 // and we just need to iterate `i` along `reduce_dim`.
                 // The mapping from slice_idx to the multi-dim coords of non-reduced dims is complex.
                 // Simplified: assume slice_idx correctly corresponds to the output element index.
                 // The task is to find the input elements for this output element.
                 // This requires mapping the output flat index (slice_idx) to an output N-1 dim coord,
                 // then inserting the reduction dim to get an input N dim coord.

                 // Reconstruct input_coord from slice_idx and `i` for reduce_dim
                 // This is the hard part.
                 // Let input_flat_idx_base be the flat index of the first element of the current slice if reduce_dim index were 0.
                 var current_dim_prod = 1u;
                 var input_flat_idx_base_calc = 0u;
                 var out_dim_counter = params.output_rank;

                 // Iterate from the highest dimension of the output tensor downwards
                 for (var od_rev = 0u; od_rev < params.output_rank; od_rev = od_rev + 1u) {
                    var out_d = params.output_rank - 1u - od_rev;
                    var in_d = out_d;
                    if (in_d >= params.reduce_dim) { in_d = in_d + 1u; } // map output dim to input dim

                    let out_dim_size = get_input_shape_val(in_d);
                    let coord_in_out_d = (temp_slice_idx_for_input_coord / current_dim_prod) % out_dim_size;
                    input_coord_full[in_d] = coord_in_out_d;
                    current_dim_prod = current_dim_prod * out_dim_size;
                 }
                 temp_slice_idx_for_input_coord = slice_idx; // Reset for each iteration if needed by logic
                 input_coord_full[params.reduce_dim] = i; // Current element along reduction axis
            }
             current_read_dim_out_tracker = current_read_dim_out_tracker + 1u;
        }
    }

    // Convert full N-D input coordinate to flat index
    var input_flat_idx = 0u;
    for (var d = 0u; d < params.input_rank; d = d + 1u) {
      input_flat_idx = input_flat_idx + input_coord_full[d] * get_input_stride_val(d);
    }
    thread_sum = thread_sum + input_tensor[input_flat_idx];
  }

  // Store partial sum in shared memory
  s_reduce_buffer[tid] = thread_sum;
  workgroupBarrier(); // Ensure all threads have written their partial sums

  // --- Parallel reduction within the workgroup ---
  // This assumes WORKGROUP_SIZE_X is a power of two
  for (var s = WORKGROUP_SIZE_X / 2u; s > 0u; s = s / 2u) {
    if (tid < s) {
      s_reduce_buffer[tid] = s_reduce_buffer[tid] + s_reduce_buffer[tid + s];
    }
    workgroupBarrier(); // Synchronize after each step of reduction
  }

  // The first thread (tid == 0) now holds the sum for the slice for this workgroup
  if (tid == 0u) {
    if (params.reduce_dim_size == 0u) { // Avoid division by zero for empty reduction dimension
        output_tensor[slice_idx] = 0.0; // Or NaN, depending on desired behavior
    } else {
        output_tensor[slice_idx] = s_reduce_buffer[0] / f32(params.reduce_dim_size);
    }
  }
}
