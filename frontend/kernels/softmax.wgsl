const MAX_DIMS: u32 = 8u;
const MAX_DIMS_VEC4: u32 = MAX_DIMS / 4u; // Should be 2u

struct Params {
  // Use vec4 arrays for 16-byte alignment required for uniform buffers
  shape_vecs: array<vec4<u32>, MAX_DIMS_VEC4>,
  strides_vecs: array<vec4<u32>, MAX_DIMS_VEC4>,
  ndims: u32,
  // Padding to ensure the struct size is a multiple of 16 (max alignment of vec4)
  // Total size = 16*2 (shape) + 16*2 (strides) + 4 (ndims) + 12 (padding) = 80 bytes
  _padding: vec3<u32>,
  // Output tensor will have the same metadata as input.
};

// Must be a power of two.
const workgroup_size_x: u32 = 256u;

// Shared memory for performing reductions (max and sum) within a workgroup.
// Its size is determined by workgroup_size_x.
var<workgroup> s_reduce_buffer: array<f32, workgroup_size_x>;

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> softmax_dim_tensor: array<u32, 1>;


@compute @workgroup_size(workgroup_size_x, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let sm_dim_raw = softmax_dim_tensor[0];
  let ndims = params.ndims;

  // It's expected that sm_dim_raw is already validated on the CPU (0 <= sm_dim_raw < ndims).
  // Adding a basic guard here for safety in case of issues.
  if (sm_dim_raw >= ndims || ndims == 0u || ndims > MAX_DIMS) {
    return; // Invalid configuration
  }
  let sm_dim = sm_dim_raw;

  // slice_id identifies which "1D vector" (along sm_dim) this workgroup is processing.
  let slice_id = global_id.x;
  let tid = local_id.x;

  // Calculate num_total_slices to guard against over-dispatch.
  // num_total_slices = product of shape[d] for d != sm_dim.
  var num_total_slices = 1u;
  if (params.shape_vecs[sm_dim / 4u][sm_dim % 4u] == 0u && ndims > 0u) { // Avoid division by zero if sm_dim size is 0
      num_total_slices = 0u; // No slices to process if softmax dimension is empty
  } else if ndims > 0u { // Calculate total elements and divide by size of softmax dimension
    var total_elements = 1u;
    for (var d_idx = 0u; d_idx < ndims; d_idx = d_idx + 1u) {
        total_elements = total_elements * params.shape_vecs[d_idx / 4u][d_idx % 4u];
    }
    if (params.shape_vecs[sm_dim / 4u][sm_dim % 4u] > 0u) {
        num_total_slices = total_elements / params.shape_vecs[sm_dim / 4u][sm_dim % 4u];
    } else { // if shape_vecs[sm_dim / 4u][sm_dim % 4u] is 0 but total_elements is also 0 (e.g. shape [0])
        num_total_slices = 1u; // Special case for 0-element tensor leading to 1 "empty" slice
        if (total_elements > 0u) { num_total_slices = 0u;} // Should not happen if shape_vecs[sm_dim / 4u][sm_dim % 4u] == 0
    }
  } else { // ndims == 0
    num_total_slices = 1u; // A 0D tensor can be thought of as one slice with one element
  }


  if (slice_id >= num_total_slices) {
    return;
  }

  let elements_along_sm_dim = params.shape_vecs[sm_dim / 4u][sm_dim % 4u];
  if (elements_along_sm_dim == 0u) {
      // If the dimension being softmaxed over is empty, there's nothing to compute for this slice.
      // Depending on desired behavior for empty slices (e.g. fill with NaN, 0, or 1/N),
      // one might write specific values or simply return.
      // For now, just return, output will remain uninitialized or as per buffer default.
      return;
  }


  // Calculate the N-D coordinate for the start of this slice
  // current_coord will have current_coord[sm_dim] = 0.
  var current_coord: array<u32, MAX_DIMS>;
  var temp_slice_id_for_coord_calc = slice_id;

  for (var d_loop = 0u; d_loop < ndims; d_loop = d_loop + 1u) {
    let d = ndims - 1u - d_loop; // d iterates from ndims-1 down to 0

    if (d == sm_dim) {
      current_coord[d] = 0u; // Placeholder for iteration along sm_dim
    } else {
      let dim_size = params.shape_vecs[d / 4u][d % 4u];
      if (dim_size == 0u) { // Should not happen if num_total_slices > 0 and slice_id is valid
          // This case implies an inconsistent state or a 0-sized dimension not part of sm_dim,
          // which would mean num_total_slices should be 0.
          // If somehow reached, returning might be safest.
          return;
      }
      current_coord[d] = temp_slice_id_for_coord_calc % dim_size;
      temp_slice_id_for_coord_calc = temp_slice_id_for_coord_calc / dim_size;
    }
  }

  // Calculate the initial flat offset for this slice (where element at sm_dim index 0 would be)
  var initial_flat_offset_for_slice = 0u;
  for (var d_calc = 0u; d_calc < ndims; d_calc = d_calc + 1u) {
    initial_flat_offset_for_slice = initial_flat_offset_for_slice + current_coord[d_calc] * params.strides_vecs[d_calc / 4u][d_calc % 4u];
  }
  let stride_along_sm_dim = params.strides_vecs[sm_dim / 4u][sm_dim % 4u];

  // --- Stage 1: Find the maximum value in the current slice ---
  var thread_max_val: f32 = -3.402823466e+38; // Initialize with smallest f32
  for (var i: u32 = tid; i < elements_along_sm_dim; i = i + workgroup_size_x) {
    let element_flat_idx = initial_flat_offset_for_slice + i * stride_along_sm_dim;
    thread_max_val = max(thread_max_val, input_tensor[element_flat_idx]);
  }

  s_reduce_buffer[tid] = thread_max_val;
  workgroupBarrier();

  for (var s: u32 = workgroup_size_x / 2u; s > 0u; s = s / 2u) {
    if (tid < s) {
      s_reduce_buffer[tid] = max(s_reduce_buffer[tid], s_reduce_buffer[tid + s]);
    }
    workgroupBarrier();
  }
  let slice_max_val: f32 = s_reduce_buffer[0];
  workgroupBarrier();


  // --- Stage 2: Calculate the sum of exponentials: sum(exp(x_i - slice_max_val)) ---
  var thread_sum_exp: f32 = 0.0;
  for (var i: u32 = tid; i < elements_along_sm_dim; i = i + workgroup_size_x) {
    let element_flat_idx = initial_flat_offset_for_slice + i * stride_along_sm_dim;
    thread_sum_exp = thread_sum_exp + exp(input_tensor[element_flat_idx] - slice_max_val);
  }

  s_reduce_buffer[tid] = thread_sum_exp;
  workgroupBarrier();

  for (var s: u32 = workgroup_size_x / 2u; s > 0u; s = s / 2u) {
    if (tid < s) {
      s_reduce_buffer[tid] = s_reduce_buffer[tid] + s_reduce_buffer[tid + s];
    }
    workgroupBarrier();
  }
  let slice_sum_exp: f32 = s_reduce_buffer[0];
  workgroupBarrier();


  // --- Stage 3: Calculate softmax values and store them in the output tensor ---
  var inv_slice_sum_exp: f32 = 0.0;
  if (slice_sum_exp <= 1e-9) { // Using a small epsilon
    // If sum_exp is zero or very small, implies all exp(inputs - max) were zero.
    // PyTorch behavior for softmax([value], dim=0) where value results in sum_exp=0 is typically NaN for the output if inputs were NaN,
    // or if all inputs were -inf, then exp results are 0, sum_exp is 0. Output is often 0 or 1/N.
    // Let's default to 0.0 for stability if sum_exp is tiny/zero.
    // If elements_along_sm_dim > 0, and sum_exp is zero, this means all exp(...) were zero.
    // Uniform distribution 1.0 / f32(elements_along_sm_dim) could be an alternative if sum_exp is strictly 0.
    // However, if inputs led to this, 0.0 is often safer to avoid NaN propagation from 0/0.
    inv_slice_sum_exp = 0.0;
  } else {
    inv_slice_sum_exp = 1.0 / slice_sum_exp;
  }

  for (var i: u32 = tid; i < elements_along_sm_dim; i = i + workgroup_size_x) {
    let element_flat_idx = initial_flat_offset_for_slice + i * stride_along_sm_dim;
    let original_val: f32 = input_tensor[element_flat_idx];
    let exp_val: f32 = exp(original_val - slice_max_val);
    output_tensor[element_flat_idx] = exp_val * inv_slice_sum_exp;
  }
}