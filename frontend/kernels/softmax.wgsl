struct Params {
  // num_cols is the size of the dimension over which softmax is applied (e.g., width of the matrix)
  num_cols: u32,
  // num_rows is implicitly handled by the dispatch grid (global_id.x)
};

// Pipeline-overridable constant for workgroup size in the X dimension.
// This value is set during pipeline creation in JavaScript/Rust.
// IMPORTANT: The reduction logic assumes this is a power of two (e.g., 64, 128, 256).
@id(0) override workgroup_size_x: u32 = 256u;

// Shared memory for performing reductions (max and sum) within a workgroup.
// Its size is determined by the actual workgroup_size_x used.
var<workgroup> s_reduce_buffer: array<f32, workgroup_size_x>;

// Define the compute shader entry point and workgroup size.
// The workgroup size (X, Y, Z) is (workgroup_size_x, 1, 1).
@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  // row_idx determines which row of the tensor this workgroup is processing.
  // Assumes dispatch is (num_rows, 1, 1).
  let row_idx: u32 = global_id.x;
  // tid is the thread's index within the workgroup (0 to workgroup_size_x - 1).
  let tid: u32 = local_id.x;

  let row_size: u32 = params.num_cols; // Number of elements in the current row.
  let row_offset: u32 = row_idx * row_size; // Starting index of the current row in the flat tensor.

  // --- Stage 1: Find the maximum value in the current row ---
  // This is the first step for numerically stable softmax.
  var thread_max_val: f32 = -3.402823466e+38; // Initialize with smallest f32

  // Each thread iterates over a part of the row, calculating its local maximum.
  // The loop `i = i + workgroup_size_x` ensures all elements are covered
  // if row_size > workgroup_size_x.
  // If tid >= row_size, this thread doesn't process any elements for this row,
  // and thread_max_val remains -infinity, which is correctly handled by max().
  for (var i: u32 = tid; i < row_size; i = i + workgroup_size_x) {
    thread_max_val = max(thread_max_val, input_tensor[row_offset + i]);
  }

  // Store the thread's local maximum into shared memory.
  s_reduce_buffer[tid] = thread_max_val;
  // Synchronize all threads in the workgroup to ensure all writes to s_reduce_buffer are complete.
  workgroupBarrier();

  // Perform parallel reduction in shared memory to find the overall maximum for the row.
  // This loop halves the number of active threads in each step.
  // Assumes workgroup_size_x is a power of two.
  for (var s: u32 = workgroup_size_x / 2u; s > 0u; s = s / 2u) {
    if (tid < s) {
      s_reduce_buffer[tid] = max(s_reduce_buffer[tid], s_reduce_buffer[tid + s]);
    }
    workgroupBarrier(); // Synchronize after each reduction step.
  }
  // The final maximum value for the row is now in s_reduce_buffer[0].
  let row_max_val: f32 = s_reduce_buffer[0];
  // Ensure all threads read the correct row_max_val before proceeding.
  workgroupBarrier();


  // --- Stage 2: Calculate the sum of exponentials: sum(exp(x_i - row_max_val)) ---
  var thread_sum_exp: f32 = 0.0; // Initialize sum for this thread.

  // Each thread iterates over its assigned elements again.
  // It calculates exp(element - row_max_val) and accumulates the sum.
  // If tid >= row_size, this loop is skipped, and thread_sum_exp remains 0.0, which is correct for summation.
  for (var i: u32 = tid; i < row_size; i = i + workgroup_size_x) {
    thread_sum_exp = thread_sum_exp + exp(input_tensor[row_offset + i] - row_max_val);
  }

  // Store the thread's partial sum into shared memory.
  s_reduce_buffer[tid] = thread_sum_exp;
  workgroupBarrier(); // Ensure all writes are complete.

  // Perform parallel reduction for the sum.
  for (var s: u32 = workgroup_size_x / 2u; s > 0u; s = s / 2u) {
    if (tid < s) {
      s_reduce_buffer[tid] = s_reduce_buffer[tid] + s_reduce_buffer[tid + s];
    }
    workgroupBarrier(); // Synchronize.
  }
  // The final sum of exponentials for the row is now in s_reduce_buffer[0].
  let row_sum_exp: f32 = s_reduce_buffer[0];
  // Ensure all threads read the correct row_sum_exp.
  workgroupBarrier();


  // --- Stage 3: Calculate softmax values and store them in the output tensor ---
  // Softmax_i = exp(x_i - row_max_val) / row_sum_exp.

  // Handle potential division by zero or very small sum_exp.
  // If row_sum_exp is effectively zero (e.g., all inputs were -infinity, or exp underflowed),
  // the output for that element will be 0.
  // A small epsilon (1e-9f) is used as a threshold.
  var inv_row_sum_exp: f32 = 0;
  if (row_sum_exp <= 1e-9) {
    inv_row_sum_exp = 0.0;
    // Alternative for sum_exp == 0: distribute probability, e.g., 1.0 / f32(row_size).
    // However, if exp results are all zero, 0 is often the desired output.
  } else {
    inv_row_sum_exp = 1.0 / row_sum_exp;
  }

  // Each thread calculates and writes the softmax values for its assigned elements.
  // The loop `for (var i: u32 = tid; i < row_size; ...)` ensures that only elements
  // belonging to the current row are processed and written.
  for (var i: u32 = tid; i < row_size; i = i + workgroup_size_x) {
    let original_val: f32 = input_tensor[row_offset + i];
    // Calculate exp(original_val - row_max_val) for the numerator.
    let exp_val: f32 = exp(original_val - row_max_val);
    output_tensor[row_offset + i] = exp_val * inv_row_sum_exp;
  }
}