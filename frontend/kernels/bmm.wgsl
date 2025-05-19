struct Dimensions {
  B: u32, // Batch size
  M: u32, // Rows of A and Result
  K: u32, // Columns of A and Rows of B
  N: u32  // Columns of B and Result
}

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

const BLOCKSIZE_M: u32 = 16; // Workgroup size for M dimension
const BLOCKSIZE_N: u32 = 16; // Workgroup size for N dimension
// Batch dimension will be handled by global_id.z

const TILE_M: u32 = 4;  // Tile size in M dimension per thread
const TILE_N: u32 = 4;  // Tile size in N dimension per thread

@compute @workgroup_size(BLOCKSIZE_N, BLOCKSIZE_M, 1) // workgroup_size(x, y, z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;

    // Prevent out-of-bounds access for batches
    if (batch_idx >= dimensions.B) {
        return;
    }

    let tile_row_start = global_id.y * TILE_M; // Starting row in the M dimension for this thread's tile
    let tile_col_start = global_id.x * TILE_N; // Starting col in the N dimension for this thread's tile

    var sums: array<array<f32, TILE_N>, TILE_M>;
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            sums[i][j] = 0.0;
        }
    }

    let a_batch_offset = batch_idx * dimensions.M * dimensions.K;
    let b_batch_offset = batch_idx * dimensions.K * dimensions.N;
    let result_batch_offset = batch_idx * dimensions.M * dimensions.N;

    // Iterate over the K dimension
    for (var k_outer = 0u; k_outer < dimensions.K; k_outer++) {
        for (var tm = 0u; tm < TILE_M; tm++) {
            let current_m = tile_row_start + tm;
            if (current_m < dimensions.M) {
                let a_val = a[a_batch_offset + current_m * dimensions.K + k_outer];
                for (var tn = 0u; tn < TILE_N; tn++) {
                    let current_n = tile_col_start + tn;
                    if (current_n < dimensions.N) {
                        let b_val = b[b_batch_offset + k_outer * dimensions.N + current_n];
                        sums[tm][tn] += a_val * b_val;
                    }
                }
            }
        }
    }

    // Write results
    for (var tm = 0u; tm < TILE_M; tm++) {
        let output_row = tile_row_start + tm;
        if (output_row < dimensions.M) {
            for (var tn = 0u; tn < TILE_N; tn++) {
                let output_col = tile_col_start + tn;
                if (output_col < dimensions.N) {
                    result[result_batch_offset + output_row * dimensions.N + output_col] = sums[tm][tn];
                }
            }
        }
    }
}