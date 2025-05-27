struct Dimensions {
  B: u32,
  M: u32,
  K: u32,
  N: u32
}

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

const BLOCKSIZE: u32 = 16;
const TILE_M: u32 = 4;  // Tile size in M dimensiopn
const TILE_N: u32 = 4;  // Tile size in N dimension

// no more than one batch per workgroup
@compute @workgroup_size(BLOCKSIZE, BLOCKSIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let row = global_id.y * TILE_M;
    let col = global_id.x * TILE_N;
    // initialize the array with all 0s
    var sums: array<array<f32, TILE_N>, TILE_M>;
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            sums[i][j] = 0.0;
        }
    }

    // Compute the 2D tile
    for (var k = 0u; k < dimensions.K; k++) {
        // for each row
        for (var i = 0u; i < TILE_M; i++) {
            let a_offset = batch_idx * dimensions.M * dimensions.K;
            let a_element = a[a_offset + (row + i) * dimensions.K + k];
            // calculate the dot product
            for (var j = 0u; j < TILE_N; j++) {
                let b_offset = batch_idx * dimensions.K * dimensions.N;
                let b_element = b[b_offset + k * dimensions.N + (col + j)];
                sums[i][j] += a_element * b_element;
            }
        }
    }

    // Write results
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            let output_row = row + i;
            let output_col = col + j;
            if (batch_idx < dimensions.B && output_row < dimensions.M && output_col < dimensions.N) {
                let result_offset = batch_idx * dimensions.M * dimensions.N;
                result[result_offset + output_row * dimensions.N + output_col] = sums[i][j];
            }
        }
    }
}