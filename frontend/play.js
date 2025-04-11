import { 
  initializeWebGPU, 
  transferToDevice, 
  viewBuffer,
  pushErrorScopes,
  popErrorScopes
} from "./common.js";

const device = await initializeWebGPU();

// allocate a massive buffer
// Create 1MB array of ones (1MB = 1024 * 1024 bytes)
// Since Float32 is 4 bytes, we need 1024 * 1024 / 4 = 262144 elements
const size = 252144 * 4;
const data = new Float32Array(size).fill(1);
const errorScopes = ['validation', 'out-of-memory', 'internal'];
pushErrorScopes(device, errorScopes);
const buffer = await transferToDevice(
  device,
  data,
  size * 4,
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
)

const result = await viewBuffer(
  device,
  buffer,
  size * 4
)
console.log(result);