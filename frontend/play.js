// import { 
//   initializeWebGPU, 
//   transferToDevice, 
//   viewBuffer,
//   pushErrorScopes,
//   popErrorScopes
// } from "./common.js";

// const device = await initializeWebGPU();

// // allocate a massive buffer
// // Create 1MB array of ones (1MB = 1024 * 1024 bytes)
// // Since Float32 is 4 bytes, we need 1024 * 1024 / 4 = 262144 elements
// const size = 252144 * 4;
// const data = new Float32Array(size).fill(1);
// const errorScopes = ['validation', 'out-of-memory', 'internal'];
// pushErrorScopes(device, errorScopes);
// const buffer = await transferToDevice(
//   device,
//   data,
//   size * 4,
//   GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
// )

// const result = await viewBuffer(
//   device,
//   buffer,
//   size * 4
// )
// console.log(result);

import { initializeWebGPU } from "./common.js";
import { Kernel, KernelBuilder } from "./kernel_builder.js";
const device = await initializeWebGPU();
// create a matmul kernel

const matmulKernel = new Kernel({
  name: "matmul",
  shaderPath: "kernels/matmul.wgsl",
  entryPoint: "main",
  workGroupSize: {x: 16, y: 16, z: 1},
  inputConfig: [
    {
      name: "input",
      isPersistent: false,
      isOutput: false,
      type: "storage"
    },
    {
      name: "weight",
      isPersistent: true,
      isOutput: false,
      type: "storage"
    },
    {
      name: "result",
      isPersistent: false,
      isOutput: true,
      type: "storage"
    }
  ],
});

const builder = new KernelBuilder(device);
builder.beginSession();
console.log(builder);
const result = await builder.loadKernel(matmulKernel);
await builder.addTensor("input", new Float32Array(64).fill(1));
await builder.addTensor("weight", new Float32Array(64).fill(1));
await builder.addTensor("result", new Float32Array(64).fill(1));
try {
  builder.executeKernel(matmulKernel, {width: 256, height: 256, depth: 1})
} catch (error) {
  console.error(error);
}
console.log(await builder.concludeSession());