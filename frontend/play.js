import { initializeWebGPU } from "./common.js";
import { Kernel, KernelBuilder } from "./kernel_builder.js";
const device = await initializeWebGPU();
// create a matmul kernel

const matmulKernel = new Kernel({
  name: "matmul",
  shaderPath: "kernels/matmul.wgsl",
  entryPoint: "main",
  workGroupSize: {x: 16, y: 16, z: 1},
  bindingConfig: [
    {
      name: "dimensions",
      isPersistent: false,
      isOutput: false,
      type: "uniform",
    },
    {
      name: "input",
      isPersistent: false,
      isOutput: false,
      type: "read-only-storage"
    },
    {
      name: "weight",
      isPersistent: true,
      isOutput: false,
      type: "read-only-storage"
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
const dimensions = new Uint32Array([1,2048, 50000]);
const [M, K, N ] = dimensions;
await builder.loadKernel(matmulKernel);
await builder.addTensor("input", new Float32Array(M * K).fill(1));
await builder.addTensor("weight", new Float32Array(K * N).fill(1));
await builder.addTensor("result", new Float32Array(M * N).fill(1));
await builder.addTensor("dimensions", dimensions);

try {
  await builder.executeKernel(matmulKernel, {width: M, height: N, depth: 1})
} catch (error) {
  console.error(error);
}
console.log(await builder.concludeSession());