import { GPUKernel } from "./kernel.js";
import { initializeWebGPU } from "./common.js";

/** @type {any} */
const matmulKernelConfig = {
  name: "matmul",
  shaderPath: "kernels/matmul.wgsl",
  entryPoint: "main",
  workGroupSize: { x: 16, y: 16, z: 1 },
  bindingConfig: [
      {
          name: "dimensions", // M, K, N
          isOutput: false,
          type: "uniform",
      },
      {
          name: "input", // Matrix A (M x K)
          isOutput: false,
          type: "read-only-storage"
      },
      {
          name: "weight", // Matrix B (K x N)
          isOutput: false,
          type: "read-only-storage"
      },
      {
          name: "result", // Matrix C (M x N)
          isOutput: true,
          type: "storage"
      }
  ],
};

/** @type {any} */
const softmaxKernelConfig = {
  name: "softmax",
  shaderPath: "kernels/softmax.wgsl",
  entryPoint: "main",
  workGroupSize: { x: 256, y: 1, z: 1 },
  bindingConfig: [
    {
      name: "input_tensor",
      isOutput: false,
      type: "read-only-storage",
    },
    {
      name: "output_tensor",
      isOutput: true,
      type: "storage",
    },
    {
      name: "params",
      isOutput: false,
      type: "uniform",
    },
  ],
};

async function testSoftmaxKernelExecution() {
  console.log("Starting testSoftmaxKernelExecution...");
  let device;
  try {
    device = await initializeWebGPU();
    if (!device) {
      console.error("WebGPU device initialization failed.");
      return;
    }

    const config = softmaxKernelConfig;
    const shaderCode = await (await fetch(config.shaderPath)).text();
    
    const inputBindings = [];
    const outputBindings = [];
    config.bindingConfig.forEach((b, idx) => {
        const GPUToolBinding = { name: b.name, type: b.type, index: idx };
        if (b.isOutput) {
            outputBindings.push(GPUToolBinding);
        } else {
            inputBindings.push(GPUToolBinding);
        }
    });

    const kernel = new GPUKernel({
        name: config.name,
        shader: shaderCode,
        entryPoint: config.entryPoint,
        workgroupSize: config.workGroupSize,
        inputBindings: inputBindings,
        outputBindings: outputBindings
    });

    // 1. Prepare Kernel Resources
    const shaderModule = device.createShaderModule({ code: kernel.shader });
    console.log(kernel.shader)

    const allBindings = [...kernel.inputBindings, ...kernel.outputBindings].sort((a,b) => a.index - b.index);

    const bindGroupLayout = device.createBindGroupLayout({
      entries: allBindings.map(binding => ({
        binding: binding.index,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding.type },
      })),
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    
    // Restore WebGPU specific error handling and the constants field
    device.pushErrorScope('validation');
    const computePipeline = await device.createComputePipelineAsync({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: kernel.entryPoint, // Use kernel.entryPoint
        // constants: { // Restore constants
        //   "workgroup_size_x": kernel.workgroupSize.x
        // }
      },
    });
    
    const pipelineError = await device.popErrorScope();
    if (pipelineError) {
      console.error(`Pipeline creation error: ${pipelineError.message}`);
      return; // Stop execution if pipeline creation failed
    }
    console.log("Compute pipeline created successfully.");

      const inputData = new Float32Array([
        1.0, 2.0, 3.0, 4.0,  // row 0
        5.0, 1.0, 2.0, 0.5,  // row 1
      ]);
      const numRows = 2;
      const numCols = 4;
      const inputBufferSize = inputData.byteLength;
      const paramsData = new Uint32Array([numCols]);
      const outputBufferSize = inputBufferSize;

      const inputBuffer = device.createBuffer({
        size: inputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(inputBuffer.getMappedRange()).set(inputData);
      inputBuffer.unmap();

      const paramsBuffer = device.createBuffer({
        size: paramsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint32Array(paramsBuffer.getMappedRange()).set(paramsData);
      paramsBuffer.unmap();

      const outputBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const stagingBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const commandEncoder = device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(numRows, 1, 1);
      computePass.end();
      commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBufferSize);
      device.queue.submit([commandEncoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ, 0, outputBufferSize);
      const outputArray = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      console.log("Softmax Output:", outputArray);
      stagingBuffer.unmap();

  } catch (error) {
    console.error("Error during softmax kernel execution:", error);
  } finally {
    console.log("testSoftmaxKernelExecution finished.");
  }
}

console.log("testSoftmaxKernelExecution function is defined. Call it to run the test.");

Example: testSoftmaxKernelExecution().catch(console.error);