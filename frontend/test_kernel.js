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

// Add cpuMatmul function before testMatmulKernelExecution
function cpuMatmul(A_cpu, B_cpu, M_val, K_val, N_val) {
  const C_cpu = new Float32Array(M_val * N_val);
  for (let r = 0; r < M_val; ++r) {
    for (let c = 0; c < N_val; ++c) {
      let sum = 0;
      for (let k_idx = 0; k_idx < K_val; ++k_idx) {
        sum += A_cpu[r * K_val + k_idx] * B_cpu[k_idx * N_val + c];
      }
      C_cpu[r * N_val + c] = sum;
    }
  }
  return C_cpu;
}

async function testMatmulKernelExecution() {
  console.log("Starting testMatmulKernelExecution (1024x1024x1024)...");
  let device;
  try {
    device = await initializeWebGPU();
    if (!device) {
      console.error("WebGPU device initialization failed for matmul.");
      return;
    }

    // create the query set to allocate timestamp query storage on GPU
    const querySet = device.createQuerySet({
      type: "timestamp",
      count: 2
    });

    const timestampBufferSize = 2 * 8;
    const resolveBuffer = device.createBuffer({
      size: timestampBufferSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    });


    const config = matmulKernelConfig;
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
        workgroupSize: config.workGroupSize, // { x: 16, y: 16, z: 1 }
        inputBindings: inputBindings,
        outputBindings: outputBindings
    });

    const shaderModule = device.createShaderModule({ code: kernel.shader });

    const allBindings = [...kernel.inputBindings, ...kernel.outputBindings].sort((a,b) => a.index - b.index);

    const bindGroupLayout = device.createBindGroupLayout({
      entries: allBindings.map(binding => ({
        binding: binding.index,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding.type },
      })),
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    
    device.pushErrorScope('validation');
    const computePipeline = await device.createComputePipelineAsync({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: kernel.entryPoint,
      },
    });
    
    const pipelineError = await device.popErrorScope();
    if (pipelineError) {
      console.error(`Matmul Pipeline creation error: ${pipelineError.message}`);
      return;
    }
    console.log("Matmul Compute pipeline created successfully.");

    // Define matrix dimensions and data for 1024x1024x1024
    const M = 1024;
    const K = 1024;
    const N = 1024;
    console.log(`Matrix dimensions: M=${M}, K=${K}, N=${N}`);

    const dimensionsData = new Uint32Array([M, K, N]);
    
    console.log("Generating input matrices data (this might take a moment)...");
    const inputAData = new Float32Array(M * K);
    for (let i = 0; i < M * K; i++) inputAData[i] = (i % 10) + 0.1; // Simple pattern
    
    const inputBData = new Float32Array(K * N);
    for (let i = 0; i < K * N; i++) inputBData[i] = (i % 8) + 0.2; // Simple pattern
    console.log("Input matrices data generated.");

    const outputBufferSize = M * N * Float32Array.BYTES_PER_ELEMENT;

    // Create Buffers
    const dimensionsBuffer = device.createBuffer({
        size: dimensionsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(dimensionsBuffer.getMappedRange()).set(dimensionsData);
    dimensionsBuffer.unmap();

    const inputABuffer = device.createBuffer({
        size: inputAData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputABuffer, 0, inputAData);


    const inputBBuffer = device.createBuffer({
        size: inputBData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputBBuffer, 0, inputBData);


    const outputBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const stagingBuffer = device.createBuffer({
        size: outputBufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Bind Group
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0), // Use getBindGroupLayout(0) for clarity
        entries: [
          { binding: 0, resource: { buffer: dimensionsBuffer } },
          { binding: 1, resource: { buffer: inputABuffer } },
          { binding: 2, resource: { buffer: inputBBuffer } },
          { binding: 3, resource: { buffer: outputBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass({
      timestampWrites: {
        querySet: querySet,
        beginningOfPassWriteIndex: 0, // write start timestamp to index 0
        endOfPassWriteIndex: 1, // write end timestamp to index 1
      }
    });
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup);

    const TILE_M_SHADER = 4; 
    const TILE_N_SHADER = 4; 
    
    const numTileThreadsN = Math.ceil(N / TILE_N_SHADER); // 1024 / 4 = 256
    const numTileThreadsM = Math.ceil(M / TILE_M_SHADER); // 1024 / 4 = 256

    const dispatchX = Math.ceil(numTileThreadsN / config.workGroupSize.x); // 256 / 16 = 16
    const dispatchY = Math.ceil(numTileThreadsM / config.workGroupSize.y); // 256 / 16 = 16
    
    console.log(`Dispatching workgroups: X=${dispatchX}, Y=${dispatchY}, Z=1`);
    computePass.dispatchWorkgroups(dispatchX, dispatchY, 1); 
    computePass.end();

    const timestampStagingSize = resolveBuffer.size;

    const timestampStaging = device.createBuffer({
      size: timestampStagingSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });


    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBufferSize);
    
    console.time("GPU Matmul Execution");

    commandEncoder.resolveQuerySet(
      querySet,
      0, // index of first query to resolve (our start tiemstamp),
      2, // number of queries to resolve (start and end),
      resolveBuffer, // buffer to resolve the results into
      0 // offset in bytes into the destination buffer to write results
    );

    commandEncoder.copyBufferToBuffer(
      resolveBuffer, // Source buffer (results from resolveQuerySet)
      0,             // Source offset
      timestampStaging, // Destination buffer (the mappable one)
      0,             // Destination offset
      timestampStagingSize // Size to copy (entire buffer)
  );

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone(); // Wait for GPU to finish
    console.timeEnd("GPU Matmul Execution");

    console.log("Mapping GPU result...");
    await stagingBuffer.mapAsync(GPUMapMode.READ, 0, outputBufferSize);
    const gpuOutputArray = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    console.log("GPU result retrieved.");

    // CPU Verification
    console.log("Performing CPU matrix multiplication for verification (this will take time)...");
    console.time("CPU Matmul Calculation");
    const cpuResultArray = cpuMatmul(inputAData, inputBData, M, K, N);
    console.timeEnd("CPU Matmul Calculation");

    let mismatches = 0;
    const numCheckElements = 10; // Check a few random elements
    const sampleIndices = [];
    for(let i=0; i<numCheckElements; ++i) {
        sampleIndices.push(Math.floor(Math.random() * (M*N)));
    }
    sampleIndices.push(0); // check first
    sampleIndices.push(M*N-1); // check last

    console.log(`Comparing ${sampleIndices.length} elements between GPU and CPU results...`);
    for (const idx of sampleIndices) {
        if (idx >= M*N) continue; // Should not happen with current logic but as safeguard
        const gpuVal = gpuOutputArray[idx];
        const cpuVal = cpuResultArray[idx];
        if (Math.abs(gpuVal - cpuVal) > 0.1) { // Increased tolerance for large calcs
            mismatches++;
            if (mismatches < 10) { // Log first few mismatches
                const r = Math.floor(idx / N);
                const c = idx % N;
                console.error(`Mismatch at index ${idx} ([${r},${c}]): GPU=${gpuVal.toFixed(2)}, CPU=${cpuVal.toFixed(2)}, Diff=${(gpuVal - cpuVal).toFixed(2)}`);
            }
        }
    }

    if (mismatches === 0) {
        console.log(`Matmul Output VERIFIED (checked ${sampleIndices.length} elements).`);
        console.log(`  GPU[0]: ${gpuOutputArray[0].toFixed(2)}, CPU[0]: ${cpuResultArray[0].toFixed(2)}`);
        console.log(`  GPU[last]: ${gpuOutputArray[M*N-1].toFixed(2)}, CPU[last]: ${cpuResultArray[M*N-1].toFixed(2)}`);
    } else {
        console.error(`Matmul Output FAILED verification. Found ${mismatches} mismatches (logged up to 10).`);
    }
    await timestampStaging.mapAsync(
      GPUMapMode.READ,
      0,
      timestampStagingSize
    );

    const mappedTimestampBuffer = timestampStaging.getMappedRange(0, timestampStagingSize);
    console.log("Got mapped range!")
    const timestamps = new BigUint64Array(mappedTimestampBuffer);
    const startTime = timestamps[0];
    const endTime = timestamps[1];
    console.log(`Raw timestamps: Start=${startTime}ns, End=${endTime}ns`);
    timestampStaging.unmap();

    const durationBigInt = endTime - startTime;
    const durationNanoseconds = Number(durationBigInt); // Convert to regular number if needed
    const durationMilliseconds = durationNanoseconds / 1_000_000;
    console.log(`GPU Timestamp Query Duration: ${durationNanoseconds} ns (${durationMilliseconds.toFixed(3)} ms)`);

  } catch (error) {
    console.error("Error during matmul kernel execution:", error);
    if (error.stack) console.error(error.stack);
  } finally {
    console.log("testMatmulKernelExecution finished.");
  }
}

console.log("testMatmulKernelExecution function is defined. Call it to run the test.");
// Example: testSoftmaxKernelExecution().catch(console.error); // User removed this
Example: testMatmulKernelExecution().catch(console.error);