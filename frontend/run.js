import { viewBuffer  } from "./common.js";
async function requestMatrix() {
  const registration = await (await fetch("/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      capabilities: {
        maxBufferSize: 0 // Maximum buffer size in bytes
      }
    })
  })).json();
  console.log(registration);
  return registration;
}

async function initializeWebGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  console.log(adapter);
  const device = await adapter.requestDevice();

  // Add a device lost handler to catch severe errors
  device.lost.then((info) => {
    console.error(`WebGPU device was lost: ${info.message}`);
    console.error(`Reason: ${info.reason}`);
  });

  return device;
}

async function loadShader(device) {
  const shaderResponse = await fetch("kernels/matmul.wgsl");
  const shaderCode = await shaderResponse.text();
  return device.createShaderModule({
    label: "matmul",
    code: shaderCode
  });
}

async function setupBuffersAndBindGroups(device, registration) {
  const M = registration.matrix.shape[0];
  const K = registration.matrix.shape[1];
  const N = registration.matrix.shape[0];

  const matrixA = new Float32Array(registration.matrix.elements);
  const matrixB = new Float32Array(registration.matrix.elements);
  const dimensions = new Uint32Array([M, K, N]);


  // Create buffers
  const dimensionsBuffer = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
  });

  const matrixABuffer = device.createBuffer({
    size: M * K * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  });

  const matrixBBuffer = device.createBuffer({
    size: K * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  const matrixCBuffer = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const readbackBuffer = device.createBuffer({
    size: M * N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
  });


  const readbackA = device.createBuffer({
    size: M * K * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
  });

  // Write data to buffers
  device.queue.writeBuffer(matrixABuffer, 0, matrixA);

  device.queue.writeBuffer(matrixBBuffer, 0, matrixB);
  device.queue.writeBuffer(dimensionsBuffer, 0, dimensions);

  // Create bind group layout and bind group
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      }
    ]
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: dimensionsBuffer }
      },
      {
        binding: 1,
        resource: { buffer: matrixABuffer }
      },
      {
        binding: 2,
        resource: { buffer: matrixBBuffer }
      },
      {
        binding: 3,
        resource: { buffer: matrixCBuffer }
      }
    ]
  });

  return {
    dimensions: { M, K, N },
    buffers: { readbackA, matrixABuffer, matrixCBuffer, readbackBuffer },
    bindGroupLayout,
    bindGroup
  };
}

async function computeMatrixMultiplication() {
  const device = await initializeWebGPU();
  const registration = await requestMatrix();
  const shaderModule = await loadShader(device);

  const { dimensions, buffers, bindGroupLayout, bindGroup } = 
    await setupBuffersAndBindGroups(device, registration);
  
  const { M, N, K } = dimensions;
  const { readbackA, matrixABuffer, matrixCBuffer, readbackBuffer } = buffers;


  const result = await viewBuffer(device, matrixABuffer, M * K * 4);
  console.log(result);

  // Create pipeline
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });

  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "main"
    }
  });

  // Begin error scope and create command encoder
  device.pushErrorScope('validation');
  const commandEncoder = device.createCommandEncoder();

  // Setup compute pass
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, bindGroup);

  const WORKGROUP_SIZE = 16;
  computePass.dispatchWorkgroups(
    Math.ceil(M / WORKGROUP_SIZE),
    Math.ceil(N / WORKGROUP_SIZE),
  );
  computePass.end();

  // Copy results
  commandEncoder.copyBufferToBuffer(
    matrixCBuffer, 0,
    readbackBuffer, 0,
    M * N * 4
  );

  const commands = commandEncoder.finish();

  // Submit commands with error checking
  try {
    device.queue.submit([commands]);
    
    const validationError = await device.popErrorScope();
    if (validationError) {
      console.error(`WebGPU validation error: ${validationError.message}`);
      throw new Error(`WebGPU validation error: ${validationError.message}`);
    }
  } catch (e) {
    console.error("Error submitting WebGPU commands:", e);
    throw e;
  }

  const results = await viewBuffer(device, matrixCBuffer, M * N * 4);
  console.log(results);
}

// Execute the computation
computeMatrixMultiplication().catch(console.error);