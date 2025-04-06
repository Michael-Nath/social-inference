const adapter = await navigator.gpu.requestAdapter();
console.log(adapter);
const device = await adapter.requestDevice();
const shaderResponse = await fetch("kernels/matmul.wgsl");
const shaderCode = await shaderResponse.text();
const shaderModule = device.createShaderModule({
  label: "matmul",
  code: shaderCode
});

// matrixA <- POST /register
// matrixB <- GET /work
// matrixC -> POST /work


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

const M = registration.operation.matrix.shape[0];
const K = registration.operation.matrix.shape[0];
const N = registration.operation.matrix.shape[0];


const matrixA = new Float32Array(registration.operation.matrix.elements);
const matrixB = new Float32Array(registration.operation.matrix.elements);
const matrixC = new Float32Array(M * N);
const dimensions = new Uint32Array([M, K, N]);

console.log(matrixA);

// creating tensors

const dimensionsBuffer = device.createBuffer({
  size: 12, // 3 floats,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  mappedAtCreation: false
});


const matrixABuffer = device.createBuffer({
  size: M * K * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});

const matrixBBuffer = device.createBuffer({
  size: K * N * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});

const matrixCBuffer = device.createBuffer({
  size: M * N * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
})

// Create a buffer to map the results back to CPU
const readbackBuffer = device.createBuffer({
  size: M * N * 4, // 4 bytes per f32
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  mappedAtCreation: false
});


device.queue.writeBuffer(matrixABuffer, 0, matrixA);
device.queue.writeBuffer(matrixBBuffer, 0, matrixB);
device.queue.writeBuffer(dimensionsBuffer, 0, dimensions);


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


const pipelineLayout = device.createPipelineLayout({
  bindGroupLayouts: [bindGroupLayout]
})

// Create the compute pipeline
const computePipeline = device.createComputePipeline({
  layout: pipelineLayout,
  compute: {
      module: shaderModule,
      entryPoint: "main"
  }
});


// Add a device lost handler to catch severe errors
device.lost.then((info) => {
  console.error(`WebGPU device was lost: ${info.message}`);
  console.error(`Reason: ${info.reason}`);
  // You might want to attempt recovery here
});

// Begin an error scope before your operations
device.pushErrorScope('validation');

// Create command encoder, compute pass etc...
const commandEncoder = device.createCommandEncoder();

// Setup compute pass
const computePass = commandEncoder.beginComputePass();
computePass.setPipeline(computePipeline);
computePass.setBindGroup(0, bindGroup);

computePass.dispatchWorkgroups(M * N);
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
  
  // Pop the error scope and check for validation errors
  const validationError = await device.popErrorScope();
  if (validationError) {
    console.error(`WebGPU validation error: ${validationError.message}`);
    throw new Error(`WebGPU validation error: ${validationError.message}`);
  }
  
  // If you need to check for other error types, you can use nested scopes
  // device.pushErrorScope('internal');
  // device.pushErrorScope('out-of-memory');
  // and pop them in reverse order
} catch (e) {
  console.error("Error submitting WebGPU commands:", e);
  throw e;
}

// Read results as before
await readbackBuffer.mapAsync(GPUMapMode.READ);
const resultArrayBuffer = readbackBuffer.getMappedRange();
const resultArray = new Float32Array(resultArrayBuffer);
const result = Array.from(resultArray);

readbackBuffer.unmap();
console.log(result);