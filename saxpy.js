// This function initializes and runs the WebGPU SAXPY computation
async function runSaxpy(a, xArray, yArray) {
  /*
  The navigator object provides information about user's browswer and system
  we can use navigator object to identify browser capabilities, adapt to different envs
  */
  if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser.");
  }

  // Request a GPU adapter
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
      throw new Error("Failed to get GPU adapter.");
  }

  // Request a device from the adapter
  const device = await adapter.requestDevice();

  // Fetch and load the WGSL shader from a separate file
  const shaderResponse = await fetch('saxpy.wgsl');
  if (!shaderResponse.ok) {
      throw new Error(`Failed to fetch shader: ${shaderResponse.statusText}`);
  }
  // const shaderCode = await shaderResponse.text();
  const shaderCode = await shaderResponse.text();


  // Create a shader module from our shader code
  const shaderModule = device.createShaderModule({
      code: shaderCode
  });

  const vectorLength = xArray.length;

  // Create buffer for scalar 'a'
  const uniformBuffer = device.createBuffer({
      size: vectorLength * 4, // 4 bytes for f32 + 12 bytes padding (vec3<f32>)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  // Create buffer for the X input vector
  const xBuffer = device.createBuffer({
      size: vectorLength * 4, // 4 bytes per f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  // Create buffer for the Y input vector
  const yBuffer = device.createBuffer({
      size: vectorLength * 4, // 4 bytes per f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  // Create buffer for the result vector
  const resultBuffer = device.createBuffer({
      size: vectorLength * 4, // 4 bytes per f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Create a buffer to map the results back to CPU
  const readbackBuffer = device.createBuffer({
      size: vectorLength * 4, // 4 bytes per f32
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
  });

  // Write the scalar 'a' value to the uniform buffer
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([a]));
  // Write the X and Y vectors to their respective buffers
  device.queue.writeBuffer(xBuffer, 0, new Float32Array(xArray));
  device.queue.writeBuffer(yBuffer, 0, new Float32Array(yArray));
  

  // Create a bind group layout that describes the binding locations
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

  // Create the bind group based on the layout
  const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
          {
              binding: 0,
              resource: { buffer: uniformBuffer }
          },
          {
              binding: 1,
              resource: { buffer: xBuffer }
          },
          {
              binding: 2,
              resource: { buffer: yBuffer }
          },
          {
            binding: 3,
            resource: { buffer: resultBuffer }
        }
      ]
  });

  // Create a pipeline layout
  const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
  });

  // Create the compute pipeline
  const computePipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
          module: shaderModule,
          entryPoint: "saxpyKernel"
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
  const workgroupSize = 256;
  const workgroupCount = Math.ceil(vectorLength / workgroupSize);
  computePass.dispatchWorkgroups(workgroupCount, 1, 1);
  computePass.end();
  
  // Copy results
  commandEncoder.copyBufferToBuffer(
    resultBuffer, 0,
    readbackBuffer, 0,
    vectorLength * 4
  );
  
  // Finish commands
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
  return result;
}

// Export the runSaxpy function
export { runSaxpy };