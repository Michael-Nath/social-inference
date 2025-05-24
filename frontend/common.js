export async function initializeWebGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    }
  });

  // Add a device lost handler to catch severe errors
  device.lost.then((info) => {
    console.error(`WebGPU device was lost: ${info.message}`);
    console.error(`Reason: ${info.reason}`);
  });

  return device;
};

export async function viewBuffer(device, buffer, size) {
  // Create a mappable buffer to read back data
  const readbackBuffer = device.createBuffer({
    size: size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
  });

  // Create and submit commands to copy from source buffer to readback buffer
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readbackBuffer, 0, size);
  device.queue.submit([commandEncoder.finish()]);

  // Map the buffer and create array from contents
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = readbackBuffer.getMappedRange();
  const array = new Float32Array(arrayBuffer);
  const result = Array.from(array);
  readbackBuffer.unmap();

  return result;
}

export async function transferToDevice(device, data, size, usage) {
  // Create device buffer with appropriate size and usage flags
  const errorScopes = ['validation', 'out-of-memory', 'internal'];
  const deviceBuffer = device.createBuffer({
    size: size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: false
  });

  await popErrorScopes(device, errorScopes);
  // Create command encoder and encode write command
  device.queue.writeBuffer(deviceBuffer, 0, data);
  await device.queue.onSubmittedWorkDone();

  return deviceBuffer;
}

export function pushErrorScopes(device, scopes) {
  scopes.forEach(scope => device.pushErrorScope(scope));
}

export async function popErrorScopes(device, scopes) {
  const errors = [];
  // Pop in reverse order (since error scopes are stack-like)
  for (let i = scopes.length - 1; i >= 0; i--) {
const error = await device.popErrorScope();
if (error) {
    errors.push({ scope: scopes[i], message: error.message });
    console.error(`${scopes[i]} error: ${error.message}`);
}
  }
  return errors;
}

export const ALL_SCOPES = ['validation', 'out-of-memory', 'internal'];