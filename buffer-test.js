async function bufferTest() {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get GPU adapter.");
  }

  const device = await adapter.requestDevice();
  const vectorLength = 4;
  const byteLength = vectorLength * 4;

  // Create buffer for the X input vector
  const xBuffer = device.createBuffer({
    size: byteLength, // 4 bytes per f32
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  });

  device.queue.writeBuffer(xBuffer, 0, new Float32Array([1, 2, 3, 4]));
  device.queue.writeBuffer(xBuffer, 0, new Int32Array([65, 32, 16, 19]));

  const stagingBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    mappedAtCreation: false
  });

  const commandEncoder = device.createCommandEncoder();

  // this basically sets up an instruction for our GPU to copy over data from our input buffer
  // to our mappable buffer that we can read from
  commandEncoder.copyBufferToBuffer(
    xBuffer, // source buffer
    0,      // source offset
    stagingBuffer, // destination buffer
    0,      // destination offset
    byteLength // size to copy
  );

  const commands = commandEncoder.finish();
  device.queue.submit([commands]);
  
  // 5. Map the staging buffer for reading
  await stagingBuffer.mapAsync(GPUMapMode.READ);

  const mappedContent = stagingBuffer.getMappedRange()
  const copyArrayBuffer = new Int32Array(mappedContent);
  console.log(Array.from(copyArrayBuffer)); // Should output [1, 2, 3, 4]
}

// Run the example when the page loads
document.addEventListener('DOMContentLoaded', bufferTest);