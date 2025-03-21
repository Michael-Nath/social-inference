function isIOSDevice() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
}

// SAXPY FLOP Benchmark with 3 trials and GFLOP estimation
async function saxpyBenchmark() {
  try {
    // Set up GPU
    const adapter = await setupGPU();
    const device = await adapter.requestDevice();
    
    const MB = 1024 * 1024;
    // Fixed vector size: 256MB.
      const vectorSizeBytes = isIOSDevice() ? 128 * MB : 2048 * MB;
    const numElements = vectorSizeBytes / 4; // Each f32 is 4 bytes
    
    // Create input data arrays
    const inputArrayX = new Float32Array(numElements);
    const inputArrayY = new Float32Array(numElements);
    // Fill arrays with constant values (e.g. 1.0 and 2.0)
    for (let i = 0; i < numElements; i++) {
      inputArrayX[i] = 1.0;
      inputArrayY[i] = 2.0;
    }
    
    // Set up uniform data; using uniforms.x as the multiplier (e.g. 2.0)
    const uniformData = new Float32Array([2.0, 0.0, 0.0, 0.0]);
    
    // Create GPU buffers (reused across trials)
    const uniformBuffer = device.createBuffer({
      size: uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const inputBufferX = device.createBuffer({
      size: inputArrayX.byteLength,
      usage: storageUsage
    });
    const inputBufferY = device.createBuffer({
      size: inputArrayY.byteLength,
      usage: storageUsage
    });
    // Output buffer for the result
    const outputBuffer = device.createBuffer({
      size: inputArrayX.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Create the compute shader module using the provided SAXPY kernel
    const shaderModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<uniform> uniforms: vec4<f32>;
        @group(0) @binding(1) var<storage, read> inputVectorX: array<f32>;
        @group(0) @binding(2) var<storage, read> inputVectorY: array<f32>;
        @group(0) @binding(3) var<storage, read_write> outputVector: array<f32>;
        
        @compute @workgroup_size(256)
        fn saxpyKernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx < arrayLength(&inputVectorX)) {
                outputVector[idx] = uniforms.x * inputVectorX[idx] + inputVectorY[idx];
                outputVector[idx] += uniforms.y * inputVectorX[idx] + inputVectorY[idx];
                outputVector[idx] += uniforms.z * inputVectorX[idx] + inputVectorY[idx];
            }
        }
      `
    });
    
    // Create the compute pipeline
    const computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'saxpyKernel'
      }
    });
    
    // Create a bind group for the shader resources
    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBufferX } },
        { binding: 2, resource: { buffer: inputBufferY } },
        { binding: 3, resource: { buffer: outputBuffer } }
      ]
    });
    
    // Determine the number of workgroups (workgroup_size is 256)
    const workgroupSize = 256;
    const dispatchCount = Math.ceil(numElements / workgroupSize);
    
    // Run three trials and record timings
    const trialResults = [];
      for (let trial = 0; trial < 10; trial++) {
      // Data transfer: upload uniforms and input vectors
      const dtStart = performance.now();
      device.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer, uniformData.byteOffset, uniformData.byteLength);
      device.queue.writeBuffer(inputBufferX, 0, inputArrayX.buffer, inputArrayX.byteOffset, inputArrayX.byteLength);
      device.queue.writeBuffer(inputBufferY, 0, inputArrayY.buffer, inputArrayY.byteOffset, inputArrayY.byteLength);
      const dtEnd = performance.now();
      const dataTransferTime = dtEnd - dtStart;
      
      // Compute pass
      const commandEncoder = device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(dispatchCount);
      computePass.end();
      
      const commandBuffer = commandEncoder.finish();
      
      const computeStart = performance.now();
      device.queue.submit([commandBuffer]);
      await device.queue.onSubmittedWorkDone();
      const computeEnd = performance.now();
      const computeTime = computeEnd - computeStart;
      
      trialResults.push({ dataTransferTime, computeTime });
      
      // Optional: small delay between trials to allow recovery
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    // Average the results
    let totalDataTransfer = 0, totalCompute = 0;
    trialResults.forEach(trial => {
      totalDataTransfer += trial.dataTransferTime;
      totalCompute += trial.computeTime;
    });
    const avgDataTransferTime = totalDataTransfer / trialResults.length;
    const avgComputeTime = totalCompute / trialResults.length;
    
    // Estimate FLOPs: each element does 2 FLOPs (multiply and add)
    const totalFLOPs = 2 * numElements;
    const avgComputeTimeSeconds = avgComputeTime / 1000;
    const gflops = (totalFLOPs / avgComputeTimeSeconds) / 1e9;
    
    return {
      deviceName: adapter.name || 'Unknown GPU',
      vectorSizeMB: vectorSizeBytes / MB,
      avgDataTransferTime: avgDataTransferTime,
      avgComputeTime: avgComputeTime,
      gflops: gflops,
      trialCount: trialResults.length
    };
    
  } catch (error) {
    return {
      deviceName: 'Failed to initialize GPU',
      error: error.message
    };
  }
}

// Function to run the SAXPY benchmark and update the UI
async function runSaxpyBenchmark() {
  try {
    document.getElementById('saxpyStatus').textContent = 'Running SAXPY benchmark...';
    document.getElementById('saxpyStatus').style.backgroundColor = '#f5f5f5';
    // Clear previous results
    document.getElementById('saxpyResultsBody').innerHTML = '';
    document.getElementById('saxpySummary').innerHTML = '';
    
    // Short delay to update the UI
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const result = await saxpyBenchmark();
    updateSaxpyUI(result);
    
  } catch (error) {
    document.getElementById('saxpyStatus').textContent = `Error: ${error.message}`;
    document.getElementById('saxpyStatus').style.backgroundColor = '#ffe6e6';
    console.error('SAXPY Benchmark failed:', error);
  }
}

// Function to update the SAXPY benchmark UI (assumes a table structure similar to the memory benchmark)
function updateSaxpyUI(result) {
  // Update device and vector size information
  document.getElementById('saxpyGpuName').textContent = result.deviceName;
  document.getElementById('saxpyVectorSize').textContent = `${result.vectorSizeMB} MB`;
  
  document.getElementById('saxpyStatus').textContent = result.error ? 
    `Error: ${result.error}` : 'SAXPY Benchmark completed';
  
  const resultsBody = document.getElementById('saxpyResultsBody');
  
  // Data Transfer Time row
  let row = document.createElement('tr');
  let labelCell = document.createElement('td');
  labelCell.textContent = 'Average Data Transfer Time';
  row.appendChild(labelCell);
  let valueCell = document.createElement('td');
  valueCell.textContent = `${result.avgDataTransferTime.toFixed(2)} ms`;
  row.appendChild(valueCell);
  resultsBody.appendChild(row);
  
  // Compute Time row
  row = document.createElement('tr');
  labelCell = document.createElement('td');
  labelCell.textContent = 'Average Compute Time';
  row.appendChild(labelCell);
  valueCell = document.createElement('td');
  valueCell.textContent = `${result.avgComputeTime.toFixed(2)} ms`;
  row.appendChild(valueCell);
  resultsBody.appendChild(row);
  
  // FLOP Performance row
  row = document.createElement('tr');
  labelCell = document.createElement('td');
  labelCell.textContent = 'Estimated GFLOPS';
  row.appendChild(labelCell);
  valueCell = document.createElement('td');
  valueCell.textContent = `${result.gflops.toFixed(2)} GFLOPS`;
  row.appendChild(valueCell);
  resultsBody.appendChild(row);
  
  // Total Time row (Data Transfer + Compute)
  row = document.createElement('tr');
  labelCell = document.createElement('td');
  labelCell.textContent = 'Total Time (Average)';
  row.appendChild(labelCell);
  valueCell = document.createElement('td');
  const totalTime = result.avgDataTransferTime + result.avgComputeTime;
  valueCell.textContent = `${totalTime.toFixed(2)} ms`;
  row.appendChild(valueCell);
  resultsBody.appendChild(row);
  
  // Update summary
  const summary = document.getElementById('saxpySummary');
  if (!result.error) {
    summary.innerHTML = `
      <div class="summary-section">
        <h3>SAXPY Benchmark Summary</h3>
        <p>Vector size: ${result.vectorSizeMB} MB (~${(result.vectorSizeMB * 1024).toLocaleString()} KB)</p>
        <p>Average Data Transfer Time: ${result.avgDataTransferTime.toFixed(2)} ms</p>
        <p>Average Compute Time: ${result.avgComputeTime.toFixed(2)} ms</p>
        <p>Estimated Performance: ${result.gflops.toFixed(2)} GFLOPS</p>
        <p>(Averaged over ${result.trialCount} trials)</p>
      </div>
    `;
  } else {
    summary.innerHTML = `
      <div class="summary-section error">
        <h3>SAXPY Benchmark Error</h3>
        <p>${result.error}</p>
      </div>
    `;
  }
}

// Initialize the SAXPY benchmark UI: set up the run button event listener
window.addEventListener('load', () => {
  document.getElementById('runSaxpyButton').addEventListener('click', runSaxpyBenchmark);
});
