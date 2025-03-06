// GPU setup function
async function setupGPU() {
  if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in your browser");
  }
  
  try {
      return await navigator.gpu.requestAdapter();
  } catch (error) {
      throw new Error(`Failed to request GPU adapter: ${error.message}`);
  }
}

// Main memory check function
async function memCheck(maxPower = 20) {
  const adapter = await setupGPU();
  const device = await adapter.requestDevice();
  
  // Update status function for progress
  const updateStatus = (message) => {
      document.getElementById('status').textContent = message;
  };
  
  const MB = 1024 * 1024;
  const results = [];
  
  // Test allocations with increasing sizes until failure or maxPower is reached
  for (let power = 0; power <= maxPower; power++) {
      const size = Math.pow(2, power) * MB;
      updateStatus(`Testing ${(size / MB).toFixed(0)}MB (2^${power})...`);
      
      try {
          console.log(`Attempting to allocate ${size / MB}MB...`);
          const startTime = performance.now();
          
          // Create the buffer
          const buffer = device.createBuffer({
              size: size,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
          });
          
          const createTime = performance.now();
          
          // Create some data and transfer it to force actual allocation
          // (Timing for data creation excluded from benchmark)
          const dataArray = new Uint8Array(size);
          
          // Fill with some pattern (just to ensure it's actually doing something)
          for (let j = 0; j < Math.min(1024, size); j++) {
              dataArray[j] = j % 256;
          }
          
          // Start timing the data transfer
          const transferStart = performance.now();
          
          // Write the data to the buffer
          device.queue.writeBuffer(buffer, 0, dataArray);
          
          // Wait for the GPU to finish operations
          await device.queue.onSubmittedWorkDone();
          
          const endTime = performance.now();
          
          // Record successful allocation
          results.push({
              power: power,
              size: size / MB,
              unit: 'MB',
              success: true,
              createTime: createTime - startTime,
              transferTime: endTime - transferStart,
              totalTime: (createTime - startTime) + (endTime - transferStart)
          });
          
          console.log(`Successfully allocated ${size / MB}MB (2^${power}) in ${results[results.length-1].totalTime.toFixed(2)}ms`);
          console.log(`  - Buffer creation: ${results[results.length-1].createTime.toFixed(2)}ms`);
          console.log(`  - Data transfer: ${results[results.length-1].transferTime.toFixed(2)}ms`);
          
          // Clean up immediately to avoid memory accumulation affecting next test
          buffer.destroy();
          
      } catch (error) {
          // Record failed allocation
          results.push({
              power: power,
              size: size / MB,
              unit: 'MB',
              success: false,
              error: error.message
          });
          
          console.error(`Failed to allocate ${size / MB}MB (2^${power}): ${error.message}`);
          
          // Exit the loop since we've found the maximum
          break;
      }
  }
  
  return {
      deviceName: adapter.name || 'Unknown GPU',
      maxSuccessfulAllocation: results.filter(r => r.success).pop(),
      failedAllocation: results.find(r => !r.success),
      allResults: results
  };
}

// Function to update the UI with results
function updateUI(benchmarkResults) {
  // Update device info
  document.getElementById('gpuName').textContent = benchmarkResults.deviceName;
  
  // Update status
  document.getElementById('status').textContent = 'Benchmark completed';
  
  // Get the results table body
  const resultsBody = document.getElementById('resultsBody');
  resultsBody.innerHTML = '';
  
  // Find max successful allocation and failed allocation
  const maxSuccess = benchmarkResults.maxSuccessfulAllocation;
  const failedAllocation = benchmarkResults.failedAllocation;
  
  // Add rows for each result
  benchmarkResults.allResults.forEach(result => {
      const row = document.createElement('tr');
      
      // Determine row class based on success and if it's the max
      if (result.success) {
          if (maxSuccess && result.power === maxSuccess.power) {
              row.className = 'max-success';
          } else {
              row.className = 'success';
          }
      } else {
          row.className = 'failure';
      }
      
      // Size column with power notation
      const sizeCell = document.createElement('td');
      sizeCell.textContent = `${result.size} ${result.unit} (2^${result.power})`;
      row.appendChild(sizeCell);
      
      // Create Time column
      const createTimeCell = document.createElement('td');
      createTimeCell.textContent = result.success ? `${result.createTime.toFixed(2)}ms` : 'N/A';
      row.appendChild(createTimeCell);
      
      // Transfer Time column
      const transferTimeCell = document.createElement('td');
      transferTimeCell.textContent = result.success ? `${result.transferTime.toFixed(2)}ms` : 'N/A';
      row.appendChild(transferTimeCell);
      
      // Status column
      const statusCell = document.createElement('td');
      statusCell.textContent = result.success ? 'Success' : `Failed: ${result.error}`;
      row.appendChild(statusCell);
      
      resultsBody.appendChild(row);
  });
  
  // Update summary
  const summary = document.getElementById('summary');
  
  if (maxSuccess) {
      let summaryHTML = '';
      
      if (failedAllocation) {
          summaryHTML += `
              <div class="summary-section">
                  <h3>Memory Limit Detected</h3>
                  <p>Maximum memory allocation: <strong>${maxSuccess.size} ${maxSuccess.unit} (2^${maxSuccess.power})</strong></p>
                  <p>Failed at: ${failedAllocation.size} ${failedAllocation.unit} (2^${failedAllocation.power})</p>
              </div>
          `;
      } else {
          summaryHTML += `
              <div class="summary-section">
                  <h3>No Memory Limit Detected</h3>
                  <p>Successfully allocated up to: <strong>${maxSuccess.size} ${maxSuccess.unit} (2^${maxSuccess.power})</strong></p>
                  <p>Higher allocations were not tested. Your GPU may support more memory.</p>
              </div>
          `;
      }
      
      summaryHTML += `
          <div class="summary-section">
              <h3>Timing Details (Maximum Successful Allocation)</h3>
              <p>Buffer creation: ${maxSuccess.createTime.toFixed(2)}ms</p>
              <p>Data transfer: ${maxSuccess.transferTime.toFixed(2)}ms</p>
              <p>Total time: ${maxSuccess.totalTime.toFixed(2)}ms</p>
          </div>
      `;
      
      summary.innerHTML = summaryHTML;
  } else {
      summary.innerHTML = `
          <div class="summary-section">
              <h3>No Successful Allocations</h3>
              <p>Unable to allocate even 1MB of GPU memory.</p>
              <p>Your browser may not support WebGPU properly, or there might be another issue with your GPU.</p>
          </div>
      `;
  }
}

// Function to run the benchmark with automatic progression
async function runBenchmark() {
  try {
      // Update status
      document.getElementById('status').textContent = 'Running benchmark...';
      document.getElementById('status').style.backgroundColor = '#f5f5f5';
      
      // Clear previous results
      document.getElementById('resultsBody').innerHTML = '';
      document.getElementById('summary').innerHTML = '';
      
      // Small delay to allow the UI to update before running the benchmark
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Run the benchmark (with a generous maximum of 2^20 = 1GB)
      const results = await memCheck(20);
      updateUI(results);
  } catch (error) {
      document.getElementById('status').textContent = `Error: ${error.message}`;
      document.getElementById('status').style.backgroundColor = '#ffe6e6';
      console.error('Benchmark failed:', error);
  }
}

// Initialize the page
window.addEventListener('load', () => {
  // Set up event listener for the run button
  document.getElementById('runButton').addEventListener('click', runBenchmark);
});