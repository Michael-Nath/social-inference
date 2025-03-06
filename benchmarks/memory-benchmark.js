// Detect device type - focusing only on iOS vs desktop
function isIOSDevice() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
}

// Get safe maximum allocation size based on device
function getSafeMaxPower() {
  if (isIOSDevice()) {
      return 10; // Allow up to 1GB on iOS devices
  }
  return 20; // Up to 1GB (2^20) on desktop
}

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
async function memCheck() {
  // Set maximum power based on device
  const maxPower = getSafeMaxPower();
  
  try {
      const adapter = await setupGPU();
      const device = await adapter.requestDevice();
      
      // Update status function for progress
      const updateStatus = (message) => {
          document.getElementById('status').textContent = message;
      };
      
      const MB = 1024 * 1024;
      const results = [];
      
      // Start with a small allocation on iOS
      const startPower = isIOSDevice() ? 0 : 0; // 1MB on iOS, 1MB on desktop
      
      // Test allocations with increasing sizes until failure or maxPower is reached
      for (let power = startPower; power <= maxPower; power++) {
          const size = Math.pow(2, power) * MB;
          updateStatus(`Testing ${(size / MB).toFixed(0)}MB (2^${power})...`);
          
          try {
              console.log(`Attempting to allocate ${size / MB}MB...`);
              
              // For larger allocations on iOS, be extra cautious
              if (isIOSDevice() && power > 7) { // >128MB on iOS
                  updateStatus(`Preparing to test ${(size / MB).toFixed(0)}MB (2^${power}) safely...`);
                  // Longer delay on iOS to let the browser stabilize
                  await new Promise(resolve => setTimeout(resolve, 300));
              }
              
              const startTime = performance.now();
              
              // Create the buffer
              const buffer = device.createBuffer({
                  size: size,
                  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
              });
              
              const createTime = performance.now();
              
              // Create some data and transfer it to force actual allocation
              // For large allocations, be more careful
              // Use a more memory-efficient approach for data creation
              let dataArray;
              
              try {
                  // Special safe handling for larger allocations on iOS
                  if (isIOSDevice() && size > 256 * MB) {
                      // For larger allocations on iOS, skip data transfer
                      throw new Error("Skipping data transfer on iOS to prevent crash");
                  } else {
                      dataArray = new Uint8Array(size);
                      
                      // Only fill a small portion of the array to prevent excessive memory usage
                      const fillSize = Math.min(1024, size);
                      for (let j = 0; j < fillSize; j++) {
                          dataArray[j] = j % 256;
                      }
                  }
                  
                  // Start timing the data transfer
                  const transferStart = performance.now();
                  
                  // Special handling for iOS with large buffers
                  if (isIOSDevice() && size > 128 * MB) {
                      // For iOS, transfer even less data for large buffers
                      device.queue.writeBuffer(buffer, 0, dataArray, 0, Math.min(1024, size));
                  } else {
                      device.queue.writeBuffer(buffer, 0, dataArray);
                  }
                  
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
              } catch (innerError) {
                  // If we fail during data transfer, we still count the buffer creation as success
                  // but note the data transfer issue
                  results.push({
                      power: power,
                      size: size / MB,
                      unit: 'MB',
                      success: true,
                      partial: true,
                      createTime: createTime - startTime,
                      transferTime: 0,
                      totalTime: createTime - startTime,
                      transferError: innerError.message
                  });
                  
                  console.log(`Partially successful at ${size / MB}MB (2^${power}): Buffer created but data transfer skipped`);
              }
              
              // Clean up immediately and make sure the GC has a chance to run
              buffer.destroy();
              dataArray = null;
              
              // Add a small delay between tests to give the browser time to recover resources
              if (isIOSDevice() && power > 8) {
                  // Longer recovery time for iOS with large allocations
                  await new Promise(resolve => setTimeout(resolve, 500));
              } else if (power > 12) {
                  await new Promise(resolve => setTimeout(resolve, 300));
              }
              
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
          
          // Force a garbage collection pause between tests (when supported)
          // This won't work in most browsers, but worth trying
          if (typeof window.gc === 'function') {
              try {
                  window.gc();
              } catch (e) {
                  // Ignore errors
              }
          }
      }
      
      return {
          deviceName: adapter.name || 'Unknown GPU',
          deviceType: isIOSDevice() ? 'iOS' : 'Desktop',
          maxPowerTested: maxPower,
          maxSuccessfulAllocation: results.filter(r => r.success && !r.partial).pop(),
          firstPartialAllocation: results.find(r => r.partial),
          failedAllocation: results.find(r => !r.success),
          allResults: results
      };
  } catch (error) {
      // If we fail at setup level
      return {
          deviceName: 'Failed to initialize GPU',
          deviceType: isIOSDevice() ? 'iOS' : 'Desktop',
          error: error.message,
          allResults: []
      };
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

// Function to update the UI with results
function updateUI(benchmarkResults) {
  // Update device info
  document.getElementById('gpuName').textContent = benchmarkResults.deviceName;
  document.getElementById('deviceType').textContent = benchmarkResults.deviceType || 'Unknown';
  
  // Update status
  document.getElementById('status').textContent = benchmarkResults.error ? 
      `Error: ${benchmarkResults.error}` : 'Benchmark completed';
  
  // Get the results table body
  const resultsBody = document.getElementById('resultsBody');
  resultsBody.innerHTML = '';
  
  // If we have an error at setup level
  if (benchmarkResults.error) {
      document.getElementById('summary').innerHTML = `
          <div class="summary-section error">
              <h3>Benchmark Error</h3>
              <p>${benchmarkResults.error}</p>
              <p>Your browser may not support WebGPU or it might be disabled.</p>
          </div>
      `;
      return;
  }
  
  // Find max successful allocation and failed allocation
  const maxSuccess = benchmarkResults.allResults
      .filter(r => r.success && !r.partial)  // Only consider fully successful allocations
      .pop();
  const firstPartial = benchmarkResults.allResults
      .find(r => r.partial);
  const failedAllocation = benchmarkResults.allResults
      .find(r => !r.success);
  
  // Add rows for each result
  benchmarkResults.allResults.forEach(result => {
      const row = document.createElement('tr');
      
      // Determine row class based on success and if it's the max
      if (result.success) {
          if (!result.partial && maxSuccess && result.power === maxSuccess.power) {
              row.className = 'max-success';
          } else if (result.partial) {
              row.className = 'partial-success';
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
      if (result.success && !result.partial) {
          transferTimeCell.textContent = `${result.transferTime.toFixed(2)}ms`;
      } else if (result.partial) {
          transferTimeCell.textContent = 'Skipped';
      } else {
          transferTimeCell.textContent = 'N/A';
      }
      row.appendChild(transferTimeCell);
      
      // Status column
      const statusCell = document.createElement('td');
      if (result.success && !result.partial) {
          statusCell.textContent = 'Success';
      } else if (result.partial) {
          statusCell.textContent = `Partial: ${result.transferError || 'Data transfer skipped'}`;
      } else {
          statusCell.textContent = `Failed: ${result.error}`;
      }
      row.appendChild(statusCell);
      
      resultsBody.appendChild(row);
  });
  
  // Update summary
  const summary = document.getElementById('summary');
  
  if (maxSuccess) {
      let summaryHTML = '';
      
      // Add a notice for iOS devices
      if (benchmarkResults.deviceType === 'iOS') {
          summaryHTML += `
              <div class="summary-section warning">
                  <h3>iOS Device Detected</h3>
                  <p>Testing approach modified to prevent browser crashes. Maximum size tested: 2^${benchmarkResults.maxPowerTested} (${Math.pow(2, benchmarkResults.maxPowerTested)}MB)</p>
              </div>
          `;
      }
      
      // If we have partial success allocations
      if (firstPartial) {
          summaryHTML += `
              <div class="summary-section">
                  <h3>Memory Limit Detected</h3>
                  <p>Maximum full memory allocation: <strong>${maxSuccess.size} ${maxSuccess.unit} (2^${maxSuccess.power})</strong></p>
                  <p>Larger allocations (starting at ${firstPartial.size} ${firstPartial.unit}) could create buffers but not transfer data.</p>
              </div>
          `;
      } else if (failedAllocation) {
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
      
      // Only show timing details if we have a successful allocation with transfer
      if (!maxSuccess.partial) {
          summaryHTML += `
              <div class="summary-section">
                  <h3>Timing Details (Maximum Successful Allocation)</h3>
                  <p>Buffer creation: ${maxSuccess.createTime.toFixed(2)}ms</p>
                  <p>Data transfer: ${maxSuccess.transferTime.toFixed(2)}ms</p>
                  <p>Total time: ${maxSuccess.totalTime.toFixed(2)}ms</p>
              </div>
          `;
      } else {
          summaryHTML += `
              <div class="summary-section">
                  <h3>Timing Details (Maximum Successful Allocation)</h3>
                  <p>Buffer creation: ${maxSuccess.createTime.toFixed(2)}ms</p>
                  <p>Data transfer: Skipped to prevent browser crash</p>
              </div>
          `;
      }
      
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

// Initialize the page
window.addEventListener('load', () => {
  // Set up event listener for the run button
  document.getElementById('runButton').addEventListener('click', runBenchmark);
  
  // Check if we're on iOS and show warning
  if (isIOSDevice()) {
      document.getElementById('iosWarning').style.display = 'block';
  }
});