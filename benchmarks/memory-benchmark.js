// Detect device type - focusing only on iOS vs desktop
function isIOSDevice() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
}

// Get safe maximum allocation size based on device
function getSafeMaxPower() {
  if (isIOSDevice()) {
      return 10; // 2 GB
  }
  return 15; // Up to 32 GB on desktop
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

// Main memory check function with three trials per buffer size
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
      
      // Starting power (could be modified per platform)
      const startPower = 0;
      
      // Test allocations with increasing sizes until failure or maxPower is reached
      for (let power = startPower; power <= maxPower; power++) {
          const size = Math.pow(2, power) * MB;
          updateStatus(`Testing ${(size / MB).toFixed(0)}MB (2^${power})...`);
          
          const trialResults = [];
          // Run three trials for each buffer size
          for (let trial = 0; trial < 3; trial++) {
              try {
                  console.log(`Trial ${trial + 1}: Attempting to allocate ${size / MB}MB (2^${power})...`);
                  
                  // For larger allocations on iOS, be extra cautious
                  if (isIOSDevice() && power > 7) { // >128MB on iOS
                      updateStatus(`Preparing trial ${trial + 1} for ${(size / MB).toFixed(0)}MB (2^${power}) safely...`);
                      await new Promise(resolve => setTimeout(resolve, 300));
                  }
                  
                  const startTime = performance.now();
                  
                  // Create the buffer
                  const buffer = device.createBuffer({
                      size: size,
                      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                  });
                  
                  const createTime = performance.now();
                  
                  let runResult = {};
                  try {
                      // Create some data and transfer it to force actual allocation
                      const fillSize = Math.min(4 * MB, size);
                      const dataArray = new Uint8Array(fillSize);
                      for (let j = 0; j < fillSize; j++) {
                          dataArray[j] = j % 256;
                      }
                      
                      const transferStart = performance.now();
                      
                      // Write the data to the buffer in chunks
                      for (let i = 0; i < size / fillSize; i++) {
                          device.queue.writeBuffer(buffer, i * fillSize, dataArray);
                      }
                      
                      // Wait for the GPU to finish operations
                      await device.queue.onSubmittedWorkDone();
                      
                      const endTime = performance.now();
                      
                      runResult = {
                          success: true,
                          partial: false,
                          createTime: createTime - startTime,
                          transferTime: endTime - transferStart,
                          totalTime: (createTime - startTime) + (endTime - transferStart)
                      };
                      
                      console.log(`Trial ${trial + 1} successful: ${size / MB}MB allocated in ${runResult.totalTime.toFixed(2)}ms`);
                  } catch (innerError) {
                      // If data transfer fails but the buffer was created
                      runResult = {
                          success: true,
                          partial: true,
                          createTime: createTime - startTime,
                          transferTime: 0,
                          totalTime: createTime - startTime,
                          transferError: innerError.message
                      };
                      console.log(`Trial ${trial + 1} partially successful at ${size / MB}MB: Buffer created but data transfer skipped`);
                  }
                  
                  // Clean up and push trial result
                  buffer.destroy();
                  trialResults.push(runResult);
                  
                  // Small delay between trials
                  await new Promise(resolve => setTimeout(resolve, 50));
              } catch (error) {
                  // On failure for this trial, record the error result
                  trialResults.push({
                      success: false,
                      error: error.message
                  });
                  console.error(`Trial ${trial + 1} failed to allocate ${size / MB}MB (2^${power}): ${error.message}`);
              }
          } // end trial loop
          
          // Check if at least one trial succeeded
          const successfulTrials = trialResults.filter(r => r.success);
          if (successfulTrials.length === 0) {
              // Record failure for this power and exit the outer loop
              results.push({
                  power: power,
                  size: size / MB,
                  unit: 'MB',
                  success: false,
                  error: trialResults[0].error || "Unknown error"
              });
              console.error(`All trials failed for ${size / MB}MB (2^${power}). Stopping further tests.`);
              break;
          } else {
              // Average timings from all successful trials
              let totalCreate = 0, totalTransfer = 0, totalTotal = 0;
              let anyPartial = false;
              successfulTrials.forEach(trial => {
                  totalCreate += trial.createTime;
                  totalTransfer += trial.transferTime;
                  totalTotal += trial.totalTime;
                  if (trial.partial) {
                      anyPartial = true;
                  }
              });
              const count = successfulTrials.length;
              results.push({
                  power: power,
                  size: size / MB,
                  unit: 'MB',
                  success: true,
                  partial: anyPartial,
                  createTime: totalCreate / count,
                  transferTime: totalTransfer / count,
                  totalTime: totalTotal / count,
                  trialCount: count
              });
          }
          
          // Force a garbage collection pause between tests (when supported)
          if (typeof window.gc === 'function') {
              try {
                  window.gc();
              } catch (e) {
                  // Ignore errors
              }
          }
          
          // Delay between tests to let the browser recover resources
          if (isIOSDevice() && power > 8) {
              await new Promise(resolve => setTimeout(resolve, 500));
          } else if (power > 12) {
              await new Promise(resolve => setTimeout(resolve, 300));
          }
      } // end outer loop
      
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
      
      // Run the benchmark
      const results = await memCheck();
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

      // Bandwidth column
      const bandwidthCell = document.createElement('td');
      if (result.success && !result.partial) {
          bandwidthCell.textContent = `${(result.size / result.transferTime).toFixed(2)} GB/s`;
      } else if (result.partial) {
          bandwidthCell.textContent = 'Skipped';
      } else {
          bandwidthCell.textContent = 'N/A';
      }
      row.appendChild(bandwidthCell);
      
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
                  <h3>Timing Details (Average over ${maxSuccess.trialCount} trials)</h3>
                  <p>Buffer creation: ${maxSuccess.createTime.toFixed(2)}ms</p>
                  <p>Data transfer: ${maxSuccess.transferTime.toFixed(2)}ms</p>
                  <p>Total time: ${maxSuccess.totalTime.toFixed(2)}ms</p>
              </div>
          `;
      } else {
          summaryHTML += `
              <div class="summary-section">
                  <h3>Timing Details (Average over ${maxSuccess.trialCount} trials)</h3>
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
