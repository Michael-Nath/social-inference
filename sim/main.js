// main.js
// Main application demonstrating distributed SAXPY computation across a fleet of iPhones

import iPhoneFleetManager from './iphone-fleet-manager.js';

// Create a fleet manager with simulation options
const fleetManager = new iPhoneFleetManager({
  initialDeviceCount: 6,  // Start with 6 simulated iPhones
  workerPath: './iphone-worker.js',
  deviceModels: ['iPhone13', 'iPhone12', 'iPhone11', 'iPhoneX'],
  realWorldFactors: true  // Simulate real-world conditions like battery drain and network issues
});

// Helper function to format time in ms to a readable string
function formatTime(ms) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms/1000).toFixed(2)}s`;
}

// Helper function to update the UI
function updateUI(selector, content) {
  const element = document.querySelector(selector);
  if (element) element.innerHTML = content;
}

// Function to show a toast notification
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  
  document.body.appendChild(toast);
  
  // Animate in
  setTimeout(() => {
    toast.classList.add('show');
  }, 10);
  
  // Remove after 3 seconds
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => {
      document.body.removeChild(toast);
    }, 300);
  }, 3000);
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // UI elements
  const runButton = document.getElementById('run-button');
  const addDeviceButton = document.getElementById('add-device');
  const refreshStatusButton = document.getElementById('refresh-status');
  const simulateDropButton = document.getElementById('simulate-drop');
  const vectorSizeInput = document.getElementById('vector-size');
  const scalarValueInput = document.getElementById('scalar-value');
  const statusArea = document.getElementById('status-area');
  const resultsArea = document.getElementById('results-area');
  const fleetStatusArea = document.getElementById('fleet-status');
  const deviceListElement = document.getElementById('device-list');
  
  // Update fleet status initially
  updateFleetStatus();
  
  // Set up periodic fleet status updates
  setInterval(updateFleetStatus, 10000);
  
  // Add device button
  addDeviceButton.addEventListener('click', () => {
    // Get selected model from dropdown
    const modelSelect = document.getElementById('device-model');
    const selectedModel = modelSelect.value;
    
    // Add a new device
    const newDevice = fleetManager.addDevice({
      model: selectedModel
    });
    
    showToast(`Added new ${newDevice.model} (ID: ${newDevice.deviceId})`, 'success');
    
    // Update fleet status
    setTimeout(updateFleetStatus, 1000);
  });
  
  // Simulate a device dropping offline
  simulateDropButton.addEventListener('click', () => {
    const devices = fleetManager.devices;
    if (devices.length === 0) {
      showToast('No devices available to drop', 'error');
      return;
    }
    
    // Pick a random device
    const randomIndex = Math.floor(Math.random() * devices.length);
    const device = devices[randomIndex];
    
    showToast(`Simulating ${device.model} (${device.deviceId}) dropping offline...`, 'warning');
    
    // Force disconnect by triggering an error in the worker
    device.worker.postMessage({
      command: 'forceDisconnect'
    });
    
    // Update fleet status after a delay
    setTimeout(updateFleetStatus, 2000);
  });
  
  // Refresh status button
  refreshStatusButton.addEventListener('click', updateFleetStatus);
  
  // Run SAXPY button
  runButton.addEventListener('click', async () => {
    const vectorSize = parseInt(vectorSizeInput.value) || 100000;
    const scalarValue = parseFloat(scalarValueInput.value) || 2.0;
    
    // Disable button during computation
    runButton.disabled = true;
    
    // Update status
    statusArea.innerHTML = '<div class="loading-spinner"></div><p>Generating test data...</p>';
    
    try {
      // Generate test vectors
      const startDataGen = performance.now();
      const xArray = new Array(vectorSize);
      const yArray = new Array(vectorSize);
      
      for (let i = 0; i < vectorSize; i++) {
        xArray[i] = Math.random();
        yArray[i] = Math.random();
      }
      
      const dataGenTime = performance.now() - startDataGen;
      
      statusArea.innerHTML = `<div class="loading-spinner"></div>
                            <p>Running distributed SAXPY computation across iPhone fleet...</p>
                            <p>Vector size: ${vectorSize.toLocaleString()} elements</p>
                            <p>Data generation time: ${formatTime(dataGenTime)}</p>`;
      
      // Start the computation timer
      const startTime = performance.now();
      
      // Run distributed computation
      const { result, deviceUsage } = await fleetManager.runDistributedSaxpy(scalarValue, xArray, yArray);
      
      // Calculate total time
      const totalTime = performance.now() - startTime;
      
      // Update UI with results
      statusArea.innerHTML = `<p class="success">✓ SAXPY computation completed successfully!</p>
                            <p>Total computation time: ${formatTime(totalTime)}</p>
                            <p>Devices used: ${deviceUsage.length}</p>`;
      
      // Display results
      displayResults(result, totalTime, deviceUsage, scalarValue, xArray, yArray);
      
      // Show toast
      showToast(`Computation complete in ${formatTime(totalTime)}`, 'success');
      
    } catch (error) {
      // Handle errors
      statusArea.innerHTML = `<p class="error">Error: ${error.message}</p>`;
      console.error('SAXPY computation error:', error);
      showToast(`Error: ${error.message}`, 'error');
    } finally {
      // Re-enable button
      runButton.disabled = false;
      
      // Update fleet status
      updateFleetStatus();
    }
  });
  
  // Function to display computation results
  function displayResults(result, computationTime, deviceUsage, a, xArray, yArray) {
    // Validate a sample of the results
    const validateResults = [];
    for (let i = 0; i < 5; i++) {
      const index = Math.floor(Math.random() * result.length);
      const expected = a * xArray[index] + yArray[index];
      const actual = result[index];
      const difference = Math.abs(expected - actual);
      
      validateResults.push({
        index,
        expected: expected.toFixed(6),
        actual: actual.toFixed(6),
        difference: difference.toFixed(6)
      });
    }
    
    // Generate device usage summary
    const deviceSummary = deviceUsage.map(device => {
      return `<tr>
                <td>${device.model}</td>
                <td>${device.deviceId}</td>
                <td>${device.chunkSize.toLocaleString()}</td>
                <td>${formatTime(device.computationTime)}</td>
                <td>${Math.round(device.batteryLevel * 100)}%</td>
              </tr>`;
    }).join('');
    
    // Display results in the results area
    resultsArea.innerHTML = `
      <h3>Results Summary</h3>
      <div class="result-panels">
        <div class="result-panel">
          <h4>Computation Details</h4>
          <table>
            <tr><th>Vector Size</th><td>${result.length.toLocaleString()} elements</td></tr>
            <tr><th>Scalar Value (a)</th><td>${a}</td></tr>
            <tr><th>Total Time</th><td>${formatTime(computationTime)}</td></tr>
            <tr><th>Elements/Second</th><td>${Math.round(result.length / (computationTime / 1000)).toLocaleString()}</td></tr>
          </table>
        </div>
        
        <div class="result-panel">
          <h4>Result Validation</h4>
          <table>
            <tr>
              <th>Index</th>
              <th>Expected</th>
              <th>Actual</th>
              <th>Difference</th>
            </tr>
            ${validateResults.map(r => `
              <tr>
                <td>${r.index}</td>
                <td>${r.expected}</td>
                <td>${r.actual}</td>
                <td>${r.difference}</td>
              </tr>
            `).join('')}
          </table>
        </div>
      </div>
      
      <h3>Device Usage</h3>
      <table class="full-width">
        <tr>
          <th>Device Model</th>
          <th>Device ID</th>
          <th>Elements Processed</th>
          <th>Computation Time</th>
          <th>Battery Level</th>
        </tr>
        ${deviceSummary}
      </table>
      
      <h3>Results Preview</h3>
      <div class="results-preview">
        <div>
          <h4>First 10 elements:</h4>
          <pre>${JSON.stringify(result.slice(0, 10).map(v => v.toFixed(4)), null, 2)}</pre>
        </div>
        <div>
          <h4>Last 10 elements:</h4>
          <pre>${JSON.stringify(result.slice(-10).map(v => v.toFixed(4)), null, 2)}</pre>
        </div>
      </div>
    `;
  }
  
  // Function to update fleet status
  async function updateFleetStatus() {
    try {
      // Get statistics
      const stats = fleetManager.getStatistics();
      
      // Format the fleet summary
      fleetStatusArea.innerHTML = `
        <div class="stat-panel">
          <h4>Fleet Summary</h4>
          <div class="stat-grid">
            <div class="stat-item">
              <span class="stat-value">${stats.fleet.totalDevices}</span>
              <span class="stat-label">Total Devices</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">${stats.fleet.connectedDevices}</span>
              <span class="stat-label">Connected</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">${stats.fleet.processingDevices}</span>
              <span class="stat-label">Processing</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">${Math.round(stats.fleet.averageBatteryLevel * 100)}%</span>
              <span class="stat-label">Avg Battery</span>
            </div>
          </div>
        </div>
        
        <div class="stat-panel">
          <h4>Task Statistics</h4>
          <div class="stat-grid">
            <div class="stat-item">
              <span class="stat-value">${stats.tasks.completed}</span>
              <span class="stat-label">Completed</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">${stats.tasks.failed}</span>
              <span class="stat-label">Failed</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">${formatTime(stats.tasks.averageComputationTime)}</span>
              <span class="stat-label">Avg Time</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">${stats.tasks.reconnections}</span>
              <span class="stat-label">Reconnects</span>
            </div>
          </div>
        </div>
      `;
      
      // Get detailed device status
      const deviceStatus = await fleetManager.getFleetStatus();
      
      // Update device list
      updateDeviceList(deviceStatus);
      
    } catch (error) {
      console.error('Error updating fleet status:', error);
    }
  }
  
  // Function to update the device list
  function updateDeviceList(devices) {
    deviceListElement.innerHTML = '';
    
    devices.forEach(device => {
      const deviceEl = document.createElement('div');
      deviceEl.className = `device-card ${device.isConnected ? 'connected' : 'disconnected'}`;
      
      // Battery level indicator
      const batteryLevel = device.batteryLevel * 100;
      let batteryClass = 'battery-high';
      if (batteryLevel < 20) batteryClass = 'battery-critical';
      else if (batteryLevel < 50) batteryClass = 'battery-low';
      
      deviceEl.innerHTML = `
        <div class="device-header">
          <h4>${device.model}</h4>
          <div class="device-controls">
            <button class="small-button remove-device" data-id="${device.deviceId}">Remove</button>
          </div>
        </div>
        <div class="device-info">
          <div class="info-row">
            <span class="info-label">ID:</span>
            <span class="info-value">${device.deviceId}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Status:</span>
            <span class="status-indicator ${device.isConnected ? 'status-online' : 'status-offline'}">
              ${device.isConnected ? 'Online' : 'Offline'}
            </span>
          </div>
          <div class="info-row">
            <span class="info-label">Battery:</span>
            <div class="battery-indicator ${batteryClass}">
              <div class="battery-level" style="width: ${batteryLevel}%"></div>
              <span class="battery-text">${Math.round(batteryLevel)}%</span>
            </div>
          </div>
          <div class="info-row">
            <span class="info-label">Tasks:</span>
            <span class="info-value">${device.computationsPerformed}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Connection:</span>
            <span class="info-value">${Math.round(device.connectionQuality * 100)}%</span>
          </div>
        </div>
      `;
      
      // Add to the list
      deviceListElement.appendChild(deviceEl);
      
      // Add event listener for remove button
      const removeButton = deviceEl.querySelector('.remove-device');
      removeButton.addEventListener('click', () => {
        const deviceId = removeButton.getAttribute('data-id');
        if (fleetManager.removeDevice(deviceId)) {
          showToast(`Device ${deviceId} removed from fleet`, 'info');
          updateFleetStatus();
        }
      });
    });
  }
});

// Handle page unload
window.addEventListener('beforeunload', () => {
  // Clean up resources
  fleetManager.terminate();
});