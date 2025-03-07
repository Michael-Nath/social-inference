// iphone-worker.js
// This worker simulates an iPhone connecting to the SAXPY server and performing computations

// Device profiles to simulate different iPhone models
const DEVICE_PROFILES = {
  'iPhone13': { processingPower: 1.0, reliability: 0.99, batteryDrain: 0.05 },
  'iPhone12': { processingPower: 0.8, reliability: 0.98, batteryDrain: 0.06 },
  'iPhone11': { processingPower: 0.6, reliability: 0.97, batteryDrain: 0.08 },
  'iPhoneX': { processingPower: 0.4, reliability: 0.95, batteryDrain: 0.10 },
};

// State for this simulated iPhone
let deviceState = {
  model: 'iPhone13',
  batteryLevel: 1.0,  // Starts at 100%
  connectionQuality: 0.95, // Simulated connection quality (0-1)
  isConnected: false,
  computationsPerformed: 0,
  deviceId: `iPhone-${Math.floor(Math.random() * 10000)}` // Unique device ID
};

// Import SAXPY computation function
// In a real implementation, we'd need to include this directly or via importScripts()
// Assuming WebGPU will be simulated in this worker context
import { runSaxpy } from '../kernels/saxpy.js';

/**
 * Simulates connecting to the SAXPY server
 * @returns {Promise<boolean>} Connection success
 */
async function connectToServer() {
  // Simulate network connection process with some randomness
  const connectionTime = 500 + Math.random() * 1000; // 500-1500ms
  
  await new Promise(resolve => setTimeout(resolve, connectionTime));
  
  // Random connection success based on connection quality
  deviceState.isConnected = Math.random() < deviceState.connectionQuality;
  
  if (deviceState.isConnected) {
    console.log(`[${deviceState.deviceId}] Connected to SAXPY server`);
  } else {
    console.log(`[${deviceState.deviceId}] Failed to connect to SAXPY server`);
  }
  
  return deviceState.isConnected;
}

/**
 * Simulates performing the SAXPY computation on the iPhone
 * @param {number} a - Scalar value
 * @param {Array<number>} xArray - X vector
 * @param {Array<number>} yArray - Y vector
 * @returns {Promise<Array<number>>} Result vector
 */
async function performComputation(a, xArray, yArray) {
  if (!deviceState.isConnected) {
    throw new Error(`[${deviceState.deviceId}] Not connected to server`);
  }
  
  // Simulate device performance characteristics
  const profile = DEVICE_PROFILES[deviceState.model];
  
  // Simulate computation time based on processing power and data size
  const baseComputationTime = xArray.length / 10000; // Some baseline time unit
  const scaledComputationTime = baseComputationTime / profile.processingPower;
  
  // Random variation to simulate real-world conditions
  const randomFactor = 0.8 + Math.random() * 0.4; // 0.8-1.2x
  const totalComputationTime = scaledComputationTime * randomFactor;
  
  // Simulate device processing
  console.log(`[${deviceState.deviceId}] Starting computation on ${xArray.length} elements`);
  
  // Simulate occasional device failures (crash/restart)
  if (Math.random() > profile.reliability) {
    // Simulate device crash
    await new Promise(resolve => setTimeout(resolve, totalComputationTime * 0.3));
    throw new Error(`[${deviceState.deviceId}] Device crashed during computation`);
  }
  
  // Simulate actual GPU computation time
  await new Promise(resolve => setTimeout(resolve, totalComputationTime));
  
  // Actually perform the computation using the WebGPU-based SAXPY function
  // In a real device simulation, we might modify this to be more realistic
  // or even implement a simplified version that doesn't actually use WebGPU
  const result = await runSaxpy(a, xArray, yArray);
  
  // Update device state
  deviceState.batteryLevel = Math.max(0, deviceState.batteryLevel - profile.batteryDrain * (xArray.length / 100000));
  deviceState.computationsPerformed++;
  
  console.log(`[${deviceState.deviceId}] Computation complete (Battery: ${Math.round(deviceState.batteryLevel * 100)}%)`);
  
  // Simulate low battery behavior
  if (deviceState.batteryLevel < 0.2) {
    // Throttle performance when battery is low
    console.log(`[${deviceState.deviceId}] Low battery warning, throttling performance`);
    // In a more complex simulation, would actually slow down future computations
  }
  
  if (deviceState.batteryLevel < 0.05) {
    // Critical battery level - might disconnect soon
    console.log(`[${deviceState.deviceId}] Critical battery level`);
    if (Math.random() < 0.5) {
      deviceState.isConnected = false;
      throw new Error(`[${deviceState.deviceId}] Device shut down due to low battery`);
    }
  }
  
  return result;
}

// Listen for messages from the main thread
self.onmessage = async function(event) {
  if (event.data.command === 'compute') {
    const { a, xArray, yArray, taskId, chunkIndex, startIndex } = event.data.data;
    
    try {
      const startTime = performance.now();
      const result = await runSaxpy(a, xArray, yArray, taskId, chunkIndex, startIndex);
      const computationTime = performance.now() - startTime;
      
      // Send the result back with additional metadata
      self.postMessage({
        type: 'result',
        taskId: taskId,
        chunkIndex: chunkIndex,
        result: result,
        deviceStats: {
          batteryLevel: battery.level, // Update with actual battery tracking
          computationTime: computationTime
        }
      });
    } catch (error) {
      self.postMessage({
        type: 'error',
        error: error.message,
        taskId: taskId,
        chunkIndex: chunkIndex
      });
    }
  }
};

// Inform main thread that worker is ready
self.postMessage({ type: 'ready', deviceId: deviceState.deviceId });