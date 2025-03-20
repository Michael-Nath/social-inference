// iphone-worker.js
// This worker simulates an iPhone connecting to the SAXPY server and performing computations

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

// Listen for messages from the main thread
self.onmessage = async function(event) {
  if (event.data.command === 'compute') {
    const { a, xArray, yArray, taskId, chunkIndex, startIndex } = event.data.data;
    
    try {
      const startTime = performance.now();
      const result = await runSaxpy(a, xArray, yArray);
      const computationTime = performance.now() - startTime;
      
      // Send the result back with additional metadata
      self.postMessage({
        type: 'result',
        roomId: event.data.roomId,
        taskId: event.data.taskId,
        chunkIndex: chunkIndex,
        result: result,
        deviceStats: {
          batteryLevel: 100, // Update with actual battery tracking
          computationTime: computationTime
        }
      });
    } catch (error) {
      self.postMessage({
        type: 'error',
        error: "Deez nuts!\n",
        taskId: taskId,
        chunkIndex: chunkIndex
      });
    }
  }
};

// Inform main thread that worker is ready
self.postMessage({ type: 'ready', deviceId: deviceState.deviceId });