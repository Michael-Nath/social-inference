// iphone-worker.js
// Worker for iPhone device

import { runSaxpy } from '../kernels/saxpy.js';

// Simplified state for the iPhone
let deviceState = {
  model: 'iPhone13',
  batteryLevel: 1.0,
  connectionQuality: 0.95,
  isConnected: false,
  deviceId: `iPhone-${Math.floor(Math.random() * 10000)}`
};

// Listen for messages from the main thread
self.onmessage = async function(event) {
  if (event.data.command === 'compute') {
    const { a, xArray, yArray, taskId, chunkIndex, startIndex } = event.data.data;
    
    try {
      const startTime = performance.now();
      const result = await runSaxpy(a, xArray, yArray);
      const computationTime = performance.now() - startTime;
      
      // Send the result back
      self.postMessage({
        type: 'result',
        roomId: event.data.roomId,
        taskId: event.data.taskId,
        chunkIndex: chunkIndex,
        result: result,
        deviceStats: {
          batteryLevel: 100,
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