// saxpy-worker.js
// This worker file encapsulates the WebGPU SAXPY computation

// Import the original SAXPY function
// Note: In a real worker environment, you'd need to include this code directly
// or use importScripts() depending on your bundling setup
import { runSaxpy } from './saxpy.js';

// Listen for messages from the main thread
self.onmessage = async function(event) {
  try {
    const { a, xArray, yArray, taskId } = event.data;
    
    // Validate inputs
    if (typeof a !== 'number' || !Array.isArray(xArray) || !Array.isArray(yArray)) {
      throw new Error('Invalid input parameters');
    }
    
    if (xArray.length !== yArray.length) {
      throw new Error('Input arrays must have the same length');
    }
    
    console.log(`[Worker] Starting SAXPY computation with a=${a}, arrays of length ${xArray.length}`);
    
    // Execute the SAXPY computation
    const result = await runSaxpy(a, xArray, yArray);
    
    // Send the result back to the main thread
    self.postMessage({
      type: 'result',
      taskId,
      result
    });
  } catch (error) {
    // Send any errors back to the main thread
    self.postMessage({
      type: 'error',
      error: error.message,
      stack: error.stack
    });
  }
};

// Inform main thread that worker is ready
self.postMessage({ type: 'ready' });