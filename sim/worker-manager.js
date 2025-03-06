// worker-manager.js
// This class manages a pool of WebGPU worker threads for parallel computations

class WebGPUWorkerManager {
  constructor(options = {}) {
    // Extract options with defaults
    const {
      workerCount = navigator.hardwareConcurrency || 4,
      workerPath = './saxpy-worker.js',
      workerOptions = { type: 'module' }
    } = options;
    
    this.workers = [];
    this.taskQueue = [];
    this.runningTasks = new Map();
    this.workerCount = Math.min(workerCount, navigator.hardwareConcurrency || 4);
    this.workerPath = workerPath;
    this.workerOptions = workerOptions;
    this.nextTaskId = 1;
    this.readyWorkers = 0;
    
    this._initialize();
  }
  
  /**
   * Initialize the worker pool
   */
  _initialize() {
    console.log(`Initializing ${this.workerCount} WebGPU worker threads...`);
    
    for (let i = 0; i < this.workerCount; i++) {
      // Create worker using the configurable path
      const workerUrl = new URL(this.workerPath, import.meta.url);
      const worker = new Worker(workerUrl, this.workerOptions);
      
      worker.onmessage = (event) => {
        const data = event.data;
        
        if (data.type === 'ready') {
          this.readyWorkers++;
          this._processQueue();
        } else if (data.type === 'result') {
          const taskId = data.taskId;
          const task = this.runningTasks.get(taskId);
          
          if (task) {
            task.resolve(data.result);
            this.runningTasks.delete(taskId);
            worker.busy = false;
            this._processQueue();
          }
        } else if (data.type === 'error') {
          console.error(`Worker error: ${data.error}`);
          console.error(data.stack);
          
          // Find any running task on this worker and reject it
          for (const [taskId, task] of this.runningTasks.entries()) {
            if (task.worker === worker) {
              task.reject(new Error(data.error));
              this.runningTasks.delete(taskId);
              break;
            }
          }
          
          worker.busy = false;
          this._processQueue();
        }
      };
      
      worker.onerror = (error) => {
        console.error(`Worker fatal error: ${error.message}`);
        
        // Find any running task on this worker and reject it
        for (const [taskId, task] of this.runningTasks.entries()) {
          if (task.worker === worker) {
            task.reject(error);
            this.runningTasks.delete(taskId);
            break;
          }
        }
        
        // Replace the failed worker
        const index = this.workers.indexOf(worker);
        if (index !== -1) {
          this.workers.splice(index, 1);
          this._initialize();
        }
      };
      
      worker.busy = false;
      this.workers.push(worker);
    }
  }
  
  /**
   * Process the next task in the queue if any worker is available
   */
  _processQueue() {
    if (this.taskQueue.length === 0) return;
    
    // Find an available worker
    const availableWorker = this.workers.find(worker => !worker.busy);
    if (!availableWorker) return;
    
    // Get the next task and assign it to the worker
    const task = this.taskQueue.shift();
    availableWorker.busy = true;
    
    const taskId = this.nextTaskId++;
    this.runningTasks.set(taskId, {
      resolve: task.resolve,
      reject: task.reject,
      worker: availableWorker
    });
    
    // Send the data to the worker
    availableWorker.postMessage({
      a: task.a,
      xArray: task.xArray,
      yArray: task.yArray,
      taskId
    });
  }
  
  /**
   * Run a SAXPY computation across multiple worker threads
   * 
   * @param {number} a - The scalar value
   * @param {Array<number>} xArray - The x vector
   * @param {Array<number>} yArray - The y vector
   * @returns {Promise<Array<number>>} - The result vector
   */
  runSaxpy(a, xArray, yArray) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({
        a,
        xArray,
        yArray,
        resolve,
        reject
      });
      
      this._processQueue();
    });
  }
  
  /**
   * Split a large SAXPY task across multiple workers
   * 
   * @param {number} a - The scalar value
   * @param {Array<number>} xArray - The x vector
   * @param {Array<number>} yArray - The y vector
   * @returns {Promise<Array<number>>} - The result vector
   */
  runParallelSaxpy(a, xArray, yArray) {
    if (xArray.length !== yArray.length) {
      return Promise.reject(new Error('Input arrays must have the same length'));
    }
    
    const totalLength = xArray.length;
    const chunkSize = Math.ceil(totalLength / this.workerCount);
    const promises = [];
    
    // Split the computation into chunks and assign to different workers
    for (let i = 0; i < this.workerCount; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, totalLength);
      
      if (start >= totalLength) break;
      
      const xChunk = xArray.slice(start, end);
      const yChunk = yArray.slice(start, end);
      
      promises.push(
        this.runSaxpy(a, xChunk, yChunk)
          .then(result => ({ start, result }))
      );
    }
    
    // Combine the results from all workers
    return Promise.all(promises)
      .then(results => {
        const combinedResult = new Array(totalLength);
        
        results.forEach(({ start, result }) => {
          for (let i = 0; i < result.length; i++) {
            combinedResult[start + i] = result[i];
          }
        });
        
        return combinedResult;
      });
  }
  
  /**
   * Terminate all workers and clean up resources
   */
  terminate() {
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
    this.taskQueue = [];
    this.runningTasks.clear();
    console.log('All WebGPU workers terminated');
  }
}

export default WebGPUWorkerManager;