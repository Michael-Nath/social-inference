// iphone-fleet-manager.js
// This class manages a fleet of virtual iPhones for distributed SAXPY computation

class iPhoneFleetManager {
  constructor(options = {}) {
    // Extract options with defaults
    const {
      initialDeviceCount = 4,
      workerPath = './iphone-worker.js',
      workerOptions = { type: 'module' },
      deviceModels = ['iPhone13', 'iPhone12', 'iPhone11', 'iPhoneX'],
      realWorldFactors = true // Whether to simulate real-world conditions
    } = options;
    
    this.devices = [];  // Array of simulated iPhone workers
    this.taskQueue = []; // Tasks waiting to be processed
    this.runningTasks = new Map(); // Tasks currently being processed
    this.deviceModels = deviceModels;
    this.workerPath = workerPath;
    this.workerOptions = workerOptions;
    this.nextTaskId = 1;
    this.realWorldFactors = realWorldFactors;
    this.statistics = {
      totalTasksCompleted: 0,
      totalComputationTime: 0,
      devicePerformance: {},
      failedTasks: 0,
      reconnections: 0
    };
    
    // Fleet management intervals
    this.monitorInterval = null;
    this.healthCheckInterval = null;
    
    // Initialize the fleet with the specified number of devices
    this._initializeFleet(initialDeviceCount);
  }
  
  /**
   * Initialize the fleet of virtual iPhone devices
   * @param {number} count - Number of devices to create
   */
  _initializeFleet(count) {
    console.log(`Initializing a fleet of ${count} virtual iPhone devices...`);
    
    for (let i = 0; i < count; i++) {
      this._createDevice();
    }
    
    // Start monitoring the fleet
    this._startFleetMonitoring();
  }
  
  /**
   * Create a new virtual iPhone device
   * @returns {Object} The created device object
   */
  _createDevice() {
    // Create a worker to simulate the iPhone
    const workerUrl = new URL(this.workerPath, import.meta.url);
    const worker = new Worker(workerUrl, this.workerOptions);
    
    // Generate random device characteristics
    const randomModel = this.deviceModels[Math.floor(Math.random() * this.deviceModels.length)];
    const initialBatteryLevel = this.realWorldFactors ? 0.7 + Math.random() * 0.3 : 1.0; // 70-100%
    const connectionQuality = this.realWorldFactors ? 0.8 + Math.random() * 0.2 : 1.0; // 80-100%
    const deviceId = `iPhone-${Math.floor(Math.random() * 10000)}`;
    
    // Create device state object
    const device = {
      worker,
      model: randomModel,
      deviceId,
      batteryLevel: initialBatteryLevel,
      connectionQuality,
      isConnected: false,
      isReady: false,
      isProcessing: false,
      computationsPerformed: 0,
      lastActiveTime: Date.now(),
      totalComputationTime: 0,
      reconnectAttempts: 0
    };
    
    // Set up message handler for this device
    worker.onmessage = (event) => this._handleDeviceMessage(device, event);
    
    // Handle errors from the device
    worker.onerror = (error) => {
      console.error(`Error from device ${device.deviceId}:`, error.message);
      
      // Find any task that was running on this device and mark it as failed
      for (const [taskId, task] of this.runningTasks.entries()) {
        if (task.device === device) {
          task.reject(new Error(`Device ${device.deviceId} encountered an error: ${error.message}`));
          this.runningTasks.delete(taskId);
          this.statistics.failedTasks++;
          break;
        }
      }
      
      // Reset the device state
      device.isConnected = false;
      device.isProcessing = false;
      
      // Attempt to recover the device
      this._recoverDevice(device);
    };
    
    // Initialize the device with its characteristics
    worker.postMessage({
      command: 'initialize',
      data: {
        model: device.model,
        batteryLevel: device.batteryLevel,
        connectionQuality: device.connectionQuality,
        deviceId: device.deviceId
      }
    });
    
    // Add to the fleet
    this.devices.push(device);
    console.log(`Created virtual ${device.model} (ID: ${device.deviceId})`);
    
    // Initialize performance statistics for this model
    if (!this.statistics.devicePerformance[device.model]) {
      this.statistics.devicePerformance[device.model] = {
        tasksCompleted: 0,
        totalComputationTime: 0,
        failureRate: 0,
        averageBatteryDrain: 0
      };
    }
    
    return device;
  }
  
  /**
   * Handle messages from device workers
   * @param {Object} device - The device object
   * @param {MessageEvent} event - The message event
   */
  _handleDeviceMessage(device, event) {
    const data = event.data;
    
    switch (data.type) {
      case 'ready':
        device.isReady = true;
        this._connectDevice(device);
        break;
        
      case 'initialized':
        console.log(`Device ${data.deviceId} (${data.model}) initialized with ${Math.round(data.batteryLevel * 100)}% battery`);
        break;
        
      case 'connectionStatus':
        device.isConnected = data.connected;
        if (data.connected) {
          console.log(`Device ${device.deviceId} connected to server`);
          this._processQueue();
        } else {
          console.log(`Device ${device.deviceId} failed to connect to server`);
          // Schedule a reconnection attempt
          setTimeout(() => this._connectDevice(device), 2000 + Math.random() * 3000);
        }
        
        // If this was for a specific task, resolve/reject it
        if (data.taskId && this.runningTasks.has(data.taskId)) {
          const task = this.runningTasks.get(data.taskId);
          if (data.connected) {
            task.resolve(true);
          } else {
            task.reject(new Error(`Device ${device.deviceId} failed to connect`));
          }
          this.runningTasks.delete(data.taskId);
        }
        break;
        
      case 'result':
        // Process computation result
        if (data.taskId && this.runningTasks.has(data.taskId)) {
          const task = this.runningTasks.get(data.taskId);
          const endTime = Date.now();
          const computationTime = endTime - task.startTime;
          
          // Update device state from the result
          if (data.deviceStats) {
            device.batteryLevel = data.deviceStats.batteryLevel;
            device.computationsPerformed = data.deviceStats.computationsPerformed;
            device.lastActiveTime = endTime;
            device.totalComputationTime += computationTime;
          }
          
          // Update statistics
          this.statistics.totalTasksCompleted++;
          this.statistics.totalComputationTime += computationTime;
          this.statistics.devicePerformance[device.model].tasksCompleted++;
          this.statistics.devicePerformance[device.model].totalComputationTime += computationTime;
          
          // Mark device as available for more work
          device.isProcessing = false;
          
          // Resolve the task promise with the result
          task.resolve({
            result: data.result,
            device: {
              id: device.deviceId,
              model: device.model,
              batteryLevel: device.batteryLevel,
              computationTime
            }
          });
          
          // Remove from running tasks
          this.runningTasks.delete(data.taskId);
          
          // Process next task if available
          this._processQueue();
        }
        break;
        
      case 'status':
        // Update device status if status report received
        if (data.deviceStats) {
          device.batteryLevel = data.deviceStats.batteryLevel;
          device.connectionQuality = data.deviceStats.connectionQuality;
          device.isConnected = data.deviceStats.isConnected;
          device.computationsPerformed = data.deviceStats.computationsPerformed;
        }
        
        // If this was for a specific task, resolve it
        if (data.taskId && this.runningTasks.has(data.taskId)) {
          const task = this.runningTasks.get(data.taskId);
          task.resolve(data.deviceStats);
          this.runningTasks.delete(data.taskId);
        }
        break;
        
      case 'error':
        console.error(`Error from device ${device.deviceId}:`, data.error);
        
        // Find the task that was running on this device
        for (const [taskId, task] of this.runningTasks.entries()) {
          if (task.device === device) {
            task.reject(new Error(data.error));
            this.runningTasks.delete(taskId);
            this.statistics.failedTasks++;
            break;
          }
        }
        
        // Reset device state
        device.isProcessing = false;
        
        // Check if device is still connected; if not, try to reconnect
        if (data.error.includes('Not connected')) {
          device.isConnected = false;
          this._connectDevice(device);
        }
        
        break;
    }
  }
  
  /**
   * Attempt to connect a device to the server
   * @param {Object} device - The device to connect
   * @returns {Promise<boolean>} Whether connection was successful
   */
  _connectDevice(device) {
    return new Promise((resolve) => {
      if (!device.isReady) {
        resolve(false);
        return;
      }
      
      if (device.isConnected) {
        resolve(true);
        return;
      }
      
      device.reconnectAttempts++;
      this.statistics.reconnections++;
      
      const taskId = this.nextTaskId++;
      
      this.runningTasks.set(taskId, {
        resolve,
        reject: (error) => {
          console.error(`Connection error for device ${device.deviceId}:`, error.message);
          resolve(false);
        },
        device,
        type: 'connection',
        startTime: Date.now()
      });
      
      // Send connect command to device
      device.worker.postMessage({
        command: 'connect',
        taskId
      });
    });
  }
  
  /**
   * Try to recover a device that encountered an error
   * @param {Object} device - The device to recover
   */
  async _recoverDevice(device) {
    // If we're simulating real world conditions, sometimes devices just fail
    if (this.realWorldFactors && Math.random() < 0.2) {
      console.log(`Device ${device.deviceId} could not be recovered. Replacing...`);
      
      // Remove the device from the fleet
      const index = this.devices.indexOf(device);
      if (index !== -1) {
        this.devices.splice(index, 1);
      }
      
      // Terminate the worker
      device.worker.terminate();
      
      // Create a new device to replace it
      this._createDevice();
      return;
    }
    
    console.log(`Attempting to recover device ${device.deviceId}...`);
    
    // Try to reinitialize the device
    device.worker.postMessage({
      command: 'initialize',
      data: {
        model: device.model,
        batteryLevel: device.batteryLevel,
        connectionQuality: device.connectionQuality,
        deviceId: device.deviceId
      }
    });
    
    // Wait a moment and try to reconnect
    await new Promise(resolve => setTimeout(resolve, 1000));
    await this._connectDevice(device);
  }
  
  /**
   * Start monitoring the fleet of devices
   */
  _startFleetMonitoring() {
    // Set up a health check interval to monitor devices
    this.healthCheckInterval = setInterval(() => this._healthCheck(), 30000);
    
    // Set up a monitoring interval to log status
    this.monitorInterval = setInterval(() => this._monitorFleet(), 60000);
  }
  
  /**
   * Perform a health check on all devices
   */
  async _healthCheck() {
    console.log('Performing fleet health check...');
    
    const now = Date.now();
    const reconnectPromises = [];
    
    // Check each device
    for (const device of this.devices) {
      // If device hasn't been active for a while and isn't processing
      if (!device.isProcessing && (now - device.lastActiveTime) > 60000) {
        console.log(`Device ${device.deviceId} has been inactive for ${Math.round((now - device.lastActiveTime) / 1000)} seconds`);
        
        // Check if connected
        if (!device.isConnected) {
          console.log(`Reconnecting device ${device.deviceId}...`);
          reconnectPromises.push(this._connectDevice(device));
        }
      }
      
      // Check battery levels
      if (device.batteryLevel < 0.1 && !device.isProcessing) {
        console.log(`Device ${device.deviceId} has low battery (${Math.round(device.batteryLevel * 100)}%). Simulating charging...`);
        
        // Simulate charging
        device.batteryLevel = Math.min(1.0, device.batteryLevel + 0.2);
      }
    }
    
    // Wait for all reconnection attempts
    if (reconnectPromises.length > 0) {
      await Promise.all(reconnectPromises);
    }
    
    // Add more devices if needed
    if (this.devices.length < 3) {
      console.log('Fleet is too small, adding more devices...');
      this._createDevice();
      this._createDevice();
    }
  }
  
  /**
   * Monitor the fleet and log statistics
   */
  _monitorFleet() {
    // Count connected devices
    const connectedCount = this.devices.filter(d => d.isConnected).length;
    
    // Calculate average battery level
    const totalBattery = this.devices.reduce((sum, d) => sum + d.batteryLevel, 0);
    const avgBattery = totalBattery / this.devices.length;
    
    console.log(`--- Fleet Status Report ---`);
    console.log(`Total Devices: ${this.devices.length}, Connected: ${connectedCount}`);
    console.log(`Average Battery Level: ${Math.round(avgBattery * 100)}%`);
    console.log(`Tasks Completed: ${this.statistics.totalTasksCompleted}, Failed: ${this.statistics.failedTasks}`);
    console.log(`Reconnections: ${this.statistics.reconnections}`);
    console.log(`Queue Length: ${this.taskQueue.length}, Running Tasks: ${this.runningTasks.size}`);
    console.log('---------------------------');
    
    // Check performance by device model
    console.log('Performance by Device Model:');
    for (const [model, stats] of Object.entries(this.statistics.devicePerformance)) {
      if (stats.tasksCompleted > 0) {
        const avgTime = stats.totalComputationTime / stats.tasksCompleted;
        console.log(`  ${model}: ${stats.tasksCompleted} tasks, avg time: ${Math.round(avgTime)}ms`);
      }
    }
  }
  
  /**
   * Process the task queue if devices are available
   */
  _processQueue() {
    if (this.taskQueue.length === 0) return;
    
    // Find available devices
    const availableDevices = this.devices.filter(
      device => device.isConnected && !device.isProcessing && device.batteryLevel > 0.05
    );
    
    if (availableDevices.length === 0) return;
    
    // Process tasks for each available device
    for (const device of availableDevices) {
      if (this.taskQueue.length === 0) break;
      
      // Get the next task
      const task = this.taskQueue.shift();
      device.isProcessing = true;
      
      // Assign task to the device
      const taskId = this.nextTaskId++;
      
      this.runningTasks.set(taskId, {
        resolve: task.resolve,
        reject: task.reject,
        device,
        type: 'computation',
        startTime: Date.now()
      });
      
      // Send the computation command to the device
      device.worker.postMessage({
        command: 'compute',
        taskId,
        data: {
          a: task.a,
          xArray: task.xArray,
          yArray: task.yArray
        }
      });
    }
  }
  
  /**
   * Run a SAXPY computation on an available iPhone device
   * 
   * @param {number} a - The scalar value
   * @param {Array<number>} xArray - The x vector
   * @param {Array<number>} yArray - The y vector
   * @returns {Promise<Object>} - The result and device info
   */
  runSaxpy(a, xArray, yArray) {
    return new Promise((resolve, reject) => {
      // Validate inputs
      if (typeof a !== 'number' || !Array.isArray(xArray) || !Array.isArray(yArray)) {
        reject(new Error('Invalid input parameters'));
        return;
      }
      
      if (xArray.length !== yArray.length) {
        reject(new Error('Input arrays must have the same length'));
        return;
      }
      
      // Add to task queue
      this.taskQueue.push({
        a,
        xArray,
        yArray,
        resolve,
        reject
      });
      
      // Try to process the queue
      this._processQueue();
    });
  }
  
  /**
   * Run a SAXPY computation distributed across multiple iPhone devices
   * 
   * @param {number} a - The scalar value
   * @param {Array<number>} xArray - The x vector
   * @param {Array<number>} yArray - The y vector
   * @returns {Promise<Array<number>>} - The combined result vector
   */
  runDistributedSaxpy(a, xArray, yArray) {
    if (xArray.length !== yArray.length) {
      return Promise.reject(new Error('Input arrays must have the same length'));
    }
    
    const totalLength = xArray.length;
    
    // Determine how many devices to use based on array size and available devices
    const availableDevices = this.devices.filter(
      device => device.isConnected && !device.isProcessing && device.batteryLevel > 0.05
    );
    
    // Use at most 1 device per 10,000 elements, but limited by available devices
    const deviceCount = Math.min(
      availableDevices.length,
      Math.max(1, Math.ceil(totalLength / 10000))
    );
    
    // Determine chunk size
    const chunkSize = Math.ceil(totalLength / deviceCount);
    
    const promises = [];
    const deviceUsage = [];
    
    // Split the computation into chunks
    for (let i = 0; i < deviceCount; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, totalLength);
      
      if (start >= totalLength) break;
      
      const xChunk = xArray.slice(start, end);
      const yChunk = yArray.slice(start, end);
      
      promises.push(
        this.runSaxpy(a, xChunk, yChunk)
          .then(response => {
            // Track which device processed which portion
            deviceUsage.push({
              deviceId: response.device.id,
              model: response.device.model,
              chunkStart: start,
              chunkEnd: end,
              chunkSize: end - start,
              computationTime: response.device.computationTime,
              batteryLevel: response.device.batteryLevel
            });
            
            return {
              start,
              result: response.result
            };
          })
      );
    }
    
    // Combine the results
    return Promise.all(promises)
      .then(results => {
        // Create the combined result array
        const combinedResult = new Array(totalLength);
        
        // Fill in the results from each chunk
        results.forEach(({ start, result }) => {
          for (let i = 0; i < result.length; i++) {
            combinedResult[start + i] = result[i];
          }
        });
        
        // Return both the combined result and information about how it was processed
        return {
          result: combinedResult,
          deviceUsage
        };
      });
  }
  
  /**
   * Get the status of all devices in the fleet
   * @returns {Promise<Array>} Array of device status objects
   */
  getFleetStatus() {
    const statusPromises = this.devices.map(device => {
      return new Promise((resolve, reject) => {
        const taskId = this.nextTaskId++;
        
        this.runningTasks.set(taskId, {
          resolve,
          reject,
          device,
          type: 'status',
          startTime: Date.now()
        });
        
        // Send status request to the device
        device.worker.postMessage({
          command: 'getStatus',
          taskId
        });
      });
    });
    
    return Promise.all(statusPromises);
  }
  
  /**
   * Get overall statistics about the fleet
   * @returns {Object} Statistics about task completion, devices, etc.
   */
  getStatistics() {
    // Calculate current fleet stats
    const connectedCount = this.devices.filter(d => d.isConnected).length;
    const processingCount = this.devices.filter(d => d.isProcessing).length;
    const totalBattery = this.devices.reduce((sum, d) => sum + d.batteryLevel, 0);
    const avgBattery = this.devices.length > 0 ? totalBattery / this.devices.length : 0;
    
    // Return combined statistics
    return {
      fleet: {
        totalDevices: this.devices.length,
        connectedDevices: connectedCount,
        processingDevices: processingCount,
        averageBatteryLevel: avgBattery,
        deviceModels: this.devices.reduce((count, device) => {
          count[device.model] = (count[device.model] || 0) + 1;
          return count;
        }, {})
      },
      tasks: {
        completed: this.statistics.totalTasksCompleted,
        failed: this.statistics.failedTasks,
        queued: this.taskQueue.length,
        running: this.runningTasks.size,
        reconnections: this.statistics.reconnections,
        averageComputationTime: this.statistics.totalTasksCompleted > 0 
          ? this.statistics.totalComputationTime / this.statistics.totalTasksCompleted 
          : 0
      },
      performanceByModel: this.statistics.devicePerformance
    };
  }
  
  /**
   * Add a new device to the fleet
   * @param {Object} options - Device options (model, batteryLevel, etc.)
   * @returns {Object} The new device
   */
  addDevice(options = {}) {
    const deviceOptions = {
      model: options.model || this.deviceModels[Math.floor(Math.random() * this.deviceModels.length)],
      batteryLevel: options.batteryLevel !== undefined ? options.batteryLevel : (0.7 + Math.random() * 0.3),
      connectionQuality: options.connectionQuality !== undefined ? options.connectionQuality : (0.8 + Math.random() * 0.2)
    };
    
    // Create the device with specified options
    const device = this._createDevice(deviceOptions);
    
    // Try to connect the device
    this._connectDevice(device);
    
    return {
      deviceId: device.deviceId,
      model: device.model,
      batteryLevel: device.batteryLevel,
      connectionQuality: device.connectionQuality
    };
  }
  
  /**
   * Remove a device from the fleet
   * @param {string} deviceId - ID of the device to remove
   * @returns {boolean} Whether the device was successfully removed
   */
  removeDevice(deviceId) {
    const deviceIndex = this.devices.findIndex(d => d.deviceId === deviceId);
    
    if (deviceIndex === -1) {
      return false;
    }
    
    const device = this.devices[deviceIndex];
    
    // Cancel any tasks running on this device
    for (const [taskId, task] of this.runningTasks.entries()) {
      if (task.device === device) {
        task.reject(new Error(`Device ${deviceId} has been removed from the fleet`));
        this.runningTasks.delete(taskId);
      }
    }
    
    // Terminate the worker
    device.worker.terminate();
    
    // Remove from the devices array
    this.devices.splice(deviceIndex, 1);
    
    console.log(`Device ${deviceId} has been removed from the fleet`);
    return true;
  }
  
  /**
   * Terminate all workers and clean up resources
   */
  terminate() {
    // Clear intervals
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = null;
    }
    
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
    
    // Terminate all device workers
    this.devices.forEach(device => device.worker.terminate());
    
    // Clear data structures
    this.devices = [];
    this.taskQueue = [];
    this.runningTasks.clear();
    
    console.log('iPhone fleet manager terminated');
  }
}

export default iPhoneFleetManager;