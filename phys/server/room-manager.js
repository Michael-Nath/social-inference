// room-manager.js
// Manages computation rooms where multiple users can connect their devices

class ComputationRoom {
  constructor(roomId, options = {}) {
    this.roomId = roomId;
    this.name = options.name || `Room ${roomId}`;
    this.description = options.description || 'A shared computation space';
    this.createdAt = new Date();
    this.createdBy = options.createdBy || 'Anonymous';
    this.isPublic = options.isPublic !== undefined ? options.isPublic : true;
    this.maxDevices = options.maxDevices || 50;
    
    // Connected users and their devices
    this.users = new Map(); // userId -> { username, joinedAt, devices }
    
    // Task management
    this.taskQueue = [];
    this.completedTasks = [];
    this.currentTask = null;
    this.taskResults = new Map();
    
    // Room statistics
    this.stats = {
      totalComputations: 0,
      totalDevicesEverConnected: 0,
      totalUsersEverJoined: 0,
      peakDeviceCount: 0,
      totalComputationTime: 0
    };
    
    // Room events
    this.events = [];
    this.addEvent('room_created', { createdBy: this.createdBy });
    
    console.log(`Room "${this.name}" (${this.roomId}) created`);
  }
  
  /**
   * Add a user to the room
   * @param {string} userId - Unique identifier for the user
   * @param {Object} userInfo - Information about the user
   * @returns {boolean} Whether the user was added successfully
   */
  addUser(userId, userInfo = {}) {
    if (this.users.has(userId)) {
      console.log(`User ${userId} already in room ${this.roomId}`);
      return false;
    }
    
    const user = {
      userId,
      username: userInfo.username || `User-${userId.substring(0, 6)}`,
      joinedAt: new Date(),
      devices: new Map(), // deviceId -> device object
      isAdmin: userInfo.isAdmin || false
    };
    
    this.users.set(userId, user);
    this.stats.totalUsersEverJoined++;
    
    this.addEvent('user_joined', { 
      userId, 
      username: user.username,
      isAdmin: user.isAdmin 
    });
    
    console.log(`User ${user.username} (${userId}) joined room ${this.roomId}`);
    return true;
  }
  
  /**
   * Remove a user from the room
   * @param {string} userId - ID of the user to remove
   * @returns {boolean} Whether the user was removed
   */
  removeUser(userId) {
    const user = this.users.get(userId);
    if (!user) {
      return false;
    }
    
    // Remove all devices owned by this user
    for (const deviceId of user.devices.keys()) {
      this.removeDevice(userId, deviceId);
    }
    
    // Remove the user
    this.users.delete(userId);
    
    this.addEvent('user_left', { userId, username: user.username });
    console.log(`User ${user.username} (${userId}) left room ${this.roomId}`);
    
    return true;
  }
  
  /**
   * Add a device to a user in the room
   * @param {string} userId - ID of the user who owns the device
   * @param {string} deviceId - Unique identifier for the device
   * @param {Object} deviceInfo - Information about the device
   * @returns {boolean} Whether the device was added successfully
   */
  addDevice(userId, deviceId, deviceInfo = {}) {
    const user = this.users.get(userId);
    if (!user) {
      console.error(`Cannot add device: User ${userId} not in room ${this.roomId}`);
      return false;
    }
    
    if (user.devices.has(deviceId)) {
      console.log(`Device ${deviceId} already registered to user ${userId}`);
      return false;
    }
    
    // Check if we've reached the maximum number of devices
    const totalDevices = this.getTotalDeviceCount();
    if (totalDevices >= this.maxDevices) {
      console.error(`Cannot add device: Room ${this.roomId} at maximum capacity (${this.maxDevices} devices)`);
      return false;
    }
    
    // Create the device object
    const device = {
      deviceId,
      userId,
      model: deviceInfo.model || 'Unknown',
      joinedAt: new Date(),
      lastActive: new Date(),
      batteryLevel: deviceInfo.batteryLevel !== undefined ? deviceInfo.batteryLevel : 1.0,
      connectionQuality: deviceInfo.connectionQuality !== undefined ? deviceInfo.connectionQuality : 0.95,
      isConnected: true,
      isProcessing: false,
      computationsPerformed: 0,
      totalComputationTime: 0,
      worker: deviceInfo.worker // Reference to the actual worker/device
    };
    
    user.devices.set(deviceId, device);
    this.stats.totalDevicesEverConnected++;
    
    // Update peak device count
    const newTotalCount = this.getTotalDeviceCount();
    if (newTotalCount > this.stats.peakDeviceCount) {
      this.stats.peakDeviceCount = newTotalCount;
    }
    
    this.addEvent('device_connected', {
      deviceId,
      userId,
      username: user.username,
      model: device.model
    });
    
    console.log(`Device ${deviceId} (${device.model}) added for user ${user.username} in room ${this.roomId}`);
    return true;
  }

  /**
   * Remove a device from the room
   * @param {string} userId - ID of the user who owns the device
   * @param {string} deviceId - ID of the device to remove
   * @returns {boolean} Whether the device was removed
   */
  removeDevice(userId, deviceId) {
    const user = this.users.get(userId);
    if (!user || !user.devices.has(deviceId)) {
      return false;
    }
    
    const device = user.devices.get(deviceId);
    
    // If the device is currently processing a task, handle the interruption
    if (device.isProcessing && this.currentTask) {
      this._handleDeviceDisconnectionDuringTask(device);
    }
    
    // Remove the device
    user.devices.delete(deviceId);
    
    this.addEvent('device_disconnected', {
      deviceId,
      userId,
      username: user.username,
      model: device.model
    });
    
    console.log(`Device ${deviceId} removed for user ${user.username} in room ${this.roomId}`);
    return true;
  }
  
  /**
   * Handle a device disconnecting while processing a task
   * @private
   */
  _handleDeviceDisconnectionDuringTask(device) {
    // If we have a current task and this device was working on it
    if (this.currentTask && device.isProcessing) {
      console.log(`Device ${device.deviceId} disconnected while processing a task`);
      
      // Find which chunk this device was processing
      const taskChunk = this.currentTask.deviceAssignments.find(
        assignment => assignment.deviceId === device.deviceId
      );
      
      if (taskChunk) {
        console.log(`Reassigning chunk ${taskChunk.chunkIndex} from disconnected device ${device.deviceId}`);
        
        // Mark this chunk as unassigned
        taskChunk.status = 'unassigned';
        taskChunk.deviceId = null;
        
        // Remove from device assignments
        this.currentTask.deviceAssignments = this.currentTask.deviceAssignments.filter(
          assignment => assignment.deviceId !== device.deviceId
        );
        
        // Queue this chunk to be processed by another device
        this.currentTask.unassignedChunks.push(taskChunk.chunkIndex);
        
        // Check if we need to assign this to another device
        this._assignPendingChunks();
      }
    }
  }
  
  /**
   * Queue a SAXPY computation task
   * @param {Object} task - The task to queue
   * @param {number} task.a - The scalar value
   * @param {Array<number>} task.xArray - The x vector
   * @param {Array<number>} task.yArray - The y vector
   * @param {string} task.initiatedBy - User ID who initiated the task
   * @returns {string} The task ID
   */
  queueTask(task) {
    const taskId = `task-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    const newTask = {
      taskId,
      a: task.a,
      xArray: task.xArray,
      yArray: task.yArray,
      vectorLength: task.xArray.length,
      initiatedBy: task.initiatedBy,
      queuedAt: new Date(),
      startedAt: null,
      completedAt: null,
      status: 'queued',
      progress: 0,
      result: null,
      error: null,
      unassignedChunks: [],
      deviceAssignments: [],
      chunksCompleted: 0,
      totalChunks: 0
    };
    
    this.taskQueue.push(newTask);
    
    this.addEvent('task_queued', {
      taskId,
      initiatedBy: task.initiatedBy,
      vectorLength: task.xArray.length
    });
    
    console.log(`Task ${taskId} queued in room ${this.roomId} (vector length: ${task.xArray.length})`);
    
    // If no task is currently running, start this one
    if (!this.currentTask) {
      this._startNextTask();
    }
    
    return taskId;
  }
  
  /**
   * Start the next task in the queue
   * @private
   */
  _startNextTask() {
    if (this.taskQueue.length === 0 || this.currentTask) {
      return;
    }
    
    this.currentTask = this.taskQueue.shift();
    this.currentTask.status = 'running';
    this.currentTask.startedAt = new Date();
    
    // Divide the task into chunks based on available devices
    this._divideTaskIntoChunks();
    
    // Assign chunks to available devices
    this._assignPendingChunks();
    
    this.addEvent('task_started', {
      taskId: this.currentTask.taskId,
      initiatedBy: this.currentTask.initiatedBy,
      vectorLength: this.currentTask.vectorLength,
      totalChunks: this.currentTask.totalChunks
    });
    
    console.log(`Task ${this.currentTask.taskId} started with ${this.currentTask.totalChunks} chunks`);
  }
  
  /**
   * Divide a task into chunks for processing
   * @private
   */
  _divideTaskIntoChunks() {
    const task = this.currentTask;
    const vectorLength = task.xArray.length;
    
    // Count available devices
    const availableDevices = this._getAvailableDevices();
    const deviceCount = Math.max(1, availableDevices.length);
    
    // Determine chunk size - aim for at least 10,000 elements per chunk
    // but also limit the maximum number of chunks to avoid overhead
    const idealElementsPerChunk = 10000;
    const maxChunks = Math.min(deviceCount * 2, Math.ceil(vectorLength / idealElementsPerChunk));
    const chunkSize = Math.ceil(vectorLength / maxChunks);
    
    // Create chunks
    task.totalChunks = Math.ceil(vectorLength / chunkSize);
    task.unassignedChunks = [];
    
    for (let i = 0; i < task.totalChunks; i++) {
      const startIndex = i * chunkSize;
      const endIndex = Math.min(startIndex + chunkSize, vectorLength);
      
      // Add this chunk index to unassigned chunks
      task.unassignedChunks.push(i);
    }
    
    console.log(`Task ${task.taskId} divided into ${task.totalChunks} chunks (${chunkSize} elements per chunk)`);
  }
  
  /**
   * Assign pending chunks to available devices
   * @private
   */
  _assignPendingChunks() {
    const task = this.currentTask;
    if (!task || task.unassignedChunks.length === 0) {
      return;
    }
    
    const availableDevices = this._getAvailableDevices();
    if (availableDevices.length === 0) {
      console.log(`No available devices to process chunks for task ${task.taskId}`);
      return;
    }
    
    console.log(`Assigning chunks for task ${task.taskId}, ${task.unassignedChunks.length} chunks remaining`);
    
    // Get the chunk size
    const vectorLength = task.xArray.length;
    const chunkSize = Math.ceil(vectorLength / task.totalChunks);
    
    // Assign chunks to devices
    for (const device of availableDevices) {
      if (task.unassignedChunks.length === 0) break;
      
      // Get the next chunk index
      const chunkIndex = task.unassignedChunks.shift();
      const startIndex = chunkIndex * chunkSize;
      const endIndex = Math.min(startIndex + chunkSize, vectorLength);
      
      // Create a slice of the vectors for this chunk
      const xSlice = task.xArray.slice(startIndex, endIndex);
      const ySlice = task.yArray.slice(startIndex, endIndex);
      
      // Assign to the device
      device.isProcessing = true;
      
      // Record the assignment
      task.deviceAssignments.push({
        chunkIndex,
        deviceId: device.deviceId,
        userId: device.userId,
        startIndex,
        endIndex,
        status: 'assigned',
        startedAt: new Date()
      });
      
      // Send the computation to the device
      this._sendComputationToDevice(device, task.taskId, task.a, xSlice, ySlice, chunkIndex, startIndex);
      
      console.log(`Chunk ${chunkIndex} assigned to device ${device.deviceId} for task ${task.taskId}`);
    }
  }
  
  /**
   * Send a computation to a device
   * @private
   */
  _sendComputationToDevice(device, taskId, a, xSlice, ySlice, chunkIndex, startIndex) {
    // This method would interface with your actual device worker
    // For now, it's a placeholder that would be implemented differently
    // depending on how you communicate with the device workers
    console.log(device) 
    throw new Error("Bruh")
    if (device.worker && typeof device.worker.postMessage === 'function') {
      // If we have a direct reference to the worker
      device.worker.postMessage({
        command: 'compute',
        taskId,
        roomId: this.roomId,
        data: {
          a,
          xArray: xSlice,
          yArray: ySlice,
          chunkIndex,
          startIndex
        }
      });
    } else {
      console.error(`Cannot send computation to device ${device.deviceId}: No worker reference`);
      // Handle this error - maybe mark the chunk as unassigned again
      this._handleDeviceComputationError(device, taskId, chunkIndex, 'No worker reference');
    }
  }
  
  /**
   * Handle a device completing a computation
   * @param {string} deviceId - ID of the device that completed the computation
   * @param {string} userId - ID of the user who owns the device
   * @param {string} taskId - ID of the task 
   * @param {number} chunkIndex - Index of the chunk that was processed
   * @param {Array<number>} result - The computation result
   * @param {Object} stats - Statistics about the computation
   */
  handleDeviceComputationComplete(deviceId, userId, taskId, chunkIndex, result, stats) {
    // Find the device
    const user = this.users.get(userId);
    if (!user) {
      console.error(`Unknown user ${userId} reported computation completion`);
      return;
    }
    
    const device = user.devices.get(deviceId);
    if (!device) {
      console.error(`Unknown device ${deviceId} for user ${userId} reported computation completion`);
      return;
    }
    
    // Check if this is for the current task
    if (!this.currentTask || this.currentTask.taskId !== taskId) {
      console.error(`Received completion for unknown or outdated task ${taskId}`);
      return;
    }
    
    // Find the assignment
    const assignment = this.currentTask.deviceAssignments.find(
      a => a.deviceId === deviceId && a.chunkIndex === chunkIndex
    );
    
    if (!assignment) {
      console.error(`No assignment found for device ${deviceId}, chunk ${chunkIndex}`);
      return;
    }
    
    // Mark the assignment as completed
    assignment.status = 'completed';
    assignment.completedAt = new Date();
    assignment.computationTime = stats.computationTime;
    
    // Update device stats
    device.isProcessing = false;
    device.lastActive = new Date();
    device.computationsPerformed++;
    device.totalComputationTime += stats.computationTime;
    device.batteryLevel = stats.batteryLevel;
    
    // Store the result
    if (!this.taskResults.has(taskId)) {
      this.taskResults.set(taskId, []);
    }
    
    this.taskResults.get(taskId).push({
      chunkIndex,
      startIndex: assignment.startIndex,
      endIndex: assignment.endIndex,
      result
    });
    
    // Update task progress
    this.currentTask.chunksCompleted++;
    this.currentTask.progress = (this.currentTask.chunksCompleted / this.currentTask.totalChunks) * 100;
    
    console.log(`Device ${deviceId} completed chunk ${chunkIndex} for task ${taskId}. Progress: ${this.currentTask.progress.toFixed(1)}%`);
    
    // Check if we should assign more chunks
    if (this.currentTask.unassignedChunks.length > 0) {
      this._assignPendingChunks();
    }
    
    // Check if the task is complete
    if (this.currentTask.chunksCompleted >= this.currentTask.totalChunks) {
      this._finalizeTask();
    }
  }
  
  /**
   * Handle an error in device computation
   * @param {Object} device - The device object
   * @param {string} taskId - ID of the task
   * @param {number} chunkIndex - Index of the chunk that failed
   * @param {string} error - Error message
   */
  _handleDeviceComputationError(device, taskId, chunkIndex, error) {
    console.error(`Error in device ${device.deviceId} computing chunk ${chunkIndex}: ${error}`);
    
    // Check if this is for the current task
    if (!this.currentTask || this.currentTask.taskId !== taskId) {
      return;
    }
    
    // Find the assignment
    const assignment = this.currentTask.deviceAssignments.find(
      a => a.deviceId === device.deviceId && a.chunkIndex === chunkIndex
    );
    
    if (assignment) {
      // Mark the assignment as failed
      assignment.status = 'failed';
      assignment.error = error;
      
      // Remove from device assignments
      this.currentTask.deviceAssignments = this.currentTask.deviceAssignments.filter(
        a => !(a.deviceId === device.deviceId && a.chunkIndex === chunkIndex)
      );
      
      // Put the chunk back in the unassigned queue
      this.currentTask.unassignedChunks.push(chunkIndex);
    }
    
    // Update device status
    device.isProcessing = false;
    
    // Attempt to reassign the chunk
    this._assignPendingChunks();
  }
  
  /**
   * Finalize a completed task
   * @private
   */
  _finalizeTask() {
    const task = this.currentTask;
    
    console.log(`Finalizing task ${task.taskId}`);
    
    // Combine all the results
    const results = this.taskResults.get(task.taskId) || [];
    results.sort((a, b) => a.startIndex - b.startIndex);
    
    // Create the final result array
    const finalResult = new Array(task.vectorLength);
    
    for (const chunk of results) {
      for (let i = 0; i < chunk.result.length; i++) {
        finalResult[chunk.startIndex + i] = chunk.result[i];
      }
    }
    
    // Set task completion
    task.completedAt = new Date();
    task.status = 'completed';
    task.result = finalResult;
    
    // Calculate execution time
    const executionTime = task.completedAt - task.startedAt;
    
    // Update room statistics
    this.stats.totalComputations++;
    this.stats.totalComputationTime += executionTime;
    
    this.addEvent('task_completed', {
      taskId: task.taskId,
      initiatedBy: task.initiatedBy,
      executionTimeMs: executionTime,
      deviceCount: new Set(task.deviceAssignments.map(a => a.deviceId)).size
    });
    
    console.log(`Task ${task.taskId} completed in ${executionTime}ms`);
    
    // Add to completed tasks
    this.completedTasks.push(task);
    
    // Clear current task
    this.currentTask = null;
    
    // Start the next task if one is queued
    if (this.taskQueue.length > 0) {
      this._startNextTask();
    }
  }
  
  /**
   * Get available devices for computation
   * @private
   * @returns {Array<Object>} Array of available device objects
   */
  _getAvailableDevices() {
    const availableDevices = [];
    
    for (const user of this.users.values()) {
      for (const device of user.devices.values()) {
        if (device.isConnected && !device.isProcessing && device.batteryLevel > 0.05) {
          availableDevices.push(device);
        }
      }
    }
    
    return availableDevices;
  }
  
  /**
   * Get the total number of devices in the room
   * @returns {number} Total device count
   */
  getTotalDeviceCount() {
    let count = 0;
    for (const user of this.users.values()) {
      count += user.devices.size;
    }
    return count;
  }
  
  /**
   * Get the count of connected devices
   * @returns {number} Connected device count
   */
  getConnectedDeviceCount() {
    let count = 0;
    for (const user of this.users.values()) {
      for (const device of user.devices.values()) {
        if (device.isConnected) {
          count++;
        }
      }
    }
    return count;
  }
  
  /**
   * Add an event to the room event log
   * @param {string} type - Type of event
   * @param {Object} data - Event data
   */
  addEvent(type, data) {
    const event = {
      type,
      timestamp: new Date(),
      data
    };
    
    this.events.push(event);
    
    // Limit event log size
    if (this.events.length > 1000) {
      this.events = this.events.slice(-1000);
    }
  }
  
  /**
   * Get room information
   * @returns {Object} Room information
   */
  getRoomInfo() {
    return {
      roomId: this.roomId,
      name: this.name,
      description: this.description,
      createdAt: this.createdAt,
      createdBy: this.createdBy,
      isPublic: this.isPublic,
      userCount: this.users.size,
      deviceCount: this.getTotalDeviceCount(),
      connectedDeviceCount: this.getConnectedDeviceCount(),
      taskQueueLength: this.taskQueue.length,
      currentTask: this.currentTask ? {
        taskId: this.currentTask.taskId,
        progress: this.currentTask.progress,
        status: this.currentTask.status
      } : null,
      stats: this.stats
    };
  }
  
  /**
   * Get detailed room status including all users and devices
   * @returns {Object} Detailed room status
   */
  getDetailedStatus() {
    const userDetails = [];
    
    for (const user of this.users.values()) {
      const deviceDetails = [];
      
      for (const device of user.devices.values()) {
        deviceDetails.push({
          deviceId: device.deviceId,
          model: device.model,
          isConnected: device.isConnected,
          isProcessing: device.isProcessing,
          batteryLevel: device.batteryLevel,
          computationsPerformed: device.computationsPerformed,
          joinedAt: device.joinedAt
        });
      }
      
      userDetails.push({
        userId: user.userId,
        username: user.username,
        isAdmin: user.isAdmin,
        joinedAt: user.joinedAt,
        deviceCount: user.devices.size,
        devices: deviceDetails
      });
    }
    
    return {
      roomInfo: this.getRoomInfo(),
      users: userDetails,
      currentTask: this.currentTask,
      recentCompletedTasks: this.completedTasks.slice(-5),
      recentEvents: this.events.slice(-20)
    };
  }
  
  /**
   * Update a device's status
   * @param {string} userId - ID of the user who owns the device
   * @param {string} deviceId - ID of the device to update
   * @param {Object} updates - Status updates to apply
   * @returns {boolean} Whether the update was successful
   */
  updateDeviceStatus(userId, deviceId, updates) {
    const user = this.users.get(userId);
    if (!user) return false;
    
    const device = user.devices.get(deviceId);
    if (!device) return false;
    
    // Apply updates
    if (updates.batteryLevel !== undefined) {
      device.batteryLevel = updates.batteryLevel;
    }
    
    if (updates.connectionQuality !== undefined) {
      device.connectionQuality = updates.connectionQuality;
    }
    
    if (updates.isConnected !== undefined) {
      device.isConnected = updates.isConnected;
    }
    
    device.lastActive = new Date();
    
    return true;
  }
}

/**
 * Manager for computation rooms
 */
class RoomManager {
  constructor() {
    this.rooms = new Map(); // roomId -> Room
    this.users = new Map(); // userId -> { username, rooms: Set<roomId> }
  }

  /**
   * Get a list of all rooms
   * @param {boolean} publicOnly - Whether to only include public rooms
   * @returns {Array<Object>} Array of room info objects
  */
  listRooms(publicOnly = true) {
    const roomList = [];
    
    for (const [roomId, room] of this.rooms.entries()) {
      if (!publicOnly || room.isPublic) {
        roomList.push(room.getRoomInfo());
      }
    }
    
    return roomList;
  }
  
  /**
   * Create a new computation room
   * @param {Object} options - Room options
   * @returns {string} The room ID
   */
  createRoom(options = {}) {
    const roomId = options.roomId || `room-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    if (this.rooms.has(roomId)) {
      throw new Error(`Room with ID ${roomId} already exists`);
    }
    
    const room = new ComputationRoom(roomId, options);
    this.rooms.set(roomId, room);
    
    console.log(`Created new room: ${room.name} (${roomId})`);
    return roomId;
  }

    /**
   * Add a device to a specific room
   * @param {string} userId - ID of the user adding the device
   * @param {string} roomId - ID of the room to add the device to
   * @param {string} deviceId - Unique identifier for the device
   * @param {Object} deviceInfo - Additional information about the device
   * @returns {boolean} Whether the device was successfully added
   */
  addDeviceToRoom(userId, roomId, deviceId, deviceInfo = {}) {
    // Find the room
    const room = this.rooms.get(roomId);
    if (!room) {
      console.error(`Cannot add device: Room ${roomId} not found`);
      return false;
    }
    
    // Find the user
    const user = this.users.get(userId);
    if (!user) {
      console.error(`Cannot add device: User ${userId} not found`);
      return false;
    }
    
    // Check if the user is in the room
    if (!room.users.has(userId)) {
      console.error(`Cannot add device: User ${userId} not in room ${roomId}`);
      return false;
    }
    
    // Prepare device info with defaults
    const completeDeviceInfo = {
      model: deviceInfo.model || 'Unknown',
      batteryLevel: deviceInfo.batteryLevel !== undefined ? deviceInfo.batteryLevel : 1.0,
      connectionQuality: deviceInfo.connectionQuality !== undefined ? deviceInfo.connectionQuality : 0.95,
      worker: deviceInfo.worker || null,
      workerPath: deviceInfo.workerPath || null
    };
    
    // Use the room's existing addDevice method
    return room.addDevice(userId, deviceId, completeDeviceInfo);
  }
  
  /**
   * Get a room by ID
   * @param {string} roomId - The room ID
   * @returns {ComputationRoom|null} The room or null if not found
   */
  getRoom(roomId) {
    return this.rooms.get(roomId) || null;
  }

    /**
   * Check if a room is empty
   * @param {string} roomId - The room ID to check
   * @returns {boolean} Whether the room is empty
   */
  isRoomEmpty(roomId) {
    const room = this.rooms.get(roomId);
    if (!room) {
      console.error(`Room ${roomId} not found`);
      return true; // Consider non-existent rooms as "empty"
    }
    
    return room.users.size === 0;
  }

  /**
   * Automatically clean up empty rooms
   * @returns {Array<string>} List of room IDs that were deleted
   */
  cleanEmptyRooms() {
    const deletedRooms = [];
    
    for (const [roomId, room] of this.rooms.entries()) {
      if (this.isRoomEmpty(roomId)) {
        // Delete the room
        if (this.deleteRoom(roomId)) {
          deletedRooms.push(roomId);
        }
      }
    }
    
    if (deletedRooms.length > 0) {
      console.log(`Cleaned up ${deletedRooms.length} empty room(s): ${deletedRooms.join(', ')}`);
    }
    
    return deletedRooms;
  }

  /**
 * Remove a user from a room
 * @param {string} userId - The user ID
 * @param {string} roomId - The room ID
 * @returns {boolean} Whether the user successfully left the room
 */
  leaveRoom(userId, roomId) {
    const room = this.rooms.get(roomId);
    if (!room) {
      console.error(`Room ${roomId} not found`);
      return false;
    }
    
    const user = this.users.get(userId);
    if (!user) {
      console.error(`User ${userId} not found`);
      return false;
    }
    
    // Check if the user is actually in this room
    if (!room.users.has(userId)) {
      console.log(`User ${userId} is not in room ${roomId}`);
      return false;
    }
    
    // Remove the user from the room
    const removeResult = room.removeUser(userId);
    
    if (removeResult) {
      // Remove the room from the user's room list
      user.rooms.delete(roomId);      
      console.log(`User ${userId} left room ${roomId}`);
    }

    // Check if the room is now empty
    if (room.users.size === 0) {
      // Delete the room if it's now empty
      this.deleteRoom(roomId);
      console.log(`Room ${roomId} deleted due to being empty`);
    }
    
    return true;
  }
  
  /**
   * Delete a room
   * @param {string} roomId - ID of the room to delete
   * @returns {boolean} Whether the room was deleted
   */
  deleteRoom(roomId) {
    const room = this.rooms.get(roomId);
    if (!room) return false;
    
    // Remove all users from the room
    for (const userId of [...room.users.keys()]) {
      room.removeUser(userId);
      
      // Update user's room list
      const user = this.users.get(userId);
      if (user) {
        user.rooms.delete(roomId);
      }
    }
    
    // Delete the room
    this.rooms.delete(roomId);
    console.log(`Deleted room ${roomId}`);
    
    return true;
  }
  
  /**
   * Register a user
   * @param {string} userId - Unique user ID
   * @param {Object} userInfo - User information
   * @returns {boolean} Whether the user was registered
   */
  registerUser(userId, userInfo = {}) {
    if (this.users.has(userId)) {
      // Update existing user
      const user = this.users.get(userId);
      user.username = userInfo.username || user.username;
      return false;
    }
    
    // Create new user
    this.users.set(userId, {
      userId,
      username: userInfo.username || `User-${userId.substring(0, 6)}`,
      registeredAt: new Date(),
      rooms: new Set()
    });
    
    console.log(`Registered user ${userInfo.username || userId}`);
    return true;
  }
  
    /**
   * Get rooms a user has joined
   * @param {string} userId - The user ID
   * @returns {Array<Object>} Array of room info objects
   */
  getUserRooms(userId) {
    if (!userId) return [];
    
    const userRooms = [];
    
    // Iterate through all rooms
    for (const [roomId, room] of this.rooms.entries()) {
      // Check if the user is a member of this room
      if (room.users.has(userId)) {
        userRooms.push(room.getRoomInfo());
      }
    }
    
    return userRooms;
  }

    /**
   * Queue a task in a specific room
   * @param {string} roomId - The ID of the room to queue the task in
   * @param {Object} taskData - Task parameters
   * @param {number} taskData.a - The scalar value
   * @param {Array<number>} taskData.xArray - The x vector
   * @param {Array<number>} taskData.yArray - The y vector
   * @param {string} initiatedBy - User ID who initiated the task
   * @returns {string|null} The task ID if queued successfully, null otherwise
   */
  queueTaskInRoom(roomId, taskData, initiatedBy) {
    // Find the room
    const room = this.rooms.get(roomId);
    if (!room) {
      console.error(`Cannot queue task: Room ${roomId} not found`);
      return null;
    }

    // Validate task parameters
    if (typeof taskData.a !== 'number') {
      console.error('Invalid scalar value');
      return null;
    }

    if (!Array.isArray(taskData.xArray) || !Array.isArray(taskData.yArray)) {
      console.error('Invalid vector inputs');
      return null;
    }

    if (taskData.xArray.length !== taskData.yArray.length) {
      console.error('Input vectors must have the same length');
      return null;
    }

    // Verify the user initiating the task is in the room
    const initiatingUser = this.users.get(initiatedBy);
    if (!initiatingUser || !room.users.has(initiatedBy)) {
      console.error(`User ${initiatedBy} is not in room ${roomId}`);
      return null;
    }

    // Prepare task data
    const fullTaskData = {
      ...taskData,
      initiatedBy
    };

    // Queue the task using the room's queueTask method
    return room.queueTask(fullTaskData);
  }


  /**
   * Get detailed status of a specific room
   * @param {string} roomId - The room ID
   * @returns {Object|null} Detailed room status or null if room not found
   */
  getRoomStatus(roomId) {
    const room = this.rooms.get(roomId);
    if (!room) {
      console.error(`Room ${roomId} not found`);
      return null;
    }
    
    return room.getDetailedStatus();
  }

  
  /**
   * Join a user to a room
   * @param {string} userId - The user ID
   * @param {string} roomId - The room ID
   * @param {Object} options - Join options
   * @returns {boolean} Whether the user joined successfully
   */
  joinRoom(userId, roomId, options = {}) {
    const room = this.rooms.get(roomId);
    if (!room) {
      console.error(`Room ${roomId} not found`);
      return false;
    }
    
    // Register user if not already registered
    if (!this.users.has(userId)) {
      this.registerUser(userId, options);
    }
    
    const user = this.users.get(userId);
    
    // Add user to room
    if (room.addUser(userId, {
      username: user.username,
      isAdmin: options.isAdmin || false
    })) {
      // Update user's room list
      user.rooms.add(roomId);
      return true;
    }
    
    return false;
  }

    /**
   * Remove a device from a specific room
   * @param {string} userId - ID of the user removing the device
   * @param {string} roomId - ID of the room to remove the device from
   * @param {string} deviceId - Unique identifier for the device
   * @returns {boolean} Whether the device was successfully removed
   */
  removeDeviceFromRoom(userId, roomId, deviceId) {
    // Find the room
    const room = this.rooms.get(roomId);
    if (!room) {
      console.error(`Cannot remove device: Room ${roomId} not found`);
      return false;
    }
    
    // Find the user
    const user = this.users.get(userId);
    if (!user) {
      console.error(`Cannot remove device: User ${userId} not found`);
      return false;
    }
    
    // Check if the user is in the room
    if (!room.users.has(userId)) {
      console.error(`Cannot remove device: User ${userId} not in room ${roomId}`);
      return false;
    }
    
    // Remove the device using the room's method
    const removeResult = room.removeDevice(userId, deviceId);
    
    if (removeResult) {
      console.log(`Device ${deviceId} removed from room ${roomId} by user ${userId}`);
      return true;
    }
    
    return false;
  }
}

module.exports = RoomManager;