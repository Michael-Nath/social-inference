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
    this.maxUsers = options.maxUsers || 50;
    
    // Connected users and their single device
    this.users = new Map(); // userId -> { username, joinedAt, device, isAdmin }
    
    // Task management
    this.taskQueue = [];
    this.completedTasks = [];
    this.currentTask = null;
    this.taskResults = new Map();
    
    // Basic stats
    this.stats = {
      totalComputations: 0
    };
  }
  
  addUser(userId, userInfo = {}) {
    if (this.users.has(userId)) {
      return false;
    }
    
    const user = {
      userId,
      username: userInfo.username || `User-${userId.substring(0, 6)}`,
      joinedAt: new Date(),
      device: null,
      isAdmin: userInfo.isAdmin || false
    };
    
    this.users.set(userId, user);
    return true;
  }
  
  removeUser(userId) {
    const user = this.users.get(userId);
    if (!user) {
      return false;
    }

    // Handle the device if the user has one
    if (user.device) {
      this._handleDeviceDisconnectionDuringTask(user.device);
    }
    
    // Remove the user
    this.users.delete(userId);
    return true;
  }
  
  setUserDevice(userId, deviceId, deviceInfo = {}) {
    const user = this.users.get(userId);
    if (!user) {
      return false;
    }
    
    // Check if user already has a device and it's different from this one
    if (user.device && user.device.deviceId !== deviceId) {
      // Disconnect the old device first
      if (user.device.isProcessing && this.currentTask) {
        this._handleDeviceDisconnectionDuringTask(user.device);
      }
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
      worker: deviceInfo.worker
    };
    
    // Set the user's device
    user.device = device;
    return true;
  }

  removeUserDevice(userId) {
    const user = this.users.get(userId);
    if (!user || !user.device) {
      return false;
    }
    
    const device = user.device;
    
    // If the device is currently processing a task, handle the interruption
    if (device.isProcessing && this.currentTask) {
      this._handleDeviceDisconnectionDuringTask(device);
    }
    
    // Remove the device
    user.device = null;
    return true;
  }
  
  _handleDeviceDisconnectionDuringTask(device) {
    // If we have a current task and this device was working on it
    if (this.currentTask && device.isProcessing) {
      // Find which chunk this device was processing
      const taskChunk = this.currentTask.deviceAssignments.find(
        assignment => assignment.deviceId === device.deviceId
      );
      
      if (taskChunk) {
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
    
    // If no task is currently running, start this one
    if (!this.currentTask) {
      this._startNextTask();
    }
    
    return taskId;
  }
  
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
  }
  
  _divideTaskIntoChunks() {
    const task = this.currentTask;
    const vectorLength = task.xArray.length;
    
    // Count available devices
    const availableDevices = this._getAvailableDevices();
    const deviceCount = Math.max(1, availableDevices.length);
    
    // Determine chunk size
    const idealElementsPerChunk = 10000;
    const maxChunks = Math.min(deviceCount * 2, Math.ceil(vectorLength / idealElementsPerChunk));
    const chunkSize = Math.ceil(vectorLength / maxChunks);
    
    // Create chunks
    task.totalChunks = Math.ceil(vectorLength / chunkSize);
    task.unassignedChunks = [];
    
    for (let i = 0; i < task.totalChunks; i++) {
      task.unassignedChunks.push(i);
    }
  }
  
  _assignPendingChunks() {
    const task = this.currentTask;
    if (!task || task.unassignedChunks.length === 0) {
      return;
    }
    
    const availableDevices = this._getAvailableDevices();
    if (availableDevices.length === 0) {
      return;
    }
    
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
    }
  }
  
  _sendComputationToDevice(device, taskId, a, xSlice, ySlice, chunkIndex, startIndex) {
    // Create computation data to be sent to the client
    const computationData = {
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
    };
    
    // Store this computation request with the device for the server to handle
    device.pendingComputation = computationData;
    return true;
  }
  
  handleDeviceComputationComplete(deviceId, userId, taskId, chunkIndex, result, stats) {
    const user = this.users.get(userId);
    if (!user) {
      return false;
    }
    
    if (!user.device || user.device.deviceId !== deviceId) {
      return false;
    }
    
    const device = user.device;
    
    if (!this.currentTask || this.currentTask.taskId !== taskId) {
      return false;
    }
    
    // Find the assignment
    const assignment = this.currentTask.deviceAssignments.find(
      a => a.deviceId === deviceId && a.chunkIndex === chunkIndex
    );
    
    if (!assignment) {
      return false;
    }
    
    // Mark the assignment as completed
    assignment.status = 'completed';
    assignment.completedAt = new Date();
    assignment.computationTime = stats.computationTime;
    
    // Update device stats
    device.isProcessing = false;
    device.isConnected = true;
    device.lastActive = new Date();
    device.computationsPerformed++;
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
    
    // Check if we should assign more chunks
    if (this.currentTask.unassignedChunks.length > 0) {
      this._assignPendingChunks();
    }
    
    // Check if the task is complete
    if (this.currentTask.chunksCompleted >= this.currentTask.totalChunks) {
      this._finalizeTask();
    }
    return true;
  }
  
  _handleDeviceComputationError(device, taskId, chunkIndex, error) {
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
  
  _finalizeTask() {
    const task = this.currentTask;
    
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
    
    // Update room statistics
    this.stats.totalComputations++;
    
    // Add to completed tasks
    this.completedTasks.push(task);
    
    // Clear current task
    this.currentTask = null;
    
    // Start the next task if one is queued
    if (this.taskQueue.length > 0) {
      this._startNextTask();
    }
  }
  
  _getAvailableDevices() {
    const availableDevices = [];
    
    for (const user of this.users.values()) {
      if (user.device && 
          user.device.isConnected && 
          !user.device.isProcessing && 
          user.device.batteryLevel > 0.05) {
        availableDevices.push(user.device);
      }
    }
    
    return availableDevices;
  }
  
  getTotalDeviceCount() {
    let count = 0;
    for (const user of this.users.values()) {
      if (user.device) count++;
    }
    return count;
  }
  
  getConnectedDeviceCount() {
    let count = 0;
    for (const user of this.users.values()) {
      if (user.device && user.device.isConnected) {
        count++;
      }
    }
    return count;
  }
  
  getRoomInfo() {
    const connectedDeviceCount = this.getConnectedDeviceCount();
    
    return {
      roomId: this.roomId,
      name: this.name,
      description: this.description,
      createdAt: this.createdAt,
      createdBy: this.createdBy,
      isPublic: this.isPublic,
      userCount: this.users.size,
      deviceCount: this.getTotalDeviceCount(),
      connectedDeviceCount,
      taskQueueLength: this.taskQueue.length,
      currentTask: this.currentTask ? {
        taskId: this.currentTask.taskId,
        progress: this.currentTask.progress,
        status: this.currentTask.status
      } : null,
      stats: this.stats
    };
  }
  
  getDetailedStatus() {
    const userDetails = [];
    
    for (const user of this.users.values()) {
      let deviceDetails = null;
      
      if (user.device) {
        deviceDetails = {
          deviceId: user.device.deviceId,
          model: user.device.model,
          isConnected: user.device.isConnected === true,
          isProcessing: user.device.isProcessing,
          batteryLevel: user.device.batteryLevel,
          computationsPerformed: user.device.computationsPerformed,
          joinedAt: user.device.joinedAt
        };
      }
      
      userDetails.push({
        userId: user.userId,
        username: user.username,
        isAdmin: user.isAdmin,
        joinedAt: user.joinedAt,
        hasDevice: !!user.device && user.device.isConnected === true,
        device: deviceDetails
      });
    }
    
    return {
      roomInfo: this.getRoomInfo(),
      users: userDetails,
      currentTask: this.currentTask,
      recentCompletedTasks: this.completedTasks.slice(-5)
    };
  }
  
  updateDeviceStatus(userId, updates) {
    const user = this.users.get(userId);
    if (!user || !user.device) return false;
    
    const device = user.device;
    
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
    this.users = new Map(); // userId -> { username, rooms: Set<roomId>, device: {} }
  }

  listRooms(publicOnly = true) {
    const roomList = [];
    
    for (const [roomId, room] of this.rooms.entries()) {
      if (!publicOnly || room.isPublic) {
        roomList.push(room.getRoomInfo());
      }
    }
    
    return roomList;
  }
  
  createRoom(options = {}) {
    const roomId = options.roomId || `room-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    if (this.rooms.has(roomId)) {
      throw new Error(`Room with ID ${roomId} already exists`);
    }
    
    const room = new ComputationRoom(roomId, options);
    this.rooms.set(roomId, room);
    
    return roomId;
  }

  setUserDevice(userId, deviceId, deviceInfo = {}) {
    const user = this.users.get(userId);
    if (!user) {
      return false;
    }

    // Ensure isConnected is set
    deviceInfo.isConnected = true;

    // Store device information with the user globally
    user.device = {
      deviceId,
      model: deviceInfo.model || 'Unknown',
      batteryLevel: deviceInfo.batteryLevel !== undefined ? deviceInfo.batteryLevel : 1.0,
      connectionQuality: deviceInfo.connectionQuality !== undefined ? deviceInfo.connectionQuality : 0.95,
      isConnected: true,
      worker: deviceInfo.worker || null
    };
    
    // Update device in all rooms user has joined
    let success = true;
    for (const roomId of user.rooms) {
      const room = this.rooms.get(roomId);
      if (room) {
        const result = room.setUserDevice(userId, deviceId, {
          ...deviceInfo,
          isConnected: true,
          worker: deviceInfo.worker
        });
        if (!result) success = false;
      }
    }
    
    return success;
  }
  
  getRoom(roomId) {
    return this.rooms.get(roomId) || null;
  }

  isRoomEmpty(roomId) {
    const room = this.rooms.get(roomId);
    if (!room) {
      return true;
    }
    
    return room.users.size === 0;
  }

  cleanEmptyRooms() {
    const deletedRooms = [];
    
    for (const [roomId, room] of this.rooms.entries()) {
      if (this.isRoomEmpty(roomId)) {
        if (this.deleteRoom(roomId)) {
          deletedRooms.push(roomId);
        }
      }
    }
    
    return deletedRooms;
  }

  leaveRoom(userId, roomId) {
    const room = this.rooms.get(roomId);
    if (!room) {
      return false;
    }
    
    const user = this.users.get(userId);
    if (!user) {
      return false;
    }
    
    if (!room.users.has(userId)) {
      return false;
    }
    
    // Remove the user from the room
    const removeResult = room.removeUser(userId);
    
    if (removeResult) {
      // Remove the room from the user's room list
      user.rooms.delete(roomId);
    }

    // Check if the room is now empty
    if (room.users.size === 0) {
      this.deleteRoom(roomId);
    }
    
    return true;
  }
  
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
    return true;
  }
  
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
      rooms: new Set(),
      device: null
    });
    
    return true;
  }
  
  getUserRooms(userId) {
    if (!userId) return [];
    
    const userRooms = [];
    
    for (const [roomId, room] of this.rooms.entries()) {
      if (room.users.has(userId)) {
        userRooms.push(room.getRoomInfo());
      }
    }
    
    return userRooms;
  }

  queueTaskInRoom(roomId, taskData, initiatedBy) {
    const room = this.rooms.get(roomId);
    if (!room) {
      return null;
    }

    if (typeof taskData.a !== 'number') {
      return null;
    }

    if (!Array.isArray(taskData.xArray) || !Array.isArray(taskData.yArray)) {
      return null;
    }

    if (taskData.xArray.length !== taskData.yArray.length) {
      return null;
    }

    const initiatingUser = this.users.get(initiatedBy);
    if (!initiatingUser || !room.users.has(initiatedBy)) {
      return null;
    }

    const fullTaskData = {
      ...taskData,
      initiatedBy
    };

    return room.queueTask(fullTaskData);
  }

  getRoomStatus(roomId) {
    const room = this.rooms.get(roomId);
    if (!room) {
      return null;
    }
    
    return room.getDetailedStatus();
  }
  
  joinRoom(userId, roomId, options = {}) {
    const room = this.rooms.get(roomId);
    if (!room) {
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
      // If user has a device, add it to this room
      if (user.device) {
        room.setUserDevice(userId, user.device.deviceId, user.device);
      }
      
      // Update user's room list
      user.rooms.add(roomId);
      return true;
    }
    
    return false;
  }

  removeUserDevice(userId) {
    const user = this.users.get(userId);
    if (!user || !user.device) {
      return false;
    }
    
    // Remove the device from all rooms the user is in
    let success = true;
    for (const roomId of user.rooms) {
      const room = this.rooms.get(roomId);
      if (room) {
        const result = room.removeUserDevice(userId);
        if (!result) success = false;
      }
    }
    
    // Remove the device from the user
    user.device = null;
    
    return success;
  }
  
  updateUserDeviceStatus(userId, updates) {
    const user = this.users.get(userId);
    if (!user || !user.device) {
      return false;
    }
    
    // Update the user's global device information
    if (updates.batteryLevel !== undefined) {
      user.device.batteryLevel = updates.batteryLevel;
    }
    
    if (updates.connectionQuality !== undefined) {
      user.device.connectionQuality = updates.connectionQuality;
    }
    
    if (updates.isConnected !== undefined) {
      user.device.isConnected = updates.isConnected;
    }
    
    // Update in all rooms
    let success = true;
    for (const roomId of user.rooms) {
      const room = this.rooms.get(roomId);
      if (room) {
        const result = room.updateDeviceStatus(userId, updates);
        if (!result) success = false;
      }
    }
    
    return success;
  }
}

module.exports = RoomManager;