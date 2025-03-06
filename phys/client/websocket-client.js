// client/websocket-client.js
// WebSocket client implementation for SAXPY Room Computing

class SAXPYWebSocketClient {
  constructor(options = {}) {
    this.userId = options.userId || this._generateUserId();
    this.username = options.username || `User-${this.userId.substring(0, 6)}`;
    this.serverUrl = options.serverUrl || 'ws://localhost:8080';
    
    // Connection state
    this.connected = false;
    this.socket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
    this.reconnectTimeout = null;
    
    // Rooms the client has joined
    this.joinedRooms = new Map(); // roomId -> roomInfo
    
    // Devices managed by this client
    this.devices = new Map(); // deviceId -> deviceInfo
    
    // Pending requests
    this.pendingRequests = new Map(); // requestId -> { resolve, reject, timeout }
    this.requestTimeout = options.requestTimeout || 10000; // 10 seconds
    
    // Event handlers
    this.eventHandlers = {
      'connect': [],
      'disconnect': [],
      'error': [],
      'roomJoined': [],
      'roomLeft': [],
      'roomUpdated': [],
      'deviceAdded': [],
      'deviceRemoved': [],
      'deviceStatusUpdated': [],
      'taskQueued': [],
      'taskStarted': [],
      'taskProgress': [],
      'taskCompleted': [],
      'userJoined': [],
      'userLeft': [],
      'message': []
    };
    
    // Auto-connect if requested
    if (options.autoConnect) {
      this.connect();
    }
  }
  
  /**
   * Generate a random user ID
   * @private
   * @returns {string} A random user ID
   */
  _generateUserId() {
    return 'user-' + Date.now().toString(36) + Math.random().toString(36).substring(2, 9);
  }
  
  /**
   * Connect to the WebSocket server
   * @returns {Promise<void>} Promise that resolves when connected
   */
  /**
 * Connect to the WebSocket server
 * @returns {Promise<void>} Promise that resolves when connected
 */
  connect() {
    // If already connected, return resolved promise
    if (this.connected && this.socket && this.socket.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    // If connection is in progress, wait for it
    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    // Create new connection promise
    this._connectionPromise = new Promise((resolve, reject) => {
      console.log(`Connecting to SAXPY WebSocket server at ${this.serverUrl}...`);
      
      try {
        // Close existing socket if any
        if (this.socket) {
          try {
            this.socket.close();
          } catch (e) {
            // Ignore errors when closing
          }
        }

        this.socket = new WebSocket(this.serverUrl);
        
        // Setup event handlers
        this.socket.onopen = () => {
          console.log('WebSocket connection established, sending auth...');
          
          // Send authentication
          const authMessage = {
            type: 'auth',
            userId: this.userId,
            username: this.username
          };
          
          try {
            this.socket.send(JSON.stringify(authMessage));
          } catch (e) {
            reject(new Error(`Failed to send auth message: ${e.message}`));
            this._connectionPromise = null;
            return;
          }
        };
        
        // Add temporary auth success handler
        const authHandler = (message) => {
          if (message.type === 'auth_success') {
            this.connected = true;
            this.reconnectAttempts = 0;
            this._emitEvent('connect', { userId: this.userId });
            
            // Remove temporary handler
            if (this.eventHandlers._authSuccess) {
              this.eventHandlers._authSuccess = this.eventHandlers._authSuccess.filter(
                handler => handler !== authHandler
              );
            }
            
            console.log('Successfully authenticated with server');
            this._connectionPromise = null;
            resolve();
            return true; // Signal that this message was handled
          }
          return false;
        };
        
        // Add temporary auth success handler
        if (!this.eventHandlers._authSuccess) {
          this.eventHandlers._authSuccess = [];
        }
        this.eventHandlers._authSuccess.push(authHandler);
        
        // Set timeout for connection
        const connectionTimeout = setTimeout(() => {
          if (!this.connected) {
            const error = new Error('Connection timeout');
            this._connectionPromise = null;
            reject(error);
          }
        }, 10000); // 10 second timeout
        
        // Setup remaining event handlers
        this.socket.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this._handleServerMessage(message);
          } catch (error) {
            console.error('Error parsing message:', error, 'Raw message:', event.data);
            console.error('Error parsing message:', error);
          }
        };
        
        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this._emitEvent('error', { error: 'WebSocket error', details: error });
          clearTimeout(connectionTimeout);
          this._connectionPromise = null;
          reject(new Error('WebSocket connection error'));
        };
        
        this.socket.onclose = (event) => {
          if (this.connected) {
            this.connected = false;
            console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
            this._emitEvent('disconnect', { 
              code: event.code, 
              reason: event.reason,
              wasClean: event.wasClean
            });
          }
          
          clearTimeout(connectionTimeout);
          this._connectionPromise = null;
          
          // Attempt to reconnect if not a clean close and wasn't rejected already
          if (!event.wasClean) {
            this._attemptReconnect();
          }
        };
        
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        this._connectionPromise = null;
        reject(error);
        
        // Attempt to reconnect
        this._attemptReconnect();
      }
    });
    
    return this._connectionPromise;
  }
  
  /**
   * Disconnect from the WebSocket server
   */
  disconnect() {
    if (!this.connected || !this.socket) {
      return;
    }
    
    // Clear any reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    // Close the connection
    try {
      this.socket.close(1000, 'Client disconnected');
    } catch (error) {
      console.error('Error closing WebSocket:', error);
    }
    
    this.connected = false;
  }
  
  /**
   * Attempt to reconnect to the server
   * @private
   */
  _attemptReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    this.reconnectAttempts++;
    
    if (this.reconnectAttempts > this.maxReconnectAttempts) {
      console.error(`Maximum reconnect attempts (${this.maxReconnectAttempts}) reached. Giving up.`);
      this._emitEvent('error', { 
        error: 'Reconnect failed',
        details: `Maximum reconnect attempts (${this.maxReconnectAttempts}) reached`
      });
      return;
    }
    
    // Exponential backoff
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
    
    this.reconnectTimeout = setTimeout(() => {
      console.log(`Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.connect().catch(() => {
        // Error handling is done in the connect method
      });
    }, delay);
  }
  
  /**
   * Send a message to the server
   * @private
   * @param {Object} message - The message to send
   * @returns {Promise<void>} Promise that resolves when the message is sent
   */
  _sendMessage(message) {
    return new Promise((resolve, reject) => {
      if (!this.connected || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
        reject(new Error('Not connected to server'));
        return;
      }
      
      try {
        this.socket.send(JSON.stringify(message));
        resolve();
      } catch (error) {
        console.error('Error sending message:', error);
        reject(error);
      }
    });
  }
  
  /**
   * Send a request to the server and wait for a response
   * @private
   * @param {string} type - Request type
   * @param {Object} data - Request data
   * @returns {Promise<any>} Promise that resolves with the response
   */
  _sendRequest(type, data = {}) {
    // First ensure we are connected
    return this.connect().then(() => {
      const requestId = `req-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      
      return new Promise((resolve, reject) => {
        // Set timeout for the request
        const timeout = setTimeout(() => {
          if (this.pendingRequests.has(requestId)) {
            this.pendingRequests.delete(requestId);
            reject(new Error(`Request ${type} timed out`));
          }
        }, this.requestTimeout);
        
        this.pendingRequests.set(requestId, { resolve, reject, timeout, type });
        
        // Send the request to the server
        this._sendMessage({
          type,
          requestId,
          ...data
        }).catch(error => {
          // Clean up if sending fails
          clearTimeout(timeout);
          this.pendingRequests.delete(requestId);
          reject(error);
        });
      });
    });
  }
  
  /**
   * Handle messages from the server
   * @private
   * @param {Object} message - The message from the server
   */
  _handleServerMessage(message) {
    // First check if this is a response to a request
    if (message.type === 'response' && message.requestId) {
      this._handleRequestResponse(message);
      return;
    }
    
    // Check for auth success messages (temporary handler)
    if (this.eventHandlers._authSuccess) {
      for (const handler of this.eventHandlers._authSuccess) {
        if (handler(message)) {
          return; // Message was handled
        }
      }
    }
    
    // Handle event messages
    switch (message.type) {
      case 'auth_success':
        this.connected = true;
        this._emitEvent('connect', { userId: this.userId });
        break;
        
      case 'roomUpdated':
        this._handleRoomUpdated(message);
        break;
        
      case 'roomJoined':
        this._handleRoomJoined(message);
        break;
        
      case 'roomLeft':
        this._handleRoomLeft(message);
        break;
        
      case 'userJoined':
        this._emitEvent('userJoined', message);
        break;
        
      case 'userLeft':
        this._emitEvent('userLeft', message);
        break;
        
      case 'deviceAdded':
        this._handleDeviceAdded(message);
        break;
        
      case 'deviceRemoved':
        this._handleDeviceRemoved(message);
        break;
        
      case 'deviceStatusUpdated':
        this._handleDeviceStatusUpdated(message);
        break;
        
      case 'taskQueued':
        this._emitEvent('taskQueued', message);
        break;
        
      case 'taskStarted':
        this._emitEvent('taskStarted', message);
        break;
        
      case 'taskProgress':
        this._emitEvent('taskProgress', message);
        break;
        
      case 'taskCompleted':
        this._emitEvent('taskCompleted', message);
        break;
        
      case 'deviceRequest':
        this._handleDeviceRequest(message);
        break;
        
      case 'error':
        this._emitEvent('error', { error: message.error, details: message.details });
        break;
        
      default:
        // For unknown message types, emit a generic message event
        this._emitEvent('message', message);
    }
  }
  
  /**
   * Handle a response to a request
   * @private
   * @param {Object} response - The response from the server
   */
  _handleRequestResponse(response) {
    const { requestId, success, data, error } = response;
    
    const pendingRequest = this.pendingRequests.get(requestId);
    if (!pendingRequest) {
      console.warn(`Received response for unknown request ${requestId}`);
      return;
    }
    
    // Clear the timeout
    clearTimeout(pendingRequest.timeout);
    
    // Remove from pending requests
    this.pendingRequests.delete(requestId);
    
    // Resolve or reject the promise
    if (success) {
      pendingRequest.resolve(data);
    } else {
      pendingRequest.reject(new Error(error || 'Unknown error'));
    }
  }
  
  /**
   * Handle room updated message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleRoomUpdated(message) {
    const { roomId, roomInfo } = message;
    
    if (this.joinedRooms.has(roomId)) {
      this.joinedRooms.set(roomId, roomInfo);
      this._emitEvent('roomUpdated', message);
    }
  }
  
  /**
   * Handle room joined message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleRoomJoined(message) {
    const { roomId, roomInfo } = message;
    
    this.joinedRooms.set(roomId, roomInfo);
    this._emitEvent('roomJoined', message);
  }
  
  /**
   * Handle room left message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleRoomLeft(message) {
    const { roomId } = message;
    
    this.joinedRooms.delete(roomId);
    
    // Remove all devices from this room
    for (const [deviceId, device] of this.devices.entries()) {
      if (device.roomId === roomId) {
        this.devices.delete(deviceId);
        this._emitEvent('deviceRemoved', { roomId, deviceId });
      }
    }
    
    this._emitEvent('roomLeft', message);
  }
  
  /**
   * Handle device added message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleDeviceAdded(message) {
    const { roomId, deviceId, deviceInfo } = message;
    
    // Only store the device if it belongs to this client
    if (deviceInfo && deviceInfo.userId === this.userId) {
      this.devices.set(deviceId, {
        ...deviceInfo,
        deviceId,
        roomId,
        status: 'connected'
      });
    }
    
    this._emitEvent('deviceAdded', message);
  }
  
  /**
   * Handle device removed message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleDeviceRemoved(message) {
    const { roomId, deviceId } = message;
    
    // Remove from local devices
    this.devices.delete(deviceId);
    
    this._emitEvent('deviceRemoved', message);
  }
  
  /**
   * Handle device status updated message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleDeviceStatusUpdated(message) {
    const { roomId, deviceId, updates } = message;
    
    // Update local device if we have it
    const device = this.devices.get(deviceId);
    if (device) {
      Object.assign(device, updates);
    }
    
    this._emitEvent('deviceStatusUpdated', message);
  }
  
  /**
   * Handle device request message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleDeviceRequest(message) {
    const { deviceId, taskId, data } = message;
    
    const device = this.devices.get(deviceId);
    if (!device) {
      console.warn(`Received request for unknown device ${deviceId}`);
      return;
    }
    
    // Process the task on the device
    if (device.worker && typeof device.worker.postMessage === 'function') {
      // Forward the request to the device worker
      device.worker.postMessage({
        command: 'compute',
        taskId,
        data
      });
    }
  }
  
  /**
   * List available computation rooms
   * @returns {Promise<Array<Object>>} Promise that resolves with an array of room info objects
   */
  listRooms() {
    return this._sendRequest('listRooms')
      .then(response => response.rooms);
  }
  
  /**
   * Create a new computation room
   * @param {Object} options - Room options
   * @returns {Promise<Object>} Promise that resolves with room info
   */
  createRoom(options = {}) {
    return this._sendRequest('createRoom', options)
      .then(response => {
        if (response.roomInfo) {
          this.joinedRooms.set(response.roomId, response.roomInfo);
        }
        return response;
      });
  }
  
  /**
   * Join a computation room
   * @param {string} roomId - The room ID
   * @param {Object} options - Join options
   * @returns {Promise<Object>} Promise that resolves with room info
   */
  joinRoom(roomId, options = {}) {
    return this._sendRequest('joinRoom', { roomId, options })
      .then(response => {
        if (response.joined) {
          this.joinedRooms.set(roomId, response.roomInfo);
        }
        return response.roomInfo;
      });
  }
  
  /**
   * Leave a computation room
   * @param {string} roomId - The room ID
   * @returns {Promise<boolean>} Promise that resolves with whether the room was left
   */
  leaveRoom(roomId) {
    return this._sendRequest('leaveRoom', { roomId })
      .then(response => {
        if (response.left) {
          this.joinedRooms.delete(roomId);
          
          // Remove all devices from this room
          for (const [deviceId, device] of this.devices.entries()) {
            if (device.roomId === roomId) {
              this.removeDevice(deviceId);
            }
          }
        }
        return response.left;
      });
  }
  
  /**
   * Get the status of a room
   * @param {string} roomId - The room ID
   * @returns {Promise<Object>} Promise that resolves with room status
   */
  getRoomStatus(roomId) {
    return this._sendRequest('getRoomStatus', { roomId })
      .then(response => response.status);
  }
  
  /**
   * Add a device to a room
   * @param {string} roomId - The room ID
   * @param {Object} deviceInfo - Device information
   * @returns {Promise<string>} Promise that resolves with the device ID
   */
  addDevice(roomId, deviceInfo = {}) {
    // Generate a device ID if not provided
    const deviceId = deviceInfo.deviceId || `device-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    // Create worker for the device if not provided
    let worker = deviceInfo.worker;
    if (!worker && typeof window !== 'undefined' && window.Worker) {
      worker = new Worker(new URL(deviceInfo.workerPath || './iphone-worker.js', import.meta.url), {
        type: 'module'
      });
      
      // Set up message handler
      worker.onmessage = (event) => {
        this._handleDeviceMessage(deviceId, event);
      };
    }
    
    // Prepare device info for the server
    const serverDeviceInfo = {
      model: deviceInfo.model || 'iPhone13',
      batteryLevel: deviceInfo.batteryLevel !== undefined ? deviceInfo.batteryLevel : 1.0,
      connectionQuality: deviceInfo.connectionQuality !== undefined ? deviceInfo.connectionQuality : 0.95,
      userId: this.userId
    };
    
    // Send request to server
    return this._sendRequest('addDevice', {
      roomId,
      deviceId,
      deviceInfo: serverDeviceInfo
    })
      .then(response => {
        if (response.added) {
          // Store the device locally
          this.devices.set(deviceId, {
            deviceId,
            roomId,
            model: serverDeviceInfo.model,
            batteryLevel: serverDeviceInfo.batteryLevel,
            connectionQuality: serverDeviceInfo.connectionQuality,
            worker,
            addedAt: new Date(),
            status: 'connected'
          });
          
          // Initialize the worker
          if (worker) {
            worker.postMessage({
              command: 'initialize',
              data: {
                deviceId,
                model: serverDeviceInfo.model,
                batteryLevel: serverDeviceInfo.batteryLevel,
                connectionQuality: serverDeviceInfo.connectionQuality
              }
            });
          }
        }
        return deviceId;
      });
  }
  
  /**
   * Remove a device from a room
   * @param {string} deviceId - The device ID
   * @returns {Promise<boolean>} Promise that resolves with whether the device was removed
   */
  removeDevice(deviceId) {
    const device = this.devices.get(deviceId);
    if (!device) {
      return Promise.resolve(false);
    }
    
    const roomId = device.roomId;
    
    return this._sendRequest('removeDevice', { roomId, deviceId })
      .then(response => {
        if (response.removed) {
          // Terminate the worker if we have one
          if (device.worker) {
            device.worker.terminate();
          }
          
          // Remove from local devices
          this.devices.delete(deviceId);
        }
        return response.removed;
      });
  }
  
  /**
   * Handle a message from a device worker
   * @private
   * @param {string} deviceId - The device ID
   * @param {MessageEvent} event - The message event
   */
  _handleDeviceMessage(deviceId, event) {
    const message = event.data;
    const device = this.devices.get(deviceId);
    
    if (!device) {
      console.warn(`Received message from unknown device ${deviceId}`);
      return;
    }
    
    // Handle different message types
    switch (message.type) {
      case 'result':
        // Send computation result to server
        this._sendRequest('computationResult', {
          roomId: device.roomId,
          deviceId,
          taskId: message.taskId,
          chunkIndex: message.chunkIndex,
          result: message.result,
          stats: message.deviceStats
        }).catch(error => {
          console.error(`Error sending computation result for device ${deviceId}:`, error);
        });
        break;
        
      case 'status':
        // Update device status
        if (message.deviceStats) {
          device.batteryLevel = message.deviceStats.batteryLevel;
          device.connectionQuality = message.deviceStats.connectionQuality;
          device.isConnected = message.deviceStats.isConnected;
          
          // Update server with new status
          this._sendRequest('updateDeviceStatus', {
            roomId: device.roomId,
            deviceId,
            updates: {
              batteryLevel: device.batteryLevel,
              connectionQuality: device.connectionQuality,
              isConnected: device.isConnected
            }
          }).catch(error => {
            console.error(`Error updating device status for ${deviceId}:`, error);
          });
        }
        break;
        
      case 'error':
        console.error(`Error from device ${deviceId}:`, message.error);
        break;
    }
  }
  
  /**
   * Queue a SAXPY computation task in a room
   * @param {string} roomId - The room ID
   * @param {number} a - The scalar value
   * @param {Array<number>} xArray - The x vector
   * @param {Array<number>} yArray - The y vector
   * @returns {Promise<string>} Promise that resolves with the task ID
   */
  queueTask(roomId, a, xArray, yArray) {
    return this._sendRequest('queueTask', {
      roomId,
      a,
      xArray,
      yArray
    })
      .then(response => response.taskId);
  }
  
  /**
   * Register an event handler
   * @param {string} event - The event name
   * @param {Function} handler - The event handler
   */
  on(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event].push(handler);
    }
  }
  
  /**
   * Remove an event handler
   * @param {string} event - The event name
   * @param {Function} handler - The handler to remove
   */
  off(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
    }
  }
  
  /**
   * Emit an event
   * @private
   * @param {string} event - The event name
   * @param {Object} data - The event data
   */
  _emitEvent(event, data) {
    if (this.eventHandlers[event]) {
      for (const handler of this.eventHandlers[event]) {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in ${event} handler:`, error);
        }
      }
    }
  }
}

export default SAXPYWebSocketClient;