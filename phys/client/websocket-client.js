// client/websocket-client.js
// WebSocket client implementation for SAXPY Room Computing with one device per user

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
    
    // User's single device
    this.device = null;
    
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
      'deviceUpdated': [],
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
    
    // Handle device computation requests
    if (message.type === 'deviceComputation') {
      this._handleDeviceComputation(message);
      return;
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
        
      case 'deviceUpdated':
        this._handleDeviceUpdated(message);
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
        
      case 'error':
        this._emitEvent('error', { error: message.error, details: message.details });
        break;
        
      default:
        // For unknown message types, emit a generic message event
        this._emitEvent('message', message);
    }
  }
  
  /**
   * Handle computation request from server
   * @private
   * @param {Object} message - The computation message
   */
  _handleDeviceComputation(message) {
    const { deviceId, computation } = message;
    
    // Check if this is our device
    if (this.device && this.device.deviceId === deviceId) {
      // Check if we have a worker
      if (this.device.worker && typeof this.device.worker.postMessage === 'function') {
        try {
          // Forward the computation to our local worker
          this.device.worker.postMessage(computation);
          console.log(`Forwarded computation to local worker for device ${deviceId}`);
        } catch (error) {
          console.error(`Error forwarding computation to worker:`, error);
          
          // Report error back to server
          this._sendRequest('computationError', {
            roomId: computation.roomId,
            deviceId: deviceId,
            taskId: computation.taskId,
            chunkIndex: computation.data.chunkIndex,
            error: error.message
          }).catch(err => {
            console.error('Failed to report computation error to server:', err);
          });
        }
      } else {
        console.error(`Received computation for device ${deviceId} but no worker is available`);
        
        // Report error back to server
        this._sendRequest('computationError', {
          roomId: computation.roomId,
          deviceId: deviceId,
          taskId: computation.taskId,
          chunkIndex: computation.data.chunkIndex,
          error: 'No worker available'
        }).catch(err => {
          console.error('Failed to report computation error to server:', err);
        });
      }
    } else {
      console.warn(`Received computation for unknown device ${deviceId}`);
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
    this._emitEvent('roomLeft', message);
  }
  
  /**
   * Handle device updated message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleDeviceUpdated(message) {
    const { deviceId, deviceInfo } = message;
    
    
    const workerRef = this.device ? this.device.worker : null;

    // Update our local device information
    if (deviceInfo && deviceInfo.userId === this.userId) {
      this.device = {
        deviceId,
        ...deviceInfo,
        status: 'connected',
        isConnected: true, // Explicitly set isConnected to true
        worker: workerRef,
      };
      
      
      console.log(`Device updated: ${deviceId} connected = ${this.device.isConnected}`);
    }
    
    this._emitEvent('deviceUpdated', message);
  }
  
  /**
   * Handle device removed message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleDeviceRemoved(message) {
    const { userId } = message;
    
    // If this is our device, clear it
    if (userId === this.userId) {
      this.device = null;
    }
    
    this._emitEvent('deviceRemoved', message);
  }
  
  /**
   * Handle device status updated message
   * @private
   * @param {Object} message - The message from the server
   */
  _handleDeviceStatusUpdated(message) {
    const { userId, updates } = message;
    
    // Update our device if it's ours
    if (userId === this.userId && this.device) {
      Object.assign(this.device, updates);
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
    
    // Only process if this is our device
    if (this.device && this.device.deviceId === deviceId) {
      if (this.device.worker && typeof this.device.worker.postMessage === 'function') {
        // Forward the request to the device worker
        this.device.worker.postMessage({
          command: 'compute',
          taskId,
          data
        });
      }
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
   * Set the user's device
   * @param {Object} deviceInfo - Device information
   * @returns {Promise<string>} Promise that resolves with the device ID
   */
  setDevice(deviceInfo = {}) {
    // Create worker for the device if not provided
    let worker = deviceInfo.worker;
    if (!worker && typeof window !== 'undefined' && window.Worker) {
      try {
        worker = new Worker(new URL(deviceInfo.workerPath || './client/iphone-worker.js', import.meta.url), {
          type: 'module'
        });

        
        // Set up message handler
        worker.onmessage = (event) => {
          this._handleDeviceMessage(event);
        };
      } catch (error) {
        console.warn('Error creating worker:', error);
        // Continue without a worker
      }
    }
    
    // Prepare device info for the server
    const serverDeviceInfo = {
      model: deviceInfo.model || 'iPhone13',
      batteryLevel: deviceInfo.batteryLevel !== undefined ? deviceInfo.batteryLevel : 1.0,
      connectionQuality: deviceInfo.connectionQuality !== undefined ? deviceInfo.connectionQuality : 0.95,
      userId: this.userId,
      isConnected: true // Explicitly set isConnected to true
    };
    
    // Send request to server
    return this._sendRequest('setDevice', {
      deviceInfo: serverDeviceInfo
    })
      .then(response => {
        if (response.added) {
          const deviceId = response.deviceId;
          
          // Store the device locally
          this.device = {
            deviceId,
            model: serverDeviceInfo.model,
            batteryLevel: serverDeviceInfo.batteryLevel,
            connectionQuality: serverDeviceInfo.connectionQuality,
            worker,
            addedAt: new Date(),
            status: 'connected',
            isConnected: true // Explicitly set isConnected to true
          };

          // Initialize the worker
          if (worker) {
            try {
              worker.postMessage({
                command: 'initialize',
                data: {
                  deviceId,
                  model: serverDeviceInfo.model,
                  batteryLevel: serverDeviceInfo.batteryLevel,
                  connectionQuality: serverDeviceInfo.connectionQuality
                }
              });
            } catch (error) {
              console.warn('Error initializing worker:', error);
            }
          }
          
          return deviceId;
        }
        
        return response.deviceId;
      });
  }
  
  /**
   * Remove the user's device
   * @returns {Promise<boolean>} Promise that resolves with whether the device was removed
   */
  removeDevice() {
    if (!this.device) {
      return Promise.resolve(false);
    }
    
    return this._sendRequest('removeDevice')
      .then(response => {
        if (response.removed) {
          // Safely terminate the worker if it exists and has a terminate method
          if (this.device.worker && typeof this.device.worker.terminate === 'function') {
            try {
              this.device.worker.terminate();
            } catch (error) {
              console.warn('Error terminating worker:', error);
            }
          }
          
          // Remove local device
          this.device = null;
        }
        return response.removed;
      });
  }
  
  /**
   * Handle a message from a device worker
   * @private
   * @param {MessageEvent} event - The message event
   */
  _handleDeviceMessage(event) {
    const message = event.data;
    
    if (!this.device) {
      console.warn(`Received message from device worker but no device is registered`);
      return;
    }
    
    // Handle different message types
    switch (message.type) {
      case 'result':
        // Send computation result to server
        console.log(message.roomId);
        this._sendRequest('computationResult', {
          roomId: message.roomId,
          deviceId: this.device.deviceId,
          taskId: message.taskId,
          chunkIndex: message.chunkIndex,
          result: message.result,
          stats: message.deviceStats
        }).catch(error => {
          console.error(`Error sending computation result:`, error);
        });
        break;
        
      case 'status':
        // Update device status
        if (message.deviceStats) {
          this.device.batteryLevel = message.deviceStats.batteryLevel;
          this.device.connectionQuality = message.deviceStats.connectionQuality;
          this.device.isConnected = message.deviceStats.isConnected;
          
          // Update server with new status
          this._sendRequest('updateDeviceStatus', {
            updates: {
              batteryLevel: this.device.batteryLevel,
              connectionQuality: this.device.connectionQuality,
              isConnected: this.device.isConnected
            }
          }).catch(error => {
            console.error(`Error updating device status:`, error);
          });
        }
        break;
        
      case 'error':
        console.error(`Error from device:`, message.error);
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