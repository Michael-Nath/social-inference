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
    
    // User's single device
    this.device = null;
    
    // Pending requests
    this.pendingRequests = new Map(); // requestId -> { resolve, reject, timeout }
    this.requestTimeout = options.requestTimeout || 10000;
    
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
  
  _generateUserId() {
    return 'user-' + Date.now().toString(36) + Math.random().toString(36).substring(2, 9);
  }
  
  connect() {
    if (this.connected && this.socket && this.socket.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    if (this._connectionPromise) {
      return this._connectionPromise;
    }

    this._connectionPromise = new Promise((resolve, reject) => {
      try {
        if (this.socket) {
          try {
            this.socket.close();
          } catch (e) {
            // Ignore errors when closing
          }
        }

        this.socket = new WebSocket(this.serverUrl);
        
        this.socket.onopen = () => {
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
            
            this._connectionPromise = null;
            resolve();
            return true;
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
        }, 10000);
        
        this.socket.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this._handleServerMessage(message);
          } catch (error) {
            console.error('Error parsing message:', error);
          }
        };
        
        this.socket.onerror = (error) => {
          this._emitEvent('error', { error: 'WebSocket error', details: error });
          clearTimeout(connectionTimeout);
          this._connectionPromise = null;
          reject(new Error('WebSocket connection error'));
        };
        
        this.socket.onclose = (event) => {
          if (this.connected) {
            this.connected = false;
            this._emitEvent('disconnect', { 
              code: event.code, 
              reason: event.reason,
              wasClean: event.wasClean
            });
          }
          
          clearTimeout(connectionTimeout);
          this._connectionPromise = null;
          
          if (!event.wasClean) {
            this._attemptReconnect();
          }
        };
        
      } catch (error) {
        this._connectionPromise = null;
        reject(error);
        this._attemptReconnect();
      }
    });
    
    return this._connectionPromise;
  }
  
  disconnect() {
    if (!this.connected || !this.socket) {
      return;
    }
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    try {
      this.socket.close(1000, 'Client disconnected');
    } catch (error) {
      console.error('Error closing WebSocket:', error);
    }
    
    this.connected = false;
  }
  
  _attemptReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    this.reconnectAttempts++;
    
    if (this.reconnectAttempts > this.maxReconnectAttempts) {
      this._emitEvent('error', { 
        error: 'Reconnect failed',
        details: `Maximum reconnect attempts reached`
      });
      return;
    }
    
    // Exponential backoff
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
    
    this.reconnectTimeout = setTimeout(() => {
      this.connect().catch(() => {
        // Error handling is done in the connect method
      });
    }, delay);
  }
  
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
        reject(error);
      }
    });
  }
  
  _sendRequest(type, data = {}) {
    return this.connect().then(() => {
      const requestId = `req-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          if (this.pendingRequests.has(requestId)) {
            this.pendingRequests.delete(requestId);
            reject(new Error(`Request ${type} timed out`));
          }
        }, this.requestTimeout);
        
        this.pendingRequests.set(requestId, { resolve, reject, timeout, type });
        
        this._sendMessage({
          type,
          requestId,
          ...data
        }).catch(error => {
          clearTimeout(timeout);
          this.pendingRequests.delete(requestId);
          reject(error);
        });
      });
    });
  }
  
  _handleServerMessage(message) {
    if (message.type === 'response' && message.requestId) {
      this._handleRequestResponse(message);
      return;
    }
    
    if (this.eventHandlers._authSuccess) {
      for (const handler of this.eventHandlers._authSuccess) {
        if (handler(message)) {
          return;
        }
      }
    }
    
    if (message.type === 'deviceComputation') {
      this._handleDeviceComputation(message);
      return;
    }
    
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
        this._emitEvent('message', message);
    }
  }
  
  _handleDeviceComputation(message) {
    const { deviceId, computation } = message;
    
    if (this.device && this.device.deviceId === deviceId) {
      if (this.device.worker && typeof this.device.worker.postMessage === 'function') {
        try {
          this.device.worker.postMessage(computation);
        } catch (error) {
          this._sendRequest('computationError', {
            roomId: computation.roomId,
            deviceId: deviceId,
            taskId: computation.taskId,
            chunkIndex: computation.data.chunkIndex,
            error: error.message
          }).catch(err => {
            console.error('Failed to report computation error:', err);
          });
        }
      } else {
        this._sendRequest('computationError', {
          roomId: computation.roomId,
          deviceId: deviceId,
          taskId: computation.taskId,
          chunkIndex: computation.data.chunkIndex,
          error: 'No worker available'
        }).catch(err => {
          console.error('Failed to report error:', err);
        });
      }
    }
  }
  
  _handleRequestResponse(response) {
    const { requestId, success, data, error } = response;
    
    const pendingRequest = this.pendingRequests.get(requestId);
    if (!pendingRequest) {
      return;
    }
    
    clearTimeout(pendingRequest.timeout);
    this.pendingRequests.delete(requestId);
    
    if (success) {
      pendingRequest.resolve(data);
    } else {
      pendingRequest.reject(new Error(error || 'Unknown error'));
    }
  }
  
  _handleRoomUpdated(message) {
    const { roomId, roomInfo } = message;
    
    if (this.joinedRooms.has(roomId)) {
      this.joinedRooms.set(roomId, roomInfo);
      this._emitEvent('roomUpdated', message);
    }
  }
  
  _handleRoomJoined(message) {
    const { roomId, roomInfo } = message;
    
    this.joinedRooms.set(roomId, roomInfo);
    this._emitEvent('roomJoined', message);
  }
  
  _handleRoomLeft(message) {
    const { roomId } = message;
    
    this.joinedRooms.delete(roomId);
    this._emitEvent('roomLeft', message);
  }
  
  _handleDeviceUpdated(message) {
    const { deviceId, deviceInfo } = message;
    
    const workerRef = this.device ? this.device.worker : null;

    if (deviceInfo && deviceInfo.userId === this.userId) {
      this.device = {
        deviceId,
        ...deviceInfo,
        status: 'connected',
        isConnected: true,
        worker: workerRef,
      };
    }
    
    this._emitEvent('deviceUpdated', message);
  }
  
  _handleDeviceRemoved(message) {
    const { userId } = message;
    
    if (userId === this.userId) {
      this.device = null;
    }
    
    this._emitEvent('deviceRemoved', message);
  }
  
  _handleDeviceStatusUpdated(message) {
    const { userId, updates } = message;
    
    if (userId === this.userId && this.device) {
      Object.assign(this.device, updates);
    }
    
    this._emitEvent('deviceStatusUpdated', message);
  }
  
  _handleDeviceMessage(event) {
    const message = event.data;
    
    if (!this.device) {
      return;
    }
    
    switch (message.type) {
      case 'result':
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
        if (message.deviceStats) {
          this.device.batteryLevel = message.deviceStats.batteryLevel;
          this.device.connectionQuality = message.deviceStats.connectionQuality;
          this.device.isConnected = message.deviceStats.isConnected;
          
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
  
  // Public API methods
  
  listRooms() {
    return this._sendRequest('listRooms')
      .then(response => response.rooms);
  }
  
  createRoom(options = {}) {
    return this._sendRequest('createRoom', options)
      .then(response => {
        if (response.roomInfo) {
          this.joinedRooms.set(response.roomId, response.roomInfo);
        }
        return response;
      });
  }
  
  joinRoom(roomId, options = {}) {
    return this._sendRequest('joinRoom', { roomId, options })
      .then(response => {
        if (response.joined) {
          this.joinedRooms.set(roomId, response.roomInfo);
        }
        return response.roomInfo;
      });
  }
  
  leaveRoom(roomId) {
    return this._sendRequest('leaveRoom', { roomId })
      .then(response => {
        if (response.left) {
          this.joinedRooms.delete(roomId);
        }
        return response.left;
      });
  }
  
  getRoomStatus(roomId) {
    return this._sendRequest('getRoomStatus', { roomId })
      .then(response => response.status);
  }
  
  setDevice(deviceInfo = {}) {
    let worker = deviceInfo.worker;
    if (!worker && typeof window !== 'undefined' && window.Worker) {
      try {
        worker = new Worker(new URL(deviceInfo.workerPath || './client/iphone-worker.js', import.meta.url), {
          type: 'module'
        });
        
        worker.onmessage = (event) => {
          this._handleDeviceMessage(event);
        };
      } catch (error) {
        console.warn('Error creating worker:', error);
      }
    }
    
    const serverDeviceInfo = {
      model: deviceInfo.model || 'iPhone13',
      batteryLevel: deviceInfo.batteryLevel !== undefined ? deviceInfo.batteryLevel : 1.0,
      connectionQuality: deviceInfo.connectionQuality !== undefined ? deviceInfo.connectionQuality : 0.95,
      userId: this.userId,
      isConnected: true
    };
    
    return this._sendRequest('setDevice', { deviceInfo: serverDeviceInfo })
      .then(response => {
        if (response.added) {
          const deviceId = response.deviceId;
          
          this.device = {
            deviceId,
            model: serverDeviceInfo.model,
            batteryLevel: serverDeviceInfo.batteryLevel,
            connectionQuality: serverDeviceInfo.connectionQuality,
            worker,
            addedAt: new Date(),
            status: 'connected',
            isConnected: true
          };

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
  
  removeDevice() {
    if (!this.device) {
      return Promise.resolve(false);
    }
    
    return this._sendRequest('removeDevice')
      .then(response => {
        if (response.removed) {
          if (this.device.worker && typeof this.device.worker.terminate === 'function') {
            try {
              this.device.worker.terminate();
            } catch (error) {
              console.warn('Error terminating worker:', error);
            }
          }
          
          this.device = null;
        }
        return response.removed;
      });
  }
  
  queueTask(roomId, a, xArray, yArray) {
    return this._sendRequest('queueTask', {
      roomId,
      a,
      xArray,
      yArray
    })
      .then(response => response.taskId);
  }
  
  on(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event].push(handler);
    }
  }
  
  off(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
    }
  }
  
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