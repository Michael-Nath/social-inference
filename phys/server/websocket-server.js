// server/websocket-server.js
// WebSocket server implementation for SAXPY Room Computing

const WebSocket = require('ws');
const http = require('http');
const RoomManager = require('./room-manager');

class SAXPYWebSocketServer {
  constructor(options = {}) {
    this.port = options.port || 8080;
    this.server = http.createServer();
    this.wss = new WebSocket.Server({ server: this.server });
    this.roomManager = new RoomManager();
    
    // Track connected clients
    this.clients = new Map(); // userId -> WebSocket
    
    // Set up WebSocket connection handler
    this._setupWebSocketHandlers();
    
    // Optional cleanup interval to remove inactive rooms
    this.cleanupInterval = setInterval(() => {
      this.roomManager.cleanupInactiveRooms();
    }, 30 * 60 * 1000); // 30 minutes
  }
  
  /**
   * Start the WebSocket server
   */
  start() {
    this.server.listen(this.port, () => {
      console.log(`SAXPY WebSocket server is running on port ${this.port}`);
    });
  }
  
  /**
   * Stop the WebSocket server
   */
  stop() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    
    this.server.close(() => {
      console.log('SAXPY WebSocket server stopped');
    });
    
    // Close all client connections
    for (const client of this.wss.clients) {
      client.terminate();
    }
  }
  
  /**
   * Setup WebSocket connection handlers
   * @private
   */
  _setupWebSocketHandlers() {
    this.wss.on('connection', (ws) => {
      console.log('New client connected');
      
      let userId = null;
      let authenticated = false;
      
      ws.on('message', (message) => {
        try {
          const data = JSON.parse(message);
          
          // Handle authentication first
          if (data.type === 'auth') {
            userId = data.userId;
            authenticated = true;
            
            // Store the connection
            this.clients.set(userId, ws);
            
            // Send acknowledgment
            this._sendToClient(ws, {
              type: 'auth_success',
              userId
            });
            
            console.log(`Client authenticated: ${userId}`);
          } else if (!authenticated) {
            // Reject non-authenticated requests
            this._sendToClient(ws, {
              type: 'error',
              error: 'Authentication required'
            });
            return;
          } else {
            // Process message based on type
            this._handleClientMessage(ws, userId, data);
          }
        } catch (error) {
          console.error('Error processing message:', error);
          this._sendToClient(ws, {
            type: 'error',
            error: 'Invalid message format'
          });
        }
      });
      
      ws.on('close', () => {
        console.log(`Client disconnected${userId ? ': ' + userId : ''}`);
        
        if (userId) {
          // Clean up client connections
          this.clients.delete(userId);
          
          // Handle disconnection from rooms
          this._handleClientDisconnect(userId);
        }
      });
      
      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });
    });
  }
  
  /**
   * Handle client messages
   * @private
   * @param {WebSocket} ws - The WebSocket connection
   * @param {string} userId - The user ID
   * @param {Object} message - The message data
   */
  _handleClientMessage(ws, userId, message) {
    const { type, requestId } = message;
    
    switch (type) {
      case 'listRooms':
        this._handleListRooms(ws, userId, requestId);
        break;
        
      case 'createRoom':
        this._handleCreateRoom(ws, userId, requestId, message);
        break;
        
      case 'joinRoom':
        this._handleJoinRoom(ws, userId, requestId, message);
        break;
        
      case 'leaveRoom':
        this._handleLeaveRoom(ws, userId, requestId, message);
        break;
        
      case 'setDevice':
        this._handleSetDevice(ws, userId, requestId, message);
        break;
        
      case 'removeDevice':
        this._handleRemoveDevice(ws, userId, requestId, message);
        break;
        
      case 'getRoomStatus':
        this._handleGetRoomStatus(ws, userId, requestId, message);
        break;
        
      case 'queueTask':
        this._handleQueueTask(ws, userId, requestId, message);
        break;
        
      case 'computationResult':
        this._handleComputationResult(ws, userId, requestId, message);
        break;
        
      case 'computationError':
        this._handleComputationError(ws, userId, requestId, message);
        break;
        
      case 'updateDeviceStatus':
        this._handleUpdateDeviceStatus(ws, userId, requestId, message);
        break;
        
      default:
        this._sendToClient(ws, {
          type: 'error',
          requestId,
          error: `Unknown message type: ${type}`
        });
    }
  }
  
  /**
   * Handle computation error from client
   * @private
   */
  _handleComputationError(ws, userId, requestId, message) {
    try {
      const { roomId, deviceId, taskId, chunkIndex, error } = message;
      
      if (!roomId || !deviceId || !taskId || chunkIndex === undefined) {
        throw new Error('Invalid computation error parameters');
      }
      
      // Get the room
      const room = this.roomManager.getRoom(roomId);
      if (!room) {
        throw new Error(`Room ${roomId} not found`);
      }
      
      // Find the user
      const user = room.users.get(userId);
      if (!user) {
        throw new Error(`User ${userId} not found in room ${roomId}`);
      }
      
      // Check if this is the user's device
      if (!user.device || user.device.deviceId !== deviceId) {
        throw new Error(`Device ${deviceId} not found for user ${userId}`);
      }
      
      // Handle the error through the room's error handler
      const device = user.device;
      room._handleDeviceComputationError(device, taskId, chunkIndex, error || 'Client reported error');
      
      // Response to the client
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: true,
        data: { handled: true }
      });
      
      // Check if there are any pending computations to send
      this._sendPendingComputations(room);
      
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle client disconnection
   * @private
   * @param {string} userId - The user ID
   */
  _handleClientDisconnect(userId) {
    // Get rooms the user has joined
    const rooms = this.roomManager.getUserRooms(userId);
    
    // Mark the user's device as disconnected
    this.roomManager.updateUserDeviceStatus(userId, {
      isConnected: false
    });
    
    // Notify other room members about the device status change
    for (const room of rooms) {
      this._broadcastToRoom(room.roomId, {
        type: 'deviceStatusUpdated',
        roomId: room.roomId,
        userId,
        updates: { isConnected: false },
        roomInfo: this.roomManager.getRoom(room.roomId).getRoomInfo()
      }, userId); // Exclude the disconnected user
    }
  }
  
  /**
   * Handle list rooms request
   * @private
   */
  _handleListRooms(ws, userId, requestId) {
    const rooms = this.roomManager.listRooms();
    
    this._sendToClient(ws, {
      type: 'response',
      requestId,
      success: true,
      data: {
        rooms
      }
    });
  }
  
  /**
   * Handle create room request
   * @private
   */
  _handleCreateRoom(ws, userId, requestId, message) {
    try {
      const { name, description, isPublic } = message;
      
      const roomId = this.roomManager.createRoom({
        name: name || `Room ${Date.now()}`,
        description: description || '',
        isPublic: isPublic !== false,
        createdBy: userId
      });
      
      // Join the user to the room automatically
      this.roomManager.joinRoom(userId, roomId, { isAdmin: true });
      
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: true,
        data: {
          roomId,
          roomInfo: this.roomManager.getRoom(roomId).getRoomInfo()
        }
      });
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle join room request
   * @private
   */
  _handleJoinRoom(ws, userId, requestId, message) {
    try {
      const { roomId } = message;
      
      if (!roomId) {
        throw new Error('Room ID is required');
      }
      
      const joined = this.roomManager.joinRoom(userId, roomId);
      
      if (joined) {
        const roomInfo = this.roomManager.getRoom(roomId).getRoomInfo();
        
        // Notify other room members
        this._broadcastToRoom(roomId, {
          type: 'userJoined',
          roomId,
          userId,
          roomInfo
        }, userId);
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            roomId,
            joined: true,
            roomInfo
          }
        });
        
        // Also send a room-joined event for client-side event handlers
        this._sendToClient(ws, {
          type: 'roomJoined',
          roomId,
          roomInfo
        });
      } else {
        throw new Error('Failed to join room');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle leave room request
   * @private
   */
  _handleLeaveRoom(ws, userId, requestId, message) {
    try {
      const { roomId } = message;
      
      if (!roomId) {
        throw new Error('Room ID is required');
      }
      
      // Get room info before leaving for notification
      const room = this.roomManager.getRoom(roomId);
      if (!room) {
        throw new Error('Room not found');
      }
      
      const left = this.roomManager.leaveRoom(userId, roomId);
      
      if (left) {
        // Notify other room members
        this._broadcastToRoom(roomId, {
          type: 'userLeft',
          roomId,
          userId,
          roomInfo: room.getRoomInfo()
        });
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            roomId,
            left: true
          }
        });
        
        // Also send a room-left event for client-side event handlers
        this._sendToClient(ws, {
          type: 'roomLeft',
          roomId
        });
      } else {
        throw new Error('Failed to leave room');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle set device request
   * @private
   */
  _handleSetDevice(ws, userId, requestId, message) {
    try {
      const { deviceInfo } = message;
      
      if (!deviceInfo) {
        throw new Error('Device information is required');
      }
      
      // Ensure isConnected is set to true
      deviceInfo.isConnected = true;
      
      // Generate a device ID if not provided
      const deviceId = deviceInfo.deviceId || `device-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
      
      // Set the user's device
      const added = this.roomManager.setUserDevice(userId, deviceId, deviceInfo || {});
      
      if (added) {
        // Get user's rooms to send updates
        const rooms = this.roomManager.getUserRooms(userId);
        
        // Notify all room members about the new device
        for (const room of rooms) {
          const updatedRoomInfo = this.roomManager.getRoom(room.roomId).getRoomInfo();
          console.log(`Broadcasting updated room info with connected devices: ${updatedRoomInfo.connectedDeviceCount}`);
          
          this._broadcastToRoom(room.roomId, {
            type: 'deviceUpdated',
            roomId: room.roomId,
            userId,
            deviceId: deviceId,
            deviceInfo,
            roomInfo: updatedRoomInfo
          }, userId);
          
          // Also broadcast a general room update to refresh counters
          this._broadcastToRoom(room.roomId, {
            type: 'roomUpdated',
            roomId: room.roomId,
            roomInfo: updatedRoomInfo
          });
        }
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            deviceId: deviceId,
            added: true
          }
        });
        
        // Also send a deviceUpdated event for client-side event handlers
        this._sendToClient(ws, {
          type: 'deviceUpdated',
          deviceId: deviceId,
          deviceInfo: { ...deviceInfo, deviceId: deviceId }
        });
      } else {
        throw new Error('Failed to set device');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle remove device request
   * @private
   */
  _handleRemoveDevice(ws, userId, requestId, message) {
    try {
      const removed = this.roomManager.removeUserDevice(userId);
      
      if (removed) {
        // Get user's rooms to send updates
        const rooms = this.roomManager.getUserRooms(userId);
        
        // Notify all room members about the device removal
        for (const room of rooms) {
          this._broadcastToRoom(room.roomId, {
            type: 'deviceRemoved',
            roomId: room.roomId,
            userId,
            roomInfo: this.roomManager.getRoom(room.roomId).getRoomInfo()
          }, userId);
        }
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            removed: true
          }
        });
        
        // Also send a deviceRemoved event for client-side event handlers
        this._sendToClient(ws, {
          type: 'deviceRemoved',
          userId
        });
      } else {
        throw new Error('Failed to remove device');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle get room status request
   * @private
   */
  _handleGetRoomStatus(ws, userId, requestId, message) {
    try {
      const { roomId } = message;
      
      if (!roomId) {
        throw new Error('Room ID is required');
      }
      
      const status = this.roomManager.getRoomStatus(roomId);
      
      if (status) {
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            roomId,
            status
          }
        });
      } else {
        throw new Error('Room not found');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle queue task request
   * @private
   */
  _handleQueueTask(ws, userId, requestId, message) {
    try {
      const { roomId, a, xArray, yArray } = message;
      
      if (!roomId) {
        throw new Error('Room ID is required');
      }
      
      if (typeof a !== 'number' || !Array.isArray(xArray) || !Array.isArray(yArray)) {
        throw new Error('Invalid task parameters');
      }
      
      if (xArray.length !== yArray.length) {
        throw new Error('Input arrays must have the same length');
      }
      
      const taskId = this.roomManager.queueTaskInRoom(roomId, {
        a,
        xArray,
        yArray
      }, userId);
      
      if (taskId) {
        // Get the task info
        const room = this.roomManager.getRoom(roomId);
        const task = room.currentTask || room.taskQueue.find(t => t.taskId === taskId);
        
        // Notify all room members about the new task
        this._broadcastToRoom(roomId, {
          type: 'taskQueued',
          roomId,
          taskId,
          userId,
          taskInfo: {
            initiatedBy: userId,
            vectorLength: xArray.length,
            status: task ? task.status : 'queued'
          }
        });
        
        // Check if any device assignments need to be sent to clients
        this._sendPendingComputations(room);
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            roomId,
            taskId,
            queued: true
          }
        });
      } else {
        throw new Error('Failed to queue task');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Send pending computations to client devices
   * @private
   * @param {ComputationRoom} room - The room with pending computations
   */
  _sendPendingComputations(room) {
    // Iterate through all users in the room
    for (const user of room.users.values()) {
      // Check if the user has a device with pending computation
      if (user.device && user.device.pendingComputation) {
        const device = user.device;
        const computation = device.pendingComputation;
        
        // Clear the pending computation to avoid sending it multiple times
        device.pendingComputation = null;
        
        // Find the client connection for this user
        const ws = this.clients.get(user.userId);
        if (ws && ws.readyState === WebSocket.OPEN) {
          // Send the computation request to the client
          this._sendToClient(ws, {
            type: 'deviceComputation',
            deviceId: device.deviceId,
            computation: computation
          });
          
          console.log(`Sent computation request to client for device ${device.deviceId}`);
        } else {
          console.error(`Cannot find client connection for user ${user.userId} to send computation`);
          
          // Handle the error - mark the computation as failed
          if (room.currentTask && computation.taskId === room.currentTask.taskId) {
            room._handleDeviceComputationError(
              device, 
              computation.taskId, 
              computation.data.chunkIndex, 
              'Client not connected'
            );
          }
        }
      }
    }
  }
  
  /**
   * Handle computation result
   * @private
   */
  _handleComputationResult(ws, userId, requestId, message) {
    try {
      const { roomId, deviceId, taskId, chunkIndex, result, stats } = message;
      
      if (!Array.isArray(result)) {
        throw new Error('Invalid computation result parameters');
      }
      
      const room = this.roomManager.getRoom(roomId);
      if (!room) {
        throw new Error(`Room ${roomId} not found`);
      }
      
      const handled = room.handleDeviceComputationComplete(deviceId, userId, taskId, chunkIndex, result, stats || {});
      
      if (handled) {
        // Get the room to check if task is completed
        const task = room.currentTask;
        
        if (task && task.taskId === taskId) {
          // Send task progress update to all room members
          this._broadcastToRoom(roomId, {
            type: 'taskProgress',
            roomId,
            taskId,
            progress: task.progress,
            chunksCompleted: task.chunksCompleted,
            totalChunks: task.totalChunks
          });
        }
        
        // If task is now completed, notify all members with the results
        const completedTask = room.completedTasks.find(t => t.taskId === taskId);
        if (completedTask && completedTask.status === 'completed') {
          this._broadcastToRoom(roomId, {
            type: 'taskCompleted',
            roomId,
            taskId,
            executionTimeMs: completedTask.completedAt - completedTask.startedAt,
            deviceCount: new Set(completedTask.deviceAssignments.map(a => a.deviceId)).size,
            result: completedTask.result
          });
        }
        
        // Check for and send any pending computations
        this._sendPendingComputations(room);
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            handled: true
          }
        });
      } else {
        throw new Error('Failed to handle computation result');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Handle update device status
   * @private
   */
  _handleUpdateDeviceStatus(ws, userId, requestId, message) {
    try {
      const { updates } = message;
      
      if (!updates) {
        throw new Error('Invalid update parameters');
      }
      
      // Ensure isConnected is a boolean if provided
      if (updates.isConnected !== undefined) {
        updates.isConnected = updates.isConnected === true;
      }
      
      const updated = this.roomManager.updateUserDeviceStatus(userId, updates);
      
      if (updated) {
        // Get user's rooms to notify
        const rooms = this.roomManager.getUserRooms(userId);
        
        // Notify room members about the status change
        for (const room of rooms) {
          // Get a fresh detailed status to ensure all data is updated
          const roomStatus = this.roomManager.getRoomStatus(room.roomId);
          
          this._broadcastToRoom(room.roomId, {
            type: 'deviceStatusUpdated',
            roomId: room.roomId,
            userId,
            updates,
            roomInfo: roomStatus.roomInfo
          });
          
          // Also broadcast a roomUpdated event with the detailed status
          this._broadcastToRoom(room.roomId, {
            type: 'roomUpdated',
            roomId: room.roomId,
            roomInfo: roomStatus.roomInfo,
            detailedStatus: roomStatus
          });
        }
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            updated: true
          }
        });
      } else {
        throw new Error('Failed to update device status');
      }
    } catch (error) {
      this._sendToClient(ws, {
        type: 'response',
        requestId,
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Send a message to a specific client
   * @private
   * @param {WebSocket} ws - The WebSocket connection
   * @param {Object} message - The message to send
   */
  _sendToClient(ws, message) {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        const jsonString = JSON.stringify(message);
        // console.log('Sending message:', jsonString); // Log what's being sent
        ws.send(jsonString);
      } catch (error) {
        console.error('Error sending message:', error, 'Message:', message);
      }
    }
  }
  
  /**
   * Send a message to a specific user
   * @private
   * @param {string} userId - The user ID
   * @param {Object} message - The message to send
   */
  _sendToUser(userId, message) {
    const ws = this.clients.get(userId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }
  
  /**
   * Broadcast a message to all users in a room
   * @private
   * @param {string} roomId - The room ID
   * @param {Object} message - The message to broadcast
   * @param {string} [excludeUserId] - User ID to exclude from broadcast
   */
  _broadcastToRoom(roomId, message, excludeUserId) {
    const room = this.roomManager.getRoom(roomId);
    if (!room) return;
    
    // Get all users in the room
    for (const user of room.users.values()) {
      if (excludeUserId && user.userId === excludeUserId) continue;
      
      const ws = this.clients.get(user.userId);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      }
    }
  }
}

module.exports = SAXPYWebSocketServer;