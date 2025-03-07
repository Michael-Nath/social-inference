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
        
      case 'addDevice':
        this._handleAddDevice(ws, userId, requestId, message);
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
   * Handle client disconnection
   * @private
   * @param {string} userId - The user ID
   */
  _handleClientDisconnect(userId) {
    // Get rooms the user has joined
    const rooms = this.roomManager.getUserRooms(userId);
    
    // Auto-leave rooms or mark devices as disconnected
    for (const room of rooms) {
      // Option 1: Remove the user completely from the room
      // this.roomManager.leaveRoom(userId, room.roomId);
      
      // Option 2: Mark the user's devices as disconnected but keep them in the room
      const roomStatus = this.roomManager.getRoomStatus(room.roomId);
      if (roomStatus) {
        const userDevices = roomStatus.users.find(u => u.userId === userId)?.devices || [];
        
        for (const device of userDevices) {
          this.roomManager.updateDeviceStatus(room.roomId, userId, device.deviceId, {
            isConnected: false
          });
        }
        
        // Notify other room members about the device status change
        this._broadcastToRoom(room.roomId, {
          type: 'roomUpdated',
          roomId: room.roomId,
          roomInfo: this.roomManager.getRoom(room.roomId).getRoomInfo()
        }, userId); // Exclude the disconnected user
      }
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
   * Handle add device request
   * @private
   */
  _handleAddDevice(ws, userId, requestId, message) {
    try {
      const { roomId, deviceId, deviceInfo } = message;
      
      if (!roomId) {
        throw new Error('Room ID is required');
      }
      
      // Generate a device ID if not provided
      const actualDeviceId = deviceId || `device-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
      
      const added = this.roomManager.addDeviceToRoom(userId, roomId, actualDeviceId, deviceInfo || {});
      
      if (added) {
        // Get updated room info
        const roomInfo = this.roomManager.getRoom(roomId).getRoomInfo();
        
        // Notify other room members
        this._broadcastToRoom(roomId, {
          type: 'deviceAdded',
          roomId,
          userId,
          deviceId: actualDeviceId,
          deviceInfo,
          roomInfo
        }, userId);
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            roomId,
            deviceId: actualDeviceId,
            added: true
          }
        });
        
        // Also send a deviceAdded event for client-side event handlers
        this._sendToClient(ws, {
          type: 'deviceAdded',
          roomId,
          deviceId: actualDeviceId,
          deviceInfo: { ...deviceInfo, deviceId: actualDeviceId }
        });
      } else {
        throw new Error('Failed to add device');
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
      const { roomId, deviceId } = message;
      
      if (!roomId || !deviceId) {
        throw new Error('Room ID and Device ID are required');
      }
      
      const removed = this.roomManager.removeDeviceFromRoom(userId, roomId, deviceId);
      
      if (removed) {
        // Get updated room info
        const roomInfo = this.roomManager.getRoom(roomId).getRoomInfo();
        
        // Notify other room members
        this._broadcastToRoom(roomId, {
          type: 'deviceRemoved',
          roomId,
          userId,
          deviceId,
          roomInfo
        }, userId);
        
        this._sendToClient(ws, {
          type: 'response',
          requestId,
          success: true,
          data: {
            roomId,
            deviceId,
            removed: true
          }
        });
        
        // Also send a deviceRemoved event for client-side event handlers
        this._sendToClient(ws, {
          type: 'deviceRemoved',
          roomId,
          deviceId
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
   * Handle computation result
   * @private
   */
  _handleComputationResult(ws, userId, requestId, message) {
    try {
      const { roomId, deviceId, taskId, chunkIndex, result, stats } = message;
      
      if (!roomId || !deviceId || !taskId || chunkIndex === undefined || !Array.isArray(result)) {
        throw new Error('Invalid computation result parameters');
      }
      
      const handled = this.roomManager.handleDeviceComputationComplete(
        roomId, 
        userId, 
        deviceId, 
        taskId, 
        chunkIndex, 
        result, 
        stats || {}
      );
      
      if (handled) {
        // Get the room to check if task is completed
        const room = this.roomManager.getRoom(roomId);
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
      const { roomId, deviceId, updates } = message;
      
      if (!roomId || !deviceId || !updates) {
        throw new Error('Invalid update parameters');
      }
      
      const updated = this.roomManager.updateDeviceStatus(roomId, userId, deviceId, updates);
      
      if (updated) {
        // Get updated room info
        const roomInfo = this.roomManager.getRoom(roomId).getRoomInfo();
        
        // Notify room members about the status change
        this._broadcastToRoom(roomId, {
          type: 'deviceStatusUpdated',
          roomId,
          userId,
          deviceId,
          updates,
          roomInfo
        });
        
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