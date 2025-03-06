// client/room-ui.js
// Updated UI implementation with WebSocket support

import SAXPYWebSocketClient from './websocket-client.js';

class SAXPYRoomUI {
  constructor(options = {}) {
    this.containerElement = options.container || document.getElementById('saxpy-room-app');
    
    // Create WebSocket client instead of the simulated client
    this.client = options.client || new SAXPYWebSocketClient({
      userId: options.userId,
      username: options.username,
      serverUrl: options.serverUrl || 'ws://localhost:8080',
      autoConnect: options.autoConnect !== false
    });
    
    // Current state
    this.currentRoom = null;
    this.selectedRoomId = null;
    
    // UI event listeners
    this.eventListeners = new Map();
    
    // Initialize the UI once connected
    this.client.on('connect', () => {
      this._initializeUI();
      this._refreshRoomsList();
    });
    
    // If not auto-connecting, initialize UI right away
    if (options.autoConnect === false) {
      this._initializeUI();
    }
    
    // Set up client event handlers
    this._setupClientEvents();
  }

  /**
   * Initialize element references after template is loaded
   * @private
   */
  _initializeElements() {
    this.elements = {
      roomsList: document.getElementById('rooms-list'),
      devicesList: document.getElementById('devices-list'),
      roomView: document.getElementById('room-view'),
      computationPanel: document.getElementById('computation-panel'),
      taskStatus: document.getElementById('task-status'),
      resultsView: document.getElementById('results-view'),
      createRoomModal: document.getElementById('create-room-modal'),
      addDeviceModal: document.getElementById('add-device-modal'),
      toastContainer: document.getElementById('toast-container'),
      connectionStatus: document.querySelector('.connection-status'),
      username: document.querySelector('.username')
    };
    
    // Set the username in the UI
    if (this.elements.username) {
      this.elements.username.textContent = this.client.username;
    }
    
    // Update connection status
    this._updateConnectionStatus(this.client.connected);
  }

  /**
   * Show a toast notification
   * @private
   * @param {string} message - The message to show
   * @param {string} type - Type of toast (info, success, error, warning)
   * @param {number} duration - How long to show the toast in milliseconds
  */
  _showToast(message, type = 'info', duration = 3000) {
    if (!this.elements || !this.elements.toastContainer) return;
    
    // Create the toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add to the container
    this.elements.toastContainer.appendChild(toast);
    
    // Animate in (using setTimeout to ensure the transition works)
    setTimeout(() => {
      toast.classList.add('show');
    }, 10);
    
    // Remove after specified duration
    setTimeout(() => {
      // Start fade out animation
      toast.classList.remove('show');
      
      // Remove from DOM after animation completes
      setTimeout(() => {
        if (toast.parentNode === this.elements.toastContainer) {
          this.elements.toastContainer.removeChild(toast);
        }
      }, 300); // Match this to your CSS transition duration
    }, duration);
  }

  /**
   * Add event listeners to UI elements
   * @private
   */
  _addUIEventListeners() {
    // Room management
    this._addEventListener('click', 'refresh-rooms', () => this._refreshRoomsList());
    this._addEventListener('click', 'create-room', () => this._showCreateRoomModal());
    
    // Create room modal
    this._addEventListener('click', 'create-room-submit', () => this._handleCreateRoom());
    
    // Device management
    this._addEventListener('click', 'add-device', () => this._showAddDeviceModal());
    
    // Add device modal
    this._addEventListener('click', 'add-device-submit', () => this._handleAddDevice());
    
    // Close buttons for modals
    const closeButtons = document.querySelectorAll('.close-button, .cancel-button');
    closeButtons.forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.modal').forEach(modal => {
          modal.style.display = 'none';
        });
      });
    });
    
    // Computation
    this._addEventListener('click', 'queue-task', () => this._handleQueueTask());
    
    // Range sliders
    const batterySlider = document.getElementById('device-battery');
    const batteryValue = document.getElementById('battery-value');
    if (batterySlider && batteryValue) {
      batterySlider.addEventListener('input', () => {
        batteryValue.textContent = `${Math.round(batterySlider.value * 100)}%`;
      });
    }
    
    const connectionSlider = document.getElementById('device-connection');
    const connectionValue = document.getElementById('connection-value');
    if (connectionSlider && connectionValue) {
      connectionSlider.addEventListener('input', () => {
        connectionValue.textContent = `${Math.round(connectionSlider.value * 100)}%`;
      });
    }
  }

  /**
   * Helper to add event listeners and track them
   * @private
   * @param {string} eventType - Type of event
   * @param {string} elementId - ID of the element
   * @param {Function} handler - Event handler
   */
  _addEventListener(eventType, elementId, handler) {
    const element = document.getElementById(elementId);
    if (element) {
      element.addEventListener(eventType, handler);
      
      // Track this listener so we can remove it later if needed
      const key = `${eventType}:${elementId}`;
      this.eventListeners.set(key, { element, eventType, handler });
    } else {
      console.warn(`Element with ID '${elementId}' not found when adding ${eventType} listener`);
    }
  }

  /**
   * Update connection status in the UI
   * @private
   * @param {boolean} connected - Whether connected to the server
   */
  _updateConnectionStatus(connected) {
    if (!this.elements || !this.elements.connectionStatus) return;
    
    this.elements.connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
    this.elements.connectionStatus.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
  }

  /**
   * Show the create room modal
   * @private
   */
  _showCreateRoomModal() {
    if (!this.elements || !this.elements.createRoomModal) return;
    
    this.elements.createRoomModal.style.display = 'block';
  }

  /**
   * Handle create room button click
   * @private
   */
  _handleCreateRoom() {
    const nameInput = document.getElementById('room-name');
    const descriptionInput = document.getElementById('room-description');
    const publicCheckbox = document.getElementById('room-public');
    
    if (!nameInput || !nameInput.value.trim()) {
      this._showToast('Please enter a room name', 'error');
      return;
    }
    
    this._showToast('Creating room...', 'info');
    
    this.client.createRoom({
      name: nameInput.value.trim(),
      description: descriptionInput ? descriptionInput.value.trim() : '',
      isPublic: publicCheckbox ? publicCheckbox.checked : true
    })
      .then(response => {
        if (this.elements.createRoomModal) {
          this.elements.createRoomModal.style.display = 'none';
        }
        
        this._showToast(`Room "${nameInput.value}" created`, 'success');
        
        // Refresh the rooms list
        this._refreshRoomsList();
        
        // Clear the form
        if (nameInput) nameInput.value = '';
        if (descriptionInput) descriptionInput.value = '';
        if (publicCheckbox) publicCheckbox.checked = true;
        
        // Automatically join the new room
        this._joinRoom(response.roomId);
      })
      .catch(error => {
        this._showToast(`Error creating room: ${error.message}`, 'error');
        console.error('Error creating room:', error);
      });
  }

  /**
   * Show the add device modal
   * @private
   */
  _showAddDeviceModal() {
    if (!this.currentRoom) {
      this._showToast('Please join a room first', 'error');
      return;
    }
    
    if (!this.elements || !this.elements.addDeviceModal) return;
    
    this.elements.addDeviceModal.style.display = 'block';
  }

  /**
   * Handle add device button click
   * @private
   */
  _handleAddDevice() {
    if (!this.currentRoom) {
      this._showToast('Please join a room first', 'error');
      return;
    }
    
    const modelSelect = document.getElementById('device-model');
    const batterySlider = document.getElementById('device-battery');
    const connectionSlider = document.getElementById('device-connection');
    
    if (!modelSelect || !batterySlider || !connectionSlider) {
      this._showToast('Could not find device form elements', 'error');
      return;
    }
    
    const deviceInfo = {
      model: modelSelect.value,
      batteryLevel: parseFloat(batterySlider.value),
      connectionQuality: parseFloat(connectionSlider.value),
      workerPath: './client/iphone-worker.js'
    };
    
    this.client.addDevice(this.currentRoom.roomId, deviceInfo)
      .then(deviceId => {
        if (this.elements.addDeviceModal) {
          this.elements.addDeviceModal.style.display = 'none';
        }
        
        this._showToast(`Device added: ${deviceInfo.model}`, 'success');
        this._updateDevicesList();
      })
      .catch(error => {
        this._showToast(`Error adding device: ${error.message}`, 'error');
        console.error('Error adding device:', error);
      });
  }
  
  /**
   * Initialize the user interface
   * @private
   */
  _initializeUI() {
    if (!this.containerElement) {
      throw new Error('Container element not found');
    }
    
    // Import HTML template from separate file
    fetch('templates/room-ui-template.html')
      .then(response => response.text())
      .then(html => {
        // Insert the HTML template into the container
        this.containerElement.innerHTML = html;
        
        // Get references to elements we'll interact with
        this._initializeElements();
        
        // Add event listeners
        this._addUIEventListeners();
        
        // Update connection status in UI
        this._updateConnectionStatus(this.client.connected);
        
        // Load the initial data
        // this._refreshRoomsList();
      })
      .catch(error => {
        console.error('Error loading UI template:', error);
        this.containerElement.innerHTML = `
          <div class="error-state">
            <h2>Error Loading UI</h2>
            <p>${error.message}</p>
            <button id="retry-init" class="button">Retry</button>
          </div>
        `;
        
        document.getElementById('retry-init')?.addEventListener('click', () => {
          this._initializeUI();
        });
      });
  }

  /**
   * Join a room
   * @private
   * @param {string} roomId - The room ID to join
  */
  _joinRoom(roomId) {
    if (!roomId) {
      this._showToast('Invalid room ID', 'error');
      return;
    }
    
    // If already joined this room, just load the view
    if (this.client.joinedRooms.has(roomId)) {
      this.selectedRoomId = roomId;
      this._loadRoomView(roomId);
      return;
    }
    
    // Show loading toast
    this._showToast(`Joining room...`, 'info');
    
    // Join the room via the WebSocket client
    this.client.joinRoom(roomId)
      .then(roomInfo => {
        this.selectedRoomId = roomId;
        this._showToast(`Joined room: ${roomInfo.name}`, 'success');
        
        // Update the room list to reflect the joined status
        this._refreshRoomsList();
        
        // Load the room view
        this._loadRoomView(roomId);
      })
      .catch(error => {
        this._showToast(`Error joining room: ${error.message}`, 'error');
        console.error('Error joining room:', error);
      });
  }

  /**
 * Refresh the list of available rooms
 * @private
 */
  _refreshRoomsList() {
    if (!this.elements || !this.elements.roomsList) return;
    
    // Show loading indicator
    this.elements.roomsList.innerHTML = '<p class="loading">Loading rooms...</p>';
    
    // Ensure we're connected before trying to fetch rooms
    if (!this.client.connected) {
      this.client.connect()
        .then(() => {
          // Now connected, retry refreshing rooms
          this._refreshRoomsList();
        })
        .catch(error => {
          this.elements.roomsList.innerHTML = `
            <p class="error">Not connected to server</p>
            <button id="retry-connection" class="button small">Reconnect</button>
          `;
          
          document.getElementById('retry-connection')?.addEventListener('click', () => {
            this._refreshRoomsList();
          });
          
          console.error('Connection error when refreshing rooms:', error);
        });
      return;
    }
    
    // Fetch rooms from the server
    this.client.listRooms()
      .then(rooms => {
        if (rooms.length === 0) {
          this.elements.roomsList.innerHTML = '<p class="empty-message">No rooms available</p>';
          return;
        }
        
        // Generate HTML for each room
        const roomsHtml = rooms.map(room => {
          const isSelected = this.selectedRoomId === room.roomId;
          const isJoined = this.client.joinedRooms.has(room.roomId);
          
          return `
            <div class="room-item ${isSelected ? 'selected' : ''}" data-room-id="${room.roomId}">
              <div class="room-info">
                <h3>${this._escapeHtml(room.name)}</h3>
                <p>${this._escapeHtml(room.description || 'No description')}</p>
              </div>
              <div class="room-stats">
                <span class="stat" title="Users">👤 ${room.userCount}</span>
                <span class="stat" title="Devices">📱 ${room.deviceCount}</span>
              </div>
              <button class="button small join-room-button" data-room-id="${room.roomId}">
                ${isJoined ? 'View' : 'Join'}
              </button>
            </div>
          `;
        }).join('');
        
        // Update the DOM
        this.elements.roomsList.innerHTML = roomsHtml;
        
        // Add click handlers for join buttons
        const joinButtons = document.querySelectorAll('.join-room-button');
        joinButtons.forEach(button => {
          button.addEventListener('click', (event) => {
            const roomId = event.target.getAttribute('data-room-id');
            if (roomId) {
              this._joinRoom(roomId);
            }
          });
        });
        
        // Add click handlers for room items
        const roomItems = document.querySelectorAll('.room-item');
        roomItems.forEach(item => {
          item.addEventListener('click', (event) => {
            // Only handle clicks on the item itself, not on child elements like buttons
            if (!event.target.classList.contains('join-room-button')) {
              const roomId = item.getAttribute('data-room-id');
              if (roomId) {
                this._selectRoom(roomId);
              }
            }
          });
        });
      })
      .catch(error => {
        this.elements.roomsList.innerHTML = `<p class="error">Error loading rooms: ${error.message}</p>`;
        console.error('Error loading rooms:', error);
      });
  }

  /**
   * Helper method to escape HTML special characters
   * @private
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  _escapeHtml(text) {
    if (!text) return '';
    
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  /**
   * Update connection status in the UI
   * @private
   * @param {boolean} connected - Whether connected to the server
   */
  _updateConnectionStatus(connected) {
    const statusElement = document.querySelector('.connection-status');
    if (statusElement) {
      statusElement.textContent = connected ? 'Connected' : 'Disconnected';
      statusElement.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
    }
  }

    /**
   * Load the room view for a specific room
   * @private
   * @param {string} roomId - The room ID to load
   */
  _loadRoomView(roomId) {
    if (!this.elements || !this.elements.roomView) return;
    
    // Show loading indicator
    this.elements.roomView.innerHTML = `
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Loading room information...</p>
      </div>
    `;
    
    // Get the latest room status from the server
    this.client.getRoomStatus(roomId)
      .then(roomStatus => {
        // Store the current room info
        this.currentRoom = roomStatus.roomInfo;
        
        // Update room view with the room data
        this.elements.roomView.innerHTML = this._generateRoomViewHTML(roomStatus);
        
        // Show computation panel for the room
        if (this.elements.computationPanel) {
          this.elements.computationPanel.style.display = 'block';
        }
        
        // If there's a current task, show its status
        if (roomStatus.currentTask) {
          this._updateTaskStatus(roomStatus.currentTask);
        } else if (this.elements.taskStatus) {
          this.elements.taskStatus.style.display = 'none';
        }
        
        // Hide results view until needed
        if (this.elements.resultsView) {
          this.elements.resultsView.style.display = 'none';
        }
        
        // Add event listeners for room view actions
        this._addRoomViewEventListeners();
      })
      .catch(error => {
        this.elements.roomView.innerHTML = `
          <div class="error-state">
            <h2>Error Loading Room</h2>
            <p>${error.message}</p>
            <button id="retry-load-room" class="button">Retry</button>
          </div>
        `;
        
        // Add retry button click handler
        document.getElementById('retry-load-room')?.addEventListener('click', () => {
          this._loadRoomView(roomId);
        });
        
        console.error('Error loading room view:', error);
      });
  }

  /**
   * Generate HTML for the room view
   * @private
   * @param {Object} roomStatus - Room status data
   * @returns {string} HTML for the room view
   */
  _generateRoomViewHTML(roomStatus) {
    const room = roomStatus.roomInfo;
    const users = roomStatus.users || [];
    
    // Count total connected devices
    const totalConnectedDevices = users.reduce((count, user) => {
      return count + (user.devices?.filter(d => d.isConnected)?.length || 0);
    }, 0);
    
    // Generate the users list
    const usersHtml = users.map(user => {
      const isCurrentUser = user.userId === this.client.userId;
      const connectedDevices = user.devices?.filter(d => d.isConnected)?.length || 0;
      const totalDevices = user.devices?.length || 0;
      
      return `
        <div class="user-item ${isCurrentUser ? 'current-user' : ''}">
          <div class="user-info">
            <span class="username">${this._escapeHtml(user.username)}</span>
            ${isCurrentUser ? '<span class="badge">You</span>' : ''}
            ${user.isAdmin ? '<span class="badge admin">Admin</span>' : ''}
          </div>
          <div class="user-devices">
            <span class="device-count" title="Connected Devices">
              📱 ${connectedDevices}/${totalDevices}
            </span>
          </div>
        </div>
      `;
    }).join('');
    
    // Generate HTML for the room stats
    return `
      <div class="room-header">
        <h2>${this._escapeHtml(room.name)}</h2>
        <div class="room-actions">
          <button id="leave-room" class="button small">Leave Room</button>
        </div>
      </div>
      
      <div class="room-stats">
        <div class="stat-card">
          <div class="stat-value">${room.userCount}</div>
          <div class="stat-label">Users</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${totalConnectedDevices}</div>
          <div class="stat-label">Connected Devices</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${room.deviceCount}</div>
          <div class="stat-label">Total Devices</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${roomStatus.stats?.totalComputations || 0}</div>
          <div class="stat-label">Computations</div>
        </div>
      </div>
      
      <div class="room-description">
        ${room.description ? `<p>${this._escapeHtml(room.description)}</p>` : ''}
      </div>
      
      <div class="room-users">
        <h3>Connected Users</h3>
        <div class="users-list">
          ${usersHtml.length > 0 ? usersHtml : '<p class="empty-message">No users in this room</p>'}
        </div>
      </div>
    `;
  }

  /**
   * Add event listeners specific to the room view
   * @private
   */
  _addRoomViewEventListeners() {
    // Leave room button
    const leaveButton = document.getElementById('leave-room');
    if (leaveButton) {
      leaveButton.addEventListener('click', () => {
        this._leaveRoom(this.currentRoom.roomId);
      });
    }
  }

  /**
   * Leave a room
   * @private
   * @param {string} roomId - The room ID to leave
   */
  _leaveRoom(roomId) {
    if (!roomId) return;
    
    this._showToast('Leaving room...', 'info');
    
    this.client.leaveRoom(roomId)
      .then(left => {
        if (left) {
          this._showToast('Left room successfully', 'success');
          this._clearRoomView();
          this._refreshRoomsList();
        }
      })
      .catch(error => {
        this._showToast(`Error leaving room: ${error.message}`, 'error');
        console.error('Error leaving room:', error);
      });
  }

  /**
   * Clear the room view
   * @private
   */
  _clearRoomView() {
    this.currentRoom = null;
    
    // Reset the room view to empty state
    if (this.elements.roomView) {
      this.elements.roomView.innerHTML = `
        <div class="empty-state">
          <h2>Select a Room</h2>
          <p>Select a computation room from the sidebar or create a new one to get started.</p>
        </div>
      `;
    }
    
    // Hide computation panels
    if (this.elements.computationPanel) {
      this.elements.computationPanel.style.display = 'none';
    }
    
    if (this.elements.taskStatus) {
      this.elements.taskStatus.style.display = 'none';
    }
    
    if (this.elements.resultsView) {
      this.elements.resultsView.style.display = 'none';
    }
  }
  
  // Rest of the implementation remains largely the same
  // But with these key differences:
  
  // 1. All server communication now happens through WebSockets
  // 2. Real-time updates from the server are handled via event listeners
  // 3. When the connection is lost, the UI updates to reflect this
  // 4. Reconnection attempts are handled automatically by the client
  
  /**
   * Set up event handlers for client events
   * @private
   */
  _setupClientEvents() {
    // Connection status events
    this.client.on('connect', () => {
      this._updateConnectionStatus(true);
      this._showToast('Connected to server', 'success');
    });
    
    this.client.on('disconnect', () => {
      this._updateConnectionStatus(false);
      this._showToast('Disconnected from server', 'error');
    });
    
    // Room events
    this.client.on('roomJoined', (data) => {
      this._showToast(`Joined room: ${data.roomInfo.name}`, 'success');
      this._loadRoomView(data.roomId);
    });
    
    this.client.on('roomLeft', (data) => {
      this._showToast(`Left room: ${data.roomId}`, 'info');
      this._clearRoomView();
    });
    
    this.client.on('roomUpdated', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._updateRoomView(data.roomInfo);
      }
    });
    
    // User events
    this.client.on('userJoined', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`User joined: ${data.username || data.userId}`, 'info');
        // Refresh room view to show the new user
        this._loadRoomView(data.roomId);
      }
    });
    
    this.client.on('userLeft', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`User left: ${data.username || data.userId}`, 'info');
        // Refresh room view to update user list
        this._loadRoomView(data.roomId);
      }
    });
    
    // Device events
    this.client.on('deviceAdded', (data) => {
      this._showToast(`Device added: ${data.deviceInfo.model}`, 'success');
      this._updateDevicesList();
      
      // If the device was added to the current room, refresh the view
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._loadRoomView(data.roomId);
      }
    });
    
    this.client.on('deviceRemoved', (data) => {
      this._showToast(`Device removed: ${data.deviceId}`, 'info');
      this._updateDevicesList();
      
      // If the device was removed from the current room, refresh the view
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._loadRoomView(data.roomId);
      }
    });
    
    this.client.on('deviceStatusUpdated', (data) => {
      // Update the device in the list
      this._updateDevicesList();
      
      // If the device is in the current room, consider refreshing the view
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        // For minor updates, we might not want to refresh the entire view
        // but for significant changes (like connection status), we might
        if (data.updates.isConnected !== undefined) {
          this._loadRoomView(data.roomId);
        }
      }
    });
    
    // Task events
    this.client.on('taskQueued', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`Task queued by ${data.userId === this.client.userId ? 'you' : 'another user'}`, 'info');
      }
    });
    
    this.client.on('taskStarted', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`Task started: ${data.taskId}`, 'info');
        this._updateTaskStatus({
          taskId: data.taskId,
          status: 'running',
          progress: 0
        });
      }
    });
    
    this.client.on('taskProgress', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._updateTaskProgress({
          taskId: data.taskId,
          status: 'running',
          progress: data.progress
        });
      }
    });
    
    this.client.on('taskCompleted', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`Task completed: ${data.taskId}`, 'success');
        this._showTaskResults(data);
      }
    });
    
    // Error events
    this.client.on('error', (data) => {
      this._showToast(`Error: ${data.error}`, 'error');
      console.error('SAXPY client error:', data.error, data.details);
    });
  }
  
  // The remainder of the class would include all the UI methods from the original implementation,
  // with updates to use the WebSocket client instead of the simulated one.
  // For brevity, we'll assume those methods are largely unchanged.
}

export default SAXPYRoomUI;