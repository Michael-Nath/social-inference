// client/room-ui.js
// Simplified UI implementation with WebSocket support

import SAXPYWebSocketClient from './websocket-client.js';

class SAXPYRoomUI {
  constructor(options = {}) {
    this.containerElement = options.container || document.getElementById('saxpy-room-app');
    
    // Create WebSocket client
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

  // Private methods
  _selectRoom(roomId) {
    // Deselect all room items
    const roomItems = document.querySelectorAll('.room-item');
    roomItems.forEach(item => item.classList.remove('selected'));

    // Select the clicked room
    const selectedItem = document.querySelector(`.room-item[data-room-id="${roomId}"]`);
    if (selectedItem) {
      selectedItem.classList.add('selected');
    }

    // Store the selected room ID
    this.selectedRoomId = roomId;
  }

  _updateRoomView(roomInfo) {
    if (!this.elements || !this.elements.roomView) return;

    // Update room name and description
    const roomHeader = this.elements.roomView.querySelector('.room-header h2');
    const roomDescription = this.elements.roomView.querySelector('.room-description p');

    if (roomHeader) {
      roomHeader.textContent = this._escapeHtml(roomInfo.name);
    }

    if (roomDescription) {
      roomDescription.textContent = this._escapeHtml(roomInfo.description || 'No description');
    }

    // Update room stats
    const statCards = this.elements.roomView.querySelectorAll('.stat-card');
    if (statCards.length >= 3) {
      statCards[0].querySelector('.stat-value').textContent = roomInfo.userCount;
      statCards[1].querySelector('.stat-value').textContent = roomInfo.connectedDeviceCount;
    }
  }

  _updateTaskStatus(task) {
    if (!this.elements || !this.elements.taskStatus) return;

    // Show task status panel
    this.elements.taskStatus.style.display = 'block';

    // Update task status details
    const statusHtml = `
      <h3>Current Task</h3>
      <div class="task-details">
        <p>Task ID: ${this._escapeHtml(task.taskId)}</p>
        <p>Status: ${this._escapeHtml(task.status)}</p>
        <div class="progress-bar">
          <div class="progress" style="width: ${task.progress || 0}%"></div>
        </div>
        <p>Progress: ${(task.progress || 0).toFixed(1)}%</p>
      </div>
    `;

    this.elements.taskStatus.innerHTML = statusHtml;
  }

  _updateTaskProgress(taskProgress) {
    if (!this.elements || !this.elements.taskStatus) return;

    const progressBar = this.elements.taskStatus.querySelector('.progress');
    const progressText = this.elements.taskStatus.querySelector('.progress-bar + p');

    if (progressBar) {
      progressBar.style.width = `${taskProgress.progress || 0}%`;
    }

    if (progressText) {
      progressText.textContent = `Progress: ${(taskProgress.progress || 0).toFixed(1)}%`;
    }
  }

  _showTaskResults(taskData) {
    if (!this.elements || !this.elements.resultsView) return;

    // Show results view
    this.elements.resultsView.style.display = 'block';

    // Parse and display results
    const resultsHtml = `
      <h3>Task Results</h3>
      <div class="task-results">
        <p>Task ID: ${this._escapeHtml(taskData.taskId)}</p>
        <p>Completed At: ${taskData.completedAt ? new Date(taskData.completedAt).toLocaleString() || new Date().toLocaleString() : new Date().toLocaleString()}</p>
        <p>Execution Time: ${taskData.executionTimeMs} ms</p>
        <p>Devices Used: ${taskData.deviceCount}</p>
        ${taskData.result ? `
          <div class="result-preview">
            <h4>Result Preview</h4>
            <pre>${JSON.stringify(taskData.result.slice(0, 10), null, 2)}${taskData.result.length > 10 ? '...' : ''}</pre>
          </div>
        ` : ''}
      </div>
    `;

    this.elements.resultsView.innerHTML = resultsHtml;
  }

  _initializeElements() {
    this.elements = {
      roomsList: document.getElementById('rooms-list'),
      devicesList: document.getElementById('devices-list'),
      roomView: document.getElementById('room-view'),
      computationPanel: document.getElementById('computation-panel'),
      taskStatus: document.getElementById('task-status'),
      resultsView: document.getElementById('results-view'),
      createRoomModal: document.getElementById('create-room-modal'),
      setDeviceModal: document.getElementById('set-device-modal') || document.getElementById('add-device-modal'),
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

  _showToast(message, type = 'info', duration = 3000) {
    if (!this.elements || !this.elements.toastContainer) return;
    
    // Create the toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add to the container
    this.elements.toastContainer.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
      toast.classList.add('show');
    }, 10);
    
    // Remove after specified duration
    setTimeout(() => {
      toast.classList.remove('show');
      
      setTimeout(() => {
        if (toast.parentNode === this.elements.toastContainer) {
          this.elements.toastContainer.removeChild(toast);
        }
      }, 300);
    }, duration);
  }

  _addUIEventListeners() {
    // Room management
    this._addEventListener('click', 'refresh-rooms', () => this._refreshRoomsList());
    this._addEventListener('click', 'create-room', () => this._showCreateRoomModal());
    this._addEventListener('click', 'create-room-submit', () => this._handleCreateRoom());
    
    // Device management
    const addDeviceBtn = document.getElementById('add-device');
    if (addDeviceBtn) {
      addDeviceBtn.addEventListener('click', () => this._showSetDeviceModal());
    }
    
    const setDeviceBtn = document.getElementById('set-device');
    if (setDeviceBtn) {
      setDeviceBtn.addEventListener('click', () => this._showSetDeviceModal());
    }
    
    const addDeviceSubmitBtn = document.getElementById('add-device-submit');
    if (addDeviceSubmitBtn) {
      addDeviceSubmitBtn.addEventListener('click', () => this._handleSetDevice());
    }
    
    const setDeviceSubmitBtn = document.getElementById('set-device-submit');
    if (setDeviceSubmitBtn) {
      setDeviceSubmitBtn.addEventListener('click', () => this._handleSetDevice());
    }
    
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

  _addEventListener(eventType, elementId, handler) {
    const element = document.getElementById(elementId);
    if (element) {
      element.addEventListener(eventType, handler);
      
      // Track this listener so we can remove it later if needed
      const key = `${eventType}:${elementId}`;
      this.eventListeners.set(key, { element, eventType, handler });
    }
  }

  _updateConnectionStatus(connected) {
    if (!this.elements || !this.elements.connectionStatus) return;
    
    this.elements.connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
    this.elements.connectionStatus.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
  }

  _showCreateRoomModal() {
    if (!this.elements || !this.elements.createRoomModal) return;
    
    this.elements.createRoomModal.style.display = 'block';
  }

  _generateVector(length, type = 'random', options = {}) {
    switch(type) {
      case 'sequential':
        return Array.from(
          { length }, 
          (_, i) => options.start !== undefined 
            ? options.start + i * (options.step || 1) 
            : i + 1
        );
      
      case 'constant':
        return Array(length).fill(options.value || 1);
      
      case 'random':
      default:
        const min = options.min !== undefined ? options.min : 0;
        const max = options.max !== undefined ? options.max : 100;
        return Array.from(
          { length }, 
          () => min + Math.random() * (max - min)
        );
    }
  }

  _handleQueueTask() {
    // Ensure we have a current room selected
    if (!this.currentRoom) {
      this._showToast('Please join a room first', 'error');
      return;
    }

    // Verify the current user is the room creator/root user
    const roomStatus = this.currentRoom;
    if (roomStatus.createdBy !== this.client.userId) {
      this._showToast('Only the room creator can queue tasks', 'error');
      return;
    }

    // Get input elements
    const vectorSizeInput = document.getElementById('vector-size');
    const scalarInput = document.getElementById('scalar-value');

    // Validate inputs
    if (!vectorSizeInput || !scalarInput) {
      this._showToast('Could not find task input elements', 'error');
      return;
    }

    // Parse inputs
    const vectorLength = parseInt(vectorSizeInput.value);
    const a = parseFloat(scalarInput.value);

    // Validate parsed inputs
    if (isNaN(vectorLength) || vectorLength <= 0) {
      this._showToast('Invalid vector length', 'error');
      return;
    }

    if (isNaN(a)) {
      this._showToast('Invalid scalar value', 'error');
      return;
    }

    // Generate vectors (default to sequential)
    const xArray = this._generateVector(vectorLength, 'sequential');
    const yArray = this._generateVector(vectorLength, 'sequential');

    this._showToast(`Queuing SAXPY task...`, 'info');

    // Send task to server
    this.client.queueTask(this.currentRoom.roomId, a, xArray, yArray)
      .then(taskId => {
        this._showToast(`Task queued successfully`, 'success');
      })
      .catch(error => {
        this._showToast(`Error queuing task: ${error.message}`, 'error');
      });
  }

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
        
        this._showToast(`Room created`, 'success');
        
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
      });
  }

  _showSetDeviceModal() {
    const modal = this.elements?.setDeviceModal || document.getElementById('set-device-modal') || document.getElementById('add-device-modal');
    
    if (modal) {
      modal.style.display = 'block';
    }
  }

  _handleSetDevice() {
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
      workerPath: './iphone-worker.js'
    };
    
    this.client.setDevice(deviceInfo)
      .then(deviceId => {
        // Close the modal
        const modal = this.elements?.setDeviceModal || document.getElementById('set-device-modal') || document.getElementById('add-device-modal');
        if (modal) {
          modal.style.display = 'none';
        }
        
        this._showToast(`Device set`, 'success');
        this._updateDevicesList();
      })
      .catch(error => {
        this._showToast(`Error setting device: ${error.message}`, 'error');
      });
  }
  
  _handleRemoveDevice() {
    if (!this.client.device) {
      this._showToast('You don\'t have a device to remove', 'error');
      return;
    }
    
    this.client.removeDevice()
      .then(removed => {
        if (removed) {
          this._showToast('Device removed', 'success');
          this._updateDevicesList();
        } else {
          this._showToast('Failed to remove device', 'error');
        }
      })
      .catch(error => {
        this._showToast(`Error removing device: ${error.message}`, 'error');
      });
  }
  
  _initializeUI() {
    if (!this.containerElement) {
      throw new Error('Container element not found');
    }
    
    // Import HTML template
    fetch('templates/room-ui-template.html')
      .then(response => response.text())
      .then(html => {
        // Insert the HTML template into the container
        this.containerElement.innerHTML = html;
        
        // Get element references
        this._initializeElements();
        
        // Add event listeners
        this._addUIEventListeners();
        
        // Update connection status in UI
        this._updateConnectionStatus(this.client.connected);
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
    
    this._showToast(`Joining room...`, 'info');
    
    // Join the room via the WebSocket client
    this.client.joinRoom(roomId)
      .then(roomInfo => {
        this.selectedRoomId = roomId;
        this._showToast(`Joined room`, 'success');
        
        // Update the room list
        this._refreshRoomsList();
        
        // Load the room view
        this._loadRoomView(roomId);
      })
      .catch(error => {
        this._showToast(`Error joining room: ${error.message}`, 'error');
      });
  }

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
                <span class="stat" title="Connected Devices">📱 ${room.connectedDeviceCount}</span>
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
      });
  }

  _escapeHtml(text) {
    if (!text) return '';
    
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  _loadRoomView(roomId) {
    if (!this.elements || !this.elements.roomView) return;
    
    // Show loading indicator
    this.elements.roomView.innerHTML = `
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Loading room...</p>
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
        
        // Update the devices list
        this._updateDevicesList();
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
      });
  }

  _generateRoomViewHTML(roomStatus) {
    const room = roomStatus.roomInfo;
    const users = roomStatus.users || [];
    
    // Generate the users list
    const usersHtml = users.map(user => {
      const isCurrentUser = user.userId === this.client.userId;
      const hasDevice = user.hasDevice || user.device;
      const isDeviceConnected = hasDevice && (user.device && user.device.isConnected);
      
      return `
        <div class="user-item ${isCurrentUser ? 'current-user' : ''}">
          <div class="user-info">
            <span class="username">${this._escapeHtml(user.username)}</span>
            ${isCurrentUser ? '<span class="badge">You</span>' : ''}
            ${user.isAdmin ? '<span class="badge admin">Admin</span>' : ''}
          </div>
          <div class="user-device">
            <span class="device-status" title="Device Status">
              ${isDeviceConnected ? 
                `📱 Connected` : 
                `❌ No Device`}
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
          <div class="stat-value">${room.connectedDeviceCount}</div>
          <div class="stat-label">Connected Devices</div>
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
      
      <div class="device-section">
        <h3>Your Device</h3>
        <div id="user-device-details"></div>
      </div>
    `;
  }

  _addRoomViewEventListeners() {
    // Leave room button
    const leaveButton = document.getElementById('leave-room');
    if (leaveButton) {
      leaveButton.addEventListener('click', () => {
        this._leaveRoom(this.currentRoom.roomId);
      });
    }
  }

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
      });
  }

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

  _updateDevicesList() {
    const deviceDetailsEl = document.getElementById('user-device-details');
    if (!deviceDetailsEl) return;
    
    if (!this.client.device) {
      // Show no device state
      deviceDetailsEl.innerHTML = `
        <div class="empty-device">
          <p>You don't have a device set up.</p>
          <button class="button set-device-btn">Set Up Device</button>
        </div>
      `;
      
      // Add click handler for set device button
      const setDeviceButtons = deviceDetailsEl.querySelectorAll('.set-device-btn');
      setDeviceButtons.forEach(btn => {
        btn.addEventListener('click', () => this._showSetDeviceModal());
      });
    } else {
      // Show the user's device
      const device = this.client.device;
      
      // Generate a unique ID for the remove button
      const deviceId = `device-${Date.now()}`;
      deviceDetailsEl.innerHTML = `
        <div class="device-item ${device.status === 'connected' ? 'connected' : 'disconnected'}">
          <div class="device-header">
            <h4>${this._escapeHtml(device.model)}</h4>
            <span class="device-status ${device.status === 'connected' ? 'online' : 'offline'}">
              ${device.status === 'connected' ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div class="device-details">
            <div class="device-stats">
              <span class="battery" title="Battery Level">
                🔋 ${Math.round(device.batteryLevel * 100)}%
              </span>
              <span class="connection" title="Connection Quality">
                📶 ${Math.round(device.connectionQuality * 100)}%
              </span>
              <span class="computations" title="Computations Performed">
                🖥️ ${device.computationsPerformed || 0} computations
              </span>
            </div>
            <div class="device-actions">
              <button id="${deviceId}" class="button small remove-device-btn">Remove Device</button>
            </div>
          </div>
        </div>
      `;
      
      // Add click handler for remove device button
      const removeDeviceButton = document.getElementById(deviceId);
      if (removeDeviceButton) {
        removeDeviceButton.addEventListener('click', () => this._handleRemoveDevice());
      }
    }
  }
  
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
      this._showToast(`Left room`, 'info');
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
        this._showToast(`User joined`, 'info');
        this._loadRoomView(data.roomId);
      }
    });
    
    this.client.on('userLeft', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`User left`, 'info');
        this._loadRoomView(data.roomId);
      }
    });
    
    // Device events
    this.client.on('deviceUpdated', (data) => {
      this._showToast(`Device updated`, 'success');
      this._updateDevicesList();
      
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._loadRoomView(data.roomId);
      }
    });
    
    this.client.on('deviceRemoved', (data) => {
      this._showToast(`Device removed`, 'info');
      this._updateDevicesList();
      
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._loadRoomView(data.roomId);
      }
    });
    
    this.client.on('deviceStatusUpdated', (data) => {
      this._updateDevicesList();
      
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        if (data.updates.isConnected !== undefined) {
          this._loadRoomView(data.roomId);
        }
      }
    });
    
    // Task events
    this.client.on('taskQueued', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`Task queued`, 'info');
      }
    });
    
    this.client.on('taskStarted', (data) => {
      if (this.currentRoom && this.currentRoom.roomId === data.roomId) {
        this._showToast(`Task started`, 'info');
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
        this._showToast(`Task completed`, 'success');
        this._showTaskResults(data);
      }
    });
    
    // Error events
    this.client.on('error', (data) => {
      this._showToast(`Error: ${data.error}`, 'error');
    });
  }
}

export default SAXPYRoomUI;