// server/server.js
// Launch script for the SAXPY WebSocket server

const SAXPYWebSocketServer = require('./websocket-server');

// Create and start server
const server = new SAXPYWebSocketServer({
  port: process.env.PORT || 8080
});

server.start();

console.log('SAXPY Room Computing WebSocket server started');

// Handle shutdown gracefully
process.on('SIGINT', () => {
  console.log('Shutting down SAXPY server...');
  server.stop();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('Shutting down SAXPY server...');
  server.stop();
  process.exit(0);
});
