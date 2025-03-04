import { create, globals } from 'webgpu';
// Simple Node.js HTTP Server
import http from "http";

async function init() {
  Object.assign(globalThis, globals);
  const navigator = { gpu: create([])};
  const device = await(await navigator.gpu.requestAdapter()).requestDevice();
  console.log(device.adapterInfo.architecture);
  console.log(device.adapterInfo.description);
}


// Configuration
const PORT = 3000;
const HOST = 'localhost';

// Create HTTP server
const server = http.createServer(async (req, res) => {
  // Set response header
  res.writeHead(200, {'Content-Type': 'text/html'});
  
  init()
  
  // Send a simple response
  res.end('<h1>Hello World</h1>');
});

// Start the server
server.listen(PORT, HOST, () => {
  console.log(`Server is running at http://${HOST}:${PORT}/`);
});