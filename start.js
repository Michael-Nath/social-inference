const http = require('http');
const fs = require('fs');
const path = require('path');

// Port for the server to listen on
const PORT = process.env.PORT || 3000;

// Create the HTML file with the WebGPU code
const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGPU Device Information</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            font-size: 24px;
            margin-top: 0;
        }
        .info-panel {
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
            margin-top: 20px;
            background-color: #f8f9fa;
        }
        .error {
            color: #d9534f;
            background-color: #fdf7f7;
            border: 1px solid #f5c6cb;
            padding: 10px;
            border-radius: 6px;
            margin-top: 20px;
        }
        .feature-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .feature-table th, .feature-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .feature-table th {
            background-color: #f2f2f2;
        }
        .feature-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        pre {
            background-color: #282c34;
            color: #abb2bf;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>WebGPU Device Information</h1>
        
        <div id="gpuStatus">Checking WebGPU support...</div>
        
        <div id="gpuInfo" class="info-panel" style="display: none;">
            <h2>GPU Information</h2>
            <div id="adapterInfo"></div>
            
            <h3>Features</h3>
            <div id="features"></div>
            
            <h3>Limits</h3>
            <div id="limits"></div>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
    </div>

    <script>
        async function checkWebGPU() {
            const statusElement = document.getElementById('gpuStatus');
            const gpuInfoElement = document.getElementById('gpuInfo');
            const errorElement = document.getElementById('error');
            const adapterInfoElement = document.getElementById('adapterInfo');
            const featuresElement = document.getElementById('features');
            const limitsElement = document.getElementById('limits');

            try {
                // Check if WebGPU is supported
                if (!navigator.gpu) {
                    throw new Error('WebGPU is not supported on this browser/device.');
                }

                statusElement.textContent = 'WebGPU is supported! Requesting adapter...';
                
                // Request adapter
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    throw new Error('Couldn\\'t request WebGPU adapter.');
                }

                // Get adapter info
                const adapterInfo = await adapter.requestAdapterInfo();
                
                // Display adapter info
                let adapterInfoHTML = '<ul>';
                adapterInfoHTML += \`<li><strong>Vendor:</strong> \${adapterInfo.vendor || 'Unknown'}</li>\`;
                adapterInfoHTML += \`<li><strong>Architecture:</strong> \${adapterInfo.architecture || 'Unknown'}</li>\`;
                adapterInfoHTML += \`<li><strong>Device:</strong> \${adapterInfo.device || 'Unknown'}</li>\`;
                adapterInfoHTML += \`<li><strong>Description:</strong> \${adapterInfo.description || 'Unknown'}</li>\`;
                adapterInfoHTML += '</ul>';
                
                adapterInfoElement.innerHTML = adapterInfoHTML;

                // Get adapter features
                const features = [];
                for (const feature of adapter.features) {
                    features.push(feature);
                }
                
                if (features.length > 0) {
                    let featuresHTML = '<table class="feature-table">';
                    featuresHTML += '<tr><th>Supported Features</th></tr>';
                    features.forEach(feature => {
                        featuresHTML += \`<tr><td>\${feature}</td></tr>\`;
                    });
                    featuresHTML += '</table>';
                    featuresElement.innerHTML = featuresHTML;
                } else {
                    featuresElement.innerHTML = '<p>No specific features reported.</p>';
                }

                // Get adapter limits
                const limits = [];
                for (const [key, value] of Object.entries(adapter.limits)) {
                    limits.push({ name: key, value: value });
                }
                
                if (limits.length > 0) {
                    let limitsHTML = '<table class="feature-table">';
                    limitsHTML += '<tr><th>Limit Name</th><th>Value</th></tr>';
                    limits.forEach(limit => {
                        limitsHTML += \`<tr><td>\${limit.name}</td><td>\${limit.value}</td></tr>\`;
                    });
                    limitsHTML += '</table>';
                    limitsElement.innerHTML = limitsHTML;
                } else {
                    limitsElement.innerHTML = '<p>No limits reported.</p>';
                }

                // Request a device to see if it works
                try {
                    const device = await adapter.requestDevice();
                    statusElement.textContent = '✅ WebGPU is fully supported and working on your device!';
                } catch (deviceError) {
                    statusElement.textContent = '⚠️ WebGPU adapter detected, but device creation failed.';
                    console.error('Device creation error:', deviceError);
                }

                // Show GPU info section
                gpuInfoElement.style.display = 'block';
                
            } catch (error) {
                console.error('WebGPU error:', error);
                statusElement.textContent = '❌ WebGPU check failed.';
                errorElement.textContent = \`Error: \${error.message}\`;
                errorElement.style.display = 'block';
            }
        }

        // Run the WebGPU check when the page loads
        window.addEventListener('load', checkWebGPU);
    </script>
</body>
</html>`;

// Save the HTML content to a file
fs.writeFileSync(path.join(__dirname, 'webgpu-info.html'), htmlContent);

// Create HTTP server
const server = http.createServer((req, res) => {
  // Set headers for all responses
  res.setHeader('Content-Type', 'text/html');
  
  // Serve our WebGPU page for all requests
  fs.readFile(path.join(__dirname, 'webgpu-info.html'), (err, data) => {
    if (err) {
      res.statusCode = 500;
      res.end(`Error: ${err.message}`);
      return;
    }
    
    res.statusCode = 200;
    res.end(data);
  });
});

// Start the server
server.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log(`Access this URL from your phone to see WebGPU information.`);
  console.log(`Make sure your phone is on the same network and use your computer's IP address instead of localhost.`);
});