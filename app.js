// Import the SAXPY function from our module
import { runSaxpy } from './saxpy.js';
import { summarizeGPU } from './gpu-utils.js';

// Descriptions for WebGPU limits
const limitDescriptions = {
    "maxTextureDimension1D": "Maximum width for a 1D texture",
    "maxTextureDimension2D": "Maximum width and height for a 2D texture",
    "maxTextureDimension3D": "Maximum width, height, and depth for a 3D texture",
    "maxTextureArrayLayers": "Maximum number of layers in a texture array",
    "maxBindGroups": "Maximum number of bind groups that can be bound at once",
    "maxBindingsPerBindGroup": "Maximum number of bindings in a single bind group",
    "maxDynamicUniformBuffersPerPipelineLayout": "Maximum number of dynamic uniform buffers per pipeline layout",
    "maxDynamicStorageBuffersPerPipelineLayout": "Maximum number of dynamic storage buffers per pipeline layout",
    "maxSampledTexturesPerShaderStage": "Maximum number of sampled textures per shader stage",
    "maxSamplersPerShaderStage": "Maximum number of samplers per shader stage",
    "maxStorageBuffersPerShaderStage": "Maximum number of storage buffers per shader stage",
    "maxStorageTexturesPerShaderStage": "Maximum number of storage textures per shader stage",
    "maxUniformBuffersPerShaderStage": "Maximum number of uniform buffers per shader stage",
    "maxUniformBufferBindingSize": "Maximum size of a uniform buffer binding in bytes",
    "maxStorageBufferBindingSize": "Maximum size of a storage buffer binding in bytes",
    "minUniformBufferOffsetAlignment": "Minimum alignment for uniform buffer offsets",
    "minStorageBufferOffsetAlignment": "Minimum alignment for storage buffer offsets",
    "maxVertexBuffers": "Maximum number of vertex buffers",
    "maxBufferSize": "Maximum size of a buffer in bytes",
    "maxVertexAttributes": "Maximum number of vertex attributes",
    "maxVertexBufferArrayStride": "Maximum stride between elements in a vertex buffer",
    "maxInterStageShaderComponents": "Maximum number of components that can be passed between shader stages",
    "maxInterStageShaderVariables": "Maximum number of variables that can be passed between shader stages",
    "maxColorAttachments": "Maximum number of color attachments in a render pass",
    "maxColorAttachmentBytesPerSample": "Maximum number of bytes per sample for all color attachments",
    "maxComputeWorkgroupStorageSize": "Maximum size of workgroup memory in compute shaders",
    "maxComputeInvocationsPerWorkgroup": "Maximum number of invocations in a compute workgroup",
    "maxComputeWorkgroupSizeX": "Maximum size of a compute workgroup in the X dimension",
    "maxComputeWorkgroupSizeY": "Maximum size of a compute workgroup in the Y dimension",
    "maxComputeWorkgroupSizeZ": "Maximum size of a compute workgroup in the Z dimension",
    "maxComputeWorkgroupsPerDimension": "Maximum number of workgroups in any dimension"
};

// Helper function to format values for display
function formatValue(value) {
    if (typeof value === 'number') {
        // Format large numbers with commas
        if (value >= 1000) {
            return value.toLocaleString();
        }
        return value.toString();
    }
    return String(value);
}

// Add input form to the page
function addInputForm() {
  const formHTML = `
    <div class="input-form">
      <h3>SAXPY Calculator</h3>
      <p>Enter parameters for the SAXPY computation (a*x + y)</p>
      
      <div class="form-group">
        <label for="scalar-a">Scalar value (a):</label>
        <input type="number" id="scalar-a" step="any" value="2.0">
      </div>
      
      <div class="form-group">
        <label for="vector-size">Vector Size:</label>
        <input type="number" id="vector-size" min="1" max="1000000" value="5">
      </div>
      
      <div class="form-group">
        <label for="x-pattern">Vector X Pattern:</label>
        <select id="x-pattern">
          <option value="sequential">Sequential (1, 2, 3, ...)</option>
          <option value="random">Random Numbers</option>
          <option value="constant">Constant Value</option>
        </select>
      </div>
      
      <div class="form-group" id="x-constant-container" style="display: none;">
        <label for="x-constant-value">X Constant Value:</label>
        <input type="number" id="x-constant-value" step="any" value="1">
      </div>
      
      <div class="form-group">
        <label for="y-pattern">Vector Y Pattern:</label>
        <select id="y-pattern">
          <option value="sequential">Sequential (10, 20, 30, ...)</option>
          <option value="random">Random Numbers</option>
          <option value="constant">Constant Value</option>
        </select>
      </div>
      
      <div class="form-group" id="y-constant-container" style="display: none;">
        <label for="y-constant-value">Y Constant Value:</label>
        <input type="number" id="y-constant-value" step="any" value="10">
      </div>
      
      <button id="run-button" class="run-button">Run SAXPY</button>
      <div id="validation-error" class="error-message"></div>
    </div>
    
    <style>
      .input-form {
        margin-bottom: 20px;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
      }
      
      .form-group {
        margin-bottom: 15px;
      }
      
      .form-group label {
        display: block;
        margin-bottom: 5px;
        font-weight: bold;
      }
      
      .form-group input, .form-group select {
        width: 100%;
        padding: 8px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 14px;
      }
      
      .run-button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
      }
      
      .run-button:hover {
        background-color: #2980b9;
      }
      
      .error-message {
        color: #e74c3c;
        margin-top: 10px;
        font-size: 14px;
      }
    </style>
  `;
  
  // Create a container for the form
  const formContainer = document.createElement('div');
  formContainer.innerHTML = formHTML;
  
  // Insert the form at the beginning of the results container
  const resultsContainer = document.getElementById('results');
  resultsContainer.insertBefore(formContainer, resultsContainer.firstChild);
  
  // Add event listeners for the pattern selectors
  document.getElementById('x-pattern').addEventListener('change', function() {
    document.getElementById('x-constant-container').style.display = 
      this.value === 'constant' ? 'block' : 'none';
  });
  
  document.getElementById('y-pattern').addEventListener('change', function() {
    document.getElementById('y-constant-container').style.display = 
      this.value === 'constant' ? 'block' : 'none';
  });
}

// Generate vector based on the selected pattern
function generateVector(size, pattern, constantValue = 1) {
  const vector = new Array(size);
  
  switch (pattern) {
    case 'sequential':
      for (let i = 0; i < size; i++) {
        vector[i] = pattern === 'sequential' ? (i + 1) * (pattern === 'x-sequential' ? 1 : 10) : (i + 1);
      }
      break;
    case 'random':
      for (let i = 0; i < size; i++) {
        vector[i] = Math.random() * 100; // Random numbers between 0 and 100
      }
      break;
    case 'constant':
      for (let i = 0; i < size; i++) {
        vector[i] = constantValue;
      }
      break;
    default:
      for (let i = 0; i < size; i++) {
        vector[i] = i + 1;
      }
  }
  
  return vector;
}

// Validate inputs before running
function validateInputs(a, size) {
  // Check if a is a valid number
  if (isNaN(a)) {
    return "Scalar value 'a' must be a valid number";
  }
  
  // Check if size is a valid positive integer
  if (!Number.isInteger(size) || size <= 0) {
    return "Vector size must be a positive integer";
  }
  
  if (size > 1000000) {
    return "Vector size is too large (maximum: 1,000,000)";
  }
  
  return null; // No validation errors
}

// Modified runExample function to use user input
async function runExample() {
    try {
        // Check if we're running with user input or default values
        const isFormAvailable = document.getElementById('scalar-a') !== null;
        
        let a, size, x, y;
        
        if (isFormAvailable) {
            // Get values from input fields
            a = parseFloat(document.getElementById('scalar-a').value);
            size = parseInt(document.getElementById('vector-size').value);
            
            // Validate inputs
            const validationError = validateInputs(a, size);
            if (validationError) {
                document.getElementById('validation-error').textContent = validationError;
                return;
            } else {
                document.getElementById('validation-error').textContent = '';
            }
            
            // Get pattern selections
            const xPattern = document.getElementById('x-pattern').value;
            const yPattern = document.getElementById('y-pattern').value;
            
            // Get constant values if applicable
            const xConstant = parseFloat(document.getElementById('x-constant-value').value || 1);
            const yConstant = parseFloat(document.getElementById('y-constant-value').value || 10);
            
            // Generate vectors
            x = generateVector(size, xPattern, xConstant);
            y = generateVector(size, yPattern, yConstant);
        } else {
            // Use default values on initial load
            a = 2.0;
            size = a;
            x = [1, 2, 3, 4, 5];
            y = [10, 20, 30, 40, 50];
        }

        console.log("Input a:", a);
        console.log("Vector size:", size);
        console.log("Input x (first 5 elements):", x.slice(0, 5));
        console.log("Input y (first 5 elements):", y.slice(0, 5));

        // Run the SAXPY computation
        const result = await runSaxpy(a, x, y);
        console.log("SAXPY result (first 5 elements):", result.slice(0, 5));

        // Verify the results for the first few elements as a sanity check
        const expectedResult = x.map((xi, i) => a * xi + y[i]);
        console.log("Expected result (first 5 elements):", expectedResult.slice(0, 5));
        
        // Update UI or perform additional processing with the results
        const adapter = await navigator.gpu.requestAdapter();
        displayResults(a, x, y, result, adapter, size);
    } catch (error) {
        console.error("Error running SAXPY:", error);
        document.getElementById('error').textContent = `Error: ${error.message}`;
    }
}

// Function to display results in the UI
function displayResults(a, x, y, result, adapter, size) {
    
    const limits = adapter.limits;
    const resultContainer = document.getElementById('results');

    // Clear previous results except the form (if it exists)
    const form = resultContainer.querySelector('.input-form');
    resultContainer.innerHTML = '';
    
    // Add the form back if it was there before
    if (form) {
        resultContainer.appendChild(form);
    } else {
        // Add form for first time
        addInputForm();
    }

    const GPUInfo = `
        <h3>Your GPU's name is: ${adapter.name}</h3>
    `;
    
    // Determine how many elements to show in the table (max 20)
    const displayLimit = Math.min(size, 20);
    
    // Create SAXPY results section
    const saxpySection = `
        <h3>SAXPY Results (showing first ${displayLimit} of ${size} elements)</h3>
        <p>Formula: result = ${a} * x + y</p>
        <table class="result-table">
            <thead>
                <tr>
                    <th>Index</th>
                    <th>x</th>
                    <th>y</th>
                    <th>result</th>
                </tr>
            </thead>
            <tbody>
                ${Array.from({length: displayLimit}, (_, i) => `
                    <tr>
                        <td>${i}</td>
                        <td>${x[i].toFixed(4)}</td>
                        <td>${y[i].toFixed(4)}</td>
                        <td>${result[i].toFixed(4)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    // Create limits section with improved formatting
    let limitsSection = `<h3>WebGPU Adapter Limits</h3>`;
    
    // Add limits table with descriptions
    limitsSection += `
        <div class="limits-container">
            <table class="limits-table">
                <thead>
                    <tr>
                        <th>Limit Name</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Get all limit descriptions and sort them alphabetically
    const limitNames = Object.keys(limitDescriptions).sort();
    
    // Add each limit as a table row
    for (const limitName of limitNames) {
        const value = limits[limitName] !== undefined ? limits[limitName] : 'N/A';
        const description = limitDescriptions[limitName];
        
        limitsSection += `
            <tr>
                <td class="limit-name">${limitName}</td>
                <td class="limit-value">${formatValue(value)}</td>
                <td class="limit-description">${description}</td>
            </tr>
        `;
    }
    
    // Add any additional limits that might not be in our descriptions
    const additionalLimits = Object.keys(limits).filter(key => !limitDescriptions[key]).sort();
    
    for (const limitName of additionalLimits) {
        const value = limits[limitName];
        
        limitsSection += `
            <tr>
                <td class="limit-name">${limitName}</td>
                <td class="limit-value">${formatValue(value)}</td>
                <td class="limit-description">No description available</td>
            </tr>
        `;
    }
    
    limitsSection += `
                </tbody>
            </table>
        </div>
        <button id="copy-limits" class="copy-button">Copy Limits to Clipboard</button>
    `;
    
    // Performance metrics section
    const performanceSection = `
        <h3>Performance Metrics</h3>
        <div class="performance-container">
            <p>Vector size: <strong>${size.toLocaleString()}</strong> elements</p>
            <p>Total bytes processed: <strong>${(size * 3 * 4).toLocaleString()}</strong> bytes (3 vectors × 4 bytes per float)</p>
        </div>
    `;
    
    // Append content to the results container
    resultContainer.insertAdjacentHTML('beforeend', GPUInfo);
    resultContainer.insertAdjacentHTML('beforeend', performanceSection);
    resultContainer.insertAdjacentHTML('beforeend', saxpySection);
    resultContainer.insertAdjacentHTML('beforeend', limitsSection);
    
    // Add styles
    const stylesHTML = `
        <style>
            .limits-table, .result-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            }
            
            .limits-table th, .limits-table td,
            .result-table th, .result-table td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            
            .limits-table th, .result-table th {
                background-color: #3498db;
                color: white;
                font-weight: 500;
            }
            
            .limits-table tbody tr:hover, .result-table tbody tr:hover {
                background-color: #f5f5f5;
            }
            
            .limit-name {
                font-weight: 500;
                color: #2c3e50;
            }
            
            .limit-value {
                font-family: monospace;
                font-size: 14px;
            }
            
            .limit-description {
                color: #7f8c8d;
                font-size: 14px;
            }
            
            .performance-container {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
                border: 1px solid #ddd;
            }
            
            .copy-button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 20px;
                font-size: 14px;
            }
            
            .copy-button:hover {
                background-color: #2980b9;
            }
        </style>
    `;
    resultContainer.insertAdjacentHTML('beforeend', stylesHTML);
    
    // Add event listener for the copy button
    document.getElementById('copy-limits').addEventListener('click', () => {
        // Get all limit names from both sources
        const allLimitNames = [...new Set([...Object.keys(limitDescriptions), ...Object.keys(limits)])].sort();
        const limitsText = allLimitNames
            .map(name => {
                const value = limits[name] !== undefined ? limits[name] : 'N/A';
                return `${name}: ${value}`;
            })
            .join('\n');
        navigator.clipboard.writeText(limitsText).then(() => {
            alert('Limits copied to clipboard!');
        });
    });
    
    // Add event listener to the run button
    document.getElementById('run-button').addEventListener('click', runExample);
    
    // Also allow Enter key to submit
    document.querySelectorAll('.input-form input').forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                runExample();
            }
        });
    });
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check if we already have a results container
    if (!document.getElementById('results')) {
        // Create results container if it doesn't exist
        const resultsContainer = document.createElement('div');
        resultsContainer.id = 'results';
        document.body.appendChild(resultsContainer);
    }
    
    // Create error container if it doesn't exist
    if (!document.getElementById('error')) {
        const errorContainer = document.createElement('div');
        errorContainer.id = 'error';
        errorContainer.style.color = '#e74c3c';
        errorContainer.style.marginTop = '20px';
        document.body.appendChild(errorContainer);
    }
    
    // Run the example with default values
    runExample();
});