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
      <p>Enter values for the SAXPY computation (a*x + y)</p>
      
      <div class="form-group">
        <label for="scalar-a">Scalar value (a):</label>
        <input type="number" id="scalar-a" step="any" value="2.0">
      </div>
      
      <div class="form-group">
        <label for="vector-x">Vector X (comma-separated numbers):</label>
        <input type="text" id="vector-x" value="1, 2, 3, 4, 5">
      </div>
      
      <div class="form-group">
        <label for="vector-y">Vector Y (comma-separated numbers):</label>
        <input type="text" id="vector-y" value="10, 20, 30, 40, 50">
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
      
      .form-group input {
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
}

// Parse comma-separated values into a numeric array
function parseVector(inputString) {
  return inputString
    .split(',')
    .map(item => item.trim())
    .filter(item => item !== '')
    .map(item => parseFloat(item));
}

// Validate inputs before running
function validateInputs(a, x, y) {
  // Check if a is a valid number
  if (isNaN(a)) {
    return "Scalar value 'a' must be a valid number";
  }
  
  // Check if x and y are valid arrays
  if (!Array.isArray(x) || x.length === 0 || x.some(isNaN)) {
    return "Vector X must contain valid numbers";
  }
  
  if (!Array.isArray(y) || y.length === 0 || y.some(isNaN)) {
    return "Vector Y must contain valid numbers";
  }
  
  // Check if x and y have the same length
  if (x.length !== y.length) {
    return "Vectors X and Y must have the same length";
  }
  
  return null; // No validation errors
}

// Modified runExample function to use user input
async function runExample() {
    try {
        // Check if we're running with user input or default values
        const isFormAvailable = document.getElementById('scalar-a') !== null;
        
        let a, x, y;
        
        if (isFormAvailable) {
            // Get values from input fields
            a = parseFloat(document.getElementById('scalar-a').value);
            x = parseVector(document.getElementById('vector-x').value);
            y = parseVector(document.getElementById('vector-y').value);
            
            // Validate inputs
            const validationError = validateInputs(a, x, y);
            if (validationError) {
                document.getElementById('validation-error').textContent = validationError;
                return;
            } else {
                document.getElementById('validation-error').textContent = '';
            }
        } else {
            // Use default values on initial load
            a = 2.0;
            x = [1, 2, 3, 4, 5];
            y = [10, 20, 30, 40, 50];
        }

        console.log("Input a:", a);
        console.log("Input x:", x);
        console.log("Input y:", y);

        // Run the SAXPY computation
        const result = await runSaxpy(a, x, y);
        console.log("SAXPY result (a*x + y):", result);

        // Verify the results
        const expectedResult = x.map((xi, i) => a * xi + y[i]);
        console.log("Expected result:", expectedResult);
        console.log("Results match:", JSON.stringify(result) === JSON.stringify(expectedResult));

        // Update UI or perform additional processing with the results
        const adapter = await navigator.gpu.requestAdapter();
        displayResults(a, x, y, result, adapter);
    } catch (error) {
        console.error("Error running SAXPY:", error);
        document.getElementById('error').textContent = `Error: ${error.message}`;
    }
}

// Function to display results in the UI
function displayResults(a, x, y, result, adapter) {
    
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
    
    // Create SAXPY results section
    const saxpySection = `
        <h3>SAXPY Results</h3>
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
                ${x.map((xi, i) => `
                    <tr>
                        <td>${i}</td>
                        <td>${xi}</td>
                        <td>${y[i]}</td>
                        <td>${result[i]}</td>
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
    
    // Append content to the results container
    resultContainer.insertAdjacentHTML('beforeend', GPUInfo);
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