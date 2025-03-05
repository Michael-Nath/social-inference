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

// Example usage
async function runExample() {
    try {
        // Define inputs
        const a = 2.0;
        const x = [1, 2, 3, 4, 5];
        const y = [10, 20, 30, 40, 50];



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
        adapter.limits
        displayResults(a, x, y, result, adapter.limits);
    } catch (error) {
        console.error("Error running SAXPY:", error);
        document.getElementById('error').textContent = `Error: ${error.message}`;
    }
}

// Function to display results in the UI
function displayResults(a, x, y, result, limits) {
    const resultContainer = document.getElementById('results');
    
    // Create SAXPY results section
    const saxpySection = `
        <h3>SAXPY Results</h3>
        <p>Formula: result = ${a} * ${x} + ${y}</p>
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
    
    // Combine SAXPY results and limits
    resultContainer.innerHTML = `
        ${saxpySection}
        ${limitsSection}
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
}

// Run the example when the page loads
document.addEventListener('DOMContentLoaded', runExample);