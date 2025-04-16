/**
 * Represents a WebGPU kernel that can be executed by the KernelBuilder.
 */
export class Kernel {
  /**
   * @param {Object} options - Kernel configuration options
   * @param {string} options.name - Name of the kernel
   * @param {string} options.shaderPath - Path to the WGSL shader file
   * @param {string} options.entryPoint - Entry point function in the shader
   * @param {Object} options.workgroupSize - Workgroup size configuration
   * @param {number} options.workgroupSize.x - X dimension of workgroup
   * @param {number} options.workgroupSize.y - Y dimension of workgroup
   * @param {number} options.workgroupSize.z - Z dimension of workgroup (optional)
   * @param {Array<Object>} options.inputConfig - Configuration for input tensors
   * @param {string} options.inputConfig[].name - Name of the tensor
   * @param {boolean} [options.inputConfig[].isPersistent] - Whether the tensor should persist between kernel executions
   * @param {boolean} [options.inputConfig[].isOutput] - Whether the tensor is an output tensor
   * @param {string} [options.inputConfig[].type] - Buffer type (e.g., "storage", "uniform")
   */

  constructor(options) {
    this.name = options.name;
    this.shaderPath = options.shaderPath;
    this.entryPoint = options.entryPoint || "main";
    this.workgroupSize = options.workgroupSize || { x: 16, y: 16, z: 1 };
    this.inputConfig = options.inputConfig || [];
    
    // Initialize persistentTensors map with null values for each persistent tensor name
    this.persistentTensors = new Map();
    if (options.inputConfig && Array.isArray(options.inputConfig)) {
      for (const config of options.inputConfig) {
        if (config.isPersistent) {
          this.persistentTensors.set(config.name, null);
        }
      }
    }
    
    this.shaderModule = null;
    this.bindGroupLayout = null;
    this.pipelineLayout = null;
    this.computePipeline = null;
  }

  /**
   * Calculates the number of workgroups based on input dimensions and workgroup size
   * @param {Object} dimensions - Input dimensions
   * @returns {Object} - Workgroup counts for each dimension
   */
  calculateWorkgroups(dimensions) {
    return {
      x: Math.ceil(dimensions.width / this.workgroupSize.x),
      y: Math.ceil(dimensions.height / this.workgroupSize.y),
      z: Math.ceil(dimensions.depth || 1 / this.workgroupSize.z || 1)
    };
  }

  /**
   * Adds a persistent tensor to the kernel
   * @param {string} name - Name of the tensor
   * @param {Object} tensor - Tensor data with elements and shape
   * @returns {boolean} - True if the tensor was added, false if it's not a persistent tensor
   */
  addPersistentTensor(name, tensor) {
    if (this.persistentTensors.has(name)) {
      this.persistentTensors.set(name, tensor);
      return true;
    }
    return false;
  }

  /**
   * Checks if all persistent tensors have been provided
   * @returns {boolean} - True if all persistent tensors are available
   */
  hasPersistentTensorsReady() {
    for (const tensor of this.persistentTensors.values()) {
      if (tensor === null) return false;
    }
    return true;
  }
}

/**
 * Represents a WebGPU computation session with associated resources.
 */
class ComputeSession {
  /**
   * @param {GPUDevice} device - The WebGPU device
   */
  constructor(device) {
    this.device = device;
    this.tensors = new Map();
    this.buffers = new Map();
    this.bindGroups = new Map();
    this.commandEncoder = null;
    this.readbackBuffers = [];
    this.errorScopes = [];
  }

  /**
   * Adds a tensor to the session
   * @param {string} name - Name of the tensor
   * @param {Object} tensor - Tensor data with elements and shape
   * @returns {ComputeSession} - This session instance for chaining
   */
  addTensor(name, tensor) {
    this.tensors.set(name, tensor);
    return this;
  }

  /**
   * Pushes an error scope and tracks it
   * @param {string} scopeType - Type of error scope ('validation' or 'out-of-memory')
   * @returns {ComputeSession} - This session instance for chaining
   */
  pushErrorScope(scopeType) {
    this.device.pushErrorScope(scopeType);
    this.errorScopes.push(scopeType);
    return this;
  }

  /**
   * Pops an error scope and checks for errors
   * @returns {Promise<string|null>} - Error message or null if no error
   */
  async popErrorScope() {
    if (this.errorScopes.length === 0) {
      return null;
    }
    
    const scopeType = this.errorScopes.pop();
    const error = await this.device.popErrorScope();
    
    if (error) {
      return `${scopeType} error: ${error.message}`;
    }
    
    return null;
  }

  /**
   * Creates a command encoder with error scope
   * @returns {GPUCommandEncoder} - The created command encoder
   */
  createCommandEncoder() {
    this.pushErrorScope('validation');
    return this.device.createCommandEncoder();
  }

  /**
   * Exports a buffer to CPU for debugging
   * @param {string} bufferName - Name of the buffer to export
   * @returns {Promise<Float32Array>} - The buffer data
   */
  async exportBufferToCPU(bufferName) {
    const buffer = this.buffers.get(bufferName);
    if (!buffer) {
      throw new Error(`Buffer '${bufferName}' not found in the session.`);
    }
    const readbackBuffer = this.device.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readbackBuffer, 0, buffer.size);
    
    this.device.queue.submit([encoder.finish()]);
    
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readbackBuffer.getMappedRange());
    const result = new Float32Array(data.length);
    result.set(data);
    readbackBuffer.unmap();
    
    return result;
  }

  /**
   * Visualizes the buffer layout for debugging
   * @param {string} bufferName - Name of the buffer to visualize
   * @param {Array<number>} shape - Shape of the tensor for visualization
   * @returns {Object} - Visualization data
   */
  async visualizeBufferLayout(bufferName, shape) {
    const data = await this.exportBufferToCPU(bufferName);
    
    return {
      name: bufferName,
      shape: shape,
      data: data,
      summary: {
        min: Math.min(...data),
        max: Math.max(...data),
        mean: data.reduce((sum, val) => sum + val, 0) / data.length,
        nonZeroCount: data.filter(val => val !== 0).length
      }
    };
  }

  /**
   * Checks all pending error scopes
   * @returns {Promise<Array<string>>} - Array of error messages
   */
  async checkAllErrors() {
    const errors = [];
    
    while (this.errorScopes.length > 0) {
      const error = await this.popErrorScope();
      if (error) {
        errors.push(error);
      }
    }
    
    return errors;
  }

  /**
   * Finalizes the session and cleans up resources
   */
  async finalize() {
    // Check for any errors
    const errors = await this.checkAllErrors();
    if (errors.length > 0) {
      console.error("Errors during session execution:", errors);
    }
    
    // Unmap any mapped buffers
    for (const { readback } of this.readbackBuffers) {
      if (readback.mapState === "mapped") {
        readback.unmap();
      }
    }
  }
}

/**
 * Builder class for creating and executing WebGPU kernels.
 */
export class KernelBuilder {
  /**
   * @param {GPUDevice} device - The WebGPU device
   */
  constructor(device) {
    this.device = device;
    this.activeSession = null;
    this.kernelCache = new Map();
    this.bufferCache = new Map();
    this.persistentBufferCache = new Map(); // Cache for persistent tensor buffers
  }

  /**
   * Starts a new kernel execution session
   * @returns {KernelBuilder} - This builder instance for chaining
   */
  beginSession() {
    if (this.activeSession) {
      throw new Error("A session is already active. End the current session before starting a new one.");
    }
    
    this.activeSession = new ComputeSession(this.device);
    
    return this;
  }

  /**
   * Adds a tensor to the current session and/or to a kernel's persistent tensors
   * @param {string} name - Name of the tensor
   * @param {Object} tensor - Tensor data with elements and shape
   * @param {Array<number>} tensor.elements - Flattened array of tensor elements
   * @param {Array<number>} tensor.shape - Shape of the tensor
   * @param {Kernel} [kernel] - Optional kernel to check for persistent tensor
   * @returns {KernelBuilder} - This builder instance for chaining
   */
  addTensor(name, tensor, kernel = null) {
    // If a kernel is provided, try to add as persistent tensor first
    if (kernel && kernel.addPersistentTensor(name, tensor)) {
      // Create a buffer for this persistent tensor if it doesn't exist
      const bufferKey = `${kernel.name}_${name}`;
      if (!this.persistentBufferCache.has(bufferKey)) {
        const data = tensor;
        const buffer = this.device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          mappedAtCreation: false
        });
        
        this.device.queue.writeBuffer(buffer, 0, data);
        this.persistentBufferCache.set(bufferKey, buffer);
      }
    }
    
    // Add to current session if active
    if (this.activeSession) {
      this.activeSession.addTensor(name, tensor);
    } else if (!kernel) {
      throw new Error("No active session. Call beginSession() first.");
    }
    
    return this;
  }

  /**
   * Creates a WebGPU buffer from tensor data
   * @param {string} name - Name of the tensor
   * @param {GPUBufferUsageFlags} usage - Buffer usage flags
   * @returns {GPUBuffer} - The created buffer
   * @private
   */
  _createBufferFromTensor(name, usage) {
    const tensor = this.activeSession.tensors.get(name);
    if (!tensor) {
      throw new Error(`Tensor '${name}' not found in the current session.`);
    }
    
    this.activeSession.pushErrorScope('out-of-memory');
    
    const data = tensor;
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage,
      mappedAtCreation: false
    });

    this.activeSession.buffers.set(name, buffer);
    
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /**
   * Loads and compiles a shader module for a kernel
   * @param {Kernel} kernel - The kernel to load the shader for
   * @returns {Promise<GPUShaderModule>} - The compiled shader module
   * @private
   */
  async _loadShaderModule(kernel) {
    if (kernel.shaderModule) {
      return kernel.shaderModule;
    }
    
    const response = await fetch(kernel.shaderPath);
    const shaderCode = await response.text();
    
    this.activeSession.pushErrorScope('validation');
    
    kernel.shaderModule = this.device.createShaderModule({
      label: kernel.name,
      code: shaderCode
    });
    
    const error = await this.activeSession.popErrorScope();
    
    if (error) {
      throw new Error(`Shader compilation error for ${kernel.name}: ${error}`);
    }
    
    return kernel.shaderModule;
  }

  /**
   * Loads and prepares a kernel for execution
   * @param {Kernel} kernel - The kernel to prepare
   * @returns {Promise<Kernel>} - The prepared kernel
   */
  async loadKernel(kernel) {
    if (this.kernelCache.has(kernel.name)) {
      return this.kernelCache.get(kernel.name);
    }
    
    if (!this.activeSession) {
      throw new Error("No active session. Call beginSession() first.");
    }
    
    this.activeSession.pushErrorScope('validation');
    await this._loadShaderModule(kernel);
    
    
    // Create bind group layout based on input configuration
    const entries = kernel.inputConfig.map((input, index) => ({
      binding: index,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: input.type || "storage" }
    }));
    var error = await this.activeSession.popErrorScope();
    
    
    kernel.bindGroupLayout = this.device.createBindGroupLayout({
      entries
    });
    
    kernel.pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [kernel.bindGroupLayout]
    });
    
    kernel.computePipeline = this.device.createComputePipeline({
      layout: kernel.pipelineLayout,
      compute: {
        module: kernel.shaderModule,
        entryPoint: kernel.entryPoint
      }
    });

    
    error = await this.activeSession.popErrorScope();
    
    if (error) {
      throw new Error(`Pipeline creation error for ${kernel.name}: ${error}`);
    }
    
    this.kernelCache.set(kernel.name, kernel);
    return kernel;
  }

  /**
   * Creates a bind group for a kernel with the current session's tensors
   * @param {Kernel} kernel - The kernel to create a bind group for
   * @returns {GPUBindGroup} - The created bind group
   * @private
   */
  _createBindGroup(kernel) {
    this.activeSession.pushErrorScope('validation');
    
    const entries = [];
    
    for (let i = 0; i < kernel.inputConfig.length; i++) {
      const config = kernel.inputConfig[i];
      let buffer;
      
      if (config.isOutput) {
        // For output tensors, check if there's a tensor with this name in the session
        const outputTensor = this.activeSession.tensors.get(config.name);
        let size;
        
        if (outputTensor) {
          // Use the size of the provided tensor
          size = outputTensor.byteLength;
        } else if (config.size) {
          // Fallback to the size specified in the config
          size = config.size;
        } else {
          throw new Error(`Output tensor '${config.name}' not found and no size specified in config.`);
        }
        
        // Create output buffer
        buffer = this.device.createBuffer({
          size: size,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Create readback buffer for output
        const readbackBuffer = this.device.createBuffer({
          size: size,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        
        this.activeSession.readbackBuffers.push({
          source: buffer,
          readback: readbackBuffer,
          size: size,
          name: config.name
        });
      } else if (kernel.persistentTensors.has(config.name) && 
                 kernel.persistentTensors.get(config.name) !== null) {
        // Use persistent buffer from cache
        const bufferKey = `${kernel.name}_${config.name}`;
        buffer = this.persistentBufferCache.get(bufferKey);
        
        if (!buffer) {
          throw new Error(`Persistent buffer for '${config.name}' not found in cache.`);
        }
      } else {
        // Use input tensor from current session
        // Determine the appropriate buffer usage based on the config type
        let bufferUsage;
        if (config.type === "uniform") {
          bufferUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        } else {
          // For "storage" or "read-only-storage" types
          bufferUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        }
        buffer = this._createBufferFromTensor(config.name, bufferUsage);
      }
      
      this.activeSession.buffers.set(config.name, buffer);
      
      entries.push({
        binding: i,
        resource: { buffer }
      });
    }
    
    console.log(entries);
    const bindGroup = this.device.createBindGroup({
      layout: kernel.bindGroupLayout,
      entries
    });
    
    return bindGroup;
  }

  /**
   * Executes a kernel in the current session
   * @param {Kernel} kernel - The kernel to execute
   * @param {Object} dimensions - Dimensions for workgroup calculation
   * @returns {KernelBuilder} - This builder instance for chaining
   */
  async executeKernel(kernel, dimensions) {
    if (!this.activeSession) {
      throw new Error("No active session. Call beginSession() first.");
    }
    
    // Make sure the kernel is loaded
    if (!kernel.computePipeline) {
      await this.loadKernel(kernel);
    }
    
    if (!this.activeSession.commandEncoder) {
      this.activeSession.commandEncoder = this.activeSession.createCommandEncoder();
    }
    
    this.activeSession.pushErrorScope('validation');
    const bindGroup = this._createBindGroup(kernel);
    var error = await this.activeSession.popErrorScope();
    if (error) {
      console.error(error);
    }
    
    this.activeSession.bindGroups.set(kernel.name, bindGroup);
    
    this.activeSession.pushErrorScope('validation');
    const computePass = this.activeSession.commandEncoder.beginComputePass();
    error = await this.activeSession.popErrorScope();
    if (error) {
      console.log(error);
      throw new Error(`Kernel execution error for ${kernel.name}: ${error}`);
    }
    computePass.setPipeline(kernel.computePipeline);
    computePass.setBindGroup(0, bindGroup);
    
    const workgroups = kernel.calculateWorkgroups(dimensions);
    computePass.dispatchWorkgroups(workgroups.x, workgroups.y, workgroups.z);
    computePass.end();
    
    error = await this.activeSession.popErrorScope();
    
    if (error) {
      console.log(error);
      throw new Error(`Kernel execution error for ${kernel.name}: ${error}`);
    }
    
    return this;
  }

  /**
   * Exports a buffer from the current session to CPU for debugging
   * @param {string} bufferName - Name of the buffer to export
   * @returns {Promise<Float32Array>} - The buffer data
   */
  async exportBuffer(bufferName) {
    if (!this.activeSession) {
      throw new Error("No active session. Call beginSession() first.");
    }
    
    return this.activeSession.exportBufferToCPU(bufferName);
  }

  /**
   * Visualizes a buffer's layout for debugging
   * @param {string} bufferName - Name of the buffer to visualize
   * @param {Array<number>} shape - Shape of the tensor for visualization
   * @returns {Promise<Object>} - Visualization data
   */
  async visualizeBuffer(bufferName, shape) {
    if (!this.activeSession) {
      throw new Error("No active session. Call beginSession() first.");
    }
    
    return this.activeSession.visualizeBufferLayout(bufferName, shape);
  }

  /**
   * Ends the current session and submits all clean-up commands to the GPU
   * @returns {Promise<Object>} - Results of kernel executions
   */
  async concludeSession() {
    if (!this.activeSession) {
      throw new Error("No active session to end.");
    }
    
    const { commandEncoder, readbackBuffers } = this.activeSession;
    
    // Copy output buffers to readback buffers
    for (const { source, readback, size } of readbackBuffers) {
      console.log("source")
      console.log(source);
      console.log("readback")
      console.log(readback);
      commandEncoder.copyBufferToBuffer(source, 0, readback, 0, size);
    }
    
    const commands = commandEncoder.finish();
    
    this.activeSession.pushErrorScope('validation');
    this.device.queue.submit([commands]);
    const error = await this.activeSession.popErrorScope();
    
    if (error) {
      console.error(`Command submission error: ${error}`);
    }
    
    // Map readback buffers and collect results
    const results = {};
    for (const { readback, name } of readbackBuffers) {
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readback.getMappedRange());
      results[name] = Array.from(data);
      readback.unmap();
    }
    
    // Check for any remaining errors and clean up
    await this.activeSession.finalize();
    
    // Clean up session
    const session = this.activeSession;
    this.activeSession = null;
    
    return {\
      results,
      session
    };
  }

  /**
   * Clears the kernel and buffer caches
   * @returns {KernelBuilder} - This builder instance for chaining
   */
  clearCache() {
    // Clean up persistent buffers
    for (const buffer of this.persistentBufferCache.values()) {
      buffer.destroy();
    }
    this.persistentBufferCache.clear();
    
    this.kernelCache.clear();
    this.bufferCache.clear();
    return this;
  }
}
