export class Tensor {
  /**
   * @param {Object} options - Tensor options
   * @param {string} options.on - Device type (e.g., "cpu" or "gpu")
   */
  constructor(options) {
    this.on = options.on;
  }

  /**
   * @returns {boolean}
   */
  onCPU() {
    return this.on === "cpu";
  }

  /**
   * @returns {boolean}
   */
  onGPU() {
    return this.on === "gpu";
  }
}

/**
 * CPU-side tensor representation
 * 
 * TODO: Add support for other data types.
 */
export class CPUTensor extends Tensor {
  /**
   * @param {Object} options - CPUTensor options
   * @param {ArrayBuffer} options.data - Raw data buffer
   * @param {Array<number>} options.shape - Shape of the tensor
   * @param {string} options.dtype - Data type of the tensor
   */
  constructor(options) {
    super({ on: "cpu" });
    this.shape = options.shape;
    this.dtype = options.dtype;
    this.data = options.data;
  }

  /**
   * Performs a shallow copy of the internal data buffer to a typed array.
   * 
   * @returns {TypedArray} - Typed array of the tensor data
   */
  getTypedArray() {
    // TODO: Correctly handle other data types.
    return new Float32Array(this.data);
  }

  /**
   * @returns {ArrayBuffer} - Raw data buffer
   */
  getData() {
    return this.data;
  }

  /**
   * @returns {Array<number>} - Shape of the tensor
   */
  getShape() {
    return this.shape;
  }

  /**
   * @returns {string} - Data type of the tensor
   */
  getType() {
    return this.dtype;
  }

  /**
   * @returns {number} - Size of the tensor in bytes
   */
  getSize() {
    // Computes based on shape and dtype
    let element_size;
    
    // Determine element size based on data type
    switch (this.dtype) {
      case 'float32':
        element_size = 4; // 32 bits = 4 bytes
        break;
      case 'float16':
        element_size = 2; // 16 bits = 2 bytes
        break;
      case 'int32':
        element_size = 4; // 32 bits = 4 bytes
        break;
      case 'int16':
        element_size = 2; // 16 bits = 2 bytes
        break;
      case 'int8':
        element_size = 1; // 8 bits = 1 byte
        break;
      case 'uint8':
        element_size = 1; // 8 bits = 1 byte
        break;
      default:
        element_size = 4; // Default to float32 (4 bytes)
    }

    return this.shape.reduce((acc, dim) => acc * dim, 1) * element_size;
  }
}

export class GPUTensor extends Tensor {
  /**
   * @param {Object} options - GPUTensor options
   * @param {GPUBuffer} options.buffer - GPU buffer
   * @param {Array<number>} options.shape - Shape of the tensor
   * @param {string} options.dtype - Data type of the tensor
   */
  constructor(options) {
    super({ on: "gpu" });
    this.buffer = options.buffer;
    this.shape = options.shape;
    this.dtype = options.dtype;
  }

  getBuffer() {
    return this.buffer;
  }

  getShape() {
    return this.shape;
  }

  getType() {
    return this.dtype;
  }
}


const cyrb53 = (str, seed = 0) => {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for(let i = 0, ch; i < str.length; i++) {
      ch = str.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1  = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2  = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);

  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
};

export class CPUKernel {
  /**
   * @param {Object} options - CPUKernel options
   * @param {string} options.name - Name of the kernel
   * @param {Function} options.func - Function to execute
   * @param {string[]} options.inputs - Kernel input names
   * @param {string[]} options.outputs - Kernel output names
   */
  constructor(options) {
    this.name = options.name;
    this.func = options.func;
    this.inputs = options.inputs;
    this.outputs = options.outputs;
  }

  async execute(inputs) {
    return await this.func(inputs);
  }
}

/**
 * A GPU kernel.
 * 
 * To support cache efficeint dynamic specialization, kernels also support a
 * dimension buffer function that can be used to compute a uniform dimension
 * buffer at runtime.
 * 
 * The GPU kernel is keyed by a hash of
 * -the shader code
 * -the workgroup size
 * -the input bindings
 * -the output bindings
 * -the name
 * 
 * Note that it is not keyed by the dimension buffer configuration or workgroup
 * function. If you change this, be sure to also change the kernel name to
 * prevent cache collisions.
 * */
export class GPUKernel {
  /**
   * @param {Object} options - Kernel configuration options
   * @param {string} options.name - Name of the kernel
   * @param {string} options.shader - WGSL shader code
   * @param {string} options.entryPoint - Entry point function in the shader
   * @param {Object} options.workgroupSize - Workgroup size configuration
   * @param {number} options.workgroupSize.x - X dimension of workgroup
   * @param {number} options.workgroupSize.y - Y dimension of workgroup
   * @param {number} options.workgroupSize.z - Z dimension of workgroup (optional)
   * @param {Object} options.dimensionBuffer - Dimension buffer configuration (optional)
   * @param {function} options.dimensionBuffer.func - Function to compute the dimension buffer
   * @param {number} options.dimensionBuffer.index - Index of the dimension buffer in the shader
   * @param {Array<Object>} options.inputBindings - Configuration for buffer bindings
   * @param {string} options.inputBindings[].name - Name of the tensor
   * @param {string} [options.inputBindings[].type] - Buffer type (e.g., "storage", "uniform")
   * @param {number} options.inputBindings[].index - Index of the binding in the shader
   * @param {Array<Object>} options.outputBindings - Configuration for buffer bindings
   * @param {string} options.outputBindings[].name - Name of the tensor
   * @param {string} [options.outputBindings[].type] - Buffer type (e.g., "storage", "uniform")
   * @param {number} options.outputBindings[].index - Index of the binding in the shader
   * @param {function} options.workgroupFunction - Function to compute the workgroup size
   */

  constructor(options) {
    this.name = options.name;
    this.shader = options.shader;
    this.entryPoint = options.entryPoint || "main";
    this.workgroupSize = options.workgroupSize || { x: 16, y: 16, z: 1 };
    this.inputBindings = options.inputBindings || [];
    this.outputBindings = options.outputBindings || [];
    this.dimensionBuffer = options.dimensionBuffer || null;
    this.workgroupFunction = options.workgroupFunction;

    this.shaderModule = null;
    this.bindGroupLayout = null;
    this.pipelineLayout = null;
    this.computePipeline = null;
  }

  /**
   * @returns {number} - A unique key for the kernel
   */
  key() {
    return cyrb53(
      this.shader +
      JSON.stringify(this.workgroupSize) +
      JSON.stringify(this.inputBindings) +
      JSON.stringify(this.outputBindings) +
      this.name
    );
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
}