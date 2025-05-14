/*
 * Library to handle interfacing with coordination server
 */

import { CPUKernel, CPUTensor, GPUKernel, GPUTensor } from "./kernel.js";

/*
 * Kernels
 */

// Default node output name used by all nodes
export const DEFAULT_NODE_OUTPUT = "output";

/**
 * @class Device
 * @classdesc Base class for compute devices.
 */
export class Device {
    constructor() {};
}

/**
 * @class GPUDevice
 * @classdesc Represents a GPU compute device.
 * @extends Device
 */
export class GPUDevice extends Device {
    /**
     * @property {GPUDevice} [_instance] - Singleton instance of GPUDevice.
     * @private
     */
    static _instance;

    constructor() {
        if (GPUDevice._instance) {
            return GPUDevice._instance;
        };
        super();
        GPUDevice._instance = this;
        // TODO: will probably need a reference to the WebGPU device here at some point
    }
};
/**
 * @class CPUDevice
 * @classdesc Represents a CPU compute device.
 * @extends Device
 */
export class CPUDevice extends Device {
    /**
     * @property {CPUDevice} [_instance] - Singleton instance of CPUDevice.
     * @private
     */
    static _instance;

    constructor() {
        if (CPUDevice._instance) {
            return CPUDevice._instance;
        };
        super();
        CPUDevice._instance = this;
    }
};

/* API Objects. Should match BaseModel specs of webserver.py and inference/*.py */
/**
 * @class Registration
 * @classdesc Represents the registration response from the server.
 */
class Registration {
    /*
     * @param {Object} api_response - Registration response from server
     * @param {string} api_response.partition - Partition to register as
     */
    constructor(api_response) {
        /** @type {string} */
        this.partition = api_response.partition;
    }
}

/**
 * @class APITensor
 * @classdesc Represents a tensor in the API format.
 */
class APITensor {
    /**
     * @param {Object} api_response - Tensor response from server
     * @param {string} api_response.elements - Base64 encoded string of tensor elements
     * @param {Array<number>} api_response.shape - Shape of the tensor
     * @param {string} api_response.dtype - Data type of the tensor
     */
    constructor(api_response) {
        /** @type {string} */
        this.elements = api_response.elements;
        /** @type {Array<number>} */
        this.shape = api_response.shape;
        /** @type {string} */
        this.dtype = api_response.dtype;
    }

    /**
     * @param {CPUTensor} tensor 
     * @returns {APITensor}
     */
    static fromCPU(tensor) {
        // Convert the tensor.buffer ArrayBuffer to a base64 string
        let binary = '';
        let bytes = new Uint8Array(tensor.data);
        let len = bytes.byteLength; // is 1024, should be 4096
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        let base64 = btoa(binary);
        return new APITensor({
            elements: base64,
            shape: tensor.shape,
            dtype: tensor.dtype
        });
    }

    /**
     * @returns {CPUTensor}
     */
    toCPU() {
        const str = atob(this.elements);
        const bytes = new Uint8Array(str.length);
        for (let i = 0; i < str.length; i++) {
            bytes[i] = str.charCodeAt(i);
        }
        return new CPUTensor({
            data: bytes.buffer,
            shape: this.shape,
            dtype: this.dtype
        });
    }
}

/**
 * @class Edge
 * @classdesc Represents an edge in the computation graph.
 */
class Edge {
    /**
     * @param {Object} api_response
     * @param {string} api_response.src
     * @param {string} api_response.src_output
     * @param {string} api_response.dst
     * @param {string} api_response.dst_input
     */
    constructor(api_response) {
        /** @type {string} */
	this.src = api_response.src;
        /** @type {string} */
	this.src_output = api_response.src_output;
        /** @type {string} */
	this.dst = api_response.dst;
        /** @type {string} */
	this.dst_input = api_response.dst_input;
    }
}

/**
 * @class Node
 * @classdesc Base class for all graph nodes.
 */
export class Node {
    /** 
     * @param {Object} api_response - Node response from server
     * @param {string} api_response.type - Type of node
     * @param {string} api_response.name - Name of node
     * @property {Device} device - the device in which this node should be executed on
     */
    constructor(api_response) {
        /** @type {string} */
        this.type = api_response.type;
        /** @type {string} */
        this.name = api_response.name;
        /** @type {DevicePreference | null} */
        this.devicePreference = null;
        
        // Copy any additional properties from the API response
        for (const [key, value] of Object.entries(api_response)) {
            if (key !== "type" && key !== "name") {
                this[key] = value;
            }
        }
    }

    /**
     * Estimates the weight of the node, typically the number of elements it produces.
     * @param {Map<string, number>} inputsMap - A map of input names to their weights.
     * @returns {number} The estimated weight of the node.
     */
    estimateWeight(inputsMap) {
        // Default implementation: sum of input weights.
        // This is often a placeholder and should be overridden by subclasses
        // where a more accurate estimate of produced elements is possible.
        return Array.from(inputsMap.values()).reduce((a, b) => a + b, 0);
    }

    /**
     * @returns {Array<string>}
     */
    get_inputs() { return []; }

    /**
     * @returns {Array<string>}
     */
    get_outputs() { return []; }
}

export class ExecutionContext {
    /**
     * @param {Map<string, CPUTensor>} cpuInputs
     * @param {Map<string, GPUTensor>} gpuInputs
     */
    constructor(cpuInputs, gpuInputs) {
        this.cpuInputs = cpuInputs || new Map();
        this.gpuInputs = gpuInputs || new Map();
    }

    /**
     * @param {string} name
     * @returns {CPUTensor | null}
     */ 
    cpu(name) {
        return this.cpuInputs.get(name) || null;
    }

    /**
     * @param {string} name
     * @returns {GPUTensor | null}
     */ 
    gpu(name) {
        return this.gpuInputs.get(name) || null;
    }
}

export class DevicePreferences {
    /**
     * 
     * @param {Object} options 
     * @param {boolean} options.supportsCPU - Whether to prefer the CPU
     * @param {boolean} options.supportsGPU - Whether to prefer the GPU
     * @param {number} options.gpuWeightThreshold - The weight threshold for GPU
     */
    constructor(options) {
        this.supportsCPU = options.supportsCPU || false;
        this.supportsGPU = options.supportsGPU || false;
        this.gpuWeightThreshold = options.gpuWeightThreshold || 1000;
    }

    /**
     * @returns {boolean}
     */
    supportsCPU() {
        return this.supportsCPU;
    }

    /**
     * @returns {boolean}
     */
    supportsGPU() {
        return this.supportsGPU;
    }

    /**
     * Picks a device based on the weight of the node.
     * @param {number} weight
     * @returns {string} - "gpu" or "cpu"
     */
    pickDevice(weight) {
        // Just CPU
        if(this.supportsCPU && !this.supportsGPU) {
            return "cpu";
        }
        // Just GPU
        if(!this.supportsCPU && this.supportsGPU) {
            return "gpu";
        }

        // GPU if weight is greater than threshold, otherwise CPU
        if(weight > this.gpuWeightThreshold) {
            return "gpu";
        }
        return "cpu";
    }
}

/**
 * @class MatmulNode
 * @classdesc Represents a matrix multiplication node.
 * @extends Node
 */
class MatmulNode extends Node {
    /** @type {string} */
    static LHS = "lhs";
    /** @type {string} */
    static RHS = "rhs";
    
    /**
     * @param {Object} api_response - API response for MatmulNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    estimateWeight(inputsMap) {
        // Very rough estimate for Matmul. Output size is M*N.
        // Input weights are M*K and K*N.
        // This averages the input element counts.
        const lhsWeight = inputsMap.get(MatmulNode.LHS) || 0;
        const rhsWeight = inputsMap.get(MatmulNode.RHS) || 0;
        return (lhsWeight + rhsWeight) / 2;
    }

    get_inputs() { return [MatmulNode.LHS, MatmulNode.RHS]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    /**
     * @returns {Promise<GPUKernel>}
     */
    async getGPUKernel() {
        // From the horse's mouth: https://toji.dev/webgpu-best-practices/dynamic-shader-construction.html
        // This is a nice way to dynamically specialize kernels. The Kernel class has a key() method that allows as
        // much caching as possible. See GPUKernel for details.
        return new GPUKernel({
            name: 'matmul',
            shader: await fetch('kernels/matmul.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (executionContext) => {
                    const lhsTensor = executionContext.gpu(MatmulNode.LHS);
                    const rhsTensor = executionContext.gpu(MatmulNode.RHS);
                    if (!lhsTensor || !rhsTensor) {
                        throw new Error(`MatmulNode (${this.name}): Missing GPU tensors for dimensionBuffer calculation.`);
                    }
                    return new Uint32Array([
                        lhsTensor.shape[0],
                        lhsTensor.shape[1],
                        rhsTensor.shape[1],
                    ]);
                },
                index: 0,
            },
            workgroupFunction: (executionContext) => {
                const lhsTensor = executionContext.gpu(MatmulNode.LHS);
                if (!lhsTensor) {
                    throw new Error(`MatmulNode (${this.name}): Missing LHS GPU tensor for workgroupFunction.`);
                }
                return {
                    x: Math.ceil(lhsTensor.shape[0] / 16),
                    y: Math.ceil(lhsTensor.shape[1] / 16),
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: MatmulNode.LHS, cpu: false, binding: {type: "read-only-storage", index: 1 } },
                { name: MatmulNode.RHS, cpu: false, binding: {type: "read-only-storage", index: 2 } },
            ],
            outputs: [
                { name: DEFAULT_NODE_OUTPUT, binding: {type: "storage", index: 3 } },
            ],
        });
    }


    /**
     * Calculates the output shape for a matrix multiplication.
     * @param {ExecutionContext} executionContext
     * @returns {number[]} The output shape [M, N].
     * @throws {Error} If input shapes are missing, invalid, or incompatible.
     */
    getOutputShape(executionContext) {
        const shapeA = executionContext.gpu(MatmulNode.LHS)?.shape;
        const shapeB = executionContext.gpu(MatmulNode.RHS)?.shape;

        if (!shapeA || !shapeB) {
            throw new Error(`MatmulNode (${this.name}): Missing required input shapes ('input' or 'weight').`);
        }

        if (shapeA.length !== 2 || shapeB.length !== 2) {
            // TODO: Handle batch dimensions if necessary (e.g., [Batch, M, K])
            throw new Error(`MatmulNode (${this.name}): Currently only supports 2D matrices. Got shapes ${shapeA} and ${shapeB}.`);
        }

        const M = shapeA[0];
        const K_A = shapeA[1];
        const K_B = shapeB[0];
        const N = shapeB[1];

        if (K_A !== K_B) {
            throw new Error(`MatmulNode (${this.name}): Incompatible shapes for matrix multiplication. Inner dimensions do not match: ${shapeA} and ${shapeB}.`);
        }

        return [M, N];
    }
}

/**
 * @class ConstantNode
 * @classdesc Represents a node that outputs a constant tensor.
 * @extends Node
 */
class ConstantNode extends Node {
    /**
     * @param {Object} api_response - API response for ConstantNode.
     * @param {string} api_response.tensor_name - Name of the tensor.
     */
    constructor(api_response) {
        super(api_response);
        /** @type {string} */
        this.tensor_name = api_response.tensor_name;
    }

    estimateWeight(inputsMap) {
        // Placeholder: ConstantNode cannot determine its actual size from inputsMap
        // and doesn't store its shape/tensor directly.
        // This should ideally be seeded by _computeWeights if shape info is available
        // or ConstantNode should store its shape.
        return 1;
    }

    get_inputs() { return []; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
}

/**
 * @class SoftmaxNode
 * @classdesc Represents a softmax operation node.
 * @extends Node
 */

const MAX_DIMS_SOFTMAX = 8; // Max dimensions supported by the softmax kernel

function calculateStrides(shape) {
    const ndims = shape.length;
    if (ndims === 0) return [];
    const strides = new Array(ndims).fill(0);
    strides[ndims - 1] = 1;
    for (let i = ndims - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

class SoftmaxNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static DIM = "dim"; // Name for the input tensor that will hold the dimension scalar
    
    /**
     * @param {Object} api_response - API response for SoftmaxNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as the input.
        return inputsMap.get(SoftmaxNode.INPUT) || 0;
    }

    /**
     * Calculates the output shape for a softmax operation.
     * The output shape is identical to the input shape.
     * @param {ExecutionContext} executionContext
     * @returns {number[]} The output shape, same as the input shape.
     * @throws {Error} If the input shape is missing.
     */
    getOutputShape(executionContext) {
        const inputGPUTensor = executionContext.gpu(SoftmaxNode.INPUT);

        if (!inputGPUTensor || !inputGPUTensor.shape) {
            throw new Error(`SoftmaxNode (${this.name}): Missing required input GPU tensor or shape for 'input'.`);
        }
        // Softmax output shape is the same as the input shape
        return [...inputGPUTensor.shape]; // Return a copy
    }

    /**
     * @returns {Promise<GPUKernel>}
     */
    async getGPUKernel() {
        return new GPUKernel({
            name: 'softmax_v2',
            shader: await fetch('kernels/softmax.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (executionContext) => {
                    const inputGPUTensor = executionContext.gpu(SoftmaxNode.INPUT);
                    const dimCPUTensor = executionContext.cpu(SoftmaxNode.DIM);

                    if (!inputGPUTensor) {
                        throw new Error(`SoftmaxNode (${this.name}): Missing input GPU tensor for dimensionBuffer.`);
                    }
                    if (!dimCPUTensor) {
                        throw new Error(`SoftmaxNode (${this.name}): Missing dim CPU tensor for dimensionBuffer.`);
                    }

                    const currentInputShape = inputGPUTensor.shape;
                    let currentNdims = currentInputShape.length;
                    if (currentNdims === 0 && currentInputShape.length === 1 && currentInputShape[0] === 0) { // Handling of [] shape to mean 0D scalar
                        currentNdims = 0;
                    }

                    let smDim = dimCPUTensor.getTypedArray()[0];
                    if (currentNdims > 0) { // Normalize dim only if there are dimensions
                        if (smDim < 0) {
                            smDim = currentNdims + smDim;
                        }
                        if (smDim < 0 || smDim >= currentNdims) {
                            throw new Error(`SoftmaxNode (${this.name}): Dimension out of range. Got dim ${dimCPUTensor.getTypedArray()[0]} for input ndims ${currentNdims}`);
                        }
                    } else { // For 0D tensor, dim must be 0 or -1
                        if (smDim !== 0 && smDim !== -1) {
                             throw new Error(`SoftmaxNode (${this.name}): Dimension out of range for 0D tensor. Got dim ${dimCPUTensor.getTypedArray()[0]}`);
                        }
                        smDim = 0; // Normalize to 0 for 0D case
                    }

                    const paddedShape = new Array(MAX_DIMS_SOFTMAX).fill(1); 
                    const paddedStrides = new Array(MAX_DIMS_SOFTMAX).fill(0);

                    if (currentNdims > 0) {
                        if (currentNdims > MAX_DIMS_SOFTMAX) {
                            throw new Error(`SoftmaxNode (${this.name}): Input tensor dimensions (${currentNdims}) exceed MAX_DIMS_SOFTMAX (${MAX_DIMS_SOFTMAX}).`);
                        }
                        const strides = calculateStrides(currentInputShape);
                        for (let i = 0; i < currentNdims; i++) {
                            paddedShape[i] = currentInputShape[i];
                            paddedStrides[i] = strides[i];
                        }
                    } else { // 0D scalar tensor
                        // Padded shape is [1,1,1,1,1,1,1,1]
                        // Padded strides are [0,0,0,0,0,0,0,0]
                        // ndims is 0, smDim is 0
                    }

                    // Uniform buffer layout: shape_vecs, strides_vecs, ndims, sm_dim_resolved, padding
                    // Each vec4 takes 4 u32s. MAX_DIMS_SOFTMAX = 8.
                    // shape_vecs: MAX_DIMS_SOFTMAX u32s (equivalent to MAX_DIMS_SOFTMAX/4 vec4s)
                    // strides_vecs: MAX_DIMS_SOFTMAX u32s
                    // ndims: 1 u32
                    // sm_dim_resolved: 1 u32
                    // Total u32s before padding: 8 + 8 + 1 + 1 = 18 u32s.
                    // Total bytes = 18 * 4 = 72 bytes.
                    // Next multiple of 16 bytes is 80 bytes. Padding needed = 80 - 72 = 8 bytes (2 u32s).
                    // So, uniformData needs 18 + 2 = 20 u32 elements.
                    const uniformData = new Uint32Array(MAX_DIMS_SOFTMAX * 2 + 4); // 8 (shape) + 8 (strides) + 1 (ndims) + 1 (smDim) + 2 (padding) = 20
                    uniformData.set(paddedShape, 0);
                    uniformData.set(paddedStrides, MAX_DIMS_SOFTMAX);
                    uniformData.set([currentNdims], MAX_DIMS_SOFTMAX * 2);
                    uniformData.set([smDim], MAX_DIMS_SOFTMAX * 2 + 1);
                    // The remaining 2 u32s are padding, initialized to 0 by Uint32Array constructor.
                    return uniformData;
                },
                index: 2, // Matches @group(0) @binding(2) var<uniform> params: Params;
            },
            workgroupFunction: (executionContext) => {
                const inputGPUTensor = executionContext.gpu(SoftmaxNode.INPUT);
                const dimCPUTensor = executionContext.cpu(SoftmaxNode.DIM);

                if (!inputGPUTensor) {
                    throw new Error(`SoftmaxNode (${this.name}): Missing input GPU tensor for workgroupFunction.`);
                }
                 if (!dimCPUTensor) {
                    throw new Error(`SoftmaxNode (${this.name}): Missing dim CPU tensor for workgroupFunction.`);
                }

                const currentInputShape = inputGPUTensor.shape;
                let currentNdims = currentInputShape.length;
                if (currentNdims === 0 && currentInputShape.length === 1 && currentInputShape[0] === 0) { 
                    currentNdims = 0;
                }

                let smDim = dimCPUTensor.getTypedArray()[0];
                if (currentNdims > 0) { 
                    if (smDim < 0) {
                        smDim = currentNdims + smDim;
                    }
                     // Validation already in dimensionBuffer, but good for sanity
                    if (smDim < 0 || smDim >= currentNdims) {
                        throw new Error(`SoftmaxNode (${this.name}): Softmax dimension ${smDim} is out of bounds for input ndims ${currentNdims}.`);
                    }
                } else {
                    smDim = 0; // Normalized for 0D case
                }

                const workgroupSizeX = 256; // Must match the shader's const workgroup_size_x

                let elements_along_sm_dim = 1;
                let num_total_slices = 1;

                if (currentNdims > 0) {
                    elements_along_sm_dim = currentInputShape[smDim];
                    for (let d = 0; d < currentNdims; d++) {
                        if (d !== smDim) {
                            num_total_slices *= currentInputShape[d];
                        }
                    }
                } 
                // For 0D tensor: elements_along_sm_dim = 1, num_total_slices = 1.
                // If elements_along_sm_dim is 0 for a >0D tensor, x will be 0, which is fine (no dispatches).

                return {
                    x: Math.ceil(elements_along_sm_dim / workgroupSizeX),
                    y: num_total_slices, 
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: SoftmaxNode.INPUT, cpu: false, binding: {type: "read-only-storage", index: 0 } },
                { name: SoftmaxNode.DIM, cpu: true }
            ],
            outputs: [
                { name: DEFAULT_NODE_OUTPUT, binding: {type: "storage", index: 1 } },
            ],
        });
    }

    get_inputs() { return [SoftmaxNode.INPUT, SoftmaxNode.DIM]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
}

/**
 * @class SliceNode
 * @classdesc Represents a tensor slice operation node.
 * @extends Node
 */
class SliceNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static DIM = "dim";
    /** @type {string} */
    static START = "start";
    /** @type {string} */
    static END = "end";
    
    /**
     * @param {Object} api_response - API response for SliceNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Estimate: output elements are same as input (overestimate if not full slice).
        // A more accurate estimate would need slice parameters (start, end, dim sizes).
        return inputsMap.get(SliceNode.INPUT) || 0;
    }

    get_inputs() { return [SliceNode.INPUT, SliceNode.DIM, SliceNode.START, SliceNode.END]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
}

/**
 * @class ReshapeNode
 * @classdesc Represents a tensor reshape operation node.
 * @extends Node
 */
class ReshapeNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static DIMS = "dims";
    
    /**
     * @param {Object} api_response - API response for ReshapeNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as the input.
        return inputsMap.get(ReshapeNode.INPUT) || 0;
    }

    get_inputs() { return [ReshapeNode.INPUT, ReshapeNode.DIMS]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(inputTensor, dimsTensor) {
        if (!inputTensor) {
            throw new Error(`ReshapeNode (${this.name}): Missing input tensor.`);
        }
        if (!dimsTensor) {
            throw new Error(`ReshapeNode (${this.name}): Missing dims tensor.`);
        }

        const inputShape = inputTensor.shape;
        const inputNumElements = inputShape.reduce((acc, val) => acc * val, 1);
        
        const newDimsTypedArray = dimsTensor.getTypedArray();
        let newShape = Array.from(newDimsTypedArray);

        let unknownDimIndex = -1;
        let productOfKnownDims = 1;

        for (let i = 0; i < newShape.length; i++) {
            if (newShape[i] === -1) {
                if (unknownDimIndex !== -1) {
                    throw new Error(`ReshapeNode (${this.name}): Can only specify one unknown dimension (-1). Got shape: ${newShape}`);
                }
                unknownDimIndex = i;
            } else if (newShape[i] < 0) {
                throw new Error(`ReshapeNode (${this.name}): Dimension size cannot be negative (except -1 for unknown). Got shape: ${newShape}`);
            } else if (newShape[i] === 0 && inputNumElements !== 0) {
                // If input has elements, a zero dimension in new shape is only valid if input also has zero elements along some dimension making total zero.
                // Or, more strictly, a zero dimension generally means zero elements in that dim.
                // If inputNumElements > 0, newShape containing 0 is an error unless inputNumElements is also 0.
                // Allowing 0 dim if input is 0 elements. E.g. reshape [0,2] to [0,5]
                 productOfKnownDims = 0; // If any dim is 0, product is 0
            } else if (newShape[i] > 0) {
                productOfKnownDims *= newShape[i];
            }
        }

        if (unknownDimIndex !== -1) {
            if (productOfKnownDims === 0) {
                if (inputNumElements === 0) { // e.g. input [0,2], reshape to [-1, 0]
                    newShape[unknownDimIndex] = 0; // or 1, depends on convention for 0-element tensors
                } else {
                     throw new Error(`ReshapeNode (${this.name}): Cannot infer dimension marked -1 when other dimensions result in zero product (${productOfKnownDims}) but input has elements (${inputNumElements}).`);
                }
            } else if (inputNumElements % productOfKnownDims !== 0) {
                throw new Error(`ReshapeNode (${this.name}): Input total elements (${inputNumElements}) not divisible by product of known new dimensions (${productOfKnownDims}) for shape ${Array.from(newDimsTypedArray)}`);
            }
            newShape[unknownDimIndex] = inputNumElements / productOfKnownDims;
        }
        
        const newNumElements = newShape.reduce((acc, val) => acc * val, 1);
        if (newNumElements !== inputNumElements) {
            throw new Error(`ReshapeNode (${this.name}): Total number of elements mismatch. Input: ${inputNumElements} (${inputShape}), Requested New Shape: ${newNumElements} (${newShape}) from dims ${Array.from(newDimsTypedArray)}.`);
        }

        return newShape;
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(ReshapeNode.INPUT) || executionContext.gpu(ReshapeNode.INPUT);
        const dimsTensor = executionContext.cpu(ReshapeNode.DIMS) || executionContext.gpu(ReshapeNode.DIMS);
        return this._calculateOutputShape(inputTensor, dimsTensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'reshape',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(ReshapeNode.INPUT);
                const dimsTensor = executionContext.cpu(ReshapeNode.DIMS);

                if (!inputTensor || !dimsTensor) {
                    throw new Error(`ReshapeNode (${this.name}) kernel: Missing input or dims tensor.`);
                }

                const outputShape = this._calculateOutputShape(inputTensor, dimsTensor);
                
                // Reshape reuses the input tensor's data buffer.
                const outputTensor = new CPUTensor({
                    data: inputTensor.data, // Re-use the underlying ArrayBuffer
                    shape: outputShape,
                    dtype: inputTensor.dtype,
                });

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [ReshapeNode.INPUT, ReshapeNode.DIMS],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }

    async getGPUKernel() {
        /* TODO: We need a ShapeKernel that is IDENTICAL to a CPU kernel except operates on GPU tensors. */
        return new GPUKernel({
            name: 'reshape',
            shader: await fetch('kernels/empty.wgsl').then((res) => res.text()),
            entryPoint: 'main',
            workgroupFunction: (executionContext) => { return {x: 1}; },
            inputs: [],
            outputs: []
        });
    }
}

/**
 * @class UnsqueezeNode
 * @classdesc Represents a tensor unsqueeze operation node.
 * @extends Node
 */
class UnsqueezeNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static DIM = "dim";
    
    /**
     * @param {Object} api_response - API response for UnsqueezeNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as the input.
        return inputsMap.get(UnsqueezeNode.INPUT) || 0;
    }

    get_inputs() { return [UnsqueezeNode.INPUT, UnsqueezeNode.DIM]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    /**
     * Helper method to calculate the output shape for unsqueeze.
     * @param {CPUTensor} inputTensor 
     * @param {CPUTensor} dimTensor 
     * @returns {number[]} The calculated output shape.
     * @private
     */
    _calculateOutputShape(inputTensor, dimTensor) {
        if (!inputTensor) {
            throw new Error(`UnsqueezeNode (${this.name}): Missing required input tensor ('input').`);
        }
        if (!dimTensor) {
            throw new Error(`UnsqueezeNode (${this.name}): Missing required dimension tensor ('dim').`);
        }

        const inputShape = inputTensor.shape;
        if (dimTensor.shape.length !== 1 || dimTensor.shape[0] !== 1) {
             throw new Error(`UnsqueezeNode (${this.name}): Dimension ('dim') must be a scalar tensor. Got shape ${dimTensor.shape}`);
        }
        let dim = dimTensor.getTypedArray()[0];

        const outputRank = inputShape.length + 1;
        if (dim < 0) {
            dim = outputRank + dim;
        }

        if (dim < 0 || dim > outputRank - 1) {
             throw new Error(`UnsqueezeNode (${this.name}): Dimension out of range. Got dim ${dimTensor.data[0]} for input shape ${inputShape}`);
        }

        const outputShape = [...inputShape];
        outputShape.splice(dim, 0, 1);
        return outputShape;
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(UnsqueezeNode.INPUT);
        const dimTensor = executionContext.cpu(UnsqueezeNode.DIM);
        // Delegate calculation to the helper method
        return this._calculateOutputShape(inputTensor, dimTensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'unsqueeze',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(UnsqueezeNode.INPUT);
                const dimTensor = executionContext.cpu(UnsqueezeNode.DIM);

                if (!inputTensor || !dimTensor) {
                    throw new Error(`UnsqueezeNode (${this.name}) kernel: Missing input or dim tensor.`);
                }

                // Use the helper method to calculate the output shape
                const outputShape = this._calculateOutputShape(inputTensor, dimTensor);

                // Create the output tensor reusing the input data buffer
                const outputTensor = new CPUTensor({
                    data: inputTensor.data, // Re-use the underlying ArrayBuffer
                    shape: outputShape,
                    dtype: inputTensor.dtype,
                });

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [UnsqueezeNode.INPUT, UnsqueezeNode.DIM], // Specify input names
            outputs: [DEFAULT_NODE_OUTPUT], // Specify output name
        });
    }
}

/**
 * @class BroadcastNode
 * @classdesc Represents a tensor broadcast operation node.
 * @extends Node
 */
class BroadcastNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static DIM = "dim";
    /** @type {string} */
    static N = "n";
    
    /**
     * @param {Object} api_response - API response for BroadcastNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Estimate: output elements = input elements * n. Assume n=2 as a guess.
        const inputWeight = inputsMap.get(BroadcastNode.INPUT) || 0;
        return inputWeight * 2;
    }

    get_inputs() { return [BroadcastNode.INPUT, BroadcastNode.DIM, BroadcastNode.N]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    /**
     * Helper method to calculate the output shape for broadcast.
     * @param {CPUTensor} inputTensor
     * @param {CPUTensor} dimTensor
     * @param {CPUTensor} nTensor
     * @returns {number[]} The calculated output shape.
     * @private
     */
    _calculateOutputShape(inputTensor, dimTensor, nTensor) {
        if (!inputTensor) {
            throw new Error(`BroadcastNode (${this.name}): Missing required input tensor ('input').`);
        }
        if (!dimTensor) {
            throw new Error(`BroadcastNode (${this.name}): Missing required dimension tensor ('dim').`);
        }
        if (!nTensor) {
            throw new Error(`BroadcastNode (${this.name}): Missing required count tensor ('n').`);
        }

        const inputShape = inputTensor.shape;
        if (dimTensor.shape.length !== 1 || dimTensor.shape[0] !== 1) {
             throw new Error(`BroadcastNode (${this.name}): Dimension ('dim') must be a scalar tensor. Got shape ${dimTensor.shape}`);
        }
        if (nTensor.shape.length !== 1 || nTensor.shape[0] !== 1) {
             throw new Error(`BroadcastNode (${this.name}): Count ('n') must be a scalar tensor. Got shape ${nTensor.shape}`);
        }

        let dim = dimTensor.getTypedArray()[0];
        const n = nTensor.getTypedArray()[0];

        const outputRank = inputShape.length;
        if (dim < 0) {
            dim = outputRank + dim;
        }

        if (dim < 0 || dim >= outputRank) {
             throw new Error(`BroadcastNode (${this.name}): Dimension out of range. Got dim ${dimTensor.data[0]} for input shape ${inputShape}`);
        }
        if (n <= 0) {
             throw new Error(`BroadcastNode (${this.name}): Count ('n') must be positive. Got ${n}`);
        }
        if (inputShape[dim] !== 1) {
            throw new Error(`BroadcastNode (${this.name}): Dimension to broadcast ('dim'=${dim}) must have size 1. Got shape ${inputShape}`);
        }


        const outputShape = [...inputShape];
        outputShape[dim] = n;
        return outputShape;
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(BroadcastNode.INPUT);
        const dimTensor = executionContext.cpu(BroadcastNode.DIM);
        const nTensor = executionContext.cpu(BroadcastNode.N);
        return this._calculateOutputShape(inputTensor, dimTensor, nTensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'broadcast',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(BroadcastNode.INPUT);
                const dimTensor = executionContext.cpu(BroadcastNode.DIM);
                const nTensor = executionContext.cpu(BroadcastNode.N);

                if (!inputTensor || !dimTensor || !nTensor) {
                    throw new Error(`BroadcastNode (${this.name}) kernel: Missing input, dim, or n tensor.`);
                }

                const outputShape = this._calculateOutputShape(inputTensor, dimTensor, nTensor);
                
                // Create an uninitialized CPUTensor for the output
                const outputTensor = CPUTensor.uninitialized(outputShape, inputTensor.dtype);
                // Get a TypedArray view of its data buffer to populate
                const outputDataView = outputTensor.getTypedArray();

                const dim = dimTensor.getTypedArray()[0] >= 0 ? dimTensor.getTypedArray()[0] : inputTensor.shape.length + dimTensor.getTypedArray()[0];
                const n = nTensor.getTypedArray()[0];

                const inputData = inputTensor.getTypedArray();
                const inputStrides = calculateStrides(inputTensor.shape);
                const outputStrides = calculateStrides(outputShape);

                 // Iterate through the output tensor elements
                 // This is a generic way to handle broadcasting along any dimension
                for (let outputIndex = 0; outputIndex < outputDataView.length; outputIndex++) {
                    // Calculate the corresponding input index
                    let remainingIndex = outputIndex;
                    let inputIndex = 0;
                    let isBroadcastDim = false;

                    for (let d = 0; d < outputShape.length; d++) {
                        const coord = Math.floor(remainingIndex / outputStrides[d]) % outputShape[d];
                        remainingIndex %= outputStrides[d];

                        if (d === dim) {
                            // For the broadcast dimension, the input coordinate is always 0
                            inputIndex += 0 * inputStrides[d]; // Since inputShape[dim] is 1
                            isBroadcastDim = true;
                        } else {
                            // For other dimensions, use the calculated coordinate
                            inputIndex += coord * inputStrides[d];
                        }
                    }
                     // Get the value from the input tensor
                     const value = inputData[inputIndex];
                     // Set the value in the output tensor's view
                     outputDataView[outputIndex] = value;
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor, // Return the populated CPUTensor
                };
            },
            inputs: [BroadcastNode.INPUT, BroadcastNode.DIM, BroadcastNode.N],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }
}

/**
 * @class CatNode
 * @classdesc Represents a tensor concatenation operation node.
 * @extends Node
 */
class CatNode extends Node {
    /** @type {string} */
    static A = "a";
    /** @type {string} */
    static B = "b";
    /** @type {string} */
    static DIM = "dim";
    
    /**
     * @param {Object} api_response - API response for CatNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Output elements = elements(A) + elements(B).
        const weightA = inputsMap.get(CatNode.A) || 0;
        const weightB = inputsMap.get(CatNode.B) || 0;
        return weightA + weightB;
    }

    get_inputs() { return [CatNode.A, CatNode.B, CatNode.DIM]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    /**
     * Helper method to calculate the output shape for concatenation.
     * @param {CPUTensor} tensorA
     * @param {CPUTensor} tensorB
     * @param {CPUTensor} dimTensor
     * @returns {number[]} The calculated output shape.
     * @private
     */
    _calculateOutputShape(tensorA, tensorB, dimTensor) {
        if (!tensorA || !tensorB) {
            throw new Error(`CatNode (${this.name}): Missing required input tensors ('a' or 'b').`);
        }
        if (!dimTensor) {
            throw new Error(`CatNode (${this.name}): Missing required dimension tensor ('dim').`);
        }

        const shapeA = tensorA.shape;
        const shapeB = tensorB.shape;

        if (dimTensor.shape.length !== 1 || dimTensor.shape[0] !== 1) {
            throw new Error(`CatNode (${this.name}): Dimension ('dim') must be a scalar tensor. Got shape ${dimTensor.shape}`);
        }
        let dim = dimTensor.getTypedArray()[0];

        if (shapeA.length !== shapeB.length) {
            throw new Error(`CatNode (${this.name}): Input tensors must have the same rank. Got shapes ${shapeA} and ${shapeB}`);
        }

        const rank = shapeA.length;
        if (dim < 0) {
            dim = rank + dim;
        }

        if (dim < 0 || dim >= rank) {
            throw new Error(`CatNode (${this.name}): Dimension out of range. Got dim ${dimTensor.getTypedArray()[0]} for rank ${rank}`);
        }

        for (let i = 0; i < rank; i++) {
            if (i !== dim && shapeA[i] !== shapeB[i]) {
                throw new Error(`CatNode (${this.name}): Input tensor shapes must match except along the concatenation dimension. Got ${shapeA} and ${shapeB} for dim ${dim}`);
            }
        }

        const outputShape = [...shapeA];
        outputShape[dim] = shapeA[dim] + shapeB[dim];
        return outputShape;
    }

    getOutputShape(executionContext) {
        const tensorA = executionContext.cpu(CatNode.A);
        const tensorB = executionContext.cpu(CatNode.B);
        const dimTensor = executionContext.cpu(CatNode.DIM);
        return this._calculateOutputShape(tensorA, tensorB, dimTensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'cat',
            func: (executionContext) => {
                const tensorA = executionContext.cpu(CatNode.A);
                const tensorB = executionContext.cpu(CatNode.B);
                const dimTensor = executionContext.cpu(CatNode.DIM);

                if (!tensorA || !tensorB || !dimTensor) {
                    throw new Error(`CatNode (${this.name}) kernel: Missing input or dim tensor.`);
                }
                if (tensorA.dtype !== tensorB.dtype) {
                    throw new Error(`CatNode (${this.name}) kernel: Input tensors must have the same dtype. Got ${tensorA.dtype} and ${tensorB.dtype}.`);
                }

                const outputShape = this._calculateOutputShape(tensorA, tensorB, dimTensor);
                const outputTensor = CPUTensor.uninitialized(outputShape, tensorA.dtype);
                const outputView = outputTensor.getTypedArray();

                const dataA = tensorA.getTypedArray();
                const dataB = tensorB.getTypedArray();
                
                let dim = dimTensor.getTypedArray()[0];
                const rank = tensorA.shape.length;
                if (dim < 0) {
                    dim = rank + dim;
                }

                const stridesA = calculateStrides(tensorA.shape);
                const stridesB = calculateStrides(tensorB.shape);
                const stridesOut = calculateStrides(outputShape);

                const sizeA_dim = tensorA.shape[dim];

                // Iterate over all elements of the output tensor
                for (let i = 0; i < outputView.length; i++) {
                    let outMultiIndex = new Array(rank);
                    let remainder = i;
                    for (let d = 0; d < rank; d++) {
                        outMultiIndex[d] = Math.floor(remainder / stridesOut[d]);
                        remainder %= stridesOut[d];
                    }

                    let sourceData;
                    let sourceStrides;
                    let sourceShape;
                    let inMultiIndex = [...outMultiIndex]; // Copy for modification

                    if (outMultiIndex[dim] < sizeA_dim) {
                        // This element comes from tensorA
                        sourceData = dataA;
                        sourceStrides = stridesA;
                        sourceShape = tensorA.shape;
                        // inMultiIndex is already correct for tensorA for this part
                    } else {
                        // This element comes from tensorB
                        sourceData = dataB;
                        sourceStrides = stridesB;
                        sourceShape = tensorB.shape;
                        inMultiIndex[dim] -= sizeA_dim; // Adjust index for tensorB
                    }

                    let sourceFlatIndex = 0;
                    for (let d = 0; d < rank; d++) {
                        sourceFlatIndex += inMultiIndex[d] * sourceStrides[d];
                    }
                    outputView[i] = sourceData[sourceFlatIndex];
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [CatNode.A, CatNode.B, CatNode.DIM],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }
}

/**
 * @class FixedNode
 * @classdesc Represents a node with a fixed tensor value.
 * @extends Node
 */
class FixedNode extends Node {
    /**
     * @param {Object} api_response - API response for FixedNode.
     * @param {APITensor} api_response.tensor - The fixed tensor.
     */
    constructor(api_response) {
        super(api_response);
        /** @type {CPUTensor} */
        this.tensor = (new APITensor(api_response.tensor)).toCPU();
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Weight is the number of elements in the fixed tensor.
        if (this.tensor && this.tensor.shape) {
            return this.tensor.shape.reduce((acc, val) => acc * val, 1);
        }
        return 0; // Should not happen if tensor is always present
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'fixed',
            func: (executionContext) => {
                return {
                    [DEFAULT_NODE_OUTPUT]: this.tensor,
                };
            },
            inputs: [],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }

    getOutputShape(executionContext) {
        return this.tensor.shape;
    }

    get_inputs() { return []; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
}

/**
 * @class HadamardNode
 * @classdesc Represents an element-wise multiplication (Hadamard product) node.
 * @extends Node
 */
class HadamardNode extends Node {
    /** @type {string} */
    static A = "a";
    /** @type {string} */
    static B = "b";
    
    /**
     * @param {Object} api_response - API response for HadamardNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as input A (assuming A and B are same size).
        return inputsMap.get(HadamardNode.A) || 0;
    }

    get_inputs() { return [HadamardNode.A, HadamardNode.B]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(tensorA, tensorB) {
        if (!tensorA || !tensorB) {
            throw new Error(`HadamardNode (${this.name}): Missing required input tensors ('a' or 'b').`);
        }
        // For now, require identical shapes. Broadcasting could be added later.
        if (tensorA.shape.length !== tensorB.shape.length || 
            !tensorA.shape.every((dim, i) => dim === tensorB.shape[i])) {
            throw new Error(`HadamardNode (${this.name}): Input shapes must be identical. Got ${tensorA.shape} and ${tensorB.shape}.`);
        }
        if (tensorA.dtype !== tensorB.dtype) {
             // For element-wise ops, typically dtypes should match or be compatible (e.g. float32*float32 -> float32)
             // For simplicity, let's assume they match and output dtype is the same.
             // If they don't match, could default to higher precision or throw error.
             console.warn(`HadamardNode (${this.name}): Input dtypes are different (${tensorA.dtype}, ${tensorB.dtype}). Output dtype will be ${tensorA.dtype}.`);
        }
        return [...tensorA.shape]; // Output shape is same as input shape
    }

    getOutputShape(executionContext) {
        const tensorA = executionContext.cpu(HadamardNode.A) || executionContext.gpu(HadamardNode.A);
        const tensorB = executionContext.cpu(HadamardNode.B) || executionContext.gpu(HadamardNode.B);
        // For GPU kernels, shape calculation might need to access .gpu() if CPU tensors aren't always present
        // However, getOutputShape is often called by CPU-side logic or pre-computation steps.
        // For now, let's assume CPU versions or GPU versions with shape property are available.
        // If only GPU tensors are available in EC for GPU ops, this needs adjustment or shape info passed differently.
        const shapeA = tensorA ? tensorA.shape : null;
        const shapeB = tensorB ? tensorB.shape : null;

        if (!shapeA || !shapeB) {
             throw new Error(`HadamardNode (${this.name}): Could not retrieve shapes for input tensors for getOutputShape.`);
        }
         // This re-uses the logic, assuming CPUTensor-like objects or objects with shape/dtype for inputs
        return this._calculateOutputShape({shape: shapeA, dtype: tensorA.dtype}, {shape: shapeB, dtype: tensorB.dtype});
    }

    async getGPUKernel() {
        return new GPUKernel({
            name: 'hadamard',
            shader: await fetch('kernels/hadamard.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (executionContext) => {
                    const tensorA = executionContext.gpu(HadamardNode.A);
                    if (!tensorA) {
                        throw new Error(`HadamardNode (${this.name}): Missing GPU tensor A for dimensionBuffer.`);
                    }
                    const num_elements = tensorA.shape.reduce((acc, val) => acc * val, 1);
                    return new Uint32Array([num_elements]);
                },
                index: 3, // Assuming binding 0,1 for inputs, 2 for output
            },
            workgroupFunction: (executionContext) => {
                const tensorA = executionContext.gpu(HadamardNode.A);
                 if (!tensorA) {
                    throw new Error(`HadamardNode (${this.name}): Missing GPU tensor A for workgroupFunction.`);
                }
                const num_elements = tensorA.shape.reduce((acc, val) => acc * val, 1);
                const workgroupSizeX = 256; // Should match shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: HadamardNode.A, cpu: false, binding: {type: "read-only-storage", index: 0 } },
                { name: HadamardNode.B, cpu: false, binding: {type: "read-only-storage", index: 1 } },
            ],
            outputs: [
                { name: DEFAULT_NODE_OUTPUT, binding: {type: "storage", index: 2 } },
            ],
        });
    }
}

/**
 * @class IndexNode
 * @classdesc Represents a tensor indexing operation node.
 * @extends Node
 */
class IndexNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static INDEX = "index";
    
    /**
     * @param {Object} api_response - API response for IndexNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        const inputWeight = inputsMap.get(IndexNode.INPUT) || 0;
        // Rough estimate: output is a slice, so smaller than input.
        // This needs to be much better if we know input rank and index type.
        // For now, assume it reduces elements significantly.
        return Math.max(1, Math.floor(inputWeight / 2)); // Placeholder
    }

    get_inputs() { return [IndexNode.INPUT, IndexNode.INDEX]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(inputTensor, indexTensor) {
        if (!inputTensor) {
            throw new Error(`IndexNode (${this.name}): Missing input tensor.`);
        }
        if (!indexTensor) {
            throw new Error(`IndexNode (${this.name}): Missing index tensor.`);
        }

        const inputShape = inputTensor.shape;
        const indexData = indexTensor.getTypedArray();

        if (indexData.length !== 1) {
            throw new Error(`IndexNode (${this.name}): For simple indexing, index tensor must be a scalar (1 element). Got ${indexData.length} elements.`);
        }
        let index = indexData[0];

        if (inputShape.length === 0) {
            throw new Error(`IndexNode (${this.name}): Cannot index a 0D (scalar) tensor.`);
        }

        // Normalize index for the first dimension
        if (index < 0) {
            index = inputShape[0] + index;
        }

        if (index < 0 || index >= inputShape[0]) {
            throw new Error(`IndexNode (${this.name}): Index ${indexData[0]} out of bounds for dimension 0 with size ${inputShape[0]}.`);
        }

        // Output shape is the input shape with the first dimension removed.
        // If input is 1D, output is 0D (scalar).
        const outputShape = inputShape.slice(1);
        return outputShape; // Effectively [] for 1D input after slice(1)
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(IndexNode.INPUT);
        const indexTensor = executionContext.cpu(IndexNode.INDEX);
        return this._calculateOutputShape(inputTensor, indexTensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'index',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(IndexNode.INPUT);
                const indexTensor = executionContext.cpu(IndexNode.INDEX);

                if (!inputTensor || !indexTensor) {
                    throw new Error(`IndexNode (${this.name}) kernel: Missing input or index tensor.`);
                }

                const outputShape = this._calculateOutputShape(inputTensor, indexTensor);
                const outputTensor = CPUTensor.uninitialized(outputShape, inputTensor.dtype);
                const outputView = outputTensor.getTypedArray();
                const inputData = inputTensor.getTypedArray();
                
                let index = indexTensor.getTypedArray()[0];
                const inputShape = inputTensor.shape;
                 if (index < 0) { // Normalize again, just in case (already done in _calculateOutputShape)
                    index = inputShape[0] + index;
                }

                // Calculate the size of each slice along the first dimension
                let sliceSize = 1;
                for (let i = 1; i < inputShape.length; i++) {
                    sliceSize *= inputShape[i];
                }
                if (inputShape.length === 0) sliceSize = 0; // Should have been caught
                if (inputShape.length === 1) sliceSize = 1; // Element for 1D array

                const offset = index * sliceSize;

                if (outputView.length !== sliceSize && !(inputShape.length ===1 && outputView.length === 0) ) { // outputView.length is 0 for scalar
                     // For 1D input, outputShape is [], outputView.length is typically 0 or 1 by convention of TypedArray(0) or TypedArray(1) for scalar.
                     // If inputShape is 1D, sliceSize is 1. outputView.length for a scalar can be tricky.
                     // CPUTensor.uninitialized([], dtype) will create buffer of 1 element for some dtypes.
                     // Let's ensure logic for 1D to 0D (scalar) is robust.
                     if (inputShape.length === 1 && outputShape.length === 0) {
                        // This is indexing a 1D array to get a scalar.
                        // sliceSize is 1. offset is the index itself.
                        // outputView will be a TypedArray for the scalar value.
                        if (outputView.constructor.BYTES_PER_ELEMENT > 0 && outputView.buffer.byteLength >= outputView.constructor.BYTES_PER_ELEMENT) {
                           // It's a valid buffer for one element
                        } else {
                            throw new Error(`IndexNode (${this.name}) kernel: Output view for scalar has unexpected length/buffer for 1D->0D indexing.`);
                        }
                     } else {
                        throw new Error(`IndexNode (${this.name}) kernel: Mismatch between output view length (${outputView.length}) and calculated slice size (${sliceSize}).`);
                     }
                }

                if (inputShape.length === 1) { // Indexing a 1D array to get a scalar
                    if (outputView.length === 1 || outputView.length === 0 && outputShape.length === 0) { // check for scalar case CPUTensor.uninitialized([], dtype) can make length 1 buffer
                         outputView[0] = inputData[offset];
                    } else {
                        throw new Error(`IndexNode (${this.name}): Output view for scalar from 1D array has unexpected length.`);
                    }
                } else { // Indexing multi-dimensional array to get a slice
                    for (let i = 0; i < sliceSize; i++) {
                        outputView[i] = inputData[offset + i];
                    }
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [IndexNode.INPUT, IndexNode.INDEX],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }
}

/**
 * @class ShapeNode
 * @classdesc Represents a node that outputs the shape of a tensor.
 * @extends Node
 */
class ShapeNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    
    /**
     * @param {Object} api_response - API response for ShapeNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Output is a 1D tensor of input's rank. Estimate average rank = 4.
        return 4;
    }

    get_inputs() { return [ShapeNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(ShapeNode.INPUT);
        if (!inputTensor) {
            throw new Error(`ShapeNode (${this.name}): Missing input tensor.`);
        }
        return [inputTensor.shape.length];
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'shape',
            func: (executionContext) => {
                const shape_array = executionContext.cpu(ShapeNode.INPUT).shape;
                const outputTensor = CPUTensor.uninitialized([shape_array.length], "int32");
                const outputView = outputTensor.getTypedArray();
                for (let i = 0; i < shape_array.length; i++) {
                    outputView[i] = shape_array[i];
                }
                return { [DEFAULT_NODE_OUTPUT]: outputTensor };
            },
            inputs: [ShapeNode.INPUT],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }
    
}

/**
 * @class TransposeNode
 * @classdesc Represents a tensor transpose operation node.
 * @extends Node
 */
class TransposeNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    
    /**
     * @param {Object} api_response - API response for TransposeNode.
     * @param {number} api_response.dim0 - First dimension to transpose.
     * @param {number} api_response.dim1 - Second dimension to transpose.
     */
    constructor(api_response) {
        super(api_response);
        /** @type {number} */
        this.dim0 = api_response.dim0;
        /** @type {number} */
        this.dim1 = api_response.dim1;
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as the input.
        return inputsMap.get(TransposeNode.INPUT) || 0;
    }

    get_inputs() { return [TransposeNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _normalize_dim(dim, rank) {
        if (dim < 0) {
            return dim + rank;
        }
        return dim;
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(TransposeNode.INPUT) || executionContext.gpu(TransposeNode.INPUT);
        if (!inputTensor || !inputTensor.shape) {
            throw new Error(`TransposeNode (${this.name}): Missing input tensor or shape.`);
        }
        const inputShape = [...inputTensor.shape]; // Make a copy
        const rank = inputShape.length;

        if (rank === 0) return []; // Transpose of a scalar is the scalar itself

        const d0 = this._normalize_dim(this.dim0, rank);
        const d1 = this._normalize_dim(this.dim1, rank);

        if (d0 < 0 || d0 >= rank || d1 < 0 || d1 >= rank) {
            throw new Error(`TransposeNode (${this.name}): Invalid dimensions (${this.dim0}, ${this.dim1}) for rank ${rank}.`);
        }

        // Swap the dimensions in the shape array
        const temp = inputShape[d0];
        inputShape[d0] = inputShape[d1];
        inputShape[d1] = temp;
        return inputShape;
    }

    async getGPUKernel() {
        return new GPUKernel({
            name: 'transpose',
            shader: await fetch('kernels/transpose.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (executionContext) => {
                    const inputTensor = executionContext.gpu(TransposeNode.INPUT);
                    if (!inputTensor) {
                        throw new Error(`TransposeNode (${this.name}): Missing GPU input tensor for dimensionBuffer.`);
                    }
                    const inputShape = inputTensor.shape;
                    const rank = inputShape.length;

                    const d0 = this._normalize_dim(this.dim0, rank);
                    const d1 = this._normalize_dim(this.dim1, rank);

                    if (rank > MAX_DIMS_SOFTMAX) {
                        throw new Error(`TransposeNode (${this.name}): Input tensor rank (${rank}) exceeds MAX_DIMS_SOFTMAX (${MAX_DIMS_SOFTMAX}).`);
                    }

                    const paddedInputShape = new Array(MAX_DIMS_SOFTMAX).fill(1);
                    const paddedInputStrides = new Array(MAX_DIMS_SOFTMAX).fill(0);
                    
                    if (rank > 0) {
                        for (let i = 0; i < rank; i++) {
                            paddedInputShape[i] = inputShape[i];
                        }
                        const inputStrides = calculateStrides(inputShape);
                        for (let i = 0; i < rank; i++) {
                            paddedInputStrides[i] = inputStrides[i];
                        }
                    }
                    // For 0D tensor, shape [1,...], strides [0,...], rank 0 is fine.

                    const num_elements = rank > 0 ? inputShape.reduce((acc, val) => acc * val, 1) : 1;

                    // Uniform buffer layout: input_shape_vecs, input_strides_vecs, rank, dim0, dim1, num_elements, padding
                    // Padded sizes: 8 (shape) + 8 (strides) + 1 (rank) + 1 (d0) + 1 (d1) + 1 (num_elements) = 20 u32s
                    // Total 20 * 4 = 80 bytes. This is a multiple of 16, so no extra padding u32s needed if aligned.
                    const uniformData = new Uint32Array(MAX_DIMS_SOFTMAX * 2 + 4); 
                    uniformData.set(paddedInputShape, 0);
                    uniformData.set(paddedInputStrides, MAX_DIMS_SOFTMAX);
                    uniformData.set([rank, d0, d1, num_elements], MAX_DIMS_SOFTMAX * 2);
                    return uniformData;
                },
                index: 2, // Binding for params uniform buffer
            },
            workgroupFunction: (executionContext) => {
                const inputTensor = executionContext.gpu(TransposeNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`TransposeNode (${this.name}): Missing GPU input tensor for workgroupFunction.`);
                }
                const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                const workgroupSizeX = 256; // Must match shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: TransposeNode.INPUT, cpu: false, binding: {type: "read-only-storage", index: 0 } }
            ],
            outputs: [
                { name: DEFAULT_NODE_OUTPUT, binding: {type: "storage", index: 1 } },
            ],
        });
    }
}

/**
 * @class AddNode
 * @classdesc Represents an element-wise addition node.
 * @extends Node
 */
class AddNode extends Node {
    /** @type {string} */
    static A = "a";
    /** @type {string} */
    static B = "b";
    
    /**
     * @param {Object} api_response - API response for AddNode.
     */
    constructor(api_response) {
        super(api_response);
        // Add nodes typically run on CPU by default unless specified otherwise
        // Or they could be fused into GPU kernels. Let's assume CPU for now.
        if (!this.devicePreference) {
             this.devicePreference = new DevicePreferences({ supportsGPU: true });
        } 
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as input A (assuming A and B are same size).
        return inputsMap.get(AddNode.A) || 0;
    }

    /**
     * @returns {Promise<GPUKernel>}
     */
    async getGPUKernel() {
        // From the horse's mouth: https://toji.dev/webgpu-best-practices/dynamic-shader-construction.html
        // This is a nice way to dynamically specialize kernels. The Kernel class has a key() method that allows as
        // much caching as possible. See GPUKernel for details.
        return new GPUKernel({
            name: 'add',
            shader: await fetch('kernels/add.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (inputShapes, outputShapes) => {
                    let size = inputShapes.get(AddNode.A).reduce((a, b) => a * b, 1);
                    return new Uint32Array([size]);
                },
                index: 3,
            },
            workgroupFunction: (inputShapes, outputShapes) => {
                let size = inputShapes.get(AddNode.A).reduce((a, b) => a * b, 1);
                return {
                    x: Math.ceil(size / 256),
                    y: 1,
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: AddNode.A, cpu: false, binding: {type: "read-only-storage", index: 0 } },
                { name: AddNode.B, cpu: false, binding: {type: "read-only-storage", index: 1 } },
            ],
            outputs: [
                { name: DEFAULT_NODE_OUTPUT, binding: {type: "storage", index: 2 } },
            ],
        });
    }


    /**
     * Calculates the output shape for an element-wise addition.
     * Assumes input shapes are identical (no broadcasting yet).
     * @param {Map<string, number[]>} inputShapes - A map containing shapes for 'inputA' and 'inputB'.
     *                                            Example: Map { 'inputA' => [X, Y], 'inputB' => [X, Y] }
     * @returns {number[]} The output shape, same as input shapes.
     * @throws {Error} If input shapes are missing or incompatible.
     */
    getOutputShape(inputShapes) {
        const shapeA = inputShapes.get(AddNode.A);
        const shapeB = inputShapes.get(AddNode.B);

        if (!shapeA || !shapeB) {
            throw new Error(`AddNode (${this.name}): Missing required input shapes ('inputA' or 'inputB').`);
        }

        // Simple equality check for now (convert to string for easy comparison)
        // TODO: Implement proper broadcasting rules if needed.
        if (shapeA.length !== shapeB.length || !shapeA.every((dim, i) => dim === shapeB[i])) {
             throw new Error(`AddNode (${this.name}): Input shapes must be identical for simple addition. Got ${shapeA} and ${shapeB}.`);
        }

        // Output shape is the same as the (identical) input shapes
        return [...shapeA]; // Return a copy
    }

    get_inputs() { return [AddNode.A, AddNode.B]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
}

/**
 * @class DivNode
 * @classdesc Represents an element-wise division node. Unlike other
 * element-wise operations, this is implemented on the CPU purely because I know
 * that it is usually used in shape math
 * @extends Node
 */
class DivNode extends Node {
    /** @type {string} */
    static A = "a";
    /** @type {string} */
    static B = "b";
    
    /**
     * @param {Object} api_response - API response for DivNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true, supportsGPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as input A (assuming A and B are same size).
        return inputsMap.get(DivNode.A) || 0;
    }

    get_inputs() { return [DivNode.A, DivNode.B]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(tensorA, tensorB) {
        if (!tensorA || !tensorB) {
            throw new Error(`DivNode (${this.name}): Missing required input tensors ('a' or 'b').`);
        }
        // For now, require identical shapes. Broadcasting could be added later.
        if (tensorA.shape.length !== tensorB.shape.length || 
            !tensorA.shape.every((dim, i) => dim === tensorB.shape[i])) {
            throw new Error(`DivNode (${this.name}): Input shapes must be identical for simple division. Got ${tensorA.shape} and ${tensorB.shape}.`);
        }
        if (tensorA.dtype !== tensorB.dtype) {
            // Or we could cast, but for now require same dtype for simplicity
            // throw new Error(`DivNode (${this.name}): Input dtypes must be identical. Got ${tensorA.dtype} and ${tensorB.dtype}.`);
            // Relaxing this: output dtype will be taken from tensorA. User should ensure compatibility.
             console.warn(`DivNode (${this.name}): Input dtypes are different (${tensorA.dtype}, ${tensorB.dtype}). Output dtype will be float32.`);
        }
        // Division generally implies a float result unless explicitly integer division.
        // To handle mixed mode (like int/int resulting in float for gt check) and float/float, default to float32.
        return [...tensorA.shape]; // Output shape is same as input shape
    }

    getOutputShape(executionContext) {
        const tensorA = executionContext.cpu(DivNode.A) || executionContext.gpu(DivNode.A);
        const tensorB = executionContext.cpu(DivNode.B) || executionContext.gpu(DivNode.B);
        // Shape is determined by inputs, dtype is handled in kernel/fixed to float32
        return this._calculateOutputShape(tensorA, tensorB); 
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'div',
            func: (executionContext) => {
                const tensorA = executionContext.cpu(DivNode.A);
                const tensorB = executionContext.cpu(DivNode.B);

                if (!tensorA || !tensorB) {
                    throw new Error(`DivNode (${this.name}) kernel: Missing input tensors.`);
                }

                const outputShape = this._calculateOutputShape(tensorA, tensorB);
                // Division output is set to float32 to handle mixed types and general expectations.
                const outputTensor = CPUTensor.uninitialized(outputShape, 'float32');
                const outputView = outputTensor.getTypedArray(); // This will be a Float32Array

                const dataA = tensorA.getTypedArray();
                const dataB = tensorB.getTypedArray();

                if (dataA.length !== dataB.length) { // Should be caught by shape check already
                    throw new Error(`DivNode (${this.name}) kernel: Input arrays have mismatched lengths.`);
                }

                for (let i = 0; i < outputView.length; i++) {
                    if (dataB[i] === 0) {
                        // Handle division by zero: return NaN or Infinity based on numerator, or throw error
                        // JS default: 1/0=Infinity, 0/0=NaN, -1/0=-Infinity
                        // Let standard JS behavior apply here. Output typed array will handle conversion.
                        console.warn(`DivNode (${this.name}) kernel: Division by zero at index ${i}.`);
                    }
                    outputView[i] = dataA[i] / dataB[i];
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [DivNode.A, DivNode.B],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }

    async getGPUKernel() {
        return new GPUKernel({
            name: 'divide',
            shader: await fetch('kernels/divide.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (executionContext) => {
                    const tensorA = executionContext.gpu(DivNode.A);
                    if (!tensorA) {
                        throw new Error(`DivNode (${this.name}): Missing GPU tensor A for dimensionBuffer.`);
                    }
                    const num_elements = tensorA.shape.reduce((acc, val) => acc * val, 1);
                    return new Uint32Array([num_elements, 0]); // num_elements, padding
                },
                index: 3, // Matches @group(0) @binding(3) in shader
            },
            workgroupFunction: (executionContext) => {
                const tensorA = executionContext.gpu(DivNode.A);
                 if (!tensorA) {
                    throw new Error(`DivNode (${this.name}): Missing GPU tensor A for workgroupFunction.`);
                }
                const num_elements = tensorA.shape.reduce((acc, val) => acc * val, 1);
                const workgroupSizeX = 256; // Must match WORKGROUP_SIZE_X in shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: DivNode.A, cpu: false, binding: {type: "read-only-storage", index: 0 } },
                { name: DivNode.B, cpu: false, binding: {type: "read-only-storage", index: 1 } },
            ],
            outputs: [
                { name: DEFAULT_NODE_OUTPUT, binding: {type: "storage", index: 2 } },
            ],
        });
    }
}

/**
 * @class FloorNode
 * @classdesc Represents an element-wise floor operation node.
 * @extends Node
 */
class FloorNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    
    /**
     * @param {Object} api_response - API response for FloorNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as the input.
        return inputsMap.get(FloorNode.INPUT) || 0;
    }

    get_inputs() { return [FloorNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(tensor) {
        if (!tensor) {
            throw new Error(`FloorNode (${this.name}): Missing required input tensor.`);
        }
        return [...tensor.shape]; // Output shape is same as input shape
    }

    getOutputShape(executionContext) {
        const tensor = executionContext.cpu(FloorNode.INPUT);
        return this._calculateOutputShape(tensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'floor',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(FloorNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`FloorNode (${this.name}) kernel: Missing input tensor.`);
                }

                const outputShape = this._calculateOutputShape(inputTensor);
                // Output tensor will have int32 dtype as per PyTorch expectation for floor/ceil
                const outputTensor = CPUTensor.uninitialized(outputShape, 'int32');
                const outputView = outputTensor.getTypedArray(); // This will be an Int32Array
                const inputData = inputTensor.getTypedArray();

                for (let i = 0; i < outputView.length; i++) {
                    outputView[i] = Math.floor(inputData[i]);
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [FloorNode.INPUT],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }
}

/**
 * @class CeilNode
 * @classdesc Represents an element-wise ceil operation node.
 * @extends Node
 */
class CeilNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    
    /**
     * @param {Object} api_response - API response for CeilNode.
     */
    constructor(api_response) {
        super(api_response);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    estimateWeight(inputsMap) {
        // Output has the same number of elements as the input.
        return inputsMap.get(CeilNode.INPUT) || 0;
    }

    get_inputs() { return [CeilNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(tensor) {
        if (!tensor) {
            throw new Error(`CeilNode (${this.name}): Missing required input tensor.`);
        }
        return [...tensor.shape]; // Output shape is same as input shape
    }

    getOutputShape(executionContext) {
        const tensor = executionContext.cpu(CeilNode.INPUT);
        return this._calculateOutputShape(tensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'ceil',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(CeilNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`CeilNode (${this.name}) kernel: Missing input tensor.`);
                }

                const outputShape = this._calculateOutputShape(inputTensor);
                // Output tensor will have int32 dtype as per PyTorch expectation for floor/ceil
                const outputTensor = CPUTensor.uninitialized(outputShape, 'int32');
                const outputView = outputTensor.getTypedArray(); // This will be an Int32Array
                const inputData = inputTensor.getTypedArray();

                for (let i = 0; i < outputView.length; i++) {
                    outputView[i] = Math.ceil(inputData[i]);
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [CeilNode.INPUT],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }
}

/**
 * @class DebugNode
 * @classdesc Represents a debug node.
 * @extends Node
 */
class DebugNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    
    /**
     * @param {Object} api_response - API response for DebugNode.
     */
    constructor(api_response) {
        super(api_response);
    }

    estimateWeight(inputsMap) {
        // Pass-through node, output elements same as input.
        return inputsMap.get(DebugNode.INPUT) || 0;
    }

    get_inputs() { return [DebugNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; } // Assume pass-through for now
}

/**
 * @class Graph
 * @classdesc Represents a computation graph.
 */
export class Graph {
    /*
     * @param {Object} api_response
     * @param {Object.<string, Object>} api_response.nodes - A map of node names to node API responses.
     * @param {Array<Object>} api_response.edges - An array of edge API responses.
     */
    constructor(api_response) {
        /** @type {Object.<string, Node>} */
        this.nodes = {};
        for (const [key, value] of Object.entries(api_response.nodes)) {
            // Create the appropriate node type based on the value.type
            if (value.type === "matmul") {
                this.nodes[key] = new MatmulNode(value);
            } else if (value.type === "constant") {
                this.nodes[key] = new ConstantNode(value);
            } else if (value.type === "softmax") {
                this.nodes[key] = new SoftmaxNode(value);
            } else if (value.type === "slice") {
                this.nodes[key] = new SliceNode(value);
            } else if (value.type === "reshape") {
                this.nodes[key] = new ReshapeNode(value);
            } else if (value.type === "unsqueeze") {
                this.nodes[key] = new UnsqueezeNode(value);
            } else if (value.type === "broadcast") {
                this.nodes[key] = new BroadcastNode(value);
            } else if (value.type === "cat") {
                this.nodes[key] = new CatNode(value);
            } else if (value.type === "fixed") {
                this.nodes[key] = new FixedNode(value);
            } else if (value.type === "hadamard") {
                this.nodes[key] = new HadamardNode(value);
            } else if (value.type === "index") {
                this.nodes[key] = new IndexNode(value);
            } else if (value.type === "shape") {
                this.nodes[key] = new ShapeNode(value);
            } else if (value.type === "transpose") {
                this.nodes[key] = new TransposeNode(value);
            } else if (value.type === "add") {
                this.nodes[key] = new AddNode(value);
            } else if (value.type === "div") {
                this.nodes[key] = new DivNode(value);
            } else if (value.type === "floor") {
                this.nodes[key] = new FloorNode(value);
            } else if (value.type === "ceil") {
                this.nodes[key] = new CeilNode(value);
            } else {
                // Throw error for unknown node types
                throw new Error(`Unknown node type: ${value.type}`);
            }
        }
        /** @type {Edge[]} */
        this.edges = api_response.edges.map((e) => new Edge(e));
    }

    /*
     * Dumps graph to console
     */
    dump() {
	console.log(this);
    }

    /*
     * Returns a list of nodes in topological order
     * @returns {Node[]} - List of nodes in dependency-satisfied order
     */
    topologicalSort() {
        // Build adjacency list and in-degree count
        const adjacencyList = new Map();
        const inDegree = new Map();
        
        // Initialize adjacency list and in-degree for all nodes
        for (const nodeName of Object.keys(this.nodes)) {
            adjacencyList.set(nodeName, []);
            inDegree.set(nodeName, 0);
        }

        // Build adjacency list and update in-degree counts
        for (const edge of this.edges) {
            adjacencyList.get(edge.src).push(edge.dst);
            inDegree.set(edge.dst, inDegree.get(edge.dst) + 1);
        }

        // Find all nodes with no incoming edges
        const queue = [];
        for (const [nodeName, degree] of inDegree.entries()) {
            if (degree === 0) {
                queue.push(nodeName);
            }
        }

        const result = [];
        while (queue.length > 0) {
            const nodeName = queue.shift();
            result.push(this.nodes[nodeName]);

            // Decrease in-degree for all neighbors
            for (const neighbor of adjacencyList.get(nodeName)) {
                inDegree.set(neighbor, inDegree.get(neighbor) - 1);
                if (inDegree.get(neighbor) === 0) {
                    queue.push(neighbor);
                }
            }
        }

        // Check for cycles
        if (result.length !== Object.keys(this.nodes).length) {
            throw new Error("Graph contains a cycle");
        }

        return result;
    }
}

/**
 * @class InputAssignment
 * @classdesc Represents an input assignment for a graph node.
 */
class InputAssignment {
    /** @type {string} */
    node;
    /** @type {string} */
    input;
    /** @type {CPUTensor} */
    tensor;

    /**
     * @param {Object} api_response - Input assignment response from server
     * @param {string} api_response.node - Node to assign input to
     * @param {string} api_response.input - Input to assign
     * @param {APITensor} api_response.tensor - Tensor to assign
     */
    constructor(api_response) {
        this.node = api_response.node;
        this.input = api_response.input;
        // Directly create CPUTensor from the raw tensor data in the API response
        this.tensor = (new APITensor(api_response.tensor)).toCPU();
        console.log("constructed", this)
    }
}

/**
 * @class OutputAssignment
 * @classdesc Represents an output assignment for a graph node.
 */
export class OutputAssignment {
    /** @type {string} */
    node;
    /** @type {string} */
    output;
    /** @type {CPUTensor} */
    tensor;

    /**
     * @param {Object} options - Output assignment options
     * @param {string} options.node - Node to assign output to
     * @param {string} options.output - Output to assign
     * @param {CPUTensor} options.tensor - Tensor to assign
     */
    constructor(options) {
        this.node = options.node;
        this.output = options.output;
        this.tensor = options.tensor;
    }

    toAPI() {
        return {
            node: this.node,
            output: this.output,
            tensor: APITensor.fromCPU(this.tensor),
        };
    }
}

/**
 * @class PartitionWork
 * @classdesc Represents a unit of work for a partition.
 */
export class PartitionWork {
    /** @type {string} */ 
    correlation_id;
    /** @type {string} */
    partition;
    /** @type {Graph} */
    graph;
    /** @type {InputAssignment[]} */
    inputs;

    /**
     * @param {Object} api_response - Partition work response from server
     * @param {string} api_response.correlation_id - Correlation ID of the work
     * @param {string} api_response.partition - Partition to get work for
     * @param {Object} api_response.graph - Graph to execute
     * @param {Array<Object>} api_response.inputs - Inputs to the graph, as API responses.
     */
    constructor(api_response) {
        this.correlation_id = api_response.correlation_id;
        this.partition = api_response.partition;
        this.graph = new Graph(api_response.graph);
        this.inputs = api_response.inputs.map(ia => new InputAssignment(ia));
    }
}

/**
 * @class PartitionWorkResult
 * @classdesc Represents the result of executing partition work.
 */
export class PartitionWorkResult {
    /** @type {string} */
    correlation_id;
    /** @type {string} */
    partition;
    /** @type {OutputAssignment[]} */
    outputs;

    /**
     * @param {Object} options - Partition work result response from server
     * @param {string} options.correlation_id - Correlation ID of the work
     * @param {string} options.partition - Partition to submit work for
     * @param {Array<OutputAssignment>} options.outputs - Outputs from the graph.
     */
    constructor(options) {
        this.correlation_id = options.correlation_id;
        this.partition = options.partition;
        this.outputs = options.outputs;
    }

    toAPI() {
        return {
            correlation_id: this.correlation_id,
            partition: this.partition,
            outputs: this.outputs.map((m) => m.toAPI()),
        };
    }
}

/**
 * @class Coordinator
 * @classdesc Handles communication with the coordination server.
 */
export class Coordinator {
    /*
     * @param {Object} options - Coordinator configuration options
     * @param {string} options.url - URL of the coordination server
     */
    constructor(options) {
        /** @type {string} */
        this.url = options.url;
    }

    /*
     * Registers the worker with the coordination server
     * @returns {Promise<Registration>} - Registration response from server
     */
    async register() {
        const response = await fetch(`${this.url}/register`, {
            method: "POST",
            body: JSON.stringify({}),
        });
        return new Registration(await response.json());
    }

    /**
     * Gets the next partition work from the coordination server
     * @param {string} partition_name - Partition to get work for
     * @returns {PartitionWork | null} - Partition work from server, or null if no work is available
     */
    async get_work(partition_name) {
        const response = await fetch(`${this.url}/work/${partition_name}`);
        return new PartitionWork(await response.json());
    }

    /**
     * Submits the partition work to the coordination server
     * @param {PartitionWorkResult} work - Partition work to submit.
     * @returns {Promise<void>}
     */
    async submit_work(work) {
	await fetch(`${this.url}/work`, {
	    method: "POST",
	    body: JSON.stringify(work.toAPI()),
        headers: {
            ["Content-Type"]: 'application/json'
        }
	});
    }
}

/**
 * @class PreparedGraph
 * @classdesc Represents a graph that has been prepared for execution.
 *  (Currently a placeholder)
 */
class PreparedGraph {

}

/**
 * @class Worker
 * @classdesc Handles the execution of partition work.
 */
export class Worker {
    constructor() {
        /** @type {Object.<string, PreparedGraph>} */
        this.prepared_graphs = {};
    }

    /*
     * @param {PartitionWork} partition_work - Partition work to execute
     * @returns {Promise<PartitionWorkResult>}
     */
    async execute_work(partition_work) {
        await this._prepare_partition_work(partition_work);
        await this._execute_partition_work(partition_work);
    }

    /*
     * Prepares the partition work for execution
     *
     * @param {PartitionWork} partition_work - Partition work to prepare
     * @returns {Promise<void>}
     */
    async _prepare_partition_work(partition_work) {
        if (this.prepared_graphs[partition_work.partition]) {
            return;
        }
        // TODO: Implement graph preparation logic
        // This might involve:
        // 1. Allocating WebGPU buffers for tensors
        // 2. Compiling and caching GPUKernels
        // 3. Storing the prepared graph (buffers, kernels) in this.prepared_graphs
        this.prepared_graphs[partition_work.partition] = new PreparedGraph();
    }

    /*
     * @param {PartitionWork} partition_work - Partition work to execute
     * @returns {Promise<PartitionWorkResult>}
     */
    async _execute_partition_work(partition_work) {
        // TODO: Implement actual graph execution
        // This will involve:
        // 1. Getting the prepared graph for the partition
        // 2. Setting input tensor data from partition_work.inputs
        // 3. Executing nodes in topological order
        //    - For GPU nodes: dispatching GPUKernels
        //    - For CPU nodes: performing CPU computations
        // 4. Reading output tensor data
        // 5. Constructing and returning a PartitionWorkResult

        // Placeholder:
        return new PartitionWorkResult({
            correlation_id: partition_work.correlation_id,
            partition: partition_work.partition,
            outputs: [] // Replace with actual outputs
        });
    }
}
