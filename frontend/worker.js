/*
 * Library to handle interfacing with coordination server
 */

import { 
    readBEInt, sizeEncodedString, readEncodedString, 
    writeBEInt, writeEncodedString, 
    writeBool,
    readBool
} from "./encoding.js";
import { CPUKernel, CPUTensor, GPUKernel, GPUTensor } from "./kernel.js";
import { SafeTensorCache } from "./tensorcache.js";

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
     * @param {Object} options
     * @param {string} options.src
     * @param {string} options.src_output
     * @param {string} options.dst
     * @param {string} options.dst_input
     */
    constructor(options) {
        /** @type {string} */
	this.src = options.src;
        /** @type {string} */
	this.src_output = options.src_output;
        /** @type {string} */
	this.dst = options.dst;
        /** @type {string} */
	this.dst_input = options.dst_input;
    }

    /**
     * @param {DataView} view
     * @param {number} offset
     * @returns {[Edge, number]}
     */
    static decode(view, offset) {
        let src, src_output, dst, dst_input;
        [src, offset] = readEncodedString(view, offset);
        [src_output, offset] = readEncodedString(view, offset);
        [dst, offset] = readEncodedString(view, offset);
        [dst_input, offset] = readEncodedString(view, offset);
        return [new Edge({ src, src_output, dst, dst_input }), offset];
    }
}

/**
 * @class Node
 * @classdesc Base class for all graph nodes.
 */
export class Node {
    /** 
     * @param {Object} options - Parameters for Node.
     * @param {string} options.type - Type of node
     * @param {string} options.name - Name of node
     */
    constructor(options) {
        /** @type {string} */
        this.type = options.type;
        /** @type {string} */
        this.name = options.name;
        
        // Copy any additional properties from the options
        for (const [key, value] of Object.entries(options)) {
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
     * @param {Map<string, SafetensorCache>} safetensorCache
     */
    constructor(cpuInputs, gpuInputs, safetensorCache) {
        this.cpuInputs = cpuInputs || new Map();
        this.gpuInputs = gpuInputs || new Map();
        this.safetensorCache = safetensorCache || new SafeTensorCache();
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

    /**
     * @returns {SafeTensorCache}
     */
    cache() {
        return this.safetensorCache;
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
     * @param {Object} options - Options for MatmulNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
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

    static decode(view, offset, name, partition, type) {
        return [new MatmulNode({ name, partition, type }), offset];
    }


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
                    const lhsShape = lhsTensor.shape;
                    const rhsShape = rhsTensor.shape;
                    let B, M, K, N;

                    if (lhsShape.length === 2 && rhsShape.length === 2) { // 2D case
                        B = 1;
                        M = lhsShape[0];
                        K = lhsShape[1];
                        N = rhsShape[1];
                    } else if (
                        lhsShape.length > 2 && // more than 2D
                        lhsShape.length === rhsShape.length && // equal ranks
                        lhsShape.slice(0, -2).every((dim, i) => dim === rhsShape.slice(0, -2)[i]) // equal batch dimensions
                    ) { 
                        B = lhsShape.slice(0, -2).reduce((a, b) => a * b, 1);
                        M = lhsShape[lhsShape.length - 2];
                        K = lhsShape[lhsShape.length - 1];
                        N = rhsShape[rhsShape.length - 1];
                    } else {
                        throw new Error(`MatmulNode (${this.name}): Incompatible input tensor dimensions for dimensionBuffer. Both must be 2D or both ND with matching batch dimensions. Got ${lhsShape} and ${rhsShape}`);
                    }
                    return new Uint32Array([B, M, K, N]);
                },
                index: 0,
            },
            workgroupFunction: (executionContext) => {
                const lhsTensor = executionContext.gpu(MatmulNode.LHS);
                const rhsTensor = executionContext.gpu(MatmulNode.RHS);

                if (!lhsTensor || !rhsTensor) {
                    throw new Error(`MatmulNode (${this.name}): Missing GPU tensors for workgroupFunction.`);
                }

                const lhsShape = lhsTensor.shape;
                const rhsShape = rhsTensor.shape;
                let M_dim, N_dim, B_dim;

                if (lhsShape.length === 2 && rhsShape.length === 2) { // 2D case
                    M_dim = lhsShape[0];
                    N_dim = rhsShape[1];
                    B_dim = 1;
                } else if (
                    lhsShape.length > 2 &&
                    lhsShape.length === rhsShape.length &&
                    lhsShape.slice(0, -2).every((dim, i) => dim === rhsShape.slice(0, -2)[i])
                ) { // N-D case
                    B_dim = lhsShape.slice(0, -2).reduce((a, b) => a * b, 1);
                    M_dim = lhsShape[lhsShape.length - 2];
                    N_dim = rhsShape[rhsShape.length - 1];
                } else {
                    throw new Error(`MatmulNode (${this.name}): Incompatible input tensor dimensions for workgroupFunction. Both must be 2D or both ND with matching batch dimensions. Got ${lhsShape} and ${rhsShape}`);
                }

                const TILE_DIM = 4; // Corresponds to TILE_M, TILE_N in WGSL shader
                const WORKGROUP_XY_DIM = 16; // Corresponds to BLOCKSIZE in WGSL @workgroup_size(BLOCKSIZE, BLOCKSIZE, 1)

                const num_tiles_m = Math.ceil(M_dim / TILE_DIM);
                const num_tiles_n = Math.ceil(N_dim / TILE_DIM);

                const workgroupsX = Math.ceil(num_tiles_n / WORKGROUP_XY_DIM);
                const workgroupsY = Math.ceil(num_tiles_m / WORKGROUP_XY_DIM);
                const workgroupsZ = B_dim;
                
                return {
                    x: workgroupsX,
                    y: workgroupsY,
                    z: workgroupsZ,
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
        const shapeA = executionContext.gpu(MatmulNode.LHS)?.shape || executionContext.cpu(MatmulNode.LHS)?.shape;
        const shapeB = executionContext.gpu(MatmulNode.RHS)?.shape || executionContext.cpu(MatmulNode.RHS)?.shape;

        if (!shapeA || !shapeB) {
            throw new Error(`MatmulNode (${this.name}): Missing required input shapes for LHS or RHS.`);
        }

        const rankA = shapeA.length;
        const rankB = shapeB.length;

        if (rankA === 2 && rankB === 2) {
            const M = shapeA[0];
            const K_A = shapeA[1];
            const K_B = shapeB[0];
            const N = shapeB[1];

            if (K_A !== K_B) {
                throw new Error(`MatmulNode (${this.name}): Inner dimensions for 2D matmul do not match. LHS K=${K_A}, RHS K=${K_B}. Shapes: ${shapeA} and ${shapeB}.`);
            }
            return [M, N];
        } else if (rankA > 2 && rankA === rankB) {
            const batchDimsA = shapeA.slice(0, -2);
            const batchDimsB = shapeB.slice(0, -2);

            if (!batchDimsA.every((dim, i) => dim === batchDimsB[i])) {
                throw new Error(`MatmulNode (${this.name}): Batch dimensions for ND matmul do not match. LHS Batch: ${batchDimsA}, RHS Batch: ${batchDimsB}. Shapes: ${shapeA} and ${shapeB}.`);
            }

            const M = shapeA[rankA - 2];
            const K_A = shapeA[rankA - 1];
            const K_B = shapeB[rankB - 2];
            const N = shapeB[rankB - 1];

            if (K_A !== K_B) {
                throw new Error(`MatmulNode (${this.name}): Inner dimensions for ND matmul do not match. LHS K=${K_A}, RHS K=${K_B}. Shapes: ${shapeA} and ${shapeB}.`);
            }
            return [...batchDimsA, M, N];
        } else {
            throw new Error(`MatmulNode (${this.name}): Inputs must both be 2D or both be ND tensors of the same rank with matching batch dimensions. Got shapes A: ${shapeA} (rank ${rankA}) and B: ${shapeB} (rank ${rankB}).`);
        }
    }
}

/**
 * @class SafetensorNode
 * @classdesc Represents a node that outputs a safetensor.
 * @extends Node
 */
class SafetensorNode extends Node {
    /**
     * @param {Object} options - Options for SafetensorNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     * @param {string} options.tensor_name - Name of the tensor.
     * @param {string} options.model_name - Name of the model.
     */
    constructor(options) {
        super(options);
        this.tensor_name = options.tensor_name;
        this.model_name = options.model_name;
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        let tensor_name, model_name;
        [model_name, offset] = readEncodedString(view, offset);
        [tensor_name, offset] = readEncodedString(view, offset);
        return [new SafetensorNode({ name, partition, type, tensor_name, model_name }), offset];
    }

    estimateWeight(inputsMap) {
        // Placeholder: SafetensorNode cannot determine its actual size from inputsMap
        // and doesn't store its shape/tensor directly.
        // This should ideally be seeded by _computeWeights if shape info is available
        // or SafetensorNode should store its shape.
        return 1;
    }

    get_inputs() { return []; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getCPUKernel() {
        return new CPUKernel({
            name: 'safetensor',
            func: async (executionContext) => {
                const cache = executionContext.cache();
                const tensor = await cache.getTensor(this.model_name, this.tensor_name);
                if (!tensor) {
                    throw new Error(`SafetensorNode (${this.name}): Tensor not found in cache.`);
                }
                return {
                    [DEFAULT_NODE_OUTPUT]: tensor,
                };
            },
            inputs: [], // Specify input names
            outputs: [DEFAULT_NODE_OUTPUT], // Specify output name
        });
    }
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
     * @param {Object} options - Options for SoftmaxNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new SoftmaxNode({ name, partition, type }), offset];
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
     * @param {Object} options - Options for SliceNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new SliceNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        // Estimate: output elements are same as input (overestimate if not full slice).
        // A more accurate estimate would need slice parameters (start, end, dim sizes).
        return inputsMap.get(SliceNode.INPUT) || 0;
    }

    _calculateOutputShape(inputTensor, dimTensor, startTensor, endTensor) {
        if (!inputTensor) {
            throw new Error(`SliceNode (${this.name}): Missing input tensor.`);
        }
        if (!dimTensor) {
            throw new Error(`SliceNode (${this.name}): Missing dim tensor.`);
        }
        if (!startTensor) {
            throw new Error(`SliceNode (${this.name}): Missing start tensor.`);
        }
        if (!endTensor) {
            throw new Error(`SliceNode (${this.name}): Missing end tensor.`);
        }

        const inputShape = inputTensor.shape;
        const rank = inputShape.length;

        if (dimTensor.shape.length !== 1 || dimTensor.shape[0] !== 1) {
            throw new Error(`SliceNode (${this.name}): Dimension ('dim') must be a scalar tensor. Got shape ${dimTensor.shape}`);
        }
        let dim = dimTensor.getTypedArray()[0];
        if (dim < 0) {
            dim = rank + dim;
        }
        if (dim < 0 || dim >= rank) {
            throw new Error(`SliceNode (${this.name}): Dimension out of range. Got dim ${dimTensor.getTypedArray()[0]} for input rank ${rank}`);
        }

        if (startTensor.shape.length !== 1 || startTensor.shape[0] !== 1) {
            throw new Error(`SliceNode (${this.name}): Start index ('start') must be a scalar tensor. Got shape ${startTensor.shape}`);
        }
        let start = startTensor.getTypedArray()[0];

        if (endTensor.shape.length !== 1 || endTensor.shape[0] !== 1) {
            throw new Error(`SliceNode (${this.name}): End index ('end') must be a scalar tensor. Got shape ${endTensor.shape}`);
        }
        let end = endTensor.getTypedArray()[0];

        const dimSize = inputShape[dim];

        // Normalize start and end
        if (start < 0) {
            start = dimSize + start;
        }
        if (end < 0) {
            end = dimSize + end;
        }

        // Clamp start and end to valid range
        start = Math.max(0, Math.min(start, dimSize));
        end = Math.max(0, Math.min(end, dimSize));

        if (end < start) {
            end = start; // Produces an empty slice along this dimension
        }

        const outputShape = [...inputShape];
        outputShape[dim] = end - start;
        return outputShape;
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(SliceNode.INPUT);
        const dimTensor = executionContext.cpu(SliceNode.DIM);
        const startTensor = executionContext.cpu(SliceNode.START);
        const endTensor = executionContext.cpu(SliceNode.END);
        return this._calculateOutputShape(inputTensor, dimTensor, startTensor, endTensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'slice',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(SliceNode.INPUT);
                const dimTensor = executionContext.cpu(SliceNode.DIM);
                const startTensor = executionContext.cpu(SliceNode.START);
                const endTensor = executionContext.cpu(SliceNode.END);

                if (!inputTensor || !dimTensor || !startTensor || !endTensor) {
                    throw new Error(`SliceNode (${this.name}) kernel: Missing one or more input tensors.`);
                }

                const outputShape = this._calculateOutputShape(inputTensor, dimTensor, startTensor, endTensor);
                const outputTensor = CPUTensor.uninitialized(outputShape, inputTensor.dtype);
                const outputView = outputTensor.getTypedArray();
                const inputView = inputTensor.getTypedArray();

                const inputShape = inputTensor.shape;
                const rank = inputShape.length;
                let sliceDim = dimTensor.getTypedArray()[0];
                if (sliceDim < 0) {
                    sliceDim = rank + sliceDim;
                }
                let sliceStart = startTensor.getTypedArray()[0];
                let sliceEnd = endTensor.getTypedArray()[0];
                
                const dimSize = inputShape[sliceDim];
                if (sliceStart < 0) sliceStart = dimSize + sliceStart;
                if (sliceEnd < 0) sliceEnd = dimSize + sliceEnd;
                sliceStart = Math.max(0, Math.min(sliceStart, dimSize));
                sliceEnd = Math.max(0, Math.min(sliceEnd, dimSize));
                if (sliceEnd < sliceStart) sliceEnd = sliceStart;


                const inputStrides = calculateStrides(inputShape);
                const outputStrides = calculateStrides(outputShape);
                
                let outputFlatIndex = 0;
                const totalOutputElements = outputShape.reduce((acc, val) => acc * val, 1);

                for (outputFlatIndex = 0; outputFlatIndex < totalOutputElements; outputFlatIndex++) {
                    let inputFlatIndex = 0;
                    let currentOutputFlatIndex = outputFlatIndex;
                    
                    for (let d = 0; d < rank; d++) {
                        const outputCoord = Math.floor(currentOutputFlatIndex / outputStrides[d]) % outputShape[d];
                        currentOutputFlatIndex %= outputStrides[d];

                        let inputCoord = outputCoord;
                        if (d === sliceDim) {
                            inputCoord += sliceStart;
                        }
                        inputFlatIndex += inputCoord * inputStrides[d];
                    }
                    outputView[outputFlatIndex] = inputView[inputFlatIndex];
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [SliceNode.INPUT, SliceNode.DIM, SliceNode.START, SliceNode.END],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
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
     * @param {Object} options - Options for ReshapeNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new ReshapeNode({ name, partition, type }), offset];
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
     * @param {Object} options - Options for UnsqueezeNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new UnsqueezeNode({ name, partition, type }), offset];
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
     * @param {Object} options - Options for BroadcastNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new BroadcastNode({ name, partition, type }), offset];
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
     * @param {Object} options - Options for CatNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new CatNode({ name, partition, type }), offset];
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
     * @param {Object} params - Parameters for FixedNode.
     * @param {CPUTensor} params.tensor - The fixed tensor.
     */
    constructor(params) {
        super(params);
        /** @type {CPUTensor} */
        this.tensor = params.tensor;
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    /**
     * Decodes a FixedNode from a DataView.
     * @param {DataView} view - The DataView to decode from.
     * @param {number} offset - The offset to start decoding from.
     * @param {string} name - The name of the node.
     * @param {string} partition - The partition of the node.
     * @param {string} type - The type of the node.
     * @returns {[FixedNode, number]} A tuple containing the decoded FixedNode and the new offset.
     */
    static decode(view, offset, name, partition, type) {
        let tensor;
        [tensor, offset] = CPUTensor.decode(view, offset);
        return [new FixedNode({ name, partition, type, tensor }), offset];
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
     * @param {Object} options - Options for HadamardNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new HadamardNode({ name, partition, type }), offset];
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
     * @param {Object} options - Options for IndexNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new IndexNode({ name, partition, type }), offset];
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
        if (outputShape.length == 0) {
            return [1]
        }
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
     * @param {Object} options - Options for ShapeNode.
     * @param {string} options.name - Name of the node.
     * @param {string} options.partition - Partition of the node.
     * @param {string} options.type - Type of the node.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new ShapeNode({ name, partition, type }), offset];
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
 * @class CastNode
 * @classdesc Represents a tensor cast operation node.
 * @extends Node
 */
class CastNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    
    constructor(options) {
        super(options);
        this.dtype = options.dtype;
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        let dtype;
        [dtype, offset] = readEncodedString(view, offset);
        return [new CastNode({ name, partition, type, dtype }), offset];
    }

    get_inputs() { return [CastNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(CastNode.INPUT);
        if (!inputTensor) {
            throw new Error(`CastNode (${this.name}): Missing input tensor.`);
        }
        return inputTensor.shape;
    }
    
    getCPUKernel() {
        return new CPUKernel({
            name: 'cast',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(CastNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`CastNode (${this.name}) kernel: Missing input tensor.`);
                }
                console.debug(`CastNode (${this.name}) kernel: inputTensor.shape: ${inputTensor.shape}, dtype: ${inputTensor.dtype}, casting to: ${this.dtype}`);
                const outputTensor = CPUTensor.uninitialized(inputTensor.shape, this.dtype);
                const outputView = outputTensor.getTypedArray();
                const inputView = inputTensor.getTypedArray();
                for (let i = 0; i < inputView.length; i++) {
                    outputView[i] = inputView[i];
                }
                return { [DEFAULT_NODE_OUTPUT]: outputTensor };
            },
            inputs: [CastNode.INPUT],
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

    /** @type {number} */
    dim0;
    /** @type {number} */
    dim1;
    
    /**
     * @param {Object} options - Options for TransposeNode.
     * @param {number} options.dim0 - First dimension to transpose.
     * @param {number} options.dim1 - Second dimension to transpose.
     */
    constructor(options) {
        super(options);
        /** @type {number} */
        this.dim0 = options.dim0;
        /** @type {number} */
        this.dim1 = options.dim1;
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    static decode(view, offset, name, partition, type) {
        let dim0, dim1;
        [dim0, offset] = readBEInt(view, offset);
        [dim1, offset] = readBEInt(view, offset);
        return [new TransposeNode({ name, partition, type, dim0, dim1 }), offset];
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
        const MAX_DIMS = 8; // Define explicitly here for clarity, should match WGSL
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

                    if (rank > MAX_DIMS) { 
                        throw new Error(`TransposeNode (${this.name}): Input tensor rank (${rank}) exceeds MAX_DIMS (${MAX_DIMS}).`);
                    }

                    const paddedInputShape = new Array(MAX_DIMS).fill(1);
                    const paddedInputStrides = new Array(MAX_DIMS).fill(0);
                    
                    if (rank > 0) {
                        for (let i = 0; i < rank; i++) {
                            paddedInputShape[i] = inputShape[i];
                        }
                        const inputStrides = calculateStrides(inputShape);
                        for (let i = 0; i < rank; i++) {
                            paddedInputStrides[i] = inputStrides[i];
                        }
                    }

                    const num_elements = rank > 0 ? inputShape.reduce((acc, val) => acc * val, 1) : 1;

                    // Calculate dispatch grid and invocation parameters locally
                    const workgroupSizeX = 8; // Match shader
                    const workgroupSizeY = 8; // Match shader
                    const workgroupSizeZ = 4; // Match shader (not used for grid_invocations_per_row/slice directly but for totalWorkgroupsNeeded)
                    const invocationsPerWorkgroup = workgroupSizeX * workgroupSizeY * workgroupSizeZ;

                    let dispatchGridX = 1;
                    let dispatchGridY = 1;
                    // dispatchGridZ is not needed for these specific params

                    if (num_elements > 0) {
                        const totalWorkgroupsNeeded = Math.ceil(num_elements / invocationsPerWorkgroup);
                        const maxDispatchDim = 65535; // device.limits.maxComputeWorkgroupsPerDimension

                        if (totalWorkgroupsNeeded <= maxDispatchDim) {
                            dispatchGridX = totalWorkgroupsNeeded;
                        } else if (totalWorkgroupsNeeded <= maxDispatchDim * maxDispatchDim) {
                            dispatchGridX = maxDispatchDim;
                            dispatchGridY = Math.ceil(totalWorkgroupsNeeded / maxDispatchDim);
                        } else { // If totalWorkgroupsNeeded > maxDispatchDim^2, it implies a Z dimension or error for 2D grid calc.
                                      // For calculating grid_invocations_per_row/slice, we primarily care about X and Y dispatches.
                                      // If it goes to 3D dispatch for workgroups, grid_invocations_per_slice reflects that.
                            dispatchGridX = maxDispatchDim;
                            dispatchGridY = maxDispatchDim;
                            // We don't need to calculate dispatchGridZ here for these two params
                        }
                    }

                    const gridInvocationsPerRow = dispatchGridX * workgroupSizeX;
                    const gridInvocationsPerSlice = dispatchGridX * workgroupSizeX * dispatchGridY * workgroupSizeY;
                    
                    console.log(`TransposeNode (${this.name}) dimensionBuffer: gridInvocationsPerRow = ${gridInvocationsPerRow}, gridInvocationsPerSlice = ${gridInvocationsPerSlice}`);

                    // Shader's Params struct: 8 (shape) + 8 (strides) + 1 (rank) + 1 (d0) + 1 (d1) + 1 (num_elements) + 1 (grid_row) + 1 (grid_slice) + 2 (padding) = 24 u32s
                    const uniformData = new Uint32Array(MAX_DIMS * 2 + 4 + 2 + 2); // 16 (shape/strides based on MAX_DIMS) + 4 (rank,d0,d1,num_el) + 2 (grid_invocations) + 2 (padding)
                    let offset = 0;
                    uniformData.set(paddedInputShape, offset); offset += MAX_DIMS;
                    uniformData.set(paddedInputStrides, offset); offset += MAX_DIMS;
                    uniformData.set([rank, d0, d1, num_elements], offset); offset += 4;
                    uniformData.set([gridInvocationsPerRow, gridInvocationsPerSlice], offset); offset += 2;
                    uniformData.set([0, 0], offset); // padding0, padding1
                    
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

                // Match shader workgroup sizes
                const workgroupSizeX = 8;
                const workgroupSizeY = 8;
                const workgroupSizeZ = 4;
                const invocationsPerWorkgroup = workgroupSizeX * workgroupSizeY * workgroupSizeZ;

                if (num_elements === 0) {
                    return { x: 0, y: 0, z: 0 };
                }

                const totalWorkgroupsNeeded = Math.ceil(num_elements / invocationsPerWorkgroup);

                // Get device limits
                // const device = executionContext.device;
                // const maxDispatchDim = device.limits.maxComputeWorkgroupsPerDimension;
                const maxDispatchDim = 65535;

                let dispatchGridX = 1;
                let dispatchGridY = 1;
                let dispatchGridZ = 1;

                if (totalWorkgroupsNeeded <= maxDispatchDim) {
                    dispatchGridX = totalWorkgroupsNeeded;
                } else if (totalWorkgroupsNeeded <= maxDispatchDim * maxDispatchDim) {
                    dispatchGridX = maxDispatchDim;
                    dispatchGridY = Math.ceil(totalWorkgroupsNeeded / maxDispatchDim);
                } else if (totalWorkgroupsNeeded <= maxDispatchDim * maxDispatchDim * maxDispatchDim){
                    dispatchGridX = maxDispatchDim;
                    dispatchGridY = maxDispatchDim;
                    dispatchGridZ = Math.ceil(totalWorkgroupsNeeded / (maxDispatchDim * maxDispatchDim));
                } else {
                    throw new Error(`TransposeNode (${this.name}): num_elements (${num_elements}) is too large to dispatch with current WebGPU limits.`);
                }

                // Store these for uniform buffer population if your setup needs them pre-calculated
                // These are used by the shader to reconstruct flat_index from global_invocation_id
                // if (!executionContext.params) {
                //     executionContext.params = {};
                // }
                // executionContext.params.grid_invocations_per_row = dispatchGridX * workgroupSizeX;
                // executionContext.params.grid_invocations_per_slice = dispatchGridX * workgroupSizeX * dispatchGridY * workgroupSizeY;

                return {
                    x: dispatchGridX,
                    y: dispatchGridY,
                    z: dispatchGridZ,
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
     * @param {Object} options - Options for AddNode.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new AddNode({ name, partition, type }), offset];
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
                func: (executionContext) => {
                    const inputTensorA = executionContext.gpu(AddNode.A);
                    const inputTensorB = executionContext.gpu(AddNode.B);
                    if (!inputTensorA || !inputTensorB) {
                        throw new Error(`AddNode (${this.name}): Missing GPU input tensors.`);
                    }
                    const size = inputTensorA.shape.reduce((a, b) => a * b, 1);
                    return new Uint32Array([size]);
                },
                index: 3,
            },
            workgroupFunction: (executionContext) => {
                const inputTensorA = executionContext.gpu(AddNode.A);
                const inputTensorB = executionContext.gpu(AddNode.B);
                if (!inputTensorA || !inputTensorB) {
                    throw new Error(`AddNode (${this.name}): Missing GPU input tensors.`);
                }
                const size = inputTensorA.shape.reduce((a, b) => a * b, 1);
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
     * @param {ExecutionContext} executionContext - The execution context for the node.
     * @returns {number[]} The output shape, same as input shapes.
     * @throws {Error} If input shapes are missing or incompatible.
     */
    getOutputShape(executionContext) {
        const shapeA = executionContext.gpu(AddNode.A).shape;
        const shapeB = executionContext.gpu(AddNode.B).shape;

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
     * @param {Object} options - Options for DivNode.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true, supportsGPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new DivNode({ name, partition, type }), offset];
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

                // Figure out output dtype
                let outputDtype;
                outputDtype = 'float32';
                // if(tensorA.dtype === 'int32' || tensorB.dtype === 'int32') {
                //     outputDtype = 'int32';
                // } else {
                //     outputDtype = 'float32';
                // }

                const outputShape = this._calculateOutputShape(tensorA, tensorB);
                // Division output is set to float32 to handle mixed types and general expectations.
                const outputTensor = CPUTensor.uninitialized(outputShape, outputDtype);
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
     * @param {Object} options - Options for FloorNode.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new FloorNode({ name, partition, type }), offset];
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
     * @param {Object} options - Options for CeilNode.
     */
    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new CeilNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        return inputsMap.get(CeilNode.INPUT) || 0;
    }

    get_inputs() { return [CeilNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(tensor) {
        if (!tensor) {
            throw new Error(`CeilNode (${this.name}): Missing required input tensor.`);
        }
        return [...tensor.shape];
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
     * @param {Object} options - Options for DebugNode.
     */
    constructor(options) {
        super(options);
    }

    static decode(view, offset, name, partition, type) {
        return [new DebugNode({ name, partition, type }), offset];
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
    /**
     * @param {Object} options
     * @param {Object.<string, Object>} options.nodes - A map of node names to node API responses.
     * @param {Array<Object>} options.edges - An array of edge API responses.
     */
    constructor(options) {
        this.nodes = options.nodes;
        this.edges = options.edges;
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

    static decodeNode(view, offset) {
        let nodeName, nodePartition, nodeType;
        [nodeName, offset] = readEncodedString(view, offset);
        [nodePartition, offset] = readEncodedString(view, offset);
        [nodeType, offset] = readEncodedString(view, offset);
        if (nodeType === "matmul") {
            return MatmulNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "safetensor") {
            return SafetensorNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "softmax") {
            return SoftmaxNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "slice") {
            return SliceNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "reshape") {
            return ReshapeNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "unsqueeze") {
            return UnsqueezeNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "broadcast") {
            return BroadcastNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "cat") {
            return CatNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "cast") {
            return CastNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "fixed") {
            return FixedNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "hadamard") {
            return HadamardNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "index") {
            return IndexNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "shape") {
            return ShapeNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "transpose") {
            return TransposeNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "add") {
            return AddNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "div") {
            return DivNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "floor") {
            return FloorNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "ceil") {
            return CeilNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "cos") {
            return CosNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "debug") {
            return DebugNode.decode(view, offset, nodeName, nodePartition, nodeType);            
        } else if (nodeType === "index_select") {
            return IndexSelectNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "reduce_mean") {
            return ReduceMeanNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "rsqrt") {
            return RsqrtNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "silu") {
            return SiluNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "sin") {
            return SinNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else if (nodeType === "squared") {
            return SquaredNode.decode(view, offset, nodeName, nodePartition, nodeType);
        } else {
            // Throw error for unknown node types
            throw new Error(`Unknown node type: ${nodeType}`);
        }
    }

    /**
     * Decodes a graph from a DataView
     * @param {DataView} view - The DataView to decode from
     * @param {number} offset - The offset to start decoding from
     * @returns {[Graph, number]} A tuple containing the decoded graph and the new offset
     */
    static decode(view, offset) {
        let numNodes, numEdges;
        [numNodes, offset] = readBEInt(view, offset);
        let nodes = {};
        for (let i = 0; i < numNodes; i++) {
            let node;
            [node, offset] = Graph.decodeNode(view, offset);
            nodes[node.name] = node;
        }
        [numEdges, offset] = readBEInt(view, offset);
        let edges = [];
        for (let i = 0; i < numEdges; i++) {
            let edge;
            [edge, offset] = Edge.decode(view, offset);
            edges.push(edge);
        }
        return [new Graph({ nodes, edges }), offset];
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
     * @param {Object} options - Input assignment options
     * @param {string} options.node - Node to assign input to
     * @param {string} options.input - Input to assign
     * @param {CPUTensor} options.tensor - Tensor to assign
     */
    constructor(options) {
        this.node = options.node;
        this.input = options.input;
        // Directly create CPUTensor from the raw tensor data in the API response
        this.tensor = options.tensor;
        console.log("constructed", this)
    }

    /**
     * Reads an input assignment from the DataView.
     * @param {DataView} view The DataView to read from.
     * @param {number} offset The offset to start reading at.
     * @returns {[InputAssignment, number]} A tuple containing the decoded input assignment and new offset.
     */
    static decode(view, offset) {
        let node, inputName, tensorData;
        [node, offset] = readEncodedString(view, offset);
        [inputName, offset] = readEncodedString(view, offset);
        [tensorData, offset] = CPUTensor.decode(view, offset);
        return [new InputAssignment({ node, input: inputName, tensor: tensorData }), offset];
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

    /**
     * Calculates the size of an encoded output assignment.
     * @returns {number} The size in bytes.
     */
    encodedSize() {
        return sizeEncodedString(this.node) +
               sizeEncodedString(this.output) +
               this.tensor.encodedSize();
    }

    /**
     * Writes an output assignment to the DataView.
     * @param {DataView} view The DataView to write to.
     * @param {number} offset The offset to start writing at.
     * @returns {number} The new offset after writing.
     */
    encode(view, offset) {
        offset = writeEncodedString(view, offset, this.node);
        offset = writeEncodedString(view, offset, this.output);
        offset = this.tensor.encode(view, offset);
        return offset;
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
    /** @type {boolean} */
    shouldTrace;

    /**
     * @param {Object} options - Partition work options
     * @param {string} options.correlation_id - Correlation ID of the work
     * @param {string} options.partition - Partition to get work for
     * @param {Graph} options.graph - Graph to execute
     * @param {Array<InputAssignment>} options.inputs - Inputs to the graph.
     * @param {boolean} options.shouldTrace - Whether to trace the execution of the graph.
     */
    constructor(options) {
        this.correlation_id = options.correlation_id;
        this.partition = options.partition;
        this.graph = options.graph;
        this.inputs = options.inputs;
        this.shouldTrace = options.shouldTrace;
    }

    /**
     * Reads a PartitionWork object from the DataView.
     * @param {DataView} view The DataView to read from.
     * @param {number} offset The offset to start reading at.
     * @returns {[PartitionWork, number]} A tuple containing the decoded PartitionWork and new offset.
     */
    static decode(view, offset) {
        let correlation_id, partition, graph, inputsLength, shouldTrace;
        const inputs = [];
        [correlation_id, offset] = readEncodedString(view, offset);
        [partition, offset] = readEncodedString(view, offset);
        [graph, offset] = Graph.decode(view, offset);
        [shouldTrace, offset] = readBool(view, offset);
        [inputsLength, offset] = readBEInt(view, offset);
        for (let i = 0; i < inputsLength; i++) {
            let inputAssignment;
            [inputAssignment, offset] = InputAssignment.decode(view, offset);
            inputs.push(inputAssignment);
        }
        return [new PartitionWork({ correlation_id, partition, graph, inputs, shouldTrace }), offset];
    }
}

/**
 * @class SingleStepChunk
 * @classdesc Represents a chunk of output assignments for single-step debugging.
 */
export class SingleStepChunk {
    /** @type {string} */
    correlation_id;
    /** @type {string} */ // PartitionName is a string
    partition;
    /** @type {Array<OutputAssignment>} */
    outputs;
    /** @type {boolean} */
    last_chunk


    /**
     * @param {Object} options
     * @param {string} options.correlation_id
     * @param {string} options.partition
     * @param {Array<OutputAssignment>} options.outputs - Array of OutputAssignment instances.
     * * @param {boolean} options.last_chunk - whetehr this is the last result chunk of its partition
     */
    constructor(options) {
        this.correlation_id = options.correlation_id;
        this.partition = options.partition;
        this.outputs = options.outputs;
        this.last_chunk = options.last_chunk;
    }

    /**
     * @returns {Object} An API-ready representation of the chunk.
     */
    toAPI() {
        return {
            correlation_id: this.correlation_id,
            partition: this.partition,
            outputs: this.outputs.map((oa) => oa.toAPI()),
            last_chunk: this.last_chunk
        };
    }

    /**
     * Calculates the size of an encoded SingleStepChunk.
     * @returns {number} The size in bytes.
     */
    encodedSize() {
        let size = sizeEncodedString(this.correlation_id);
        size += sizeEncodedString(this.partition);
        size += 4; // outputs_length
        for (const output of this.outputs) {
            size += output.encodedSize();
        }
        size += 1 // for the boolean
        return size;
    }

    /**
     * Writes a PartitionWorkResult object to the DataView.
     * @param {DataView} view The DataView to write to.
     * @param {number} offset The offset to start writing at.
     * @returns {number} The new offset after writing.
     */
    encode(view, offset) {
        offset = writeEncodedString(view, offset, this.correlation_id);
        offset = writeEncodedString(view, offset, this.partition);
        offset = writeBEInt(view, offset, this.outputs.length);
        for (const output of this.outputs) {
            offset = output.encode(view, offset);
        }
        offset = writeBool(view, offset, this.last_chunk);
        return offset;
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

    /**
     * Calculates the size of an encoded PartitionWorkResult.
     * @returns {number} The size in bytes.
     */
    encodedSize() {
        let size = sizeEncodedString(this.correlation_id);
        size += sizeEncodedString(this.partition);
        size += 4; // outputs_length
        for (const output of this.outputs) {
            size += output.encodedSize();
        }
        return size;
    }

    /**
     * Writes a PartitionWorkResult object to the DataView.
     * @param {DataView} view The DataView to write to.
     * @param {number} offset The offset to start writing at.
     * @param {PartitionWorkResult} result The PartitionWorkResult object.
     * @returns {number} The new offset after writing.
     */
    encode(view, offset) {
        offset = writeEncodedString(view, offset, this.correlation_id);
        offset = writeEncodedString(view, offset, this.partition);
        offset = writeBEInt(view, offset, this.outputs.length);
        for (const output of this.outputs) {
            offset = output.encode(view, offset);
        }
        return offset;
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
     * @returns {Promise<PartitionWork | null>} - Partition work from server, or null if no work is available
     */
    async get_work(partition_name) {
        const response = await fetch(`${this.url}/work/${partition_name}`);
        const buffer = await response.arrayBuffer();
        if (buffer.byteLength === 0) {
            return null;
        }
        console.debug("got work", buffer);
        const view = new DataView(buffer);
        const [work] = PartitionWork.decode(view, 0);
        return work;
    }

    /**
     * Submits some work to the coordination server to be checked
     * @param {SingleStepChunk} work - Partition chunk to submit.
     * @returns {Promise<void>}
     */
    async check_work(work) {

        const size = work.encodedSize();
        const buffer = new ArrayBuffer(size);
        const view = new DataView(buffer);
        work.encode(view, 0);
        const start = performance.now(); 
        await fetch(`${this.url}/check-work`, {
            method: "POST",
            body: buffer,
            headers: {
                ["Content-Type"]: 'application/octet-stream'
            }
        });
        const end = performance.now();
        console.log(`check_work took ${end - start}ms`);
        }

    /**
     * Submits the partition work to the coordination server
     * @param {PartitionWorkResult} work - Partition work to submit.
     * @returns {Promise<void>}
     */
    async submit_work(work) {
        const size = work.encodedSize();
        const buffer = new ArrayBuffer(size);
        const view = new DataView(buffer);
        work.encode(view, 0);
        
        await fetch(`${this.url}/work`, {
            method: "POST",
            body: buffer,
            headers: {
                ["Content-Type"]: 'application/octet-stream'
            }
        });
    }
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

// Insert new node class skeletons alphabetically here

/**
 * @class CosNode
 * @classdesc Represents an element-wise cosine operation node.
 * @extends Node
 */
class CosNode extends Node {
    /** @type {string} */
    static INPUT = "input";

    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true, gpuWeightThreshold: 512 });
    }

    static decode(view, offset, name, partition, type) {
        return [new CosNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        return inputsMap.get(CosNode.INPUT) || 0;
    }

    get_inputs() { return [CosNode.INPUT]; }
    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.gpu(CosNode.INPUT) || executionContext.cpu(CosNode.INPUT);
        if (!inputTensor) throw new Error(`CosNode (${this.name}): Missing input tensor.`);
        return [...inputTensor.shape];
    }

    async getGPUKernel() {
        // console.warn(`CosNode (${this.name}): GPU kernel not yet implemented. Fetching empty.`);
        return new GPUKernel({
            name: 'cos',
            shader: await fetch('kernels/cos.wgsl').then(r => r.text()), 
            entryPoint: 'main',
            dimensionBuffer: { 
                func: (executionContext) => {
                    const inputTensor = executionContext.gpu(CosNode.INPUT);
                    if (!inputTensor) {
                        throw new Error(`CosNode (${this.name}): Missing GPU input tensor for dimensionBuffer.`);
                    }
                    const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                    return new Uint32Array([num_elements]); // num_elements
                },
                index: 2 
            },
            workgroupFunction: (executionContext) => {
                const inputTensor = executionContext.gpu(CosNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`CosNode (${this.name}): Missing GPU input tensor for workgroupFunction.`);
                }
                const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                const workgroupSizeX = 256; // Must match WORKGROUP_SIZE_X in shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            inputs: [{ name: CosNode.INPUT, cpu: false, binding: { type: "read-only-storage", index: 0 } }],
            outputs: [{ name: DEFAULT_NODE_OUTPUT, binding: { type: "storage", index: 1 } }],
        });
    }
}

/**
 * @class IndexSelectNode
 * @classdesc Selects slices from an input tensor along a given dimension at specified indices.
 * @extends Node
 */
class IndexSelectNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static DIM = "dim";
    /** @type {string} */
    static INDEX = "index";

    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsCPU: true });
    }

    static decode(view, offset, name, partition, type) {
        return [new IndexSelectNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        const inputWeight = inputsMap.get(IndexSelectNode.INPUT) || 0;
        const indexWeight = inputsMap.get(IndexSelectNode.INDEX) || 1;
        // Output weight is roughly indexWeight * (inputWeight / inputShape[0])
        // This is a simplification.
        if (inputWeight > 0 && indexWeight > 0) {
             // Assuming inputShape[0] is at least 1 for this rough estimate
            return indexWeight * (inputWeight / (inputsMap.get('_inputShape0SizePlaceholder_') || Math.max(1, inputWeight / indexWeight) ));
        }
        return inputWeight; 
    }

    get_inputs() { return [IndexSelectNode.INPUT, IndexSelectNode.DIM, IndexSelectNode.INDEX]; }
    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _calculateOutputShape(inputTensor, indexTensor) {
        if (!inputTensor || !inputTensor.shape) {
            throw new Error(`IndexSelectNode (${this.name}): Missing input tensor or shape.`);
        }
        if (!indexTensor || !indexTensor.shape) {
            throw new Error(`IndexSelectNode (${this.name}): Missing index tensor or shape.`);
        }
        const inputShape = inputTensor.shape;
        const indexShape = indexTensor.shape;

        if (inputShape.length === 0) {
            throw new Error(`IndexSelectNode (${this.name}): Cannot index a 0D (scalar) input tensor with index_select semantics.`);
        }

        // Output shape is index.shape + input.shape[1:]
        const outputShape = [...indexShape, ...inputShape.slice(1)];
        return outputShape;
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(IndexSelectNode.INPUT);
        const indexTensor = executionContext.cpu(IndexSelectNode.INDEX);
        // const dimTensor = executionContext.cpu(IndexSelectNode.DIM); // DIM is ignored
        return this._calculateOutputShape(inputTensor, indexTensor);
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'index_select',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(IndexSelectNode.INPUT);
                const indexTensor = executionContext.cpu(IndexSelectNode.INDEX);
                // DIM is ignored: const dimTensor = executionContext.cpu(IndexSelectNode.DIM);

                if (!inputTensor || !indexTensor) {
                    throw new Error(`IndexSelectNode (${this.name}) kernel: Missing input or index tensor.`);
                }

                const outputShape = this._calculateOutputShape(inputTensor, indexTensor);
                const outputTensor = CPUTensor.uninitialized(outputShape, inputTensor.dtype);
                
                const outputView = outputTensor.getTypedArray();
                const inputData = inputTensor.getTypedArray();
                const indexData = indexTensor.getTypedArray();

                const inputShape = inputTensor.shape;
                const inputStrides = calculateStrides(inputShape);

                // Size of one slice from the input tensor (all dimensions except the first)
                let sliceSize = 1;
                for (let i = 1; i < inputShape.length; i++) {
                    sliceSize *= inputShape[i];
                }
                if (inputShape.length === 0) { // Should be caught by _calculateOutputShape
                    sliceSize = 0; 
                } else if (inputShape.length === 1) { // Input is 1D, slice is a single element
                    sliceSize = 1;
                }

                const numIndices = indexData.length; // Total number of indices to select
                let outputBufferOffset = 0;

                for (let i = 0; i < numIndices; i++) {
                    let selectedInputDim0Index = indexData[i];
                    
                    // Bounds checking for the index
                    if (selectedInputDim0Index < 0 || selectedInputDim0Index >= inputShape[0]) {
                        throw new Error(`IndexSelectNode (${this.name}) kernel: Index ${selectedInputDim0Index} at index tensor position ${i} is out of bounds for input dimension 0 size ${inputShape[0]}.`);
                    }

                    const inputBufferOffset = selectedInputDim0Index * sliceSize * inputStrides[0] / inputShape[0]; // More robust: selectedInputDim0Index * inputStrides[0]
                    // Corrected offset: index along dim 0 * stride of dim 0
                    const currentInputSliceOffset = selectedInputDim0Index * (inputShape.length > 1 ? inputStrides[0] : 1) ; 
                    // If input is 1D, inputStrides[0] is 1.
                    // If input is >1D, inputStrides[0] is product of shape[1]*shape[2]*...
                    // sliceSize is also product of shape[1]*shape[2]*...
                    // So, selectedInputDim0Index * sliceSize is the correct start for N-D.
                    // For 1D, inputStrides[0] is 1, selectedInputDim0Index * 1 is the offset.

                    let actualInputOffset;
                    if (inputShape.length === 1) { // Input is 1D
                        actualInputOffset = selectedInputDim0Index; 
                    } else { // Input is N-D (N > 1)
                        actualInputOffset = selectedInputDim0Index * sliceSize; 
                    }

                    // Copy the slice
                    for (let j = 0; j < sliceSize; j++) {
                        outputView[outputBufferOffset + j] = inputData[actualInputOffset + j];
                    }
                    outputBufferOffset += sliceSize;
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [IndexSelectNode.INPUT, IndexSelectNode.DIM, IndexSelectNode.INDEX],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }
}

/**
 * @class ReduceMeanNode
 * @classdesc Reduces a tensor by taking the mean along a given dimension.
 * @extends Node
 */
class ReduceMeanNode extends Node {
    /** @type {string} */
    static INPUT = "input";
    /** @type {string} */
    static DIM = "dim"; // This will be a CPUTensor containing the dimension(s) to reduce

    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true, gpuWeightThreshold: 512 }); 
    }

    static decode(view, offset, name, partition, type) {
        return [new ReduceMeanNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        const inputWeight = inputsMap.get(ReduceMeanNode.INPUT) || 0;
        // Estimate output weight based on input rank and typical reduction
        // This is a rough heuristic.
        const inputRank = inputsMap.get('_inputRankPlaceholder_') || 3; // Need actual rank if possible
        const reduceDimSize = inputsMap.get('_reduceDimSizePlaceholder_') || 2;
        if (inputRank > 1 && reduceDimSize > 0) {
            return Math.max(1, Math.floor(inputWeight / reduceDimSize));
        }
        return Math.max(1, Math.floor(inputWeight / 2)); 
    }

    get_inputs() { return [ReduceMeanNode.INPUT, ReduceMeanNode.DIM]; }
    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    _normalize_dim(dim, rank) {
        if (rank === 0) return 0; // No dims to normalize for a scalar
        if (dim < 0) {
            return dim + rank;
        }
        return dim;
    }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.cpu(ReduceMeanNode.INPUT) || executionContext.gpu(ReduceMeanNode.INPUT);
        const dimCPUTensor = executionContext.cpu(ReduceMeanNode.DIM) || executionContext.gpu(ReduceMeanNode.DIM); // DIM can be CPU or GPU

        if (!inputTensor || !inputTensor.shape) {
            throw new Error(`ReduceMeanNode (${this.name}): Missing input tensor or shape.`);
        }
        if (!dimCPUTensor) {
            throw new Error(`ReduceMeanNode (${this.name}): Missing dim tensor.`);
        }

        const inputShape = [...inputTensor.shape];
        const rank = inputShape.length;
        
        if (rank === 0) return []; // ReduceMean of a scalar is the scalar itself

        const dimToReduce = this._normalize_dim(dimCPUTensor.getTypedArray()[0], rank);

        if (dimToReduce < 0 || dimToReduce >= rank) {
            throw new Error(`ReduceMeanNode (${this.name}): Invalid dimension ${dimCPUTensor.getTypedArray()[0]} for input rank ${rank}.`);
        }

        const outputShape = [];
        for (let i = 0; i < rank; i++) {
            if (i !== dimToReduce) {
                outputShape.push(inputShape[i]);
            }
        }
        // If input was 1D and reduced, output is scalar (shape [])
        // If input was N-D and all non-reduced dims were 1, and reduced dim was >1, output is also scalar.
        // If outputShape is empty, it means the result is a scalar.
        return outputShape; 
    }

    async getGPUKernel() {
        return new GPUKernel({
            name: 'reduce_mean',
            shader: await fetch('kernels/reduce_mean.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (executionContext) => {
                    const inputTensor = executionContext.gpu(ReduceMeanNode.INPUT);
                    const dimCPUTensor = executionContext.cpu(ReduceMeanNode.DIM);

                    if (!inputTensor) {
                        throw new Error(`ReduceMeanNode (${this.name}): Missing GPU input tensor for dimensionBuffer.`);
                    }
                    if (!dimCPUTensor) {
                        throw new Error(`ReduceMeanNode (${this.name}): Missing CPU dim tensor for dimensionBuffer.`);
                    }

                    const inputShape = inputTensor.shape;
                    const inputRank = inputShape.length;
                    const reduceDimRaw = dimCPUTensor.getTypedArray()[0];
                    const reduceDim = this._normalize_dim(reduceDimRaw, inputRank);

                    if (inputRank > MAX_DIMS_SOFTMAX) { // Using MAX_DIMS_SOFTMAX as MAX_DIMS
                        throw new Error(`ReduceMeanNode (${this.name}): Input tensor rank (${inputRank}) exceeds MAX_DIMS (${MAX_DIMS_SOFTMAX}).`);
                    }
                    if (reduceDim < 0 || reduceDim >= inputRank && inputRank > 0) {
                        throw new Error(`ReduceMeanNode (${this.name}): Invalid reduce_dim ${reduceDimRaw} for rank ${inputRank}.`);
                    }

                    const paddedInputShape = new Array(MAX_DIMS_SOFTMAX).fill(1);
                    const paddedInputStrides = new Array(MAX_DIMS_SOFTMAX).fill(0);
                    if (inputRank > 0) {
                        for (let i = 0; i < inputRank; i++) paddedInputShape[i] = inputShape[i];
                        const inputStrides = calculateStrides(inputShape);
                        for (let i = 0; i < inputRank; i++) paddedInputStrides[i] = inputStrides[i];
                    }

                    let outputRank = 0;
                    let numOutputElements = 1;
                    if (inputRank > 0) {
                        outputRank = inputRank - 1;
                        for(let i=0; i < inputRank; i++) {
                            if (i !== reduceDim) {
                                numOutputElements *= inputShape[i];
                            }
                        }
                        if (inputRank === 1) { // Reducing a 1D tensor results in a scalar
                           outputRank = 0; // Scalar output
                           numOutputElements = 1;
                        }
                    } // else for inputRank 0, outputRank 0, numOutputElements 1 (scalar)
                    

                    const reduceDimSize = (inputRank > 0 && inputShape[reduceDim] !== undefined) ? inputShape[reduceDim] : 1;
                    if (reduceDimSize === 0 && inputRank > 0) {
                        console.warn(`ReduceMeanNode (${this.name}): Reducing along a dimension of size 0. Mean is ill-defined (output will be 0).`);
                    }

                    // Params: input_shape_vecs, input_strides_vecs, input_rank, 
                    //         output_rank, num_output_elements, 
                    //         reduce_dim, reduce_dim_size, padding (3)
                    // Total elements: (MAX_DIMS_SOFTMAX * 2) + 5 data scalars + 3 padding scalars = 16 + 5 + 3 = 24 u32s.
                    const uniformData = new Uint32Array(MAX_DIMS_SOFTMAX * 2 + 5 + 3); 
                    uniformData.set(paddedInputShape, 0);
                    uniformData.set(paddedInputStrides, MAX_DIMS_SOFTMAX);
                    uniformData.set([inputRank, outputRank, numOutputElements, reduceDim, reduceDimSize], MAX_DIMS_SOFTMAX * 2);
                    return uniformData;
                },
                index: 2, // Binding for params
            },
            workgroupFunction: (executionContext) => {
                const inputTensor = executionContext.gpu(ReduceMeanNode.INPUT);
                const dimCPUTensor = executionContext.cpu(ReduceMeanNode.DIM);
                if (!inputTensor || !dimCPUTensor) {
                    throw new Error(`ReduceMeanNode (${this.name}): Missing GPU input or CPU dim tensor for workgroupFunction.`);
                }
                const inputShape = inputTensor.shape;
                const inputRank = inputShape.length;
                const reduceDim = this._normalize_dim(dimCPUTensor.getTypedArray()[0], inputRank);

                let numOutputElements = 1; // This is the number of slices
                 if (inputRank > 0) {
                    for(let i=0; i < inputRank; i++) {
                        if (i !== reduceDim) {
                            numOutputElements *= inputShape[i];
                        }
                    }
                    if (inputRank === 1) numOutputElements = 1; // Scalar output
                } // else for inputRank 0, numOutputElements is 1 (scalar)


                const workgroupSizeX = 256; // Must match shader
                return {
                    x: workgroupSizeX, // We dispatch enough threads to cover reduce_dim_size in one go per slice.
                                       // The shader loop handles elements > WORKGROUP_SIZE_X.
                                       // Actual number of workgroups in X is 1 for this strategy.
                                       // More accurately, x is related to reduce_dim_size, and y is numOutputElements.
                                       // Let's re-think the dispatch to match the shader's expectation.
                                       // Shader uses workgroup_id.y as slice_idx.
                    y: numOutputElements, // Number of output elements (slices)
                    x: 1, // Each workgroup processes one full slice reduction internally.
                           // The parallelism is over the elements *within* the reduction dimension for that slice.
                           // The shader loops if reduce_dim_size > WORKGROUP_SIZE_X.
                           // So, we need WORKGROUP_SIZE_X threads in x, and numOutputElements workgroups in y.
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: ReduceMeanNode.INPUT, cpu: false, binding: {type: "read-only-storage", index: 0 } },
                { name: ReduceMeanNode.DIM, cpu: true } // DIM is used by JS helpers
            ],
            outputs: [
                { name: DEFAULT_NODE_OUTPUT, binding: {type: "storage", index: 1 } },
            ],
        });
    }
}

/**
 * @class RsqrtNode
 * @classdesc Represents an element-wise reciprocal square root node.
 * @extends Node
 */
class RsqrtNode extends Node {
    /** @type {string} */
    static INPUT = "input";

    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true, gpuWeightThreshold: 512 });
    }

    static decode(view, offset, name, partition, type) {
        return [new RsqrtNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        return inputsMap.get(RsqrtNode.INPUT) || 0;
    }

    get_inputs() { return [RsqrtNode.INPUT]; }
    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.gpu(RsqrtNode.INPUT) || executionContext.cpu(RsqrtNode.INPUT);
        if (!inputTensor) throw new Error(`RsqrtNode (${this.name}): Missing input tensor.`);
        return [...inputTensor.shape];
    }

    async getGPUKernel() {
        // console.warn(`RsqrtNode (${this.name}): GPU kernel not yet implemented. Fetching empty.`);
        return new GPUKernel({
            name: 'rsqrt',
            shader: await fetch('kernels/rsqrt.wgsl').then(r => r.text()), 
            entryPoint: 'main',
            dimensionBuffer: { 
                func: (executionContext) => {
                    const inputTensor = executionContext.gpu(RsqrtNode.INPUT);
                    if (!inputTensor) {
                        throw new Error(`RsqrtNode (${this.name}): Missing GPU input tensor for dimensionBuffer.`);
                    }
                    const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                    return new Uint32Array([num_elements]); // num_elements
                },
                index: 2 // Binding for params
            },
            workgroupFunction: (executionContext) => {
                const inputTensor = executionContext.gpu(RsqrtNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`RsqrtNode (${this.name}): Missing GPU input tensor for workgroupFunction.`);
                }
                const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                const workgroupSizeX = 256; // Must match WORKGROUP_SIZE_X in shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            inputs: [{ name: RsqrtNode.INPUT, cpu: false, binding: { type: "read-only-storage", index: 0 } }],
            outputs: [{ name: DEFAULT_NODE_OUTPUT, binding: { type: "storage", index: 1 } }],
        });
    }
}

/**
 * @class SiluNode
 * @classdesc Represents an SiLU activation function node.
 * @extends Node
 */
class SiluNode extends Node {
    /** @type {string} */
    static INPUT = "input";

    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true, gpuWeightThreshold: 512 });
    }

    static decode(view, offset, name, partition, type) {
        return [new SiluNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        return inputsMap.get(SiluNode.INPUT) || 0;
    }

    get_inputs() { return [SiluNode.INPUT]; }
    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.gpu(SiluNode.INPUT) || executionContext.cpu(SiluNode.INPUT);
        if (!inputTensor) throw new Error(`SiluNode (${this.name}): Missing input tensor.`);
        return [...inputTensor.shape];
    }

    async getGPUKernel() {
        // console.warn(`SiluNode (${this.name}): GPU kernel not yet implemented. Fetching empty.`);
        return new GPUKernel({
            name: 'silu',
            shader: await fetch('kernels/silu.wgsl').then(r => r.text()), 
            entryPoint: 'main',
            dimensionBuffer: { 
                func: (executionContext) => {
                    const inputTensor = executionContext.gpu(SiluNode.INPUT);
                    if (!inputTensor) {
                        throw new Error(`SiluNode (${this.name}): Missing GPU input tensor for dimensionBuffer.`);
                    }
                    const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                    return new Uint32Array([num_elements]); // num_elements
                },
                index: 2 // Binding for params
            },
            workgroupFunction: (executionContext) => {
                const inputTensor = executionContext.gpu(SiluNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`SiluNode (${this.name}): Missing GPU input tensor for workgroupFunction.`);
                }
                const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                const workgroupSizeX = 256; // Must match WORKGROUP_SIZE_X in shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            inputs: [{ name: SiluNode.INPUT, cpu: false, binding: { type: "read-only-storage", index: 0 } }],
            outputs: [{ name: DEFAULT_NODE_OUTPUT, binding: { type: "storage", index: 1 } }],
        });
    }
}

/**
 * @class SinNode
 * @classdesc Represents an element-wise sine operation node.
 * @extends Node
 */
class SinNode extends Node {
    /** @type {string} */
    static INPUT = "input";

    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true, gpuWeightThreshold: 512 });
    }

    static decode(view, offset, name, partition, type) {
        return [new SinNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        return inputsMap.get(SinNode.INPUT) || 0;
    }

    get_inputs() { return [SinNode.INPUT]; }
    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.gpu(SinNode.INPUT) || executionContext.cpu(SinNode.INPUT);
        if (!inputTensor) throw new Error(`SinNode (${this.name}): Missing input tensor.`);
        return [...inputTensor.shape];
    }

    async getGPUKernel() {
        // console.warn(`SinNode (${this.name}): GPU kernel not yet implemented. Fetching empty.`);
        return new GPUKernel({
            name: 'sin',
            shader: await fetch('kernels/sin.wgsl').then(r => r.text()), 
            entryPoint: 'main',
            dimensionBuffer: { 
                func: (executionContext) => {
                    const inputTensor = executionContext.gpu(SinNode.INPUT);
                    if (!inputTensor) {
                        throw new Error(`SinNode (${this.name}): Missing GPU input tensor for dimensionBuffer.`);
                    }
                    const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                    return new Uint32Array([num_elements]); // num_elements
                },
                index: 2 
            },
            workgroupFunction: (executionContext) => {
                const inputTensor = executionContext.gpu(SinNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`SinNode (${this.name}): Missing GPU input tensor for workgroupFunction.`);
                }
                const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                const workgroupSizeX = 256; // Must match WORKGROUP_SIZE_X in shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            inputs: [{ name: SinNode.INPUT, cpu: false, binding: { type: "read-only-storage", index: 0 } }],
            outputs: [{ name: DEFAULT_NODE_OUTPUT, binding: { type: "storage", index: 1 } }],
        });
    }
}

/**
 * @class SquaredNode
 * @classdesc Represents an element-wise square operation node.
 * @extends Node
 */
class SquaredNode extends Node {
    /** @type {string} */
    static INPUT = "input";

    constructor(options) {
        super(options);
        this.devicePreference = new DevicePreferences({ supportsGPU: true, supportsCPU: true, gpuWeightThreshold: 512 });
    }

    static decode(view, offset, name, partition, type) {
        return [new SquaredNode({ name, partition, type }), offset];
    }

    estimateWeight(inputsMap) {
        return inputsMap.get(SquaredNode.INPUT) || 0;
    }

    get_inputs() { return [SquaredNode.INPUT]; }
    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }

    getOutputShape(executionContext) {
        const inputTensor = executionContext.gpu(SquaredNode.INPUT) || executionContext.cpu(SquaredNode.INPUT);
        if (!inputTensor) throw new Error(`SquaredNode (${this.name}): Missing input tensor.`);
        return [...inputTensor.shape];
    }

    getCPUKernel() {
        return new CPUKernel({
            name: 'squared',
            func: (executionContext) => {
                const inputTensor = executionContext.cpu(SquaredNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`SquaredNode (${this.name}) kernel: Missing input tensor.`);
                }

                const outputShape = this.getOutputShape(executionContext); // Re-use for consistency
                // Output tensor will have the same dtype as the input tensor
                const outputTensor = CPUTensor.uninitialized(outputShape, inputTensor.dtype);
                const outputView = outputTensor.getTypedArray();
                const inputData = inputTensor.getTypedArray();

                for (let i = 0; i < outputView.length; i++) {
                    const val = inputData[i];
                    outputView[i] = val * val;
                }

                return {
                    [DEFAULT_NODE_OUTPUT]: outputTensor,
                };
            },
            inputs: [SquaredNode.INPUT],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }

    async getGPUKernel() {
        return new GPUKernel({
            name: 'squared',
            shader: await fetch('kernels/squared.wgsl').then(r => r.text()), 
            entryPoint: 'main',
            dimensionBuffer: { 
                func: (executionContext) => {
                    const inputTensor = executionContext.gpu(SquaredNode.INPUT);
                    if (!inputTensor) {
                        throw new Error(`SquaredNode (${this.name}): Missing GPU input tensor for dimensionBuffer.`);
                    }
                    const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                    return new Uint32Array([num_elements]); // num_elements
                },
                index: 2 // Binding for params
            },
            workgroupFunction: (executionContext) => {
                const inputTensor = executionContext.gpu(SquaredNode.INPUT);
                if (!inputTensor) {
                    throw new Error(`SquaredNode (${this.name}): Missing GPU input tensor for workgroupFunction.`);
                }
                const num_elements = inputTensor.shape.length > 0 ? inputTensor.shape.reduce((acc, val) => acc * val, 1) : 1;
                const workgroupSizeX = 256; // Must match WORKGROUP_SIZE_X in shader
                return {
                    x: Math.ceil(num_elements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            inputs: [{ name: SquaredNode.INPUT, cpu: false, binding: { type: "read-only-storage", index: 0 } }],
            outputs: [{ name: DEFAULT_NODE_OUTPUT, binding: { type: "storage", index: 1 } }],
        });
    }
}

