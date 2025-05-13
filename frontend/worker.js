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
     * @param {boolean} options.prefersGPU - Whether to prefer WebGPU
     * @param {boolean} options.prefersCPU - Whether to prefer CPU
     */
    constructor(options) {
        this.supportsCPU = options.supportsCPU || false;
        this.supportsGPU = options.supportsGPU || false;
        this.prefersGPU = options.prefersGPU || false;
        this.prefersCPU = options.prefersCPU || false;
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
     * @returns {boolean}
     */
    prefersGPU() {
        return this.prefersGPU || (this.supportsGPU && !this.supportsCPU);
    }

    /**
     * @returns {boolean}
     */
    prefersCPU() {
        return this.prefersCPU || (this.supportsCPU && !this.supportsGPU);
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
            throw new Error(`SoftmaxNode (${this.name}): Missing required input GPU tensor or shape for \'input\'.`);
        }
        // Softmax output shape is the same as the input shape
        return [...inputGPUTensor.shape]; // Return a copy
    }

    /**
     * @returns {Promise<GPUKernel>}
     */
    async getGPUKernel() {
        return new GPUKernel({
            name: 'softmax',
            shader: await fetch('kernels/softmax.wgsl').then(r => r.text()),
            dimensionBuffer: { // Corresponds to 'params: Params' in WGSL
                func: (inputGPUTensors, inputCPUTensors) => {
                    const currentInputShape = inputGPUTensors.get(SoftmaxNode.INPUT);
                    const currentNdims = currentInputShape.length;

                    const paddedShape = new Array(MAX_DIMS_SOFTMAX).fill(1); // Pad with 1s for shape
                    const paddedStrides = new Array(MAX_DIMS_SOFTMAX).fill(0); // Pad with 0s for strides

                    if (currentNdims > 0) {
                        if (currentNdims > MAX_DIMS_SOFTMAX) {
                            throw new Error(`SoftmaxNode (${this.name}): Input tensor dimensions (${currentNdims}) exceed MAX_DIMS_SOFTMAX (${MAX_DIMS_SOFTMAX}).`);
                        }
                        const strides = calculateStrides(currentInputShape);
                        for (let i = 0; i < currentNdims; i++) {
                            paddedShape[i] = currentInputShape[i];
                            paddedStrides[i] = strides[i];
                        }
                    }
                    // For 0D tensor (currentNdims = 0), shape/strides are effectively empty for padding.

                    const uniformData = new Uint32Array(MAX_DIMS_SOFTMAX * 2 + 1);
                    uniformData.set(paddedShape, 0);
                    uniformData.set(paddedStrides, MAX_DIMS_SOFTMAX);
                    uniformData.set([currentNdims], MAX_DIMS_SOFTMAX * 2);
                    return uniformData;
                },
                index: 2, // Matches @group(0) @binding(2) var<uniform> params: Params;
            },
            workgroupFunction: (inputGPUTensors, inputCPUTensors) => {
                const currentInputShape = inputGPUTensors.get(SoftmaxNode.INPUT);
                const currentNdims = currentInputShape.length;
                const workgroupSizeX = 256; // Must match the shader's const workgroup_size_x

                // Get the dimension to apply softmax along from CPU tensor
                const dimTensor = inputCPUTensors.get(SoftmaxNode.DIM);
                const dim = dimTensor ? dimTensor.data[0] : 0; // Default to 0 if not provided
                
                // Calculate total elements and elements per slice
                let totalElements = 1;
                let elementsPerSlice = 1;
                
                if (currentNdims > 0) {
                    totalElements = currentInputShape.reduce((acc, val) => acc * val, 1);
                    
                    // Calculate the size of each softmax slice
                    const normalizedDim = dim < 0 ? dim + currentNdims : dim;
                    if (normalizedDim >= 0 && normalizedDim < currentNdims) {
                        elementsPerSlice = currentInputShape[normalizedDim];
                    }
                } else {
                    // For a 0D tensor (scalar), totalElements is 1.
                    totalElements = 1;
                    elementsPerSlice = 1;
                }

                return {
                    // Dispatch enough workgroups to cover all elements in each slice
                    x: Math.ceil(elementsPerSlice / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            entryPoint: "main",
            inputs: [
                { name: SoftmaxNode.INPUT, cpu: false, binding: {type: "read-only-storage", index: 0 } },
                { name: SoftmaxNode.DIM, cpu: true, binding: {type: "read-only-storage", index: 3 } }
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

    get_inputs() { return [ReshapeNode.INPUT, ReshapeNode.DIMS]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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

    get_inputs() { return [HadamardNode.A, HadamardNode.B]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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
    }

    get_inputs() { return [IndexNode.INPUT, IndexNode.INDEX]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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
    }

    get_inputs() { return [ShapeNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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
    }

    get_inputs() { return [TransposeNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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
 * @classdesc Represents an element-wise division node.
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
    }

    get_inputs() { return [DivNode.A, DivNode.B]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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
    }

    get_inputs() { return [FloorNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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
    }

    get_inputs() { return [CeilNode.INPUT]; }

    get_outputs() { return [DEFAULT_NODE_OUTPUT]; }
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
