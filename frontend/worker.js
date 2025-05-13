/*
 * Library to handle interfacing with coordination server
 */

import { CPUKernel, CPUTensor, GPUKernel } from "./kernel.js";

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
        /** @type {Device | null} */
        this.device = null;
        
        // Copy any additional properties from the API response
        for (const [key, value] of Object.entries(api_response)) {
            if (key !== "type" && key !== "name") {
                this[key] = value;
            }
        }
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
        this.device = new GPUDevice();
    }

    /**
     * @param {Map<string, number[]>} inputShapes - Shapes of the input tensors.
     * @param {Map<string, number[]>} outputShapes - Shapes of the output tensors.
     * @returns {Promise<GPUKernel>}
     */
    async getKernel(inputShapes, outputShapes) {
        // From the horse's mouth: https://toji.dev/webgpu-best-practices/dynamic-shader-construction.html
        // This is a nice way to dynamically specialize kernels. The Kernel class has a key() method that allows as
        // much caching as possible. See GPUKernel for details.
        return new GPUKernel({
            name: 'matmul',
            shader: await fetch('kernels/matmul.wgsl').then(r => r.text()),
            dimensionBuffer: {
                func: (inputShapes, outputShapes) => {
                    return new Uint32Array([
                        inputShapes.get(MatmulNode.LHS)[0],
                        inputShapes.get(MatmulNode.LHS)[1],
                        inputShapes.get(MatmulNode.RHS)[1],
                    ]);
                },
                index: 0,
            },
            workgroupFunction: (inputShapes, outputShapes) => {
                return {
                    x: Math.ceil(inputShapes.get(MatmulNode.LHS)[0] / 16),
                    y: Math.ceil(inputShapes.get(MatmulNode.LHS)[1] / 16),
                    z: 1,
                };
            },
            entryPoint: "main",
            inputBindings: [
                { name: MatmulNode.LHS, type: "read-only-storage", index: 1 },
                { name: MatmulNode.RHS, type: "read-only-storage", index: 2 },
            ],
            outputBindings: [
                { name: DEFAULT_NODE_OUTPUT, type: "storage", index: 3 },
            ],
        });
    }


    /**
     * Calculates the output shape for a matrix multiplication.
     * @param {Map<string, number[]>} inputShapes - A map containing shapes for 'input' and 'weight'.
     *                                            Example: Map { 'input' => [M, K], 'weight' => [K, N] }
     * @returns {number[]} The output shape [M, N].
     * @throws {Error} If input shapes are missing, invalid, or incompatible.
     */
    getOutputShape(inputShapes) {
        const shapeA = inputShapes.get(MatmulNode.LHS);
        const shapeB = inputShapes.get(MatmulNode.RHS);

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
        this.device = new GPUDevice();
    }

    /**
     * Calculates the output shape for a softmax operation.
     * The output shape is identical to the input shape.
     * @param {Map<string, number[]>} inputShapes - A map containing the shape for \'input\'.
     *                                            Example: Map { \'input\' => [Batch, Features] }
     * @returns {number[]} The output shape, same as the input shape.
     * @throws {Error} If the input shape is missing.
     */
    getOutputShape(inputShapes) {
        const inputShape = inputShapes.get(SoftmaxNode.INPUT);

        if (!inputShape) {
            throw new Error(`SoftmaxNode (${this.name}): Missing required input shape (\'input\').`);
        }

        // Softmax output shape is the same as the input shape
        return [...inputShape]; // Return a copy
    }

    /**
     * @param {Map<string, number[]>} inputShapes - Shapes of the input tensors.
     * @param {Map<string, number[]>} outputShapes - Shapes of the output tensors.
     * @returns {Promise<GPUKernel>}
     */
    async getKernel(inputShapes, outputShapes) {
        const inputShape = inputShapes.get(SoftmaxNode.INPUT);
        if (!inputShape) {
            throw new Error(`SoftmaxNode (${this.name}): Missing input shape for kernel creation.`);
        }

        return new GPUKernel({
            name: 'softmax',
            shader: await fetch('kernels/softmax.wgsl').then(r => r.text()),
            dimensionBuffer: { // Corresponds to 'params: Params' in WGSL
                func: (inputShapes, outputShapes) => {
                    const currentInputShape = inputShapes.get(SoftmaxNode.INPUT);
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

                    const uniformData = new Uint32Array(20);
                    uniformData.set(paddedShape, 0);
                    uniformData.set(paddedStrides, MAX_DIMS_SOFTMAX);
                    uniformData.set([currentNdims], MAX_DIMS_SOFTMAX * 2);
                    return uniformData;
                },
                index: 2, // Matches @group(0) @binding(2) var<uniform> params: Params;
            },
            workgroupFunction: (inputShapes, outputShapes) => {
                const currentInputShape = inputShapes.get(SoftmaxNode.INPUT);
                const currentNdims = currentInputShape.length;
                const workgroupSizeX = 256; // Must match the shader's const workgroup_size_x

                /* TODO: This is really, really bad. We need to assume the worst
                 * case scenario since we have no access to DIM here. What's
                 * really the issue is that the CPU needs to know what DIM is to
                 * dispatch enough workgroups, but DIM is in a GPU tensor. The
                 * best solution is to somehow tag certain inputs as needing both
                 * CPU and GPU tensors, then pass the input map to this function
                 * as well.
                 */
                let totalElements = 1;
                if (currentNdims > 0) {
                    totalElements = currentInputShape.reduce((acc, val) => acc * val, 1);
                } else {
                    // For a 0D tensor (scalar), totalElements is 1.
                    totalElements = 1;
                }

                return {
                    // Dispatch enough workgroups to cover every element if dim size was 1.
                    // The kernel handles the slice_id check internally.
                    x: Math.ceil(totalElements / workgroupSizeX),
                    y: 1,
                    z: 1,
                };
            },
            entryPoint: "main",
            inputBindings: [
                { name: SoftmaxNode.INPUT, type: "read-only-storage", index: 0 },
                { name: SoftmaxNode.DIM, type: "read-only-storage", index: 3 }
            ],
            outputBindings: [
                { name: DEFAULT_NODE_OUTPUT, type: "storage", index: 1 },
            ],
        });
    }
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
        this.device = new CPUDevice();
    }
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
        this.device = new CPUDevice();
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
        this.device = new CPUDevice();
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
        this.device = new CPUDevice();
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
        this.device = new CPUDevice();
    }

    getKernel() {
        return new CPUKernel({
            name: 'fixed',
            func: (inputs) => {
                return {
                    [DEFAULT_NODE_OUTPUT]: this.tensor,
                };
            },
            inputs: [],
            outputs: [DEFAULT_NODE_OUTPUT],
        });
    }

    getOutputShape(inputShapes) {
        return this.tensor.shape;
    }
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
        this.device = new GPUDevice();
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
        if (!this.device) {
             this.device = new GPUDevice();
        } 
    }

    /**
     * @param {Map<string, number[]>} inputShapes - Shapes of the input tensors.
     * @param {Map<string, number[]>} outputShapes - Shapes of the output tensors.
     * @returns {Promise<GPUKernel>}
     */
    async getKernel(inputShapes, outputShapes) {
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
            inputBindings: [
                { name: AddNode.A, type: "read-only-storage", index: 0 },
                { name: AddNode.B, type: "read-only-storage", index: 1 },
            ],
            outputBindings: [
                { name: DEFAULT_NODE_OUTPUT, type: "storage", index: 2 },
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
