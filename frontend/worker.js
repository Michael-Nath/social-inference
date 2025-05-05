/*
 * Library to handle interfacing with coordination server
 */

import { CPUTensor, KernelBuilder, Kernel } from "./kernel_builder.js";

/*
 * Kernels
 */

// Default node output name used by all nodes
export const DEFAULT_NODE_OUTPUT = "output";

export class Device {
    constructor() {};
}

export class GPUDevice extends Device {
    constructor() {
        if (GPUDevice._instance) {
            return GPUDevice.instance;
        };
        super();
        GPUDevice._instance = this;
        // TODO: will probably need a reference to the WebGPU device here at some point
    }
};
export class CPUDevice extends Device {
    constructor() {
        if (CPUDevice._instance) {
            return CPUDevice.instance;
        };
        super();
        CPUDevice._instance = this;
    }
};

/* API Objects. Should match BaseModel specs of webserver.py and inference/*.py */
class Registration {
    /*
     * @param {Object} api_response - Registration response from server
     * @param {string} api_response.partition - Partition to register as
     */
    constructor(api_response) {
        this.partition = api_response.partition;
    }
}

class APITensor {
    /*
     * @param {Object} api_response - Tensor response from server
     * @param {Array<number>} api_response.elements - Flattened array of tensor elements
     * @param {Array<number>} api_response.shape - Shape of the tensor
     * @param {string} api_response.dtype - Data type of the tensor
     */
    constructor(api_response) {
        this.elements = api_response.elements;
        this.shape = api_response.shape;
        this.dtype = api_response.dtype;
    }

    /*
     * @param {CPUTensor} tensor 
     * @returns {APITensor}
     */
    static fromCPU(tensor) {
	return new APITensor({
	    elements: Array.from(tensor.elements),
	    shape: tensor.shape,
	    dtype: tensor.dtype
	});
    }

    /*
     * @returns {CPUTensor}
     */
    toCPU() {
	return new CPUTensor({
	    elements: this.elements,
	    shape: this.shape,
	    dtype: this.dtype
	});
    }
}

class Edge {
    /*
     * @param {Object} api_response
     * @param {string} api_response.src
     * @param {string} api_response.src_output
     * @param {string} api_response.dst
     * @param {string} api_response.dst_input
     */
    constructor(api_response) {
	this.src = api_response.src;
	this.src_output = api_response.src_output;
	this.dst = api_response.dst;
	this.dst_input = api_response.dst_input;
    }
}

export class Node {
    /** 
     * @param {Object} api_response - Node response from server
     * @param {string} api_response.type - Type of node
     * @param {string} api_response.name - Name of node
     * @property {Device} device - the device in which this node should be executed on
     */
    constructor(api_response) {
        this.type = api_response.type;
        this.name = api_response.name;
        this.device = null;
        
        // Copy any additional properties from the API response
        for (const [key, value] of Object.entries(api_response)) {
            if (key !== "type" && key !== "name") {
                this[key] = value;
            }
        }
    }
}

class MatmulNode extends Node {
    static LHS = "lhs";
    static RHS = "rhs";
    
    constructor(api_response) {
        super(api_response);
        this.device = new GPUDevice();
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

class ConstantNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.tensor_name = api_response.tensor_name;
    }
}

class SoftmaxNode extends Node {
    static INPUT = "input";
    static DIM = "dim";
    
    constructor(api_response) {
        super(api_response);
        this.device = new GPUDevice();
    }
}

class SliceNode extends Node {
    static INPUT = "input";
    static DIM = "dim";
    static START = "start";
    static END = "end";
    
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class ReshapeNode extends Node {
    static INPUT = "input";
    static DIMS = "dims";
    
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class UnsqueezeNode extends Node {
    static INPUT = "input";
    static DIM = "dim";
    
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class BroadcastNode extends Node {
    static INPUT = "input";
    static DIM = "dim";
    static N = "n";
    
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class CatNode extends Node {
    static A = "a";
    static B = "b";
    static DIM = "dim";
    
    constructor(api_response) {
        super(api_response);
    }
}

class FixedNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.tensor = (new APITensor(api_response.tensor)).toCPU();
    }
}

class HadamardNode extends Node {
    static A = "a";
    static B = "b";
    
    constructor(api_response) {
        super(api_response);
        this.device = GPUDevice();
    }
}

class IndexNode extends Node {
    static INPUT = "input";
    static INDEX = "index";
    
    constructor(api_response) {
        super(api_response);
    }
}

class ShapeNode extends Node {
    static INPUT = "input";
    
    constructor(api_response) {
        super(api_response);
    }
}

class TransposeNode extends Node {
    static INPUT = "input";
    
    constructor(api_response) {
        super(api_response);
        this.dim0 = api_response.dim0;
        this.dim1 = api_response.dim1;
    }
}

class AddNode extends Node {
    static A = "a";
    static B = "b";
    
    constructor(api_response) {
        super(api_response);
        // Add nodes typically run on CPU by default unless specified otherwise
        // Or they could be fused into GPU kernels. Let's assume CPU for now.
        if (!this.device) {
             this.device = new CPUDevice();
        } 
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

class DivNode extends Node {
    static A = "a";
    static B = "b";
    
    constructor(api_response) {
        super(api_response);
    }
}

class FloorNode extends Node {
    static INPUT = "input";
    
    constructor(api_response) {
        super(api_response);
    }
}

class CeilNode extends Node {
    static INPUT = "input";
    
    constructor(api_response) {
        super(api_response);
    }
}

class DebugNode extends Node {
    static INPUT = "input";
    
    constructor(api_response) {
        super(api_response);
    }
}

export class Graph {
    /*
     * @param {Object} api_response
     * @param {Object} api_response.nodes
     * @param {Edge[]} api_response.edges
     */
    constructor(api_response) {
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

class InputAssignment {
    /*
     * @param {Object} api_response - Input assignment response from server
     * @param {string} api_response.node - Node to assign input to
     * @param {string} api_response.input - Input to assign
     * @param {APITensor} api_response.tensor - Tensor to assign
     */
    constructor(api_response) {
        this.node = api_response.node;
        this.input = api_response.input;
        this.tensor = (new APITensor(api_response.tensor)).toCPU();
    }
}

class OutputAssignment {
    /*
     * @param {Object} options - Output assignment options
     * @param {string} options.node - Node to assign output to
     * @param {string} options.output - Output to assign
     * @param {CPUTensor} options.tensor - Tensor to assign
     */
    constructor(options) {
        this.node = options.node;
        this.output = options.output;
        this.tensor = APITensor.fromCPU(options.tensor);
    }
}

export class PartitionWork {
    /*
     * @param {Object} api_response - Partition work response from server
     * @param {string} api_response.correlation_id - Correlation ID of the work
     * @param {string} api_response.partition - Partition to get work for
     * @param {Object} api_response.graph - Graph to execute
     * @param {InputAssignment[]} api_response.inputs - Inputs to the graph
     */
    constructor(api_response) {
        this.correlation_id = api_response.correlation_id;
        this.partition = api_response.partition;
        this.graph = new Graph(api_response.graph);
        this.inputs = api_response.inputs;
    }
}

class PartitionWorkResult {
    /*
     * @param {Object} api_response - Partition work result response from server
     * @param {string} api_response.correlation_id - Correlation ID of the work
     * @param {string} api_response.partition - Partition to submit work for
     * @param {OutputAssignment[]} api_response.outputs - Outputs from the graph
     */
    constructor(api_response) {
        this.correlation_id = api_response.correlation_id;
        this.partition = api_response.partition;
        this.outputs = api_response.outputs;
    }
}

export class Coordinator {
    /*
     * @param {Object} options - Coordinator configuration options
     * @param {string} options.url - URL of the coordination server
     */
    constructor(options) {
        this.url = options.url;
    }

    /*
     * Registers the worker with the coordination server
     * @returns {Registration} - Registration response from server
     */
    async register() {
        const response = await fetch(`${this.url}/register`, {
            method: "POST",
            body: JSON.stringify({}),
        });
        return new Registration(await response.json());
    }

    /*
     * Gets the next partition work from the coordination server
     * @param {string} partition_name - Partition to get work for
     * @returns {PartitionWork | null} - Partition work from server, or null if no work is available
     */
    async get_work(partition_name) {
        const response = await fetch(`${this.url}/work/${partition_name}`);
        return new PartitionWork(await response.json());
    }

    /*
     * Submits the partition work to the coordination server
     * @param {string} partition_name - Partition to submit to
     * @param {PartitionWorkResult} work - Partition work to submit
     */
    async submit_work(partition_name, work) {
	await fetch(`${this.url}/work`, {
	    method: "POST",
	    body: JSON.stringify(work)
	});
    }
}

class PreparedGraph {

}

export class Worker {
    constructor() {
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
    }

    /*
     * @param {PartitionWork} partition_work - Partition work to execute
     * @returns {Promise<void>}
     */
    async _execute_partition_work(partition_work) {
    }
}
