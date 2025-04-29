/*
 * Library to handle interfacing with coordination server
 */

import { CPUTensor, KernelBuilder, Kernel } from "./kernel_builder.js";

/*
 * Kernels
 */

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


const matmulKernel = new Kernel({
    name: "matmul",
    shaderPath: "kernels/matmul.wgsl",
    entryPoint: "main",
    workGroupSize: {x: 16, y: 16, z: 1},
    bindingConfig: [
      {
        name: "dimensions",
        isPersistent: false,
        isOutput: false,
        type: "uniform",
      },
      {
        name: "input",
        isPersistent: false,
        isOutput: false,
        type: "read-only-storage"
      },
      {
        name: "weight",
        isPersistent: true,
        isOutput: false,
        type: "read-only-storage"
      },
      {
        name: "result",
        isPersistent: false,
        isOutput: true,
        type: "storage"
      }
    ],
});

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
    constructor(api_response) {
        super(api_response);
        this.device = new GPUDevice();
    }
}

class ConstantNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.tensor_name = api_response.tensor_name;
    }
}

class SoftmaxNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.device = new GPUDevice();
    }
}

class SliceNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class ReshapeNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class UnsqueezeNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class BroadcastNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.device = CPUDevice();
    }
}

class CatNode extends Node {
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
    constructor(api_response) {
        super(api_response);
        this.device = GPUDevice();
    }
}

class IndexNode extends Node {
    constructor(api_response) {
        super(api_response);
    }
}

class ShapeNode extends Node {
    constructor(api_response) {
        super(api_response);
    }
}

class TransposeNode extends Node {
    constructor(api_response) {
        super(api_response);
        this.dim0 = api_response.dim0;
        this.dim1 = api_response.dim1;
    }
}

class AddNode extends Node {
    constructor(api_response) {
        super(api_response);
    }
}

class DivNode extends Node {
    constructor(api_response) {
        super(api_response);
    }
}

class FloorNode extends Node {
    constructor(api_response) {
        super(api_response);
    }
}

class CeilNode extends Node {
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
