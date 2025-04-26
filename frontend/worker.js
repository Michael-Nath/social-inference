/*
 * Library to handle interfacing with coordination server
 */

import { CPUTensor } from "./kernel_builder.js";

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

class Graph {
    /*
     * @param {Object} api_response
     * @param {Object} api_response.nodes
     * @param {Edge[]} api_response.edges
     */
    constructor(api_response) {
	this.nodes = {};
	for (const [key, value] of Object.entries(api_response.nodes)) {
	    this.nodes[key] = new Node(value);
	}
	this.edges = api_response.edges.map((e) => new Edge(e));
    }

    /*
     * Dumps graph to console
     */
    dump() {
	console.log(self);
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

class PartitionWork {
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
        this.graph = api_response.graph;
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
