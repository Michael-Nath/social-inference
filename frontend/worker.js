/*
 * Library to handle interfacing with coordination server
 */

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
        this.tensor = new APITensor(api_response.tensor);
    }
}

class OutputAssignment {
    /*
     * @param {Object} options - Output assignment options
     * @param {string} options.node - Node to assign output to
     * @param {string} options.output - Output to assign
     * @param {APITensor} options.tensor - Tensor to assign
     */
    constructor(options) {
        this.node = options.node;
        this.output = options.output;
        this.tensor = new APITensor(options.tensor);
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
     * @param {PartitionWork} work - Partition work to submit
     */
    
}
