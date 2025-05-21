import { Node, Graph, Device, GPUDevice, CPUDevice } from "./worker.js";
// We'll need Kernel eventually, assuming it's defined elsewhere (e.g., kernel_builder.js)
// For now, let's stub it if it's not imported, or assume KernelBuilder provides it.
import { GPUKernel } from "./kernel.js";

// --- Utilities ---

/**
 * Calculates the number of bytes required for a tensor.
 * @param {number[]} shape - The tensor shape.
 * @param {string} dtype - The data type (e.g., "float32", "int32").
 * @returns {number} The required byte size.
 * @throws {Error} If the dtype is unsupported.
 */
function getByteSize(shape, dtype) {
    const numElements = shape.reduce((a, b) => a * b, 1);
    let bytesPerElement;
    switch (dtype) {
        case "float32":
            bytesPerElement = 4;
            break;
        case "int32":
        case "uint32":
            bytesPerElement = 4;
            break;
        case "float16": // Note: Check JS/WebGPU handling for f16 data preparation
            bytesPerElement = 2;
            break;
        case "int16":
        case "uint16":
            bytesPerElement = 2;
            break;
        case "int8":
        case "uint8":
            bytesPerElement = 1;
            break;
        default:
            throw new Error(`Unsupported dtype for byte size calculation: ${dtype}`);
    }
    return numElements * bytesPerElement;
}

// --- Kernels --- 
// Define reusable kernels used by the compiler/executor

/** @type {GPUKernel} */
const matmulKernel = new GPUKernel({
    name: "matmul",
    shaderPath: "kernels/matmul.wgsl",
    entryPoint: "main",
    workGroupSize: { x: 16, y: 16, z: 1 },
    bindingConfig: [
        {
            name: "dimensions", // M, K, N
            isPersistent: false,
            isOutput: false,
            type: "uniform",
        },
        {
            name: "input", // Matrix A (M x K)
            isPersistent: false,
            isOutput: false,
            type: "read-only-storage"
        },
        {
            name: "weight", // Matrix B (K x N)
            isPersistent: true, // Example: weights might persist across runs
            isOutput: false,
            type: "read-only-storage"
        },
        {
            name: "result", // Matrix C (M x N)
            isPersistent: false,
            isOutput: true,
            type: "storage"
        }
    ],
});

/** @type {GPUKernel} */
const addKernel = new GPUKernel({
    name: "add",
    shaderPath: "kernels/add.wgsl",
    entryPoint: "main",
    workGroupSize: { x: 64, y: 1, z: 1 }, // Matches shader
    bindingConfig: [
        {
            name: "inputA",
            isPersistent: false,
            isOutput: false,
            type: "read-only-storage"
        },
        {
            name: "inputB",
            isPersistent: false,
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

// Base class for a computation session (a sequence of nodes on one device)
export class ComputeSession {
    /** @type {number} */
    index;
    /** @type {Device} */
    device;
    /** @type {Map<string, Node>} */ // Input requirements from other sessions? Node or Tensor?
    inputs;
    /** @type {Node[]} */
    nodes;
    /** @type {object | null} - Holds resource plan computed by the compiler */
    resourcePlan = null; // Add resourcePlan property

    /**
      * @param {Device} device - The device (CPU or GPU) this session runs on
    */
    constructor(device, index) {
        this.index = index;
        this.id = index;
        this.device = device;
        this.inputs = new Map();
        this.nodes = [];
    };

    /**
      * Adds a node to the session.
      * @param {Node} node
    */
    add(node) {
        this.nodes.push(node);
    };

    // Implemented by subclasses
    execute() {
        throw new Error("execute() has no default implementation. Must be implemented by subclasses.");
    }
    conclude() {
        throw new Error("conclude() has no default implementation. Must be implemented by subclasses.");
    }
}

// Represents a session running on the CPU
export class CPUSession extends ComputeSession {
    /** @param {CPUDevice} device */
    constructor(device, index) {
        super(device, index);
    };
    // CPU-specific methods might go here
}

// Represents a session running on the GPU
export class GPUSession extends ComputeSession {
    /** @type {Map<string, GPUBuffer>} */ // Example state
    buffers;
    /** @type {Map<string, GPUBindGroup>} */ // Example state
    bindGroups;
    /** @type {GPUCommandEncoder | null} */ // Example state
    commandEncoder;

    /** @param {GPUDevice} device */
    constructor(device, index) {
        super(device, index);
        this.buffers = new Map();
        this.bindGroups = new Map();
        this.commandEncoder = null;
    };

    /**
      * Adds a node to the session.
      * @param {Node} node
    */
    add(node) {
        if (node.type == "input") {
            // Assuming inputs map might store Nodes representing external inputs?
            this.inputs.set(node.name, node);
        }
        super.add(node);
    };
    // GPU-specific methods might go here
}


export class SessionGraphResult {
    constructor(node, should_check) {
        this.node 
    }
}

/** 
 * @typedef {import("./worker.js").Shape} Shape 
 */

/** @typedef {{ shape: Shape, dtype: string, byteSize: number }} BufferSpec */
/** @typedef {{ shape: Shape, dtype: string, byteSize: number, sourceSession: ComputeSession }} InputBufferSpec */

/** 
 * @typedef {object} ResourcePlan
 * @property {Map<string, InputBufferSpec>} requiredInputs - Map<InputKey, Spec> where InputKey is e.g. "srcNodeName:srcOutputName"
 * @property {Map<string, BufferSpec>} producedOutputs - Map<OutputKey, Spec> where OutputKey is e.g. "thisNodeName:thisOutputName"
 */

// Represents the structured DAG of compute sessions
export class SessionGraph {
    /** @type {ComputeSession[]} */
    sessions;
    /** @type {Map<string, ComputeSession>} - Maps Node name (string) to its containing session */
    _nodeToSession;
    /** @type {Map<Node, Set<ComputeSession>>} - Maps producer Node instance to set of consumer sessions */
    _outputDependencies;
    /** @type {Map<ComputeSession, Set<ComputeSession>>} - Maps session to set of sessions that depend on it (successors) */
    _sessionEdges;
    /** @type {Map<ComputeSession, Set<ComputeSession>>} - Maps session to set of sessions it depends on (predecessors) */
    _sessionPredecessorEdges;
    /** @type {Map<string, Set<string>>} - Maps final output nodes to final outputs. node => output is only present if node:output is a final output */
    _finalOutputs;
    /** @type {boolean} */
    single_step;

    /**
     * @param {ComputeSession[]} sessions - List of compute sessions
     */
    constructor(sessions, single_step) {
        this.sessions = sessions;
        this._nodeToSession = new Map();
        this._outputDependencies = new Map();
        this._sessionEdges = new Map();
        this._sessionPredecessorEdges = new Map();
        this._finalOutputs = new Map();
        // Initialize edge maps for all provided sessions
        sessions.forEach(session => this._initializeSessionEdges(session));
        this.single_step = single_step;
    }

    /**
     * Creates a new session of the appropriate type
     * @param {Device} device - The device to create the session for
     * @returns {ComputeSession} - The newly created session
     */
    static createSession(device, index) {
        // Ensure device instances are used correctly
        if (!(device instanceof Device)) {
            throw new Error("Invalid device provided to createSession");
        }
        return device instanceof GPUDevice ?
            new GPUSession(device, index) :
            new CPUSession(device, index);
    }

    /**
     * Initializes edge maps for a session
     * @param {ComputeSession} session
     */
    _initializeSessionEdges(session) {
        this._sessionEdges.set(session, new Set());
        this._sessionPredecessorEdges.set(session, new Set());
    }

    /**
     * Adds a directed edge between sessions
     * @param {ComputeSession} srcSession
     * @param {ComputeSession} dstSession
     */
    _addSessionEdge(srcSession, dstSession) {
        // Ensure source session is initialized (should be by constructor or buildFromGraph)
        if (!this._sessionEdges.has(srcSession)) {
            this._initializeSessionEdges(srcSession);
        }
        this._sessionEdges.get(srcSession).add(dstSession);

        // Ensure destination session is initialized for predecessor tracking
        if (!this._sessionPredecessorEdges.has(dstSession)) {
            this._initializeSessionEdges(dstSession);
        }
        this._sessionPredecessorEdges.get(dstSession).add(srcSession);
    }

    /**
     * Tracks that a session consumes a node's output
     * @param {Node} node
     * @param {ComputeSession} consumerSession
     */
    _addOutputDependency(node, consumerSession) {
        if (!this._outputDependencies.has(node)) {
            this._outputDependencies.set(node, new Set());
        }
        this._outputDependencies.get(node).add(consumerSession);
    }

    /**
     * Gets all sessions that consume a node's output
     * @param {Node} node
     * @returns {Set<ComputeSession>}
     */
    getOutputConsumers(node) {
        return this._outputDependencies.get(node) || new Set();
    }

    /**
     * Gets all sessions that depend on a given session (successors in DAG)
     * @param {ComputeSession} session
     * @returns {Set<ComputeSession>}
     */
    getDependentSessions(session) {
        return this._sessionEdges.get(session) || new Set();
    }

    /**
     * Gets all sessions that a given session depends on (predecessors in DAG)
     * @param {ComputeSession} session
     * @returns {Set<ComputeSession>}
     */
    getPredecessorSessions(session) {
        return this._sessionPredecessorEdges.get(session) || new Set();
    }

    /**
     * Builds a SessionGraph from a node graph.
     * This involves creating sessions, assigning nodes, and establishing dependencies.
     * @param {Graph} graph - The input graph of nodes.
     * @returns {SessionGraph} - The constructed SessionGraph.
     */
    static buildFromGraph(graph) {
        const sortedNodes = graph.topologicalSort();
        const initialSessions = []; // Temporary list to hold sessions during creation
        const nodeToSessionAssignment = new Map(); // Temporary map for node to session assignment

        let currentSession = null;
        for (const node of sortedNodes) {
            let shouldStartNewSession = false;

            if (!node.devicePreference) {
                 throw new Error(`Node ${node.name} has no devicePreference assigned.`);
            }
            if (node.weight === undefined) {
                throw new Error(`Weight not computed for node ${node.name} (${node.type}). Ensure _computeWeights runs before buildFromGraph.`);
            }

            // Determine the target device ("gpu" or "cpu") for the current node using its weight.
            const targetDeviceName = node.devicePreference.pickDevice(node.weight);
            let TargetDeviceClass;
            if (targetDeviceName === "gpu") {
                TargetDeviceClass = GPUDevice;
            } else if (targetDeviceName === "cpu") {
                TargetDeviceClass = CPUDevice;
            } else {
                // This case should ideally be caught by pickDevice or earlier checks if neither is supported
                throw new Error(`Node ${node.name}'s preference resolved to an unknown device type: ${targetDeviceName}`);
            }

            if(targetDeviceName === "cpu" && node.weight > 10000) {
                console.warn(`Node ${node.name} has a weight of ${node.weight} and is being placed on the CPU.`);
            }

            if (!currentSession || !(currentSession.device instanceof TargetDeviceClass)) {
                shouldStartNewSession = true;
            } else {
                for (const edge of graph.edges) {
                    if (edge.dst === node.name) {
                        const srcNode = graph.nodes[edge.src];
                        if (nodeToSessionAssignment.has(srcNode.name) && nodeToSessionAssignment.get(srcNode.name) !== currentSession) {
                            shouldStartNewSession = true;
                            break;
                        }
                    }
                }
            }


            if (shouldStartNewSession) {
                let chosenDevice;
                if (targetDeviceName === "gpu") {
                    chosenDevice = new GPUDevice();
                } else { // targetDeviceName === "cpu"
                    chosenDevice = new CPUDevice();
                }
                currentSession = SessionGraph.createSession(chosenDevice, initialSessions.length);
                initialSessions.push(currentSession);
            }

            currentSession.add(node); // Add node to the session
            nodeToSessionAssignment.set(node.name, currentSession); // Map node name to its assigned session
        }

        // Now that sessions are created and nodes assigned, instantiate the SessionGraph
        const sessionGraphInstance = new SessionGraph(initialSessions, true);
        sessionGraphInstance._nodeToSession = nodeToSessionAssignment; // Assign the populated map

        // Build session DAG edges (_sessionEdges and _sessionPredecessorEdges)
        // initialSessions.forEach(s => sessionGraphInstance._initializeSessionEdges(s)); // Already done by constructor

        for (const edge of graph.edges) {
            const srcNode = graph.nodes[edge.src];
            const dstNode = graph.nodes[edge.dst];

            const srcSession = nodeToSessionAssignment.get(srcNode.name);
            const dstSession = nodeToSessionAssignment.get(dstNode.name);

            if (srcSession && dstSession && srcSession !== dstSession) {
                sessionGraphInstance._addSessionEdge(srcSession, dstSession);
            } else if (!srcSession || !dstSession) {
                console.warn(`Edge ${edge.src}->${edge.dst} links nodes not found in assigned sessions during SessionGraph construction.`);
            }
        }

        // Build output dependencies (_outputDependencies)
        // _outputDependencies is cleared in the constructor, so we build it fresh here.
        for (const edge of graph.edges) {
            const srcNode = graph.nodes[edge.src];
            const dstNode = graph.nodes[edge.dst];

            if (nodeToSessionAssignment.has(srcNode.name) && nodeToSessionAssignment.has(dstNode.name)) {
                const srcSession = nodeToSessionAssignment.get(srcNode.name);
                const dstSession = nodeToSessionAssignment.get(dstNode.name);
                if (srcSession !== dstSession) {
                    sessionGraphInstance._addOutputDependency(srcNode, dstSession);
                }
            }
        }
        console.log("Session Graph built by SessionGraph.buildFromGraph:", sessionGraphInstance);
        return sessionGraphInstance;
    }
}

// Compiles a partition graph into a SessionGraph DAG
export class KernelCompiler {
    /** @type {Map<string, GPUKernel>} */
    static kernelCache = new Map();

    /** @type {GPUDevice} */
    device;

    /**
     * @param {GPUDevice} device - The WebGPU device instance.
     */
    constructor(device) {
        if (!device) {
            throw new Error("KernelCompiler requires a valid GPUDevice instance.");
        }
        this.device = device;
        // TODO: Add caches for pipelines, layouts if needed
        // this.pipelineCache = new Map();
        // this.layoutCache = new Map();
    };

    /**
     * @param {string} key - The key of the kernel to get
     * @returns {GPUKernel} - The kernel
     */
    static getKernel(key) {
        return KernelCompiler.kernelCache.get(key);
    }

    /**
     * Performs shape inference for all nodes in the graph and annotates them.
     * @param {Graph} graph - The compute graph for the partition.
     * @private // Keep static for now as it doesn't depend on this.device
     */
    static _computeWeights(graph, partitionInputs) {
        // Map to store computed weights for each node: Map<NodeName, WeightNumber>
        const nodeComputedWeights = new Map();
        const sortedNodes = graph.topologicalSort();

        for (const node of sortedNodes) {
            const nodeName = node.name;
            const currentEstimatorInputs = new Map(); // Map<inputNameOnNode, weightValue>

            // Populate currentEstimatorInputs from graph edges (weights of predecessor nodes)
            for (const edge of graph.edges) {
                if (edge.dst === nodeName) {
                    const srcNodeName = edge.src;
                    const dstInputName = edge.dst_input;

                    if (!nodeComputedWeights.has(srcNodeName)) {
                        // This should not happen if the graph is valid and sorted topologically,
                        // and all nodes correctly compute a weight.
                        throw new Error(`Weight for source node '${srcNodeName}' (providing input '${dstInputName}' to '${nodeName}') not yet computed.`);
                    }
                    const inputWeightValue = nodeComputedWeights.get(srcNodeName);
                    currentEstimatorInputs.set(dstInputName, inputWeightValue);
                }
            }

            // Populate/overwrite currentEstimatorInputs from partitionInputs (external tensors)
            // These represent the "starting weights" for inputs coming directly into the graph.
            for (const assignment of partitionInputs) {
                if (assignment.node === nodeName) {
                    const inputSlotName = assignment.input;
                    // The weight of a raw input tensor is its number of elements.
                    const tensorWeight = assignment.tensor.shape.reduce((a, b) => a * b, 1);
                    currentEstimatorInputs.set(inputSlotName, tensorWeight);
                }
            }

            // For all other node types, rely on their estimateWeight method.
            // Nodes with no inputs (e.g., ConstantNode) will call estimateWeight with an empty map.
            if (typeof node.estimateWeight !== 'function') {
                throw new Error(`Node ${nodeName} (type: ${node.type}) does not have an estimateWeight method.`);
            }
            const calculatedWeight = node.estimateWeight(currentEstimatorInputs);

            // Store the computed weight on the node instance itself.
            node.weight = calculatedWeight;
            // Store it in our map for downstream nodes that consume this node's output.
            nodeComputedWeights.set(nodeName, calculatedWeight);
        }
        // All nodes in graph.nodes are now annotated with a 'weight' property.
        // The function modifies nodes in-place, so no explicit return is necessary.
    }

    /**
     * Creates a session graph (DAG) from a graph of nodes.
     * @param {Graph} graph - The input graph to create sessions from
     * @returns {SessionGraph} - A graph of compute sessions representing the execution DAG.
     * @private // Keep static for now as it doesn't depend on this.device
     */
    static createSessionsFrom(graph) {
        // All logic is now encapsulated in SessionGraph.buildFromGraph
        const sessionGraph = SessionGraph.buildFromGraph(graph);
        console.log("Session Graph created via KernelCompiler.createSessionsFrom (delegated to SessionGraph.buildFromGraph):", sessionGraph);
        return sessionGraph;
    }

    /**
     * Analyzes sessions to determine buffer requirements (inputs, outputs, sizes).
     * Annotates each session with a `resourcePlan`.
     * @param {SessionGraph} sessionGraph - The graph with sessions and annotated nodes.
     * @param {Graph} originalGraph - The original graph structure for edge info.
     * @private // Keep static for now as it doesn't depend on this.device
     */
    static async _planSessionResources(sessionGraph, originalGraph) {
        // Precompute forward and backward edges for all sessions
        const forwardEdges = new Map(); // Map<OutputKey, Set<InputKey>>
        const backwardEdges = new Map(); // Map<nodeName,Map<inputName, OutputKey>>
        const usedNodes = new Set(); // Tracks nodes that are used by any session
        for (const edge of originalGraph.edges) {
            const outputKey = `${edge.src}:${edge.src_output}`;
            const inputKey = `${edge.dst}:${edge.dst_input}`;

            if (!backwardEdges.has(edge.dst)) {
                backwardEdges.set(edge.dst, new Map());
            }
            backwardEdges.get(edge.dst).set(edge.dst_input, outputKey);

            if (!forwardEdges.has(outputKey)) {
                forwardEdges.set(outputKey, new Set());
            }
            forwardEdges.get(outputKey).add(inputKey);

            usedNodes.add(edge.src);
        }

        for (const session of sessionGraph.sessions) {
            const resourcePlan = {
                inputOutputMappings: new Map(), // Map<InputKey, OutputKey>
                readback: [] // List[OutputKey]
            };

            // Add I/O mappings for all nodes in the session & identify final outputs
            for (const node of session.nodes) {
                if (backwardEdges.has(node.name)) {
                    for (const [inputName, outputKey] of backwardEdges.get(node.name).entries()) {
                        resourcePlan.inputOutputMappings.set(`${node.name}:${inputName}`, outputKey);
                    }
                }

                for (const outputName of node.get_outputs()) {
                    if (!forwardEdges.has(`${node.name}:${outputName}`)) {
                        if(!sessionGraph._finalOutputs.has(node.name)){
                            sessionGraph._finalOutputs.set(node.name, new Set());
                        }
                        sessionGraph._finalOutputs.get(node.name).add(outputName);
                    }
                }
            }
            // Issue readbacks if this node is in a GPU session
            if (session instanceof GPUSession) {
                // Check every output of every node...
                for (const node of session.nodes) {
                    for (const outputName of node.get_outputs()) {
                        const outputKey = `${node.name}:${outputName}`;
                        let readback = false;
                       
                        // Readback if this output is used by a CPU node
                        const inputKeys = forwardEdges.get(outputKey);
                        if(inputKeys) {
                            for (const inputKey of inputKeys) {
                                const inputNode = inputKey.split(':')[0];
                                const inputName = inputKey.split(':')[1];
                                const userSession = sessionGraph._nodeToSession.get(inputNode);
                                if (userSession instanceof CPUSession) {
                                    console.debug(`Readback flagged for ${outputKey} CPU session ${userSession.index} uses it`)
                                    readback = true;
                                    break;
                                } else {
                                    const userNode = userSession.nodes.find(n => n.name === inputNode);
                                    if(!userNode) {
                                        throw new Error(`User node ${inputNode} not found in session ${userSession.name}`);
                                    }
                                    // Check if input has CPU flag set
                                    for (const input of (await userNode.getGPUKernel()).inputs) {
                                        if (input.name === inputName && input.cpu) {
                                            console.debug(`Readback flagged for ${outputKey} because input ${inputName} of node ${inputNode} is CPU-bound.`)
                                            readback = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        // Readback if this output is a final output
                        if(
                            (sessionGraph._finalOutputs.has(node.name) &&
                            sessionGraph._finalOutputs.get(node.name).has(outputName)) ||
                            sessionGraph.single_step
                        ) {
                            readback = true;
                        }

                        if(readback) {
                            resourcePlan.readback.push(outputKey);
                        }
                    }
                }
            }

            session.resourcePlan = resourcePlan;
            console.log("Planned resources:", resourcePlan);
        }
    }

    /**
     * Pre-compiles necessary GPU resources like layouts and pipelines.
     * @param {SessionGraph} sessionGraph 
     * @private
     */
    async _prepareGPUResources(sessionGraph) {
        console.log("Preparing GPU resources (Layouts, Pipelines)...", this.device);
        /** @type {Set<GPUKernel>} */
        const requiredKernels = new Set();
        for (const session of sessionGraph.sessions) {
            if (session instanceof GPUSession) {
                for (const node of session.nodes) {
                    let unpreparedKernel = await node.getGPUKernel();
                    if (!KernelCompiler.kernelCache.has(unpreparedKernel.key())) {
                        requiredKernels.add(unpreparedKernel);
                    }
                }
            }
        }

        for (const kernel of requiredKernels) {
            // --- Create Bind Group Layout (can be cached on kernel object) --- 
            if (!kernel.bindGroupLayout) { // Check if already created/cached
                try {
                    console.debug(`Creating layout for kernel: ${kernel.name} (${kernel.key()})`);
                    let entries = []
                    // Dimension buffer
                    if (kernel.dimensionBuffer) {
                        entries.push({
                            label: `${kernel.name}.${kernel.key()}.binding.${kernel.dimensionBuffer.index}`,
                            binding: kernel.dimensionBuffer.index,
                            visibility: GPUShaderStage.COMPUTE,
                            buffer: { type: 'uniform' }
                        })
                    }
                    // Inputs with bindings
                    for (const input_def of kernel.inputs) {
                        if (input_def.binding) { // Only create layout entry if binding is defined
                            entries.push({
                                label: `${kernel.name}.${kernel.key()}.binding.${input_def.binding.index}`,
                                binding: input_def.binding.index,
                                visibility: GPUShaderStage.COMPUTE,
                                buffer: {
                                    type: input_def.binding.type
                                }
                            });
                        }
                    }
                    // Outputs (always have bindings)
                    for (const output_def of kernel.outputs) {
                        entries.push({
                            label: `${kernel.name}.${kernel.key()}.binding.${output_def.binding.index}`,
                            binding: output_def.binding.index,
                            visibility: GPUShaderStage.COMPUTE,
                            buffer: {
                                type: output_def.binding.type
                            }
                        });
                    }
                    kernel.bindGroupLayout = this.device.createBindGroupLayout({
                        label: `${kernel.name}.${kernel.key()}.bindgrouplayout`,
                        entries: entries
                    });
                } catch (error) {
                    console.error(`Failed to create bind group layout for kernel ${kernel.name} (${kernel.key()}):`, error);
                    throw error;
                }
            }

            // --- Create Shader Module & Compute Pipeline (can be cached on kernel object) ---
            if (!kernel.pipeline) { // Check if already created/cached
                try {
                    console.debug(`Compiling shader for kernel: ${kernel.name} (${kernel.key()})`);

                    const shaderModule = this.device.createShaderModule({
                        label: `${kernel.name}.${kernel.key()}.shadermodule`,
                        code: kernel.shader
                    });
                    console.debug(`Creating pipeline for kernel: ${kernel.name} (${kernel.key()})`);

                    kernel.pipeline = await this.device.createComputePipelineAsync({
                        label: `${kernel.name}.${kernel.key()}.pipeline`,
                        layout: this.device.createPipelineLayout({
                            bindGroupLayouts: [kernel.bindGroupLayout]
                        }),
                        compute: {
                            module: shaderModule,
                            entryPoint: kernel.entryPoint,
                        },
                    });
                } catch (error) {
                    console.error(`Failed to create compute pipeline for kernel ${kernel.name}:`, error);
                    throw error;
                }
            }
            console.log(`    Prepared kernel: ${kernel.name} (${kernel.key()})`);
            KernelCompiler.kernelCache.set(kernel.key(), kernel);
        }
        console.log(`GPU resources prepared. Prepared ${requiredKernels.size} kernels.`);
    }

    /**
     * Compile the partition work into an executable format (SessionGraph DAG),
     * including pre-compiling necessary GPU resources.
     * @param {PartitionWork} partition - The partition work containing the graph.
     * @returns {Promise<SessionGraph>} A promise resolving to the annotated and prepared SessionGraph.
     */
    async compile(partition) {
        /** @type {Graph} */
        const graph = partition.graph;
        console.log("Compile Step: Received Original Graph:", graph);

        // 1. Shape Inference Pass (Static)
        try {
            KernelCompiler._computeWeights(graph, partition.inputs);
            console.log("Compile Step: Shape inference complete.");
        } catch (error) {
            console.error("Compile Step: Shape inference failed:", error);
            throw error;
        }

        // 2. Create Session DAG (Static)
        const sessionGraph = KernelCompiler.createSessionsFrom(graph);
        console.log("Compile Step: Session Graph created:", sessionGraph);

        // 4. Prepare GPU Resources (Instance Method using this.device)
        try {
            await this._prepareGPUResources(sessionGraph);
            console.log("Compile Step: GPU resources prepared (layouts/pipelines).");
        } catch (error) {
            console.error("Compile Step: GPU resource preparation failed:", error);
            throw error;
        }

        // 3. Resource Planning Pass (Static)
        try {
            await KernelCompiler._planSessionResources(sessionGraph, graph);
            console.log("Compile Step: Resource planning complete.");
        } catch (error) {
            console.error("Compile Step: Resource planning failed:", error);
            throw error;
        }

        

        // SessionGraph sessions now have resourcePlan property 
        // and referenced Kernels have layouts/pipelines prepared.
        console.log("Compile Step: Final Annotated Session Graph:", sessionGraph);

        return sessionGraph; // Return the fully prepared SessionGraph
    }
} 