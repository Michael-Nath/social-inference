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

    /**
     * @param {ComputeSession[]} sessions - List of compute sessions
     */
    constructor(sessions) {
        this.sessions = sessions;
        this._nodeToSession = new Map();
        this._outputDependencies = new Map();
        this._sessionEdges = new Map();
        this._sessionPredecessorEdges = new Map();
        this._finalOutputs = new Map();
        // Initialize edge maps for all provided sessions
        sessions.forEach(session => this._initializeSessionEdges(session));
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

            if (!currentSession || node.device !== currentSession.device) {
                if (!node.device) {
                    console.warn(`Node ${node.name} has no device assigned. Assuming CPU.`);
                    node.device = new CPUDevice(); // Assign a default device
                }
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
                if (!node.device) {
                    throw new Error(`Cannot create session for node ${node.name} with null device.`);
                }
                currentSession = SessionGraph.createSession(node.device, initialSessions.length);
                initialSessions.push(currentSession);
            }

            currentSession.add(node); // Add node to the session
            nodeToSessionAssignment.set(node.name, currentSession); // Map node name to its assigned session
        }

        // Now that sessions are created and nodes assigned, instantiate the SessionGraph
        const sessionGraphInstance = new SessionGraph(initialSessions);
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
    static _computeAndAnnotateShapes(graph, partitionInputs) {
        // Map to store computed shapes during propagation: Map<NodeName, Map<OutputName, Shape>>
        const outputShapes = new Map();
        // Map to gather input shapes for each node: Map<NodeName, Map<InputName, Shape>>
        const inputShapes = new Map();


        // 1. Seed known shapes from partition inputs
        for (const assignment of partitionInputs) {
            if (!inputShapes.has(assignment.node)) {
                inputShapes.set(assignment.node, new Map());
            }
            // Store shape by the specific input name it connects to
            inputShapes.get(assignment.node).set(assignment.input, assignment.tensor.shape);
        }

        // 2. Seed shapes from constant/fixed nodes and initialize nodeInputShapes map
        for (const [nodeName, node] of Object.entries(graph.nodes)) {
            if (!inputShapes.has(nodeName)) {
                inputShapes.set(nodeName, new Map());
            }

            let nodeOutputMap;
            if (node.type === 'fixed' && node.tensor) {
                // Fixed nodes have a known tensor and shape
                nodeOutputMap = new Map();
                nodeOutputMap.set('output', [...node.tensor.shape]);
            } else if (node.type === 'constant') {
                // TODO: Handle ConstantNode shape retrieval (needs access to tensor cache/info)
                console.warn(`Shape inference for ConstantNode '${nodeName}' not implemented yet.`);
                // If shape were retrieved: nodeOutputMap = {'output': retrievedShape};
            }

            if (nodeOutputMap) {
                outputShapes.set(nodeName, nodeOutputMap);
                node.computedOutputShapes = nodeOutputMap; // Annotate node directly
            }
        }

        // 3. Topological Propagation
        const sortedNodes = graph.topologicalSort();
        for (const node of sortedNodes) {
            const nodeName = node.name;

            // Skip nodes whose shapes are already known (Fixed/Constant)
            if (outputShapes.has(nodeName)) {
                // Still need to ensure input shapes are gathered if needed for debugging?
                // Let's gather them anyway for completeness, even if output is known.
                ;
            }

            // Gather input shapes for the current node from predecessors
            const currentNodeInputMap = inputShapes.get(nodeName);
            for (const edge of graph.edges) {
                if (edge.dst === nodeName) {
                    const srcNodeName = edge.src;
                    const srcOutputName = edge.src_output || 'output'; // Assume default output name if not specified
                    const dstInputName = edge.dst_input;

                    if (!outputShapes.has(srcNodeName) || !outputShapes.get(srcNodeName).has(srcOutputName)) {
                        throw new Error(`Shape inference error: Missing shape for input '${dstInputName}' of node '${nodeName}' from output '${srcOutputName}' of node '${srcNodeName}'.`);
                    }

                    const inputShape = outputShapes.get(srcNodeName).get(srcOutputName);
                    currentNodeInputMap.set(dstInputName, inputShape);
                }
            }

            // ---> Store the computed input shapes on the node <--- 
            // Store a shallow copy of the map
            node.computedInputShapes = new Map(currentNodeInputMap);

            // Compute and store output shape for the current node
            if (outputShapes.has(nodeName)) {
                // Already processed (Fixed/Constant), skip re-computation
                continue;
            }

            if (typeof node.getOutputShape === 'function') {
                try {
                    // getOutputShape expects Map<InputName, Shape>
                    const outputShapeResult = node.getOutputShape(currentNodeInputMap);

                    // Standardize: Ensure result is Map<OutputName, Shape>
                    let nodeOutputMap = new Map();
                    if (Array.isArray(outputShapeResult)) {
                        // Assume single default output 'output'
                        nodeOutputMap.set('output', outputShapeResult);
                    } else if (outputShapeResult instanceof Map) {
                        // If it's already a Map, use it directly
                        nodeOutputMap = outputShapeResult;
                    } else if (typeof outputShapeResult === 'object') {
                        // Convert plain object to Map
                        for (const [key, value] of Object.entries(outputShapeResult)) {
                            nodeOutputMap.set(key, value);
                        }
                    } else {
                        throw new Error(`Node ${nodeName} getOutputShape returned unexpected type: ${typeof outputShapeResult}`);
                    }
                    outputShapes.set(nodeName, nodeOutputMap);
                    node.computedOutputShapes = nodeOutputMap; // Annotate node output
                } catch (error) {
                    console.error(`Error computing shape for node ${nodeName} (${node.type}):`, error);
                    throw new Error(`Shape computation failed for node ${nodeName}. Reason: ${error.message}`);
                }
            } else {
                console.warn(`Node ${nodeName} (${node.type}) does not have a getOutputShape method.`);
                // Handle nodes without shape calculation (e.g., control flow, or assume identity shape?)
                // For now, we might need to error if its output is needed by others and shape is unknown.
                // Let's assume for now such nodes won't exist or their shapes aren't critical downstream.
            }
        }
        // Nodes are annotated in place with computedInputShapes and computedOutputShapes
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
    static _planSessionResources(sessionGraph, originalGraph) {
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

                for (const [outputName, shape] of node.computedOutputShapes.entries()) {
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
                    for (const [outputName, shape] of node.computedOutputShapes.entries()) {
                        const outputKey = `${node.name}:${outputName}`;
                        let readback = false;
                       
                        // Readback if this output is used by a CPU node
                        const inputKeys = forwardEdges.get(outputKey);
                        if(inputKeys) {
                            for (const inputKey of inputKeys) {
                                const inputNode = inputKey.split(':')[0];
                                if (sessionGraph._nodeToSession.get(inputNode) instanceof CPUSession) {
                                    readback = true;
                                    break;
                                }
                            }
                        }

                        // Readback if this output is a final output
                        if(
                            sessionGraph._finalOutputs.has(node.name) &&
                            sessionGraph._finalOutputs.get(node.name).has(outputName)
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
                    let unpreparedKernel = await node.getKernel(node.computedInputShapes, node.computedOutputShapes);
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
                    // Normal bindings
                    for (const binding of kernel.inputBindings.concat(kernel.outputBindings)) {
                        entries.push({
                            label: `${kernel.name}.${kernel.key()}.binding.${binding.index}`,
                            binding: binding.index,
                            visibility: GPUShaderStage.COMPUTE,
                            buffer: {
                                type: binding.type
                            }
                        })
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
            KernelCompiler._computeAndAnnotateShapes(graph, partition.inputs);
            console.log("Compile Step: Shape inference complete.");
        } catch (error) {
            console.error("Compile Step: Shape inference failed:", error);
            throw error;
        }

        // 2. Create Session DAG (Static)
        const sessionGraph = KernelCompiler.createSessionsFrom(graph);
        console.log("Compile Step: Session Graph created:", sessionGraph);

        // 3. Resource Planning Pass (Static)
        try {
            KernelCompiler._planSessionResources(sessionGraph, graph);
            console.log("Compile Step: Resource planning complete.");
        } catch (error) {
            console.error("Compile Step: Resource planning failed:", error);
            throw error;
        }

        // 4. Prepare GPU Resources (Instance Method using this.device)
        try {
            await this._prepareGPUResources(sessionGraph);
            console.log("Compile Step: GPU resources prepared (layouts/pipelines).");
        } catch (error) {
            console.error("Compile Step: GPU resource preparation failed:", error);
            throw error;
        }

        // SessionGraph sessions now have resourcePlan property 
        // and referenced Kernels have layouts/pipelines prepared.
        console.log("Compile Step: Final Annotated Session Graph:", sessionGraph);

        return sessionGraph; // Return the fully prepared SessionGraph
    }
} 