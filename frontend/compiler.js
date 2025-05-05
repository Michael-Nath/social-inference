import { Node, Graph, Device, GPUDevice, CPUDevice } from "./worker.js";
// We'll need Kernel eventually, assuming it's defined elsewhere (e.g., kernel_builder.js)
// For now, let's stub it if it's not imported, or assume KernelBuilder provides it.
import { Kernel } from "./kernel_builder.js"; 

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

/** @type {Kernel} */
const matmulKernel = new Kernel({
    name: "matmul",
    shaderPath: "kernels/matmul.wgsl",
    entryPoint: "main",
    workGroupSize: {x: 16, y: 16, z: 1},
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

/** @type {Kernel} */
const addKernel = new Kernel({
    name: "add",
    shaderPath: "kernels/add.wgsl",
    entryPoint: "main",
    workGroupSize: {x: 64, y: 1, z: 1}, // Matches shader
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
  constructor(device) {
    this.device = device;
    this.inputs = new Map(); 
    this.nodes  = []; 
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
  constructor(device) {
    super(device);
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
  constructor(device) {
    super(device);
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
    /** @type {Map<Node, ComputeSession>} - Maps Node instance to its containing session */
    _nodeToSession;
    /** @type {Map<Node, Set<ComputeSession>>} - Maps producer Node instance to set of consumer sessions */
    _outputDependencies;
    /** @type {Map<ComputeSession, ComputeSession[]>} - Maps session to list of sessions that depend on it */
    _sessionEdges;

    /**
     * @param {ComputeSession[]} sessions - List of compute sessions
     */
    constructor(sessions) {
        this.sessions = sessions;
        this._nodeToSession = new Map();
        this._outputDependencies = new Map(); 
        this._sessionEdges = new Map(); 
    }

    /**
     * Creates a new session of the appropriate type
     * @param {Device} device - The device to create the session for
     * @returns {ComputeSession} - The newly created session
     */
    static createSession(device) {
        // Ensure device instances are used correctly
        if (!(device instanceof Device)) {
            throw new Error("Invalid device provided to createSession");
        }
        return device instanceof GPUDevice ? 
            new GPUSession(device) : 
            new CPUSession(device);
    }

    /**
     * Adds a node to a session and updates the node-to-session mapping
     * (Internal use during graph construction)
     * @param {Node} node - The node to add
     * @param {ComputeSession} session - The session to add it to
     */
    _addNodeAndMap(node, session) {
        session.add(node);
        this._nodeToSession.set(node, session);
    }

    /**
     * Initializes edges for a session
     * @param {ComputeSession} session
     */
    _initializeSessionEdges(session) {
        this._sessionEdges.set(session, []);
    }

    /**
     * Adds an edge between sessions
     * @param {ComputeSession} srcSession
     * @param {ComputeSession} dstSession
     */
    _addSessionEdge(srcSession, dstSession) {
        if (!this._sessionEdges.has(srcSession)) {
            this._initializeSessionEdges(srcSession);
        }
        const edges = this._sessionEdges.get(srcSession);
        // Use a Set temporarily or check existence
        if (!edges.includes(dstSession)) { 
            edges.push(dstSession);
        }
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
     * @returns {ComputeSession[]}
     */
    getDependentSessions(session) {
        return this._sessionEdges.get(session) || [];
    }
}

// Compiles a partition graph into a SessionGraph DAG
export class KernelCompiler {

  /** @type {Map<string, Kernel>} */
  static kernelRegistry = new Map([
      ["matmul", matmulKernel],
      ["add", addKernel]
      // Add other mappings here as needed
  ]);

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
              console.log("Already processed node:", nodeName);
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
      const nodes = graph.topologicalSort();
      const sessions = [];
      const nodeToSessionMap = new Map(); // Track final session for each node
  
      // 1. Create initial sessions based on boundaries (device changes or cross-session dependencies)
      let currentSession = null;
      for (const node of nodes) {
          let shouldStartNewSession = false;
  
          if (!currentSession || node.device !== currentSession.device) {
              // Basic check: if node.device is null, it might need a default (e.g., CPU) or indicate an error
              if (!node.device) {
                   console.warn(`Node ${node.name} has no device assigned. Assuming CPU.`);
                   node.device = new CPUDevice(); // Assign a default device
              }
              shouldStartNewSession = true;
          } else {
              // Check if any input dependency comes from a DIFFERENT *already assigned* session
              for (const edge of graph.edges) {
                  if (edge.dst === node.name) {
                      const srcNode = graph.nodes[edge.src];
                      // Crucially, check the map for the source node's finalized session
                      if (nodeToSessionMap.has(srcNode) && nodeToSessionMap.get(srcNode) !== currentSession) {
                          shouldStartNewSession = true;
                          break;
                      }
                  }
              }
          }
  
          if (shouldStartNewSession) {
              // Ensure node.device is valid before creating session
              if (!node.device) { 
                   throw new Error(`Cannot create session for node ${node.name} with null device.`);
              }
              currentSession = SessionGraph.createSession(node.device);
              sessions.push(currentSession);
          }
  
          // Add node to the *identified* current session and map it immediately
          currentSession.add(node); 
          nodeToSessionMap.set(node, currentSession);
      }
  
      // 2. Build the Session DAG edges
      const sessionEdgesMap = new Map(); // Map<ComputeSession, Set<ComputeSession>>
      sessions.forEach(s => sessionEdgesMap.set(s, new Set())); // Use Set for unique edges
  
      for (const edge of graph.edges) {
          const srcNode = graph.nodes[edge.src];
          const dstNode = graph.nodes[edge.dst];
  
          const srcSession = nodeToSessionMap.get(srcNode);
          const dstSession = nodeToSessionMap.get(dstNode);
  
          // Add edge only if sessions are different and both sessions exist
          if (srcSession && dstSession && srcSession !== dstSession) { 
              sessionEdgesMap.get(srcSession).add(dstSession);
          } else if (!srcSession || !dstSession) {
              console.warn(`Edge ${edge.src}->${edge.dst} links nodes not found in sessions.`);
          }
      }
  
      // 3. Construct the final SessionGraph and populate its internal state
      const finalSessionGraph = new SessionGraph(sessions);
      finalSessionGraph._nodeToSession = nodeToSessionMap;
      
      // Convert Set edges to Array edges for storage in SessionGraph
      sessionEdgesMap.forEach((targets, source) => {
          finalSessionGraph._sessionEdges.set(source, Array.from(targets)); 
      });
  
      // Build _outputDependencies based on the final session assignments and graph edges
      finalSessionGraph._outputDependencies.clear(); // Ensure it's empty before building
      for (const edge of graph.edges) {
          const srcNode = graph.nodes[edge.src];
          const dstNode = graph.nodes[edge.dst];
          // Ensure nodes exist in map before proceeding
          if (finalSessionGraph._nodeToSession.has(srcNode) && finalSessionGraph._nodeToSession.has(dstNode)) {
              const srcSession = finalSessionGraph._nodeToSession.get(srcNode);
              const dstSession = finalSessionGraph._nodeToSession.get(dstNode);
              if (srcSession !== dstSession) {
                  // Use the internal method of the final graph instance
                  finalSessionGraph._addOutputDependency(srcNode, dstSession); 
              }
          }
      }
      console.log("Session Graph created:", finalSessionGraph);
      return finalSessionGraph;
  }
  
  /**
   * Analyzes sessions to determine buffer requirements (inputs, outputs, sizes).
   * Annotates each session with a `resourcePlan`.
   * @param {SessionGraph} sessionGraph - The graph with sessions and annotated nodes.
   * @param {Graph} originalGraph - The original graph structure for edge info.
   * @private // Keep static for now as it doesn't depend on this.device
   */
  static _planSessionResources(sessionGraph, originalGraph) {
      for (const session of sessionGraph.sessions) {
          const resourcePlan = {
              requiredInputs: new Map(), // Map<InputKey, { shape, dtype, byteSize, sourceSession }> 
              producedOutputs: new Map() // Map<OutputKey, { shape, dtype, byteSize }>
              // TODO: Add internalTemporaries analysis later if needed for optimization
          };

          // Determine Required Inputs from other sessions
          for (const node of session.nodes) {
              for (const edge of originalGraph.edges) {
                  if (edge.dst === node.name) {
                      const srcNodeName = edge.src;
                      const srcNode = originalGraph.nodes[srcNodeName];
                      const srcSession = sessionGraph._nodeToSession.get(srcNode);

                      if (srcSession !== session) {
                          // This node needs input from an external session
                          const srcOutputName = edge.src_output || 'output';
                          const inputKey = `${srcNodeName}:${srcOutputName}`;

                          if (!resourcePlan.requiredInputs.has(inputKey) && srcNode.computedOutputShapes && srcNode.computedOutputShapes[srcOutputName]) {
                              const shape = srcNode.computedOutputShapes[srcOutputName];
                              // *** TODO: Determine correct dtype! Assuming float32 for now. ***
                              const dtype = srcNode.type === 'fixed' ? srcNode.tensor.dtype : 'float32'; 
                              const byteSize = getByteSize(shape, dtype);
                              resourcePlan.requiredInputs.set(inputKey, {
                                  shape: shape,
                                  dtype: dtype,
                                  byteSize: byteSize,
                                  sourceSession: srcSession // Track where it comes from
                              });
                          } else if (!srcNode.computedOutputShapes || !srcNode.computedOutputShapes[srcOutputName]){
                               console.warn(`Resource Planning: Cannot find shape for required input ${inputKey} for node ${node.name}`);
                          }
                      }
                  }
              }
          }

          // Determine Produced Outputs needed by other sessions
          for (const node of session.nodes) {
               if (!node.computedOutputShapes) continue; // Skip if shape inference failed

               const outputConsumers = sessionGraph.getOutputConsumers(node); // Set<ComputeSession>
               let isOutputExternal = false;
               if (outputConsumers.size > 0) {
                    for(const consumer of outputConsumers) {
                        if (consumer !== session) {
                            isOutputExternal = true;
                            break;
                        }
                    }
               }
               // TODO: Also need to check if this node is a final output of the *entire* graph/partition.

               if (isOutputExternal) {
                   for (const [outputName, shape] of Object.entries(node.computedOutputShapes)) {
                       const outputKey = `${node.name}:${outputName}`;
                       if (!resourcePlan.producedOutputs.has(outputKey)) {
                            // *** TODO: Determine correct dtype! Assuming float32 for now. ***
                           const dtype = node.type === 'fixed' ? node.tensor.dtype : 'float32'; 
                           const byteSize = getByteSize(shape, dtype);
                            resourcePlan.producedOutputs.set(outputKey, {
                               shape: shape,
                               dtype: dtype,
                               byteSize: byteSize
                           });
                       }
                   }
               }
          }
          
          // Annotate the session
          session.resourcePlan = resourcePlan;
      }
  }
  
  /**
   * Pre-compiles necessary GPU resources like layouts and pipelines.
   * @param {SessionGraph} sessionGraph 
   * @private
   */
  async _prepareGPUResources(sessionGraph) {
      console.log("Preparing GPU resources (Layouts, Pipelines)...", this.device);
      const requiredKernels = new Set();
      for (const session of sessionGraph.sessions) {
          if (session instanceof GPUSession) {
              for (const node of session.nodes) {
                  const kernel = KernelCompiler.kernelRegistry.get(node.type);
                  if (kernel) {
                      requiredKernels.add(kernel);
                  } else {
                      // Handle cases where a GPU node doesn't have a registered kernel
                      console.warn(`No kernel found in registry for GPU node type: ${node.type}`);
                  }
              }
          }
      }

      for (const kernel of requiredKernels) {
          // --- Create Bind Group Layout (can be cached on kernel object) --- 
          if (!kernel.bindGroupLayout) { // Check if already created/cached
              try {
                  console.log(`Creating layout for kernel: ${kernel.name}`);
                  kernel.bindGroupLayout = this.device.createBindGroupLayout({
                      entries: kernel.bindingConfig.map((binding, index) => ({
                          binding: index, 
                          visibility: GPUShaderStage.COMPUTE,
                          buffer: { 
                              type: binding.type // e.g., 'storage', 'read-only-storage', 'uniform'
                              // TODO: Add hasDynamicOffset, minBindingSize if needed
                          }
                          // TODO: Add sampler, texture, storageTexture if needed
                      }))
                  });
              } catch (error) {
                    console.error(`Failed to create bind group layout for kernel ${kernel.name}:`, error);
                    throw error;
              }
          }

          // --- Create Shader Module & Compute Pipeline (can be cached on kernel object) ---
          if (!kernel.pipeline) { // Check if already created/cached
                try {
                    console.log(`Fetching/Compiling shader for kernel: ${kernel.name} from ${kernel.shaderPath}`);
                    // TODO: Implement shader fetching/caching if not already part of Kernel class
                    const response = await fetch(kernel.shaderPath);
                    if (!response.ok) throw new Error(`Failed to fetch shader: ${response.statusText}`);
                    const wgslCode = await response.text();
                    
                    const shaderModule = this.device.createShaderModule({ code: wgslCode });
                    console.log(`Creating pipeline for kernel: ${kernel.name}`);

                    kernel.pipeline = await this.device.createComputePipelineAsync({
                        layout: this.device.createPipelineLayout({ 
                            bindGroupLayouts: [kernel.bindGroupLayout]
                        }),
                        compute: {
                            module: shaderModule,
                            entryPoint: kernel.entryPoint,
                            // constants: {} // Optional constants
                        },
                    });
                } catch(error) {
                     console.error(`Failed to create compute pipeline for kernel ${kernel.name}:`, error);
                     throw error;
                }
          }
      }
       console.log("GPU resources prepared.");
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