import { Coordinator, PartitionWork, Node, Graph, GPUDevice, CPUDevice } from "./worker.js";

class ComputeSession {
  // represents a runtime portion of a partition
  // a partition can be composed of multiple sessions depending on the sequence
  // of nodes in the parition:
  // ex: reshape -> elementwise -> matmul -> reshape -> elementwise
  // will become a pipeline of CPUSession([reshape]) -> GPUSession([elementwise, matmul]) -> CPUSession([reshape]) -> GPUSession([elementwise])
  // 

  /**
    * Represents a compute session
    * @property {Node} inputs - the inputs for this session
    * @property {Node[]} nodes - the list of nodes involved in this session
  */
  constructor(device) {
    this.device = device
    this.inputs = new Map()
    this.nodes  = [] // list of `Node` objects
  };

  /**
    * Adds a node to the session.
    * @param {Node} node
  */
  add(node) {
    // ASSUMPTION: `node` is of type Node
  };

  // execute the session
  execute() {
  }

  // return the result of the session
  conclude() {
  }
}

class CPUSession extends ComputeSession {
  constructor(device) {
    super(device);
  };
}

/**
 * @extends {ComputeSession}
 */
class GPUSession extends ComputeSession {
  constructor(device) {
    super(device);
    this.buffers = new Map();
    this.bindGroups = new Map();
    this.comamndEncoder = null;
    this.readbackBuffers = [];
    this.errorScopes = [];
  };

  /**
    * Adds a node to the session.
    * @param {Node} node
  */
  add(node) {
    if (node.type == "input") {
      this.inputs[node.name] = node;
    }
  };
}

class SessionGraph {
    /**
     * @param {ComputeSession[]} sessions - List of compute sessions
     */
    constructor(sessions) {
        this.sessions = sessions;
        this._nodeToSession = new Map();
        this._outputDependencies = new Map();
        // Edges: Map<ComputeSession, ComputeSession[]> representing the DAG
        this._sessionEdges = new Map(); 
    }

    /**
     * Creates a new session of the appropriate type
     * @param {Device} device - The device to create the session for
     * @returns {ComputeSession} - The newly created session
     */
    static createSession(device) {
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

class KernelCompiler {
  // it makes sense to compile at the level of partitions
  // we want to be very careful with how we are working with tensors here

  constructor(device) {
    this.activeSession = null;
  };

  /**
   * Creates a session graph (DAG) from a graph of nodes.
   * @param {Graph} graph - The input graph to create sessions from
   * @returns {SessionGraph} - A graph of compute sessions representing the execution DAG.
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
            currentSession = SessionGraph.createSession(node.device);
            sessions.push(currentSession);
        }

        // Add node to the *identified* current session and map it immediately
        // SessionGraph._addNodeAndMap would be better if SessionGraph instance existed
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

        // Add edge only if sessions are different
        if (srcSession !== dstSession) { 
            sessionEdgesMap.get(srcSession).add(dstSession);
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
        const srcSession = finalSessionGraph._nodeToSession.get(srcNode);
        const dstSession = finalSessionGraph._nodeToSession.get(dstNode);
        if (srcSession !== dstSession) {
            // Use the internal method of the final graph instance
            finalSessionGraph._addOutputDependency(srcNode, dstSession); 
        }
    }

    return finalSessionGraph;
  }

  /**
  * @param {PartitionWork} parition
  */
  compile(partition) {
    // this will create those sessions out of the partition
    /** @type {Graph} */
    const graph = partition.graph;

    console.log(graph)
    const sessions = KernelCompiler.createSessionsFrom(graph);
    console.log(sessions);
  }
};

const coordinator = new Coordinator({
  url: "",
});

const registration = await coordinator.register();
console.log(registration);
const work = await coordinator.get_work(registration.partition);
console.log(work);
// work.graph.dump();
const compiler = new KernelCompiler([]);
compiler.compile(work);