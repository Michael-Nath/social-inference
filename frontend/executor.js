// executor.js
import { SessionGraph, GPUSession, CPUSession, ComputeSession } from './compiler.js';
import { PartitionWork } from './worker.js'; 

/** 
 * @typedef {GPUBuffer | Tensor} BufferData - Represents stored data (GPU or CPU Tensor)
 * @typedef {string} DataKey - A key identifying a piece of data, e.g., "nodeName:outputName" or "nodeName:inputName"
 */

export class SessionExecutor {
    /** @type {GPUDevice} */
    device;
    /** @type {SessionGraph} */
    sessionGraph;
    /** @type {Map<ComputeSession, 'pending' | 'running' | 'complete' | 'failed'>} */
    sessionStatus;
    /** @type {Map<DataKey, BufferData>} */
    buffers;
    /** @type {ComputeSession[]} */
    readyQueue;
    /** @type {Map<ComputeSession, number>} */
    sessionInDegree;

    /**
     * @param {GPUDevice} device - The WebGPU device instance.
     * @param {SessionGraph} sessionGraph - The compiled and annotated session graph.
     */
    constructor(device, sessionGraph) {
        if (!device || !sessionGraph) {
            throw new Error("SessionExecutor requires a GPUDevice and a SessionGraph.");
        }
        this.device = device;
        this.sessionGraph = sessionGraph;

        this.sessionStatus = new Map(); 
        this.buffers = new Map(); 
        this.readyQueue = []; 
        this.sessionInDegree = new Map(); 
    }

    /**
     * Executes the entire SessionGraph.
     * @param {PartitionWork} partitionWork - Contains initial inputs.
     * @returns {Promise<Map<NodeName, Tensor>>} A map of final graph outputs (NodeName -> Tensor).
     */
    async execute(partitionWork) {
        console.log("SessionExecutor: Starting execution...");
        this._initializeState(partitionWork);
        // Initial ready sessions are found within _initializeState

        // Main execution loop (simplified view)
        let runningTasks = 0;
        const maxConcurrency = navigator.hardwareConcurrency || 4; // Limit concurrent tasks somewhat
        const promises = [];

        const processQueue = async () => {
            while (this.readyQueue.length > 0 && runningTasks < maxConcurrency) {
                 runningTasks++;
                 const sessionToRun = this.readyQueue.shift();
                 this.sessionStatus.set(sessionToRun, 'running');
                 console.log(`SessionExecutor: Starting session ${this.sessionGraph.sessions.indexOf(sessionToRun)} (${sessionToRun.constructor.name})`);

                 const taskPromise = (async () => {
                      try {
                          if (sessionToRun instanceof GPUSession) {
                              await this._executeGPUSession(sessionToRun);
                          } else if (sessionToRun instanceof CPUSession) {
                              await this._executeCPUSession(sessionToRun); // Could be sync or async
                          } else {
                              throw new Error("Unknown session type encountered.");
                          }

                          // Mark as complete and update dependents
                          this.sessionStatus.set(sessionToRun, 'complete');
                          console.log(`SessionExecutor: Completed session ${this.sessionGraph.sessions.indexOf(sessionToRun)}`);
                          this._updateReadyQueue(sessionToRun);

                      } catch (error) {
                          console.error(`Error executing session ${this.sessionGraph.sessions.indexOf(sessionToRun)}:`, error);
                          // Handle error: Mark as failed? Stop execution?
                          this.sessionStatus.set(sessionToRun, 'failed'); // Mark as failed
                           throw error; // Re-throw to be caught by Promise.all
                      } finally {
                           runningTasks--;
                           // Check if more tasks can be started
                           // Use setImmediate/setTimeout to avoid blocking the event loop excessively
                           setTimeout(processQueue, 0);
                      }
                 })();
                 promises.push(taskPromise);
            }
        };

        processQueue(); // Start processing

        // Wait for all initiated tasks to complete or fail
        await Promise.all(promises);

        // Check if all sessions completed successfully
        const allComplete = [...this.sessionStatus.values()].every(status => status === 'complete');
        if (!allComplete) {
             console.error("SessionExecutor: Execution finished with errors.");
            throw new Error("One or more sessions failed during execution.");
        } else {
             console.log("SessionExecutor: Execution finished successfully.");
        }
       
        return this._gatherFinalOutputs();
    }

    /** 
     * Initializes runtime state, calculates in-degrees, finds initial ready sessions.
     * @param {PartitionWork} partitionWork 
     * @private
     */
    _initializeState(partitionWork) {
        console.log(" Initializing executor state...");
        this.sessionStatus.clear();
        this.buffers.clear();
        this.readyQueue = [];
        this.sessionInDegree.clear();

        // Initialize status and calculate in-degrees
        const sessions = this.sessionGraph.sessions;
        sessions.forEach(s => {
            this.sessionStatus.set(s, 'pending');
            this.sessionInDegree.set(s, 0); // Initialize in-degree to 0
        });

        // Calculate actual in-degrees by traversing edges
        sessions.forEach(sourceSession => {
            const dependentSessions = this.sessionGraph.getDependentSessions(sourceSession);
            dependentSessions.forEach(targetSession => {
                this.sessionInDegree.set(targetSession, this.sessionInDegree.get(targetSession) + 1);
            });
        });

        // Find initial sessions (in-degree 0) and add to ready queue
        sessions.forEach(s => {
            if (this.sessionInDegree.get(s) === 0) {
                this.readyQueue.push(s);
                console.log(`  Session ${sessions.indexOf(s)} added to initial ready queue (in-degree 0).`);
            }
        });

        // --- Store Initial Inputs --- 
        // Keying convention: "nodeName:outputName" for data producers
        // InputAssignments tell us where the data goes (`node`, `input`) 
        // but not where it *originated* outside the partition. 
        // This mapping might need adjustment based on how external inputs are represented.
        // For now, let's store based on the target node/input, assuming it's unique.
        // A better approach might be needed if multiple external sources feed the same node input.
        console.log(" Storing initial inputs...");
        for (const assignment of partitionWork.inputs) {
             // Example Key: "TargetNodeName:TargetInputName"
             // This represents the data *required* at this input slot.
            const dataKey = `${assignment.node}:${assignment.input}`; 
            console.log(`  Storing initial input for ${dataKey}`);
            // Store raw tensor for now; conversion/buffer creation happens in session execution
            this.buffers.set(dataKey, assignment.tensor); 
        }
        console.log(" Initialization complete.");
    }

    /** 
     * Updates the ready queue after a session completes by decrementing dependent in-degrees.
     * @param {ComputeSession} completedSession
     * @private
     */
    _updateReadyQueue(completedSession) {
        const dependentSessions = this.sessionGraph.getDependentSessions(completedSession);
        
        console.log(` Session ${this.sessionGraph.sessions.indexOf(completedSession)} completed. Checking dependents: ${dependentSessions.map(s => this.sessionGraph.sessions.indexOf(s)).join(', ')}`);

        for (const dependent of dependentSessions) {
            const currentInDegree = this.sessionInDegree.get(dependent);
            if (currentInDegree > 0) { // Should always be > 0 if it's a dependent
                const newInDegree = currentInDegree - 1;
                this.sessionInDegree.set(dependent, newInDegree);

                // If all dependencies are now met (in-degree is 0), add to ready queue
                if (newInDegree === 0 && this.sessionStatus.get(dependent) === 'pending') {
                    this.readyQueue.push(dependent);
                    console.log(`  Session ${this.sessionGraph.sessions.indexOf(dependent)} added to ready queue (all dependencies met).`);
                }
            }
        }
    }

    // --- Placeholder/Stub Methods --- 
    /** 
     * @param {GPUSession} session 
     * @private 
     * @returns {Promise<void>} 
     */
    async _executeGPUSession(session) { 
        console.warn("_executeGPUSession not implemented"); 
        // Annotate session.resourcePlan usage here later
        await new Promise(r => setTimeout(r, 50)); 
    }
    /** 
     * @param {CPUSession} session 
     * @private 
     * @returns {Promise<void>} 
     */
    async _executeCPUSession(session) { 
        console.warn("_executeCPUSession not implemented"); 
        // Annotate session.resourcePlan usage here later
        await new Promise(r => setTimeout(r, 10)); 
    }
    /** 
     * @private
     * @returns {Map<NodeName, Tensor>} 
     */
    _gatherFinalOutputs() { 
        console.warn("_gatherFinalOutputs not implemented"); 
        return new Map(); 
    }
} 