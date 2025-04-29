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
        console.log(`  Executing GPUSession ${this.sessionGraph.sessions.indexOf(session)}...`);
        if (!session.resourcePlan) {
            throw new Error(`GPUSession ${this.sessionGraph.sessions.indexOf(session)} is missing its resourcePlan.`);
        }

        // --- 1. Prepare Input Buffers ---
        console.log("   Preparing required input buffers...");
        const sessionInputBuffers = new Map(); // Map<DataKey, GPUBuffer> for this session's use

        for (const [inputKey, spec] of session.resourcePlan.requiredInputs.entries()) {
            console.log(`    Processing required input: ${inputKey}`);
            
            // Retrieve the source data (should be a CPUTensor from initial inputs or a CPU session)
            const sourceData = this.buffers.get(inputKey);
            if (!sourceData) {
                throw new Error(`Executor state error: Missing required input data for key '${inputKey}' needed by session ${this.sessionGraph.sessions.indexOf(session)}.`);
            }

            // --- Handle Input Source --- 
            let inputBuffer;
            if (sourceData instanceof GPUBuffer) {
                 // Input is already a GPU buffer (from a previous GPUSession)
                 console.log(`     Reusing existing GPUBuffer for input ${inputKey}`);
                 inputBuffer = sourceData;
                 // TODO: Verify buffer usage flags? (e.g., ensure it allows STORAGE or UNIFORM reading)
                 // We assume the producing session created it with appropriate flags.

            } else if (typeof sourceData === 'object' && sourceData.elements && sourceData.shape && sourceData.dtype) { 
                // Input is likely a CPUTensor (from initial input or CPUSession)
                 console.log(`     Creating new GPU buffer for CPUTensor input ${inputKey} (Size: ${spec.byteSize}, Shape: ${spec.shape}, DType: ${spec.dtype})`);

                // Create the GPU buffer on the device
                const newGpuBuffer = this.device.createBuffer({
                    size: spec.byteSize,
                    // Usage: Needs to be written to (COPY_DST) and used as input in shaders (STORAGE or UNIFORM)
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM, // Add UNIFORM just in case
                    mappedAtCreation: true, // Map it initially for writing
                });

                // Get the mapped range and copy the data
                const mappedRange = newGpuBuffer.getMappedRange();
                let typedArray;
                // Select the correct TypedArray type based on dtype
                switch (spec.dtype) {
                    case 'float32':
                        typedArray = new Float32Array(mappedRange);
                        break;
                    case 'int32':
                        typedArray = new Int32Array(mappedRange);
                        break;
                    // Add cases for other dtypes (float16, int16, int8, uints) as needed
                    default:
                        newGpuBuffer.unmap(); // Unmap before throwing
                        throw new Error(`Unsupported dtype for GPU buffer creation: ${spec.dtype}`);
                }
                
                // Copy data from the source CPU tensor's elements
                typedArray.set(sourceData.elements); 
                newGpuBuffer.unmap();
                inputBuffer = newGpuBuffer; // Use the newly created buffer
                 console.log(`     New GPU buffer created and data written for ${inputKey}`);

            } else {
                // Unexpected data type found in the buffer map
                 throw new Error(`Unexpected data type found for required input '${inputKey}'. Expected GPUBuffer or CPUTensor-like object, got ${typeof sourceData}`);
            }

            // Store the GPU buffer (either reused or newly created) for use within this session's kernels
            sessionInputBuffers.set(inputKey, inputBuffer);
        }

        // --- 2. Prepare Output Buffers ---
        // TODO: Create buffers for outputs specified in session.resourcePlan.producedOutputs
        // These need STORAGE | COPY_SRC usage if they need to be read back later.
        console.log("   Preparing produced output buffers...");
        const sessionOutputBuffers = new Map(); // Map<DataKey, GPUBuffer> for this session's outputs

        for (const [outputKey, spec] of session.resourcePlan.producedOutputs.entries()) {
            console.log(`    Processing produced output: ${outputKey} (Size: ${spec.byteSize}, Shape: ${spec.shape}, DType: ${spec.dtype})`);

             // Create the GPU buffer on the device
            // We don't map this at creation, as the GPU will write to it.
            const gpuBuffer = this.device.createBuffer({
                size: spec.byteSize,
                // Usage: Kernel will write to it (STORAGE), and might need to be copied out (COPY_SRC)
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 
            });

            // Store the created (empty) GPU buffer. Kernels will populate it.
            // Keyed by the original outputKey (e.g., "thisNodeName:thisOutputName")
            sessionOutputBuffers.set(outputKey, gpuBuffer);
             console.log(`     GPU buffer created for output ${outputKey}`);
        }

        // --- 3. Execute Kernels ---
        console.log("   Executing kernels...");
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass(); // Single compute pass for the session for now

        // We assume session.nodes are already topologically sorted *within* the session
        for (const node of session.nodes) {
            const kernel = KernelCompiler.kernelRegistry.get(node.type);
            if (!kernel) {
                // Skip nodes that don't have a registered GPU kernel (e.g., placeholder nodes?)
                 console.log(`    Skipping node ${node.name} (${node.type}): No registered GPU kernel.`);
                continue; 
            }

            console.log(`    Preparing to execute kernel for node ${node.name} (${node.type})`);

            // --- a. Gather Buffers for Bind Group ---
            const bindGroupEntries = [];
            // TODO: This mapping logic is complex and CRITICAL
            // We need to map kernel.bindingConfig entries (by name/index) 
            // to the correct GPUBuffer instances from sessionInputBuffers, sessionOutputBuffers,
            // potentially uniform buffers, or persistent weight buffers.
            // Example (HIGHLY SIMPLIFIED - needs actual logic based on node inputs/outputs):
            for (let i = 0; i < kernel.bindingConfig.length; i++) {
                const bindingConfig = kernel.bindingConfig[i];
                let bufferToBind = null;

                // VERY Placeholder Logic - Needs proper mapping!
                if (bindingConfig.isOutput) {
                    // Find the corresponding output buffer created in Step 2
                    const outputKey = `${node.name}:${bindingConfig.name}`; // Assuming output name matches binding name
                    bufferToBind = sessionOutputBuffers.get(outputKey);
                     console.log(`      Binding output: ${bindingConfig.name} -> ${outputKey}`);
                } else {
                    // Find the corresponding input buffer prepared in Step 1
                    // How do we know *which* input edge corresponds to this binding?
                    // We might need to look at originalGraph.edges or node.computedInputShapes keys?
                    // Let's assume node.computedInputShapes map keys match binding names for now (risky assumption)
                    const inputKey = null; // <<< NEED TO DETERMINE THE CORRECT inputKey (e.g., "previousNode:previousOutput")
                    // bufferToBind = sessionInputBuffers.get(inputKey); 
                    console.warn(`      Binding input: ${bindingConfig.name} -> Cannot determine source buffer yet.`);
                }

                if (!bufferToBind) {
                     console.error(`      Failed to find buffer for binding ${i} (${bindingConfig.name}) on node ${node.name}`);
                     // Should probably throw an error here
                    continue; // Skip binding if buffer not found (for now)
                }

                bindGroupEntries.push({ 
                    binding: i, 
                    resource: { buffer: bufferToBind }
                });
            }

            // --- b. Create Bind Group ---
            // Requires kernel.bindGroupLayout to be prepared by compiler
            if (!kernel.bindGroupLayout) {
                throw new Error(`Kernel ${kernel.name} is missing its bindGroupLayout.`);
            }
            const bindGroup = this.device.createBindGroup({
                layout: kernel.bindGroupLayout,
                entries: bindGroupEntries,
            });

            // --- c. Set Pipeline & Bind Group ---
             if (!kernel.pipeline) {
                throw new Error(`Kernel ${kernel.name} is missing its pipeline.`);
            }
            computePass.setPipeline(kernel.pipeline);
            computePass.setBindGroup(0, bindGroup); // Assuming bind group index 0

            // --- d. Dispatch Kernel ---
            // TODO: Calculate dispatch size based on output shape and kernel workgroup size
            const dispatchX = 1; // Placeholder
            const dispatchY = 1; // Placeholder
            const dispatchZ = 1; // Placeholder
             console.log(`    Dispatching kernel ${kernel.name} for node ${node.name} with workgroups (${dispatchX}, ${dispatchY}, ${dispatchZ})`);
            computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);

        }

        computePass.end();
        this.device.queue.submit([commandEncoder.finish()]);
        console.log(`   GPU commands for session ${this.sessionGraph.sessions.indexOf(session)} submitted.`);

        // --- 4. Handle Output Transfers / Cleanup ---
        // TODO: If outputs need to be read back (e.g., for CPU sessions or final results):
        //  - commandEncoder.copyBufferToBuffer(...) to a staging buffer
        //  - device.queue.submit(...)
        //  - stagingBuffer.mapAsync(GPUMapMode.READ)
        //  - Process mapped data and store in this.buffers
        // TODO: Destroy temporary buffers?

        console.log(`  GPUSession ${this.sessionGraph.sessions.indexOf(session)} execution steps need further implementation.`);
        // Placeholder delay
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