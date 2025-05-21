// executor.js
import { SessionGraph, GPUSession, CPUSession, ComputeSession, KernelCompiler } from './compiler.js';
import { PartitionWork, ExecutionContext } from './worker.js';
import { GPUTensor, CPUTensor } from './kernel.js';
import { SafeTensorCache } from './tensorcache.js';
import { ALL_SCOPES, popErrorScopes, pushErrorScopes } from './common.js';
import { UIManager } from './uiManager.js';

/** @typedef {import('./kernel.js').Tensor} Tensor */
/** @typedef {import('./kernel.js').GPUKernel} GPUKernel */

/** 
 * @typedef {GPUBuffer | Tensor} BufferData - Represents stored data (GPU or CPU Tensor)
 * @typedef {string} DataKey - A key nodename:outputname identifying a piece of data, e.g., "nodeName:outputName" 
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

    /** @type {Map<DataKey, CPUTensor>} */
    cpuInputs; // edge case (keyed by input, not output); inputs from the partition work
    /** @type {Map<DataKey, GPUTensor>} */
    gpuInputs; // edge case (keyed by input, not output); inputs from the partition work
    /** @type {Map<DataKey, CPUTensor>} */
    cpuOutputs; // outputs from nodes, keyed by output
    /** @type {Map<DataKey, GPUTensor>} */
    gpuOutputs; // outputs from nodes, keyed by output

    /** @type {SafeTensorCache} */
    safetensorCache;

    /** @type {import('./uiManager.js').UIManager} */
    uiManager;

    /**
     * @param {GPUDevice} device - The WebGPU device instance.
     * @param {SessionGraph} sessionGraph - The compiled and annotated session graph.
     * @param {import('./uiManager.js').UIManager} uiManager - The UI manager instance.
     */
    constructor(device, sessionGraph, uiManager) {
        if (!device || !sessionGraph) {
            throw new Error("SessionExecutor requires a GPUDevice and a SessionGraph.");
        }
        this.device = device;
        this.sessionGraph = sessionGraph;
        this.uiManager = uiManager;

        this.sessionStatus = new Map();
        this.buffers = new Map();
        this.readyQueue = [];
        this.sessionInDegree = new Map();

        this.cpuInputs = new Map();
        this.gpuInputs = new Map();
        this.cpuOutputs = new Map();
        this.gpuOutputs = new Map();

        this.safetensorCache = new SafeTensorCache();
    }

    /**
     * @param {CPUTensor} cpuTensor
     * @param {string} [label] - Optional label for the GPU buffer.
     * @returns {Promise<GPUTensor>}
     */
    async _cpuToGPU(cpuTensor, label) {
        const gpuBuffer = await this.device.createBuffer({
            label: label,
            size: cpuTensor.getSize(),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        const mappedRange = gpuBuffer.getMappedRange();
        const typedArray = new Float32Array(mappedRange);
        typedArray.set(cpuTensor.getTypedArray()); // Copies here
        gpuBuffer.unmap(); // Boundary here

        const gpuTensor = new GPUTensor({
            buffer: gpuBuffer,
            shape: cpuTensor.getShape(),
            dtype: cpuTensor.getType(),
        });

        return gpuTensor;
    }

    /**
     * Executes the entire SessionGraph.
     * @param {PartitionWork} partitionWork - Contains initial inputs.
     * @returns {Promise<Map<NodeName, Tensor>>} A map of final graph outputs (NodeName -> Tensor).
     */
    async execute(partitionWork) {
        console.log("SessionExecutor: Starting execution...");
        await this._initializeState(partitionWork);
        // Initial ready sessions are found within _initializeState

        // Main execution loop (simplified view)
        let runningTasks = 0;
        const maxConcurrency = navigator.hardwareConcurrency || 4; // Limit concurrent tasks somewhat


        const promises = [];
        const sessionTask = (session) => {
            return (async () => {
                // Wait on all dependency promises
                const dependencyPromises = [];
                for (const dependency of this.sessionGraph.getPredecessorSessions(session)) {
                    dependencyPromises.push(promises[dependency.index]);
                }
                console.log(`Session ${session.index} waiting on ${dependencyPromises.length} dependencies.`);
                await Promise.all(dependencyPromises);
                console.log(`Session ${session.index} ready to run!`);

                if (this.uiManager) {
                    this.uiManager.onSessionStart(session.id, session.index);
                }

                try {
                    if (session instanceof GPUSession) {
                        await this._executeGPUSession(session);
                    } else if (session instanceof CPUSession) {
                        await this._executeCPUSession(session);
                    } else {
                        throw new Error("Unknown session type encountered.");
                    }
                    if (this.uiManager) {
                        this.uiManager.onSessionEnd(session.id, session.index, true);
                    }
                } catch (error) {
                    console.error(`Error executing session ${session.index}:`, error);
                    if (this.uiManager) {
                        this.uiManager.onSessionEnd(session.id, session.index, false);
                    }
                    throw error;
                }

                console.log(`Session ${session.index} done!`);
            })();
        }

        for (const session of this.sessionGraph.sessions) {
            // promises.push(sessionTask(session));
            await sessionTask(session);
        }

        // await Promise.all(promises);

        return this._gatherFinalOutputs();

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
    async _initializeState(partitionWork) {
        console.group("Initialization");
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
                console.log(`Session ${sessions.indexOf(s)} added to initial ready queue (in-degree 0).`);
            }
        });

        // --- Store Initial Inputs --- 
        console.group("Loading Inputs");
        for (const assignment of partitionWork.inputs) {
            const dataKey = `${assignment.node}:${assignment.input}`;
            // Check if input is needed by CPUSession
            if (this.sessionGraph._nodeToSession.get(assignment.node) instanceof CPUSession) {
                this.cpuInputs.set(dataKey, assignment.tensor);
                console.log(`Stored initial input for ${dataKey}:`, assignment.tensor);
            } else {
                const gpuAssignment = await this._cpuToGPU(assignment.tensor, `input_${dataKey}`);
                this.gpuInputs.set(dataKey, gpuAssignment);
                console.log(`Stored initial input for ${dataKey}:`, gpuAssignment);
            }
            
        }
        console.groupEnd();
        console.groupEnd();
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

    /** 
     * @param {GPUSession} session 
     * @private 
     * @returns {Promise<void>} 
     */
    async _executeGPUSession(session) {
        pushErrorScopes(this.device, ALL_SCOPES);
        const start = performance.now();
        console.groupCollapsed(`Executing GPUSession ${session.index}`);
        console.log(`Session ID: ${session.id}, Session Index: ${session.index}`);

        if (!session.resourcePlan) {
            throw new Error(`GPUSession ${session.index} is missing its resourcePlan.`);
        }

        const commandEncoder = this.device.createCommandEncoder({ label: `Command Encoder for Session ${session.index}` });
        console.debug("Command encoder created:", commandEncoder);
        
        console.log("Executing kernels...");

        const computePassParams = { label: `ComputePass_Session_${session.index}` }
        const computePass = commandEncoder.beginComputePass(computePassParams);
        console.debug("Compute pass begin:", computePass);

        // We assume session.nodes are already topologically sorted *within* the session
        for (const [nodeIndexInSession, node] of session.nodes.entries()) {
            console.log(node);
            console.groupCollapsed(`Node ${node.name}`);
            console.log(`Node ID: ${node.id}, Node Index in Session: ${nodeIndexInSession}`);
            console.log(this.uiManager);

            if (this.uiManager) {
                this.uiManager.onNodeStart(session.id, session.index, node.id || node.name, nodeIndexInSession);
            }

            let success = false;
            try {
                /** @type {GPUKernel} */
                const rawKernel = await node.getGPUKernel();
                const kernelKey = rawKernel.key();
                const kernel = KernelCompiler.getKernel(kernelKey);
                console.log(`Kernel: ${kernel.name} (${kernelKey})`);

                if (!kernel) {
                    throw new Error(`Missing GPU kernel for node ${node.name} of type ${node.type}`);
                }

                const bindGroupEntries = [];
                const inputGPUTensors = new Map();
                const inputCPUTensors = new Map();
                for (const input of kernel.inputs) {
                    const inputKey = `${node.name}:${input.name}`;
                    let inputGPUTensor;
                    let inputCPUTensor;

                    // Determine the source of the input data
                    let sourceKey; // Key to look up in cpu/gpuOutputs or cpu/gpuInputs
                    let dataSource; // Map (cpuOutputs, gpuOutputs, cpuInputs, gpuInputs)
                    let isInputSource = false; // Is the source from partitionWork inputs?

                    if (!session.resourcePlan.inputOutputMappings.has(inputKey)) {
                        // Input comes directly from partitionWork.inputs
                        sourceKey = inputKey;
                        isInputSource = true;
                    } else {
                        // Input comes from the output of a previous node in the graph
                        sourceKey = session.resourcePlan.inputOutputMappings.get(inputKey);
                        isInputSource = false;
                    }

                    // Check CPU sources first if needed
                    if (input.cpu) {
                        dataSource = isInputSource ? this.cpuInputs : this.cpuOutputs;
                        if (dataSource.has(sourceKey)) {
                            inputCPUTensor = dataSource.get(sourceKey);
                        } else {
                            // Allow GPU tensor to provide CPU input if needed and CPU isn't available
                             console.warn(`CPU input ${inputKey} requested but not found directly. Will check GPU sources.`);
                            // Still need to check GPU sources below
                        }
                    }

                    // Check GPU sources (always needed for binding, might be needed for CPU fallback)
                    dataSource = isInputSource ? this.gpuInputs : this.gpuOutputs;
                    if (dataSource.has(sourceKey)) {
                        inputGPUTensor = dataSource.get(sourceKey);
                    } else {
                        // If GPU source not found, check if a corresponding CPU source exists and can be moved
                        const cpuDataSource = isInputSource ? this.cpuInputs : this.cpuOutputs;
                        if (cpuDataSource.has(sourceKey)) {
                            console.debug(`Moving ${isInputSource ? 'input' : 'output'} ${sourceKey} from CPU to GPU for input ${inputKey}`);
                            const cpuSourceTensor = cpuDataSource.get(sourceKey);
                            inputGPUTensor = await this._cpuToGPU(cpuSourceTensor, `moved_${sourceKey}_for_${inputKey}`);
                            // Store the moved tensor in the appropriate GPU map
                            if (isInputSource) {
                                this.gpuInputs.set(sourceKey, inputGPUTensor);
                            } else {
                                this.gpuOutputs.set(sourceKey, inputGPUTensor);
                            }
                            // If CPU input was also requested but not found, use the original CPU tensor
                            if (input.cpu && !inputCPUTensor) {
                                 console.debug(`Using original CPU tensor ${sourceKey} for CPU input ${inputKey} after GPU move.`);
                                inputCPUTensor = cpuSourceTensor;
                            }
                        } else if (!input.cpu || !inputCPUTensor) {
                             // If no GPU or CPU source found (and CPU input wasn't already satisfied)
                            throw new Error(`Missing required input/output source '${sourceKey}' for node ${node.name}, input ${input.name}`);
                        }
                    }
                    
                    // Final check if CPU input was requested but still missing
                    if (input.cpu && !inputCPUTensor) {
                        throw new Error(`Could not satisfy CPU input requirement for ${inputKey}`);
                    }

                    // Store the resolved tensors
                    if (inputGPUTensor) { inputGPUTensors.set(input.name, inputGPUTensor); }
                    if (inputCPUTensor) { inputCPUTensors.set(input.name, inputCPUTensor); }

                    // Add GPU tensor buffer to bind group entries if it exists AND binding is defined
                    if (input.binding && inputGPUTensor) {
                        bindGroupEntries.push({
                            binding: input.binding.index,
                            resource: { buffer: inputGPUTensor.getBuffer() }
                        });
                    } else if (input.binding && !inputGPUTensor) {
                        // This case should ideally be caught by earlier checks if a binding exists but no GPU tensor could be resolved.
                        throw new Error(`Input ${inputKey} has a binding defined but no GPUTensor was resolved for it.`);
                    }
                }

                // Create ExecutionContext for kernel calls
                const executionContext = new ExecutionContext(inputCPUTensors, inputGPUTensors);

                for (const output of kernel.outputs) {
                    const outputKey = `${node.name}:${output.name}`;
                    if (this.gpuOutputs.has(outputKey) || this.cpuOutputs.has(outputKey)) {
                        throw new Error(`Output ${outputKey} is registered. This should not happen.`);
                    }

                    // Create a new buffer for the output tensor
                    const outputShape = node.getOutputShape(executionContext);
                    if (!outputShape) {
                        throw new Error(`Missing output shape for ${outputKey}`);
                    }

                    // Calculate buffer size based on shape and data type
                    const elementSize = 4; // Assuming float32 (4 bytes)
                    const numElements = outputShape.reduce((a, b) => a * b, 1);
                    const bufferSize = numElements * elementSize;

                    // Create the GPU buffer
                    const buffer = this.device.createBuffer({
                        label: `${node.name}.${output.name}`,
                        size: bufferSize,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                    });

                    // Create a GPUTensor with the buffer
                    const outputTensor = new GPUTensor({
                        buffer: buffer,
                        shape: outputShape,
                        dtype: 'float32' // TODO: Real dtype
                    });
                    this.gpuOutputs.set(outputKey, outputTensor);

                    console.log(`Created output buffer ${outputKey} with shape [${outputShape}]`);

                    bindGroupEntries.push({
                        binding: output.binding.index,
                        resource: { buffer: outputTensor.getBuffer() }
                    });
                }

                if (kernel.dimensionBuffer) {
                    // Create uniform buffer for dimensions
                    const dimensionData = kernel.dimensionBuffer.func(executionContext);
                    console.debug("Dimension data:", dimensionData);
                    const dimensionBuffer = this.device.createBuffer({
                        label: `${node.name}.dimensions`,
                        size: dimensionData.byteLength,
                        usage: GPUBufferUsage.UNIFORM,
                        mappedAtCreation: true,
                    });

                    const mappedRange = dimensionBuffer.getMappedRange();
                    const typedArray = new Uint32Array(mappedRange);
                    typedArray.set(dimensionData);
                    dimensionBuffer.unmap();

                    bindGroupEntries.push({
                        binding: kernel.dimensionBuffer.index,
                        resource: { buffer: dimensionBuffer }
                    });
                }

                console.debug(`Bind group entries:`, bindGroupEntries);
                for(const binding of bindGroupEntries) {
                    console.debug(`Binding ${binding.binding}: `, binding.resource.buffer);
                }

                // --- b. Create Bind Group ---
                // Requires kernel.bindGroupLayout to be prepared by compiler
                if (!kernel.bindGroupLayout) {
                    throw new Error(`Kernel ${kernel.name} is missing its bindGroupLayout.`);
                }
                const bindGroup = this.device.createBindGroup({
                    label: `${node.name}.bindgroup`,
                    layout: kernel.bindGroupLayout,
                    entries: bindGroupEntries,
                });

                // --- c. Set Pipeline & Bind Group ---
                if (!kernel.pipeline) {
                    throw new Error(`Kernel ${kernel.name} is missing its pipeline.`);
                }
                computePass.setPipeline(kernel.pipeline);
                console.debug("Pipeline set:", kernel.pipeline);
                computePass.setBindGroup(0, bindGroup); // Assuming bind group index 0
                console.debug("Bindgroup set:", bindGroup)


                // --- d. Dispatch Kernel ---
                const dispatch = kernel.workgroupFunction(executionContext);
                const dispatchX = dispatch.x;
                const dispatchY = dispatch.y;
                const dispatchZ = dispatch.z;
                console.log(`Workgroups: (${dispatchX}, ${dispatchY}, ${dispatchZ})`);
                computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
                console.debug("Command submitted: dispatchWorkgroups(x,y,z)", dispatchX, dispatchY, dispatchZ);
                console.groupEnd();

                success = true;
            } catch (e) {
                console.error(`Error during GPU kernel execution for node ${node.name} in session ${session.index}:`, e);
                success = false;
                throw e; // Re-throw to be caught by session-level error handling
            } finally {
                if (this.uiManager) {
                    this.uiManager.onNodeEnd(session.id, session.index, node.id || node.name, nodeIndexInSession, success);
                }
                console.groupEnd(); // End Node group
            }
        }
        computePass.end();

        console.group("Preparing Readbacks");
        const readbackOperations = [];
        for (const outputKey of session.resourcePlan.readback) {
            const gpuTensor = this.gpuOutputs.get(outputKey);
            if (!gpuTensor) {
                console.warn(`Readback requested for ${outputKey}, but GPUTensor not found in gpuOutputs. Skipping readback for this key.`);
                continue;
            }

            const gpuBuffer = gpuTensor.getBuffer();
            if (!gpuBuffer) {
                console.warn(`GPUTensor for ${outputKey} does not have a valid buffer. Skipping readback.`);
                continue;
            }

            const stagingBuffer = this.device.createBuffer({
                label: `staging_readback_${outputKey}`,
                size: gpuBuffer.size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });

            commandEncoder.copyBufferToBuffer(
                gpuBuffer,             // source
                0,                     // sourceOffset
                stagingBuffer,         // destination
                0,                     // destinationOffset
                gpuBuffer.size         // size
            );
            console.debug("Command submitted: copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size)", gpuBuffer, 0, stagingBuffer, 0, gpuBuffer.size);

            readbackOperations.push({
                outputKey,
                stagingBuffer,
                shape: gpuTensor.getShape(),
                dtype: gpuTensor.getType(),
            });
            console.log(`Readback requested for ${outputKey}. Staging buffer:`, stagingBuffer);
        }
        console.groupEnd();

        const commandBuffer = commandEncoder.finish();
        console.debug("Command buffer:", commandBuffer);

        popErrorScopes(this.device, ALL_SCOPES);

        const prepEnd = performance.now();
        this.device.queue.submit([commandBuffer]);
        console.log(`GPU commands (kernels + readback copies) submitted.`);
        
        // Wait for all submitted GPU work to complete
        await this.device.queue.onSubmittedWorkDone();
        const workEnd = performance.now();
        console.log(`GPU work completed.`);

        // Process actual readbacks now that GPU is done
        console.group("Processing Readbacks");
        for (const op of readbackOperations) {
            console.log(`Processing mapped readback for ${op.outputKey}`);
            try {
                await op.stagingBuffer.mapAsync(GPUMapMode.READ);
                const arrayBufferContent = op.stagingBuffer.getMappedRange();
                
                // Data must be copied out of the mapped range before unmap is called.
                const dataCopy = arrayBufferContent.slice(0);

                op.stagingBuffer.unmap();
                op.stagingBuffer.destroy(); // Clean up the staging buffer

                const cpuTensor = new CPUTensor({
                    data: dataCopy,
                    shape: op.shape,
                    dtype: op.dtype,
                });

                this.cpuOutputs.set(op.outputKey, cpuTensor);
                console.log(`Readback complete for ${op.outputKey}. CPUTensor stored.`);
                console.debug("Readback tensor: ", cpuTensor);
            } catch (error) {
                console.error(`Error during readback for ${op.outputKey}:`, error);
                // Ensure buffer is destroyed even if an error occurs during mapping/reading
                if (op.stagingBuffer) {
                    // If mapAsync failed, it might not be mapped, unmap might throw.
                    // Destroy should be safe.
                    try { op.stagingBuffer.unmap(); } catch (e) { /* ignore */ }
                    op.stagingBuffer.destroy();
                }
            }
        }
        console.groupEnd(); // End of console.group for readbacks

        const end = performance.now();
        console.log(`Prep done in ${prepEnd - start}ms, work done in ${workEnd - prepEnd}ms, cleanup in ${end - workEnd}ms, total in ${end - start}ms.`);
        console.groupEnd(); // End of console.group for GPUSession execution
        
    }
    /** 
     * @param {CPUSession} session 
     * @private 
     * @returns {Promise<void>} 
     */
    async _executeCPUSession(session) {
        pushErrorScopes(this.device, ALL_SCOPES);
        console.groupCollapsed(`Executing CPUSession ${session.index}`);
        console.log(`Session ID: ${session.id}, Session Index: ${session.index}`);

        for (const [nodeIndexInSession, node] of session.nodes.entries()) {
            console.groupCollapsed(`Node ${node.name}`);
            console.log(`Node ID: ${node.id}, Node Index in Session: ${nodeIndexInSession}`);

            if (this.uiManager) {
                this.uiManager.onNodeStart(session.id, session.index, node.id || node.name, nodeIndexInSession);
            }
            let success = false;
            try {
                const kernel = await node.getCPUKernel(); // Ensure kernel is awaited if getKernel is async
                console.log(`Executing CPU node ${node.name}, kernel:`, kernel);

                const inputCPUTensors = new Map();
                for(const inputDef of kernel.inputs) { // Assuming kernel.inputs is array of names or objects with name
                    const inputName = typeof inputDef === 'string' ? inputDef : inputDef.name;
                    const inputKey = `${node.name}:${inputName}`;

                    let inputTensor;
                    if(!session.resourcePlan.inputOutputMappings.has(inputKey)) {
                        // Check partitionWork inputs
                        if(this.cpuInputs.has(inputKey)) {
                            inputTensor = this.cpuInputs.get(inputKey);
                        } else {
                            // Try GPU inputs (shouldn't happen for CPUSession unless explicitly designed)
                            if (this.gpuInputs.has(inputKey)) {
                                 console.warn(`CPUSession node ${node.name} using GPU input ${inputKey}. Ensure this is intended.`);
                                 // TODO: Need GPU->CPU transfer mechanism if this is a valid path
                                 throw new Error(`Automatic GPU->CPU transfer for CPUSession input ${inputKey} not implemented.`);
                            }
                            throw new Error(`Missing input ${inputKey} in both cpuInputs and gpuInputs`);
                        }
                    } else {
                        const sourceOutputKey = session.resourcePlan.inputOutputMappings.get(inputKey);
                        // Check previous node outputs
                        if(this.cpuOutputs.has(sourceOutputKey)) {
                            inputTensor = this.cpuOutputs.get(sourceOutputKey);
                        } else {
                             // Try GPU outputs (needs transfer)
                             if (this.gpuOutputs.has(sourceOutputKey)) {
                                console.warn(`CPUSession node ${node.name} using GPU output ${sourceOutputKey} as input ${inputKey}. Ensure this is intended.`);
                                // TODO: Need GPU->CPU transfer mechanism
                                throw new Error(`Automatic GPU->CPU transfer for CPUSession input ${inputKey} from output ${sourceOutputKey} not implemented.`);
                             }
                            throw new Error(`Missing output ${sourceOutputKey} (backing input ${inputKey}) in both cpuOutputs and gpuOutputs`);
                        }
                    }
                    inputCPUTensors.set(inputName, inputTensor);
                }

                // Create ExecutionContext for CPU kernel
                const executionContext = new ExecutionContext(inputCPUTensors, new Map(), this.safetensorCache); // No GPU inputs expected

                console.log(`Invoking kernel ${kernel.name} for node ${node.name} with context:`, executionContext);
                const outputs = await kernel.execute(executionContext); // Pass ExecutionContext
                console.log(`Kernel ${kernel.name} produced:`, outputs);

                for(const outputDef of kernel.outputs) { // Assuming kernel.outputs is array of names or objects with name
                    const outputName = typeof outputDef === 'string' ? outputDef : outputDef.name;
                    const outputKey = `${node.name}:${outputName}`;
    
                    let outputTensor = outputs[outputName]; // Access output from the result object
                    
                    if (!outputTensor) {
                        console.warn(`Kernel ${kernel.name} did not produce expected output '${outputName}'.`);
                        // Or throw new Error(...)? Depends on strictness required.
                        continue; 
                    }
    
                    if (this.cpuOutputs.has(outputKey) || this.gpuOutputs.has(outputKey)) {
                         console.warn(`Overwriting existing output key ${outputKey} during CPUSession execution.`);
                         // Potentially throw error if overwrite is not allowed
                    }
    
                    if (outputTensor instanceof CPUTensor) {
                        this.cpuOutputs.set(outputKey, outputTensor);
                    } else if (outputTensor instanceof GPUTensor) {
                         console.warn(`CPU Kernel ${kernel.name} produced a GPUTensor for output ${outputName}. Ensure this is intended.`);
                         // TODO: Need CPU->GPU transfer? Or is this an error?
                         this.gpuOutputs.set(outputKey, outputTensor);
                    } else {
                        throw new Error(`Kernel ${kernel.name} produced invalid output type for ${outputName}: ${typeof outputTensor}`);
                    }
                }
                // Annotate session.resourcePlan usage here later
                await new Promise(r => setTimeout(r, 10));
            } catch (e) {
                console.error(`Error during CPU kernel execution for node ${node.name} in session ${session.index}:`, e);
                success = false;
                throw e; // Re-throw to be caught by session-level error handling
            } finally {
                if (this.uiManager) {
                    this.uiManager.onNodeEnd(session.id, session.index, node.id || node.name, nodeIndexInSession, success);
                }
                console.groupEnd(); // End Node group
            }
        }
        // Annotate session.resourcePlan usage here later
        await new Promise(r => setTimeout(r, 10));
        console.groupEnd();
        popErrorScopes(this.device, ALL_SCOPES);
    }
    /** 
     * @private
     * @returns {Map<NodeName, Map<string, CPUTensor>>} 
     */
    _gatherFinalOutputs() {
        const finalOutputs = this.sessionGraph._finalOutputs;

        const m = new Map();
        for(const [outputNode, outputs] of finalOutputs.entries()) {
            m.set(outputNode, new Map());
            for(const output of outputs) {
                const outputKey = `${outputNode}:${output}`;
                m.get(outputNode).set(output, this.cpuOutputs.get(outputKey));
            }
        }

        return m;
    }
} 