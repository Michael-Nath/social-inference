// executor.js
import { SessionGraph, GPUSession, CPUSession, ComputeSession, KernelCompiler } from './compiler.js';
import { PartitionWork } from './worker.js';
import { GPUTensor, CPUTensor } from './kernel.js';
import { ALL_SCOPES, popErrorScopes, pushErrorScopes } from './common.js';

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

        this.cpuInputs = new Map();
        this.gpuInputs = new Map();
        this.cpuOutputs = new Map();
        this.gpuOutputs = new Map();
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
        typedArray.set(cpuTensor.getData()); // Copies here
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

                try {
                    if (session instanceof GPUSession) {
                        await this._executeGPUSession(session);
                    } else if (session instanceof CPUSession) {
                        await this._executeCPUSession(session);
                    } else {
                        throw new Error("Unknown session type encountered.");
                    }
                } catch (error) {
                    console.error(`Error executing session ${session.index}:`, error);
                    throw error;
                }

                console.log(`Session ${session.index} done!`);
            })();
        }

        for (const session of this.sessionGraph.sessions) {
            promises.push(sessionTask(session));
        }

        await Promise.all(promises);

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
        const sessionIndex = this.sessionGraph.sessions.indexOf(session);
        const start = performance.now();
        console.group(`Executing GPUSession ${sessionIndex}...`);
        if (!session.resourcePlan) {
            throw new Error(`GPUSession ${sessionIndex} is missing its resourcePlan.`);
        }

        pushErrorScopes(this.device, ALL_SCOPES)

        const commandEncoder = this.device.createCommandEncoder({ label: `CommandEncoder_Session_${sessionIndex}` });
        console.debug("Command encoder created:", commandEncoder);
        
        console.log("Executing kernels...");

        const computePassParams = { label: `ComputePass_Session_${sessionIndex}` }
        const computePass = commandEncoder.beginComputePass(computePassParams);
        console.debug("Compute pass begin:", computePass);

        // We assume session.nodes are already topologically sorted *within* the session
        for (const node of session.nodes) {
            console.group(`Node: ${node.name} (${node.type})`);

            /** @type {GPUKernel} */
            const rawKernel = await node.getKernel();
            const kernelKey = rawKernel.key();
            const kernel = KernelCompiler.getKernel(kernelKey);
            console.log(`Kernel: ${kernel.name} (${kernelKey})`);

            if (!kernel) {
                throw new Error(`Missing GPU kernel for node ${node.name} of type ${node.type}`);
            }

            const bindGroupEntries = [];
            for (const inputBinding of kernel.inputBindings) {
                const inputKey = `${node.name}:${inputBinding.name}`;
                let inputTensor;
                if (!session.resourcePlan.inputOutputMappings.has(inputKey)) {
                    // Check inputs
                    if (this.gpuInputs.has(inputKey)) {
                        // Use that
                        inputTensor = this.gpuInputs.get(inputKey);
                    } else if (this.cpuInputs.has(inputKey)) {
                        // Do move
                        console.debug("Moving CPU input to GPU:", inputKey);
                        inputTensor = await this._cpuToGPU(this.cpuInputs.get(inputKey), `input_${inputKey}`);
                        this.gpuInputs.set(inputKey, inputTensor);
                    } else {
                        // Error
                        throw new Error(`Missing input for binding ${inputBinding.name} on node ${node.name}`);
                    }
                } else {
                    // Use output
                    const outputKey = session.resourcePlan.inputOutputMappings.get(inputKey);
                    if (this.gpuOutputs.has(outputKey)) {
                        // Use that
                        inputTensor = this.gpuOutputs.get(outputKey);
                    } else if (this.cpuOutputs.has(outputKey)) {
                        // Do move
                        console.debug("Moving CPU output to GPU:", outputKey);
                        inputTensor = await this._cpuToGPU(this.cpuOutputs.get(outputKey), `input_from_output_${outputKey}`);
                        this.gpuOutputs.set(outputKey, inputTensor);
                    } else {
                        // Error
                        throw new Error(`Missing output for binding ${inputBinding.name} on node ${node.name}`);
                    }
                }

                bindGroupEntries.push({
                    binding: inputBinding.index,
                    resource: { buffer: inputTensor.getBuffer() }
                });
            }

            for (const outputBinding of kernel.outputBindings) {
                const outputKey = `${node.name}:${outputBinding.name}`;
                if (this.gpuOutputs.has(outputKey) || this.cpuOutputs.has(outputKey)) {
                    throw new Error(`Output ${outputKey} is registered. This should not happen.`);
                }

                // Create a new buffer for the output tensor
                const outputShape = node.computedOutputShapes.get(outputBinding.name);
                if (!outputShape) {
                    throw new Error(`Missing output shape for ${outputKey}`);
                }

                // Calculate buffer size based on shape and data type
                const elementSize = 4; // Assuming float32 (4 bytes)
                const numElements = outputShape.reduce((a, b) => a * b, 1);
                const bufferSize = numElements * elementSize;

                // Create the GPU buffer
                const buffer = this.device.createBuffer({
                    label: `${node.name}.${outputBinding.name}`,
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
                    binding: outputBinding.index,
                    resource: { buffer: outputTensor.getBuffer() }
                });
            }

            if (kernel.dimensionBuffer) {
                // Create uniform buffer for dimensions
                const dimensionData = kernel.dimensionBuffer.func(node.computedInputShapes, node.computedOutputShapes);
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
            const dispatch = kernel.workgroupFunction(node.computedInputShapes, node.computedOutputShapes);
            const dispatchX = dispatch.x;
            const dispatchY = dispatch.y;
            const dispatchZ = dispatch.z;
            console.log(`Workgroups: (${dispatchX}, ${dispatchY}, ${dispatchZ})`);
            computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
            console.debug("Command submitted: dispatchWorkgroups(x,y,z)", dispatchX, dispatchY, dispatchZ);
            console.groupEnd();
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

                let typedData;
                switch (op.dtype) {
                    case 'float32':
                        typedData = new Float32Array(dataCopy);
                        break;
                    case 'int32':
                        typedData = new Int32Array(dataCopy);
                        break;
                    // TODO: Add other data types as supported by CPUTensor and your kernels
                    default:
                        console.error(`Unsupported dtype for CPUTensor readback: ${op.dtype} for key ${op.outputKey}`);
                        // Skip creating CPUTensor for unsupported type
                        continue; 
                }
                
                op.stagingBuffer.unmap();
                op.stagingBuffer.destroy(); // Clean up the staging buffer

                const cpuTensor = new CPUTensor({
                    data: typedData,
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
        // Assume in topo order
        for(const node of session.nodes) {
            const kernel = node.getKernel();
            console.log("Executing CPU kernel:", kernel);

            const inputs = {};
            for(const input of kernel.inputs) {
                const inputKey = `${node.name}:${input}`;

                let inputTensor;
                if(!session.resourcePlan.inputOutputMappings.has(inputKey)) {
                    // Check inputs
                    if(this.cpuInputs.has(inputKey)) {
                        inputTensor = this.cpuInputs.get(inputKey);
                    } else {
                        throw new Error(`Missing input ${inputKey}`);
                    }
                } else {
                    const outputKey = session.resourcePlan.inputOutputMappings.get(inputKey);
                    // Use output
                    if(this.cpuOutputs.has(outputKey)) {
                        // Use that
                        inputTensor = this.cpuOutputs.get(outputKey);
                    } else {
                        throw new Error(`Missing output ${outputKey} that backs input ${inputKey}`);
                    }
                }
                inputs[input] = inputTensor;
            }

            console.log("Invoking kernel with:", inputs);
            const outputs = await kernel.execute(inputs);
            console.log("Kernel produced:", outputs);

            for(const output of kernel.outputs) {
                const outputKey = `${node.name}:${output}`

                let outputTensor;
                if(typeof outputs === Map) {
                    outputTensor = outputs.get(output);
                } else {
                    outputTensor = outputs[output];
                }
                this.cpuOutputs.set(outputKey, outputTensor);
            }
        }
        // Annotate session.resourcePlan usage here later
        await new Promise(r => setTimeout(r, 10));
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