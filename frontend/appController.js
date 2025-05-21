// frontend/appController.js
import { Coordinator, PartitionWorkResult, OutputAssignment, SingleStepChunk } from "./worker.js"; // Assuming worker.js path
import { KernelCompiler } from "./compiler.js"; // Assuming compiler.js path
import { SessionExecutor } from "./executor.js"; // Assuming executor.js path

export class AppController {
    device;
    uiManager;
    coordinator;
    compiler;
    executor;

    constructor(webGPUDevice, uiManagerInstance) {
        this.device = webGPUDevice;
        this.uiManager = uiManagerInstance;

        this.coordinator = new Coordinator({ url: "" }); // Configure URL if needed
        this.compiler = new KernelCompiler(this.device);
    }

    async runMainWorkflow() {
        this.uiManager.clearError();
        try {
            console.log("AppController: Registering with coordinator...");
            const registration = await this.coordinator.register();
            console.log("AppController: Registered for partition:", registration.partition);
            this.uiManager.displayCurrentPartition(registration.partition);

            while (true) {
                console.log("AppController: Getting work for partition:", registration.partition);
                const work = await this.coordinator.get_work(registration.partition);
                if (!work) {
                    console.log("AppController: No work available for partition:", registration.partition);
                    this.uiManager.displayError("No work available from the coordinator."); // Inform user
                    await new Promise(resolve => setTimeout(resolve, 1000)); // Add a 1 second delay
                    continue;
                }
                console.log("AppController: Received work with correlation ID:", work.correlation_id, work);

                console.log("AppController: Starting compilation...");
                const sessionGraph = await this.compiler.compile(work); // work should contain the graph definition
                console.log("AppController: Compilation complete.", sessionGraph);

                if (!sessionGraph || !sessionGraph.sessions || sessionGraph.sessions.length === 0) {
                    console.error("AppController: Compilation resulted in an empty or invalid session graph.");
                    this.uiManager.displayError("Failed to compile a valid execution graph.");
                    // Potentially submit an empty/error result back to coordinator if required by protocol
                    return;
                }

                this.uiManager.renderSessionGraph(sessionGraph);

                console.log("AppController: Starting execution...");
                this.executor = new SessionExecutor(this.device, sessionGraph, this.uiManager);
                const finalOutputs = await this.executor.execute(work); // Pass work for initial inputs
                console.log("AppController: Execution complete. Final outputs:", finalOutputs);

                // Collect and submit outputs
                let outputAssignments = [];
                if (finalOutputs && finalOutputs.size > 0) {
                    for (const [nodeName, outputsMap] of finalOutputs.entries()) {
                        for (const [outputName, tensor] of outputsMap.entries()) {
                            // Ensure tensor is serializable/CPUTensor for OutputAssignment
                            // This might require a conversion from GPUTensor if not handled by executor._gatherFinalOutputs
                            outputAssignments.push(new OutputAssignment({
                                node: nodeName,
                                output: outputName,
                                tensor: tensor // This tensor needs to be in the CPUTensor/serializable format.
                            }));
                        }
                    }
                } else {
                    console.warn("AppController: No final outputs were gathered by the executor or finalOutputs is empty.");
                }
                
                console.log("AppController: Checking work...")
                
                console.log("AppController: Submitting work results...");
                const CHUNK_SIZE = 8;
                const totalAssignments = outputAssignments.length;
                for (let i = 0; i < totalAssignments; i += CHUNK_SIZE) {
                    const chunk = outputAssignments.slice(i, i + CHUNK_SIZE);
                    console.log(`AppController: Checking work chunk ${Math.floor(i / CHUNK_SIZE) + 1}/${Math.ceil(totalAssignments / CHUNK_SIZE)}`);
                    await this.coordinator.check_work(new SingleStepChunk({
                        partition: work.partition,
                        correlation_id: work.correlation_id,
                        outputs: chunk,
                        last_chunk: i + CHUNK_SIZE >= totalAssignments 
                    }));
                }

                // await this.coordinator.submit_work(new PartitionWorkResult({
                //     partition: work.partition,
                //     correlation_id: work.correlation_id,
                //     outputs: outputAssignments,
                // }));
                console.log("AppController: Work results submitted successfully.");
                // Optionally display a success message via UIManager
            }
        } catch (error) {
            console.error("AppController: An error occurred in the main workflow:", error);
            this.uiManager.displayError(error.message || "An unknown error occurred in the application workflow.");
            // Depending on the error, may need to inform the coordinator
        }
    }
} 