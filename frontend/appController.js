// frontend/appController.js
import { Coordinator, PartitionWorkResult, OutputAssignment, SingleStepChunk, DEFAULT_NODE_OUTPUT } from "./worker.js"; // Assuming worker.js path
import { KernelCompiler } from "./compiler.js"; // Assuming compiler.js path
import { SessionExecutor } from "./executor.js"; // Assuming executor.js path
import { SafeTensorCache, OutputCache } from "./tensorcache.js";
import { Profiler } from "./utils/profiler.js";
import { CPUTensor } from "./kernel.js";

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

    async runRunChatWorkflow() {
        while(true) {
            const cid = this.uiManager.chatCid;
            if(cid != null) {
                const response = await fetch(`/output/${cid}`);
                const data = await response.json();
                this.uiManager.updateChatOutput(data.decoded_text);
                this.uiManager.setStatus(data.status);
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }

    async runMainWorkflow() {
        this.uiManager.clearError();
        try {
            console.log("AppController: Getting partition from local storage...");
            const sessionId = localStorage.getItem('sessionId');
            const partition = localStorage.getItem('partition');
            if(partition) {
                await this.coordinator.revived(partition, sessionId);
            } else {
                console.log("We do not have a valid partition to work with")
            }

            console.log("AppController: Registering with coordinator...");
            const registration = await this.coordinator.register();
            console.log("AppController: Registered for partition:", registration.partition);
            this.uiManager.displayCurrentPartition(registration.partition);

            // Store partition & current session in local storage
            localStorage.setItem('partition', registration.partition);
            localStorage.setItem('sessionId', registration.sessionId);

            const outputCache = new OutputCache();
            const profiler = new Profiler();

            console.log("AppController: Prefilling cache...");
            const prefillResponse = await fetch(`/prefill/${registration.partition}`);
            const prefillData = await prefillResponse.json();

            const promises = [];
            let done = 0;
            let total = prefillData.safetensors.length;
            for(const safetensor of prefillData.safetensors) {
                const p = (async () => {
                    const response = await fetch(`/safetensor/${btoa(safetensor.model_name)}/${safetensor.tensor_name}`);
                    if (!response.ok) {
                        throw new Error(`Failed to fetch tensor: ${response.status} ${response.statusText}`);
                    }
                    const buffer = await response.arrayBuffer();
                    const view = new DataView(buffer);
                    const [tensor] = CPUTensor.decode(view, 0); 
                    done++;
                    this.uiManager.displayError(`Prefilling cache... ${done}/${total}`);
                    return {
                        node_name: safetensor.node_name,
                        output_name: DEFAULT_NODE_OUTPUT,
                        tensor: tensor
                    };
                })();
                promises.push(p);
            }
            this.uiManager.displayError("Prefilling cache...");
            const prefillResults = await Promise.all(promises);
            for(const result of prefillResults) {
                outputCache.put(`${result.node_name}:${result.output_name}`, result.tensor);
            }
            console.log(`AppController: Cache prefilled with ${prefillResults.length} tensors.`);

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
                const executor = new SessionExecutor(this.device, sessionGraph, this.uiManager, null, profiler, work.shouldTrace, outputCache);
                const { finalOutputs, trace } = await executor.execute(work); // Pass work for initial inputs
                console.log("AppController: Execution complete. Final outputs:", finalOutputs);

                // Collect and submit outputs
                let outputAssignments = [];
                if (finalOutputs && finalOutputs.size > 0) {
                    for (const [nodeName, outputsMap] of finalOutputs.entries()) {
                        //if (nodeName.includes("embeds_matrix")) continue;
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
               
                if(work.shouldTrace) {
                    console.log("AppController: Checking work...")
                    const chunks = trace.getChunks(1024 * 200);
                    for(let i = 0; i < chunks.length; i++) {
                        console.log("AppController: Checking work chunk", chunks[i]);
                        await this.coordinator.check_work(new SingleStepChunk({
                            partition: work.partition,
                            correlation_id: work.correlation_id,
                            outputs: chunks[i],
                            last_chunk: i == chunks.length - 1
                        }));
                    }
                    console.log("AppController: Submitting work results...");
                }

                await this.coordinator.submit_work(new PartitionWorkResult({
                    partition: work.partition,
                    correlation_id: work.correlation_id,
                    outputs: outputAssignments,
                }));
                console.log("AppController: Work results submitted successfully.");
                // const nextToken = submitResponse.next_token;
                // const decodedText = submitResponse.decoded_text;
                // this.tokens.push(nextToken);
                // console.log(submitResponse);
                // // Display the decoded text
                // console.log("Decoded text:", decodedText);
                // if (decodedText && decodedText.trim()) {
                //     this.decodedTokens += decodedText;
                // }
                if (profiler) {
                    this.uiManager.displayProfiling(profiler);
                    profiler.clear();
                }
            }
        } catch (error) {
            console.error("AppController: An error occurred in the main workflow:", error);
            this.uiManager.displayError(error.message || "An unknown error occurred in the application workflow.");
            // Depending on the error, may need to inform the coordinator
        }
    }
} 