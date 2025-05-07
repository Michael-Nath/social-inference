import { Coordinator, PartitionWork, PartitionWorkResult, OutputAssignment } from "./worker.js";
import { KernelCompiler } from "./compiler.js";
import { initializeWebGPU } from "./common.js"; // Import the initializer
import { SessionExecutor } from "./executor.js"; // Import the executor

// Session-related classes (ComputeSession, CPUSession, GPUSession, SessionGraph) 
// and KernelCompiler are now defined in compiler.js

const coordinator = new Coordinator({
  url: "", // Assuming coordinator runs locally or URL is configured elsewhere
});

async function main() {
  try {
    // --- Initialize WebGPU Device --- 
    const device = await initializeWebGPU();
    if (!device) {
        console.error("WebGPU not supported or initialization failed.");
        return; // Stop execution if no device
    }
    console.log("WebGPU Device Initialized:", device);
    // ----------------------------------

    const registration = await coordinator.register();
    console.log("Registered for partition:", registration.partition);
    
    const work = await coordinator.get_work(registration.partition);
    if (!work) {
      console.log("No work available for partition:", registration.partition);
      return;
    }
    console.log("Received work with correlation ID:", work.correlation_id);
    console.log(work);
    
    // 1. Compile
    console.log("--- Starting Compilation ---");
    const compiler = new KernelCompiler(device);
    const sessionGraph = await compiler.compile(work);
    console.log("--- Compilation Complete ---");

    // 2. Execute
    console.log("--- Starting Execution ---");
    const executor = new SessionExecutor(device, sessionGraph);
    const finalOutputs = await executor.execute(work); // Pass work for initial inputs
    console.log("--- Execution Complete ---");
    console.log("Final Outputs:", finalOutputs); // Log the final results

    // Collect outputs
    let o = []
    for(const [outputNode, outputs] of finalOutputs.entries()) {
      for(const [output, tensor] of outputs.entries()) {
        o.push(new OutputAssignment({
          node: outputNode,
          output: output,
          tensor: tensor
        }));
      }
    }

    await coordinator.submit_work(new PartitionWorkResult({
      partition: work.partition,
      correlation_id: work.correlation_id,
      outputs: o 
    }))

  } catch (error) {
    console.error("An error occurred during main execution:", error);
  }
}

main();