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

    // --- Update HTML with dynamic info ---
    document.getElementById('current-partition').textContent = registration.partition;

    const sessionsContainer = document.getElementById('sessions-container');
    sessionsContainer.innerHTML = ''; // Clear previous content

    let sessionsToDisplay = [];
    let graphTitleForConsole = "Detected sessionGraph structure";

    if (sessionGraph) {
        if (sessionGraph.sessions && Array.isArray(sessionGraph.sessions)) {
            sessionsToDisplay = sessionGraph.sessions;
            graphTitleForConsole = "sessionGraph.sessions (Array)";
        } else if (Array.isArray(sessionGraph) && sessionGraph.every(s => s && typeof s === 'object' && s.nodes)) {
            sessionsToDisplay = sessionGraph;
            graphTitleForConsole = "sessionGraph (Array of sessions)";
        } else if (typeof sessionGraph === 'object' && !Array.isArray(sessionGraph) && Object.values(sessionGraph).every(s => s && typeof s === 'object' && s.nodes && typeof s.id !== 'undefined')) {
            sessionsToDisplay = Object.values(sessionGraph);
            graphTitleForConsole = "sessionGraph (Object Map of sessions)";
        } else if (sessionGraph.nodes && Array.isArray(sessionGraph.nodes)) {
            sessionsToDisplay = [sessionGraph]; // Treat as a single session
            if (!sessionGraph.id) sessionsToDisplay[0].id = "default_session"; // Assign a default ID if missing
            graphTitleForConsole = "sessionGraph (Single session object with .nodes)";
        }
    }

    console.log(graphTitleForConsole + " used for display:", sessionsToDisplay);

    if (sessionsToDisplay.length > 0) {
        sessionsToDisplay.forEach(session => {
            const sessionDiv = document.createElement('div');
            sessionDiv.className = 'session';
            sessionDiv.style.border = '1px solid #ccc';
            sessionDiv.style.marginBottom = '10px';
            sessionDiv.style.padding = '10px';

            const sessionTitle = document.createElement('h3');
            sessionTitle.textContent = `Session ID: ${session.id || 'Unknown Session'}`;
            sessionTitle.style.marginTop = '0';
            sessionDiv.appendChild(sessionTitle);

            if (session.nodes && Array.isArray(session.nodes) && session.nodes.length > 0) {
                const nodesList = document.createElement('ul');
                nodesList.style.paddingLeft = '20px';
                session.nodes.forEach(node => {
                    const nodeItem = document.createElement('li');
                    nodeItem.textContent = `Node: ${node.name || node.id || 'Unknown Node'}`;

                    if (node.dependencies && Array.isArray(node.dependencies) && node.dependencies.length > 0) {
                        const dependenciesList = document.createElement('ul');
                        dependenciesList.style.listStyleType = 'circle';
                        dependenciesList.style.marginLeft = '20px';
                        node.dependencies.forEach(dep => {
                            const dependencyItem = document.createElement('li');
                            let depText = 'Unknown Dependency';
                            if (typeof dep === 'string') {
                                depText = dep;
                            } else if (dep && typeof dep === 'object') {
                                depText = dep.node_id || dep.id || JSON.stringify(dep);
                            }
                            dependencyItem.textContent = `Depends on: ${depText}`;
                            dependenciesList.appendChild(dependencyItem);
                        });
                        nodeItem.appendChild(dependenciesList);
                    }
                    nodesList.appendChild(nodeItem);
                });
                sessionDiv.appendChild(nodesList);
            } else {
                const noNodesPara = document.createElement('p');
                noNodesPara.textContent = 'No nodes in this session.';
                sessionDiv.appendChild(noNodesPara);
            }
            sessionsContainer.appendChild(sessionDiv);
        });
    } else {
        const noSessionsPara = document.createElement('p');
        noSessionsPara.textContent = 'No sessions or compatible graph structure found to display.';
        sessionsContainer.appendChild(noSessionsPara);
    }
    // ------------------------------------

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