import { initializeWebGPU } from "./common.js"; // Import the initializer
import { AppController } from "./appController.js";
import { UIManager } from "./uiManager.js";



async function main() {
  try {
    // --- Initialize WebGPU Device --- 
    const device = await initializeWebGPU();
    if (!device) {
        console.log("WebGPU not supported or initialization failed.");
        throw new Error("WebGPU initialization failed. Please ensure your browser supports WebGPU and it is enabled.");
    }
    console.log("WebGPU Device Initialized:", device);
    // ----------------------------------

    // --- Initialize UIManager ---
    // Ensure these IDs exist in your index.html
    const uiManager = new UIManager({
      sessionsContainerId: 'sessions-container',
      currentPartitionId: 'current-partition',
      errorDisplayId: 'error-display', // Used for displaying errors by AppController
      chatContainerId: 'chat-container',
      chatOutputId: 'chat-output',
      chatInputId: 'chat-input',
      chatButtonId: 'chat-button'
    });
  
    // ----------------------------

    // --- Initialize and Run AppController ---
    const appController = new AppController(device, uiManager);
    await Promise.all([appController.runMainWorkflow(), appController.runRunChatWorkflow()]);
    // ------------------------------------

  } catch (error) {
    console.error("An error occurred during main execution in play2.js:", error);
    // Fallback error display if UIManager might not be initialized or error is outside AppController
    const errorDisplay = document.getElementById('error-display');
    if (errorDisplay) {
        errorDisplay.textContent = `A critical error occurred: ${error.message || 'Unknown error'}.`;
        errorDisplay.style.color = 'red';
    }
  }
}

main();