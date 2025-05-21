import { initializeWebGPU } from "./common.js"; // Import the initializer
import { AppController } from "./appController.js";
import { UIManager } from "./uiManager.js";

async function main() {
  try {
    // --- Initialize WebGPU Device --- 
    const device = await initializeWebGPU();
    if (!device) {
        console.error("WebGPU not supported or initialization failed.");
        // Display error to user via a dedicated error element if UIManager can't be initialized yet
        const errorDisplay = document.getElementById('error-display');
        if (errorDisplay) {
            errorDisplay.textContent = "WebGPU initialization failed. Please ensure your browser supports WebGPU and it is enabled.";
            errorDisplay.style.color = 'red';
            errorDisplay.style.display = 'block';
        }
        return; // Stop execution if no device
    }
    console.log("WebGPU Device Initialized:", device);
    // ----------------------------------

    // --- Initialize UIManager ---
    // Ensure these IDs exist in your index.html
    const uiManager = new UIManager({
        sessionsContainerId: 'sessions-container',
        currentPartitionId: 'current-partition',
        errorDisplayId: 'error-display' // Used for displaying errors by AppController
    });
    // ----------------------------

    // --- Initialize and Run AppController ---
    const appController = new AppController(device, uiManager);
    await appController.runMainWorkflow();
    // ------------------------------------

  } catch (error) {
    console.error("An error occurred during main execution in play2.js:", error);
    // Fallback error display if UIManager might not be initialized or error is outside AppController
    const errorDisplay = document.getElementById('error-display');
    if (errorDisplay) {
        errorDisplay.textContent = `A critical error occurred: ${error.message || 'Unknown error'}. Check console for details.`;
        errorDisplay.style.color = 'red';
    }
  }
}

main();