// Import the SAXPY function from our module
import { runSaxpy } from './saxpy.js';
import { summarizeGPU } from './gpu-utils.js';

// Example usage
async function runExample() {
    try {
        // Define inputs
        const a = 2.0;
        const x = [1, 2, 3, 4, 5];
        const y = [10, 20, 30, 40, 50];



        console.log("Input a:", a);
        console.log("Input x:", x);
        console.log("Input y:", y);

        // Run the SAXPY computation
        const result = await runSaxpy(a, x, y);
        console.log("SAXPY result (a*x + y):", result);

        // Verify the results
        const expectedResult = x.map((xi, i) => a * xi + y[i]);
        console.log("Expected result:", expectedResult);
        console.log("Results match:", JSON.stringify(result) === JSON.stringify(expectedResult));

        // Update UI or perform additional processing with the results
        const adapter = await navigator.gpu.requestAdapter();
        displayResults(a, x, y, result, adapter.limits);
    } catch (error) {
        console.error("Error running SAXPY:", error);
        document.getElementById('error').textContent = `Error: ${error.message}`;
    }
}

// Function to display results in the UI
function displayResults(a, x, y, result, limits) {
    const resultContainer = document.getElementById('results');
    resultContainer.innerHTML = `
        <h3>SAXPY Results</h3>
        <p>Formula: result = ${a} * ${x} + ${y}</p>
        <p> ${limits}</p>
        <table>
            <thead>
                <tr>
                    <th>Index</th>
                    <th>x</th>
                    <th>y</th>
                    <th>result</th>
                </tr>
            </thead>
            <tbody>
                ${x.map((xi, i) => `
                    <tr>
                        <td>${i}</td>
                        <td>${xi}</td>
                        <td>${y[i]}</td>
                        <td>${result[i]}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// Run the example when the page loads
document.addEventListener('DOMContentLoaded', summarizeGPU);