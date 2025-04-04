// SAXPY FLOP Benchmark with 3 trials, chunking, dispatch splitting, and GFLOPS estimation
async function saxpyBenchmark() {
    try {
	// Set up GPU and request a device with required limits.
	const device = await setupGPU();
	console.log("device limits: ", device.limits);
	const maxBindedBufferSize = device.limits.maxStorageBufferBindingSize;
	const maxWorkgroups = device.limits.maxComputeWorkgroupsPerDimension;

	const MB = 1024 * 1024;
	// Fixed vector size: 256 MB.
	const totalNumElements = 4096; 
	const vectorSizeBytes = totalNumElements * 4; // f32 elements (4 bytes each)

	if(vectorSizeBytes > maxBindedBufferSize) {
	    console.error("Vector is too big!");
	}

	// Create full input arrays for X and Y.
	const vecX = new Float32Array(totalNumElements);
	const vecY = new Float32Array(totalNumElements);
	for (let i = 0; i < totalNumElements; i++) {
	    vecX[i] = i / totalNumElements;
	    vecY[i] = 1 - i / totalNumElements;
	}

	// Fixed workgroup size for the compute kernel.
	const workgroupSize = 256;

	// Create the compute shader module (with added offset support).
	const shaderModule = device.createShaderModule({
	    code: `
        enable f16;
        
        @group(0) @binding(0) var<uniform> uniforms: vec4<f32>;
        @group(0) @binding(1) var<storage, read> inputVectorX: array<f32>;
        @group(0) @binding(2) var<storage, read> inputVectorY: array<f32>;
        @group(0) @binding(3) var<storage, read_write> outputVector: array<f32>;
        
        @compute @workgroup_size(${workgroupSize})
        fn saxpyKernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
            // uniforms.x: multiplier
            // uniforms.y: base index (as f32)
            // uniforms.z: dimension (as f32)
            let base: u32 = u32(uniforms.y);
            let idx = base + global_id.x;
            let dim: u32 = u32(uniforms.z);
			let fadsf: f16 = 0.0;
            if (idx < dim) {
                var x: f32 = inputVectorX[idx];
                var u_x: f32 = uniforms.x;
                for (var i: u32 = 0u; i < dim; i = i + 1u) {
                    x += u_x * inputVectorY[i];
                }
                outputVector[idx] = x;
            }
        }
      `
	});

	// Create the compute pipeline.
	const computePipeline = device.createComputePipeline({
	    layout: 'auto',
	    compute: {
		module: shaderModule,
		entryPoint: 'saxpyKernel'
	    }
	});

	// Error scope names to push/pop.
	const errorScopes = ['validation', 'out-of-memory', 'internal'];

	// Run three trials.
	const trialResults = [];
	for (let trial = 0; trial < 3; trial++) {
	    pushErrorScopes(device, errorScopes);

	    // For each trial, create buffers for every chunk and upload the corresponding subarray.
	    const dataTransferStart = performance.now();

	    // Create storage buffers for this chunk.
	    const inputBufferX = device.createBuffer({
		label: `inputbufferx(trial=${trial})`,
		size: vectorSizeBytes,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	    });
	    const inputBufferY = device.createBuffer({
		label: `inputbuffery(trial=${trial})`,
		size: vectorSizeBytes,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	    });
	    const outputBuffer = device.createBuffer({
		label: `outputbuffer(trial=${trial})`,
		size: vectorSizeBytes,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	    });

	    // Write the subarray data into the input buffers.
	    device.queue.writeBuffer(inputBufferX, 0, vecX.buffer, 0, vecX.byteLength);
	    device.queue.writeBuffer(inputBufferY, 0, vecY.buffer, 0, vecY.byteLength);

	    chunkBuffers.push({
		inputBufferX,
		inputBufferY,
		outputBuffer,
		chunkNumElements
	    });

	    // Explicit barrier to accurately time writes, not
	    // necessary for correctness
	    await device.queue.onSubmittedWorkDone();

	    const dataTransferEnd = performance.now();
	    const dataTransferTime = dataTransferEnd - dataTransferStart;

	    const computeStart = performance.now();

	    // Create one command encoder to record all compute passes.
	    const commandEncoder = device.createCommandEncoder();

	    // Setup compute pass
	    const computePass = commandEncoder.beginComputePass();
	    computePass.setPipeline(computePipeline);

	    let offsetDispatch = 0; // in workgroups
	    for (let dispatch = 0; dispatch < Math.ceil(totalNumElements / workgroupSize); dispatch++) {
		// Create a uniform buffer for this dispatch.
		const uniformDataForDispatch = new Float32Array([
		    1 / 16000000, // multiplier
		    dispatch * workgroupSize, // base index
		    totalNumElements, // dim
		    0 // unused
		]);

		const uniformBufferForDispatch = device.createBuffer({
		    label: `uniforminfo(trial=${trial},dispatch=${dispatch})`,
		    size: uniformDataForDispatch.byteLength,
		    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		device.queue.writeBuffer(
		    uniformBufferForDispatch, 0,
		    uniformDataForDispatch.buffer,
		    uniformDataForDispatch.byteOffset,
		    uniformDataForDispatch.byteLength
		);

		// Create a bind group for this dispatch.
		const bindGroup = device.createBindGroup({
		    layout: computePipeline.getBindGroupLayout(0),
		    entries: [
			{ binding: 0, resource: { buffer: uniformBufferForDispatch } },
			{ binding: 1, resource: { buffer: chunk.inputBufferX } },
			{ binding: 2, resource: { buffer: chunk.inputBufferY } },
			{ binding: 3, resource: { buffer: chunk.outputBuffer } }
		    ]
		});

		computePass.setBindGroup(0, bindGroup);
		computePass.dispatchWorkgroups(workgroupSize);
	    }
	    computePass.end();
	    device.queue.submit([commandEncoder.finish()]);
	    await device.queue.onSubmittedWorkDone();
	    const computeEnd = performance.now();
	    const computeTime = computeEnd - computeStart;

	    // Clean up buffers for this trial.
	    inputBufferX.destroy();
	    inputBufferY.destroy();
	    outputBuffer.destroy();

	    // Pop and log error scopes.
	    await popErrorScopes(device, errorScopes);

	    trialResults.push({ dataTransferTime, computeTime });
	    // Small delay between trials.
	    await new Promise(resolve => setTimeout(resolve, 50));
	}

	// Average the results from all trials.
	let totalDataTransfer = 0, totalCompute = 0;
	trialResults.forEach(trial => {
	    totalDataTransfer += trial.dataTransferTime;
	    totalCompute += trial.computeTime;
	});
	const avgDataTransferTime = totalDataTransfer / trialResults.length;
	const avgComputeTime = totalCompute / trialResults.length;

	const totalFLOPs = totalNumElements * totalNumElements;
	const avgComputeTimeSeconds = avgComputeTime / 1000;
	const gflops = (totalFLOPs / avgComputeTimeSeconds) / 1e9;

	return {
	    deviceName: device.name || 'Unknown GPU',
	    vectorSizeMB: vectorSizeBytes / MB,
	    avgDataTransferTime,
	    avgComputeTime,
	    gflops,
	    trialCount: trialResults.length
	};
    } catch (error) {
	throw {
	    deviceName: 'Failed to initialize GPU',
	    error: error.message
	};
    }
}

// Function to run the SAXPY benchmark and update the UI.
async function runSaxpyBenchmark() {
    try {
	document.getElementById('saxpyStatus').textContent = 'Running SAXPY benchmark...';
	document.getElementById('saxpyStatus').style.backgroundColor = '#f5f5f5';
	// Clear previous results.
	document.getElementById('saxpyResultsBody').innerHTML = '';
	document.getElementById('saxpySummary').innerHTML = '';

	// Short delay to update the UI.
	await new Promise(resolve => setTimeout(resolve, 100));

	const result = await saxpyBenchmark();
	updateSaxpyUI(result);
    } catch (error) {
	document.getElementById('saxpyStatus').textContent = `Error: ${error.message}`;
	document.getElementById('saxpyStatus').style.backgroundColor = '#ffe6e6';
	console.error('SAXPY Benchmark failed:', error);
    }
}

// Function to update the SAXPY benchmark UI.
function updateSaxpyUI(result) {
    document.getElementById('saxpyGpuName').textContent = result.deviceName;
    document.getElementById('saxpyVectorSize').textContent = `${result.vectorSizeMB} MB`;

    document.getElementById('saxpyStatus').textContent = result.error ?
	`Error: ${result.error}` : 'SAXPY Benchmark completed';

    const resultsBody = document.getElementById('saxpyResultsBody');
    // Data Transfer Time row.
    let row = document.createElement('tr');
    let labelCell = document.createElement('td');
    labelCell.textContent = 'Average Data Transfer Time';
    row.appendChild(labelCell);
    let valueCell = document.createElement('td');
    valueCell.textContent = `${result.avgDataTransferTime.toFixed(2)} ms`;
    row.appendChild(valueCell);
    resultsBody.appendChild(row);

    // Compute Time row.
    row = document.createElement('tr');
    labelCell = document.createElement('td');
    labelCell.textContent = 'Average Compute Time';
    row.appendChild(labelCell);
    valueCell = document.createElement('td');
    valueCell.textContent = `${result.avgComputeTime.toFixed(2)} ms`;
    row.appendChild(valueCell);
    resultsBody.appendChild(row);

    // GFLOPS row.
    row = document.createElement('tr');
    labelCell = document.createElement('td');
    labelCell.textContent = 'Estimated GFLOPS';
    row.appendChild(labelCell);
    valueCell = document.createElement('td');
    valueCell.textContent = `${result.gflops.toFixed(2)} GFLOPS`;
    row.appendChild(valueCell);
    resultsBody.appendChild(row);

    // Total Time row.
    row = document.createElement('tr');
    labelCell = document.createElement('td');
    labelCell.textContent = 'Total Time (Average)';
    row.appendChild(labelCell);
    valueCell = document.createElement('td');
    const totalTime = result.avgDataTransferTime + result.avgComputeTime;
    valueCell.textContent = `${totalTime.toFixed(2)} ms`;
    row.appendChild(valueCell);
    resultsBody.appendChild(row);

    // Update summary section.
    const summary = document.getElementById('saxpySummary');
    if (!result.error) {
	summary.innerHTML = `
      <div class="summary-section">
        <h3>SAXPY Benchmark Summary</h3>
        <p>Vector size: ${result.vectorSizeMB} MB (~${(result.vectorSizeMB * 1024).toLocaleString()} KB)</p>
        <p>Average Data Transfer Time: ${result.avgDataTransferTime.toFixed(2)} ms</p>
        <p>Average Compute Time: ${result.avgComputeTime.toFixed(2)} ms</p>
        <p>Estimated Performance: ${result.gflops.toFixed(2)} GFLOPS</p>
        <p>(Averaged over ${result.trialCount} trials)</p>
      </div>
    `;
    } else {
	summary.innerHTML = `
      <div class="summary-section error">
        <h3>SAXPY Benchmark Error</h3>
        <p>${result.error}</p>
      </div>
    `;
    }
}

// Initialize the SAXPY benchmark UI.
window.addEventListener('load', () => {
    document.getElementById('runSaxpyButton').addEventListener('click', runSaxpyBenchmark);
});
