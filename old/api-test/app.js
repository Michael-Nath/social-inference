function setStatus(status) {
    document.getElementById("status").innerHTML = `<p>${status}</p>`;
}

async function runTest() {
    var device;
    try {
        device = await setupGPU();
    } catch(error) {
        console.error(`Failed to initialize GPU: ${error}`);
        setStatus(`Failed to initialize GPU: ${error}`)
        return;
    }

    console.log("Device initialized! Limits: ", device.limits);


    const networkTransferStart = performance.now();
    var response;
    try {
        response = await (await fetch("http://localhost:8000/get_tensor", { method: "GET"})).json();
    } catch(error) {
        console.error(`Failed to get weights: ${error}`);
        setStatus(`Failed to get weights: ${error}`);
        return;
    }
    const networkTransferStop = performance.now();
    const networkTime = networkTransferStop - networkTransferStart;

    // Deserialize flattened tensor
    const flattenedTensorDim = response.tensor.length;
    const flattenedTensor = new Float32Array(flattenedTensorDim);
    for(let i = 0; i < flattenedTensorDim; i++) {
        flattenedTensor[i] = response.tensor[i];
    }

    // Check size
    const maxBindedBufferSize = device.limits.maxStorageBufferBindingSize;
    if(flattenedTensorDim * 4 > maxBindedBufferSize) {
        setStatus(`Received flat tensor of size ${flattenedTensorDim} (${flattenedTensorDim * 4} bytes) is too big! (max: ${maxBindedBufferSize})`);
        return;
    }

    // Load into GPU
    const errorScopes = ['validation', 'out-of-memory', 'internal'];
    pushErrorScopes(device, errorScopes);

    const dataTransferStart = performance.now();
    const gpuBuffer = device.createBuffer({
        label: `loadedBuffer`,
        size: flattenedTensor.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(gpuBuffer, 0, flattenedTensor.buffer, 0, flattenedTensor.byteLength);
    await device.queue.onSubmittedWorkDone();

    const dataTransferEnd = performance.now();
	const dataTransferTime = dataTransferEnd - dataTransferStart;

    gpuBuffer.destroy();

    await popErrorScopes(device, errorScopes);

    const lastFive = flattenedTensor.slice(-5);
    setStatus(`Loaded ${flattenedTensor.byteLength} bytes (${flattenedTensor.byteLength / 1024} KB) in ${dataTransferTime.toFixed(4)} ms (network: ${networkTime.toFixed(4)} ms). Last 5 elements: ${lastFive.join(', ')}`);
}

window.addEventListener('load', () => {
    document.getElementById('runTest').addEventListener('click', runTest);
});
