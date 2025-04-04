// Detect device type - focusing only on iOS vs desktop
function isIOSDevice() {
    return /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
}

// Get safe maximum allocation size based on device
function getSafeMaxPower() {
    if (isIOSDevice()) {
	return 10; // 2 GB
    }
    return 15; // Up to 32 GB on desktop
}

// GPU setup function
async function setupGPU() {
    if (!navigator.gpu) {
	throw new Error("WebGPU is not supported in your browser");
    }
    
    try {
	const adapter = await navigator.gpu.requestAdapter();

	const device = await adapter.requestDevice({
	    requiredLimits: {
		maxBufferSize: adapter.limits.maxBufferSize,
		maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
	    },
	    requiredFeatures: ['shader-f16']
	});

	// Listen for uncaptured errors.
	device.addEventListener('uncapturederror', event => {
	    console.error('Uncaptured error:', event.error.message);
	});

	return device;
    } catch (error) {
	throw new Error(`Failed to request GPU adapter: ${error.message}`);
    }
    
}

function pushErrorScopes(device, scopes) {
    scopes.forEach(scope => device.pushErrorScope(scope));
}

async function popErrorScopes(device, scopes) {
    const errors = [];
    // Pop in reverse order (since error scopes are stack-like)
    for (let i = scopes.length - 1; i >= 0; i--) {
	const error = await device.popErrorScope();
	if (error) {
	    errors.push({ scope: scopes[i], message: error.message });
	    console.error(`${scopes[i]} error: ${error.message}`);
	}
    }
    return errors;
}
