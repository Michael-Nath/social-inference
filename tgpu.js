async function test() {
  console.log("Michael Nath");
  
  if (!navigator.gpu) {
    console.log("WebGPU is not supported on this browser.");
    document.body.innerHTML = "<p>WebGPU is not supported on this browser.</p>";
    return;
  }

  console.log(navigator.gpu);
  document.body.innerHTML = `<pre>${JSON.stringify(navigator.gpu, null, 2)}</pre>`;
  
  const adapter = await navigator.gpu.requestAdapter();
  console.log(adapter);
  if (!adapter) {
    document.body.innerHTML += "<p>Failed to get GPU adapter.</p>";
    return;
  }
  
  const device = await adapter.requestDevice();
  console.log(device.adapterInfo);
  
  document.body.innerHTML += `<pre>${adapter.name}</pre>`;
}

// Run when the page is fully loaded
document.addEventListener("DOMContentLoaded", test);
