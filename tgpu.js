async function test() {
  const adapter = await navigator.gpu.requestAdapter();
  console.log(adapter);
  const device = await adapter.requestDevice();

  console.log(device.adapterInfo.architecture);
};

// Run when the page is fully loaded
document.addEventListener("DOMContentLoaded", test);