async function summarizeGPU() {
  if (!navigator.gpu) {
    throw new Error("NO GPU detected...\n");
  }
  
  const adapter = await navigator.gpu.requestAdapter();
  console.log(adapter);
}


export { summarizeGPU };