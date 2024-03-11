const adapter = await navigator.gpu.requestAdapter({
  powerPreference: "high-performance",
});

const info = await adapter?.requestAdapterInfo();

console.log(info);
