const scopes = ["ad-llama/model", "ad-llama/config", "ad-llama/wasm"];

scopes.forEach(async (scope) => {
  const exists = await caches.has(scope);
  console.info(`${scope} existed: ${exists}`);
  if (exists) {
    await caches.delete(scope);
    console.info("cleared");
  }
});
