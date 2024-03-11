import { detectGPUDevice } from "tvmjs";

const gpu = await detectGPUDevice();

console.info(gpu);
