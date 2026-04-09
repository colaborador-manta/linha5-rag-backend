import { pipeline, env } from '@xenova/transformers';
env.cacheDir = '/app/models';
console.log('[SETUP] Baixando modelo all-MiniLM-L6-v2...');
const start = Date.now();
const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { quantized: true });
console.log(`[SETUP] Modelo cached em ${((Date.now() - start) / 1000).toFixed(1)}s`);
process.exit(0);
