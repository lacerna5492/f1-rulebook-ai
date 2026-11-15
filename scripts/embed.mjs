// Precompute passage embeddings with the same model + runtime the browser uses.
// Usage: npm install && bash scripts/netlify-build.sh && node scripts/embed.mjs   (run from the project root)
import { pipeline, env } from '@huggingface/transformers';
import fs from 'node:fs';
env.allowRemoteModels = false;
env.localModelPath = new URL('../site/models/', import.meta.url).pathname;
const meta = JSON.parse(fs.readFileSync('site/data/passages.json', 'utf8'));
const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
const dims = 384, out = new Float32Array(meta.passages.length * dims);
for (const [i, p] of meta.passages.entries()) {
  const v = await embed(`${p.title}. ${p.text}`, { pooling: 'mean', normalize: true });
  out.set(v.data, i * dims);
}
meta.dims = dims;
fs.writeFileSync('site/data/passages.json', JSON.stringify(meta));
fs.writeFileSync('site/data/embeddings.bin', Buffer.from(out.buffer));
console.log('embedded', meta.passages.length, 'passages');
