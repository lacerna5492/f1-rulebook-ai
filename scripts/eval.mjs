import { pipeline, env } from '@huggingface/transformers';
import fs from 'node:fs';
import { buildIndex, rank } from '../site/engine.js';
env.allowRemoteModels = false;
env.localModelPath = new URL('../site/models/', import.meta.url).pathname;
const meta = JSON.parse(fs.readFileSync('site/data/passages.json', 'utf8'));
const m = new Float32Array(fs.readFileSync('site/data/embeddings.bin').buffer.slice(0));
const P = meta.passages, D = meta.dims; const IDX = buildIndex(P);
const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
const qa = await pipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad', { dtype: 'q8' });
const Qs = process.argv.slice(2).length ? process.argv.slice(2) : [
 'What is the pit lane speed limit?',
 'How many engines can a driver use in a season?',
 'How many drivers can a team use during a championship?',
 'How long is the Race Director allowed to suspend a race?',
 'What is the maximum race distance?',
 'How many points does the winner of a race get?',
 'What happens if a driver jumps the start?',
 'How many sets of tyres does each driver get?',
 'When can a driver overtake under the Safety Car?',
 'What happens when the Heat Index is greater than 31 degrees?',
 'How long does a sprint qualifying session last?',
 'How many penalty points lead to a race ban?',
 'What is the minimum time for a pit stop?',
 'How long is the first qualifying session?',
 'Who decides to deploy the virtual safety car?',
 'What is the maximum time allowed for a race?',
];
for (const q of Qs) {
  const v = (await embed(q, { pooling: 'mean', normalize: true })).data;
  const s = rank(v, q, m, D, IDX, 5).map((h) => [h.score, h.i]);
  let best = null;
  for (const [sim, i] of s.slice(0, +(process.env.QA_ON || 3))) { const r = await qa(q, P[i].text); const sc = r.score * (0.5 + sim / 1.3); if (!best || sc > best.sc) best = { sc, sim, i, ...r }; }
  console.log(`Q: ${q}\n  -> "${best.answer}"  qa=${best.score.toFixed(2)} sim=${best.sim.toFixed(2)}  [${P[best.i].article} ${P[best.i].title}]  top5: ${s.map(([,i]) => P[i].article).join(' ')}`);
}
