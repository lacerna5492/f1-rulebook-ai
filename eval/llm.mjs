// Generative answer step: the model reads the top reranked passages and answers in plain English, citing the article.
import { pipeline } from '@huggingface/transformers';
import { P, retrieve, questions } from './retrieve.mjs';
const MODEL = process.argv[2], DTYPE = process.argv[3] || 'q4', K = +(process.env.K || 3);
const gen = await pipeline('text-generation', MODEL, { dtype: DTYPE });
export const SYSTEM = 'You answer questions about the 2026 FIA Formula 1 regulations using only the rule extracts provided. The first extract is usually the most relevant.\n- Answer in one or two short sentences, using the exact numbers and wording from the rules.\n- Cite the article number in square brackets, e.g. [B1.6.3].\n- If the extracts only partly answer the question, say what they do say.\n- Only if none of the extracts relate to the question, reply exactly: Not covered in these rules.';
const t0 = Date.now();
for (const item of questions()) {
  const hits = await retrieve(item.q, K);
  const ctx = hits.map((h) => `[${P[h.i].article}] ${P[h.i].title}\n${P[h.i].text}`).join('\n\n');
  const t = Date.now();
  const out = await gen([{ role: 'system', content: SYSTEM }, { role: 'user', content: `Rule extracts:\n\n${ctx}\n\nQuestion: ${item.q}` }],
    { max_new_tokens: 90, do_sample: false });
  const ans = out[0].generated_text.at(-1).content.trim().replace(/\s+/g, ' ');
  console.log(`Q: ${item.q}\n   want ${item.none ? 'NOT COVERED' : (item.a.join('/') + ' ' + (item.ans || ''))} | ctx ${hits.map((h) => P[h.i].article).join(' ')} | ${Date.now() - t} ms\n   A: ${ans}`);
}
console.log('total', ((Date.now() - t0) / 1000).toFixed(0), 's');
