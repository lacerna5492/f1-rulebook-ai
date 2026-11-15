// Shared retrieval for the tests: MiniLM + keywords -> top 20 -> ms-marco cross-encoder rerank
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env } from '@huggingface/transformers';
import fs from 'node:fs';
import { buildIndex, rank } from '../site/engine.js';
env.cacheDir = '/home/claude/hf-cache';
const meta = JSON.parse(fs.readFileSync('site/data/passages.json', 'utf8'));
export const P = meta.passages;
const D = meta.dims, M = new Float32Array(fs.readFileSync('site/data/embeddings.bin').buffer.slice(0));
const IDX = buildIndex(P);
const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
const rtok = await AutoTokenizer.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2');
const rmod = await AutoModelForSequenceClassification.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2', { dtype: 'q8' });
export async function retrieve(q, k = 5) {
  const v = (await embed(q, { pooling: 'mean', normalize: true })).data;
  const hits = rank(v, q, M, D, IDX, 20);
  const inp = rtok(new Array(hits.length).fill(q), { text_pair: hits.map((h) => `${P[h.i].title}. ${P[h.i].text}`), padding: true, truncation: true });
  const lg = (await rmod(inp)).logits.data;
  return hits.map((h, j) => ({ ...h, rr: lg[j] })).sort((a, b) => b.rr - a.rr).slice(0, k);
}
export const questions = () => process.env.QFILE ? JSON.parse(fs.readFileSync(process.env.QFILE, 'utf8'))
  : JSON.parse(fs.readFileSync('eval/questions.json', 'utf8')).concat(JSON.parse(fs.readFileSync('eval/questions2.json', 'utf8')));
