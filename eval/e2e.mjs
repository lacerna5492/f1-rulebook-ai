// End-to-end check of the extractive pipeline: does the shown rule match, and does the short answer contain the expected words?
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env } from '@huggingface/transformers';
import fs from 'node:fs';
import { buildIndex, rank } from '../site/engine.js';
env.cacheDir = '/home/claude/hf-cache';
const FILE = process.argv[2] || 'eval/questions.json';
const RERANK = process.env.RERANK === '1', QA_ON = +(process.env.QA_ON || 3);
const meta = JSON.parse(fs.readFileSync('site/data/passages.json', 'utf8'));
const P = meta.passages, D = meta.dims, M = new Float32Array(fs.readFileSync('site/data/embeddings.bin').buffer.slice(0));
const IDX = buildIndex(P), Q = JSON.parse(fs.readFileSync(FILE, 'utf8'));
const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
const qa = await pipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad', { dtype: 'q8' });
let rtok, rmod;
if (RERANK) { rtok = await AutoTokenizer.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2'); rmod = await AutoModelForSequenceClassification.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2', { dtype: 'q8' }); }
const norm = (s) => s.toLowerCase().replace(/[’']/g, "'").replace(/\s+/g, ' ');
let r1 = 0, r3 = 0, ansOk = 0, ansN = 0, ruleOk = 0;
for (const item of Q) {
  const v = (await embed(item.q, { pooling: 'mean', normalize: true })).data;
  let hits = rank(v, item.q, M, D, IDX, RERANK ? 20 : 5);
  if (RERANK) {
    const inp = rtok(new Array(hits.length).fill(item.q), { text_pair: hits.map((h) => `${P[h.i].title}. ${P[h.i].text}`), padding: true, truncation: true });
    const lg = (await rmod(inp)).logits.data;
    hits = hits.map((h, j) => ({ ...h, rr: lg[j] })).sort((a, b) => b.rr - a.rr).slice(0, 5);
  }
  let best = null;
  for (const [k, h] of hits.slice(0, QA_ON).entries()) {
    const r = await qa(item.q, P[h.i].text);
    const prior = RERANK ? (process.env.PRIOR || '1,0.8,0.65,0.55,0.5').split(',').map(Number)[k] : 0.5 + h.score / 1.3;
    const s = r.score * prior;
    if (!best || s > best.s) best = { ...h, s, ans: r.answer, qs: r.score };
  }
  const art = P[best.i].article, arts = hits.map((h) => P[h.i].article);
  r1 += item.a.includes(arts[0]); r3 += arts.slice(0, 3).some((a) => item.a.includes(a));
  const rok = item.a.includes(art); ruleOk += rok;
  let aok = '';
  if (item.ans) { ansN++; const ok = norm(best.ans.replace(/\s*([\/()%,.])\s*/g, '$1')).includes(norm(item.ans).replace(/\s*([\/()%,.])\s*/g, '$1')) || norm(P[best.i].text).includes(norm(item.ans)) && rok && best.qs > 0.3; ansOk += ok; aok = ok ? 'OK' : 'BAD'; }
  console.log(`${item.none ? 'NONE' : rok ? '✓' : '✗'} ${aok.padEnd(3)} rr=${(hits[0].rr ?? 0).toFixed(1)} ${item.q}  ->  "${best.ans}" (${(best.qs * 100).toFixed(0)}%) [${art}]  top: ${arts.slice(0, 3).join(' ')}`);
}
console.log(`\n${FILE} rerank=${RERANK}: top1 ${r1}/${Q.length}, top3 ${r3}/${Q.length}, shown rule correct ${ruleOk}/${Q.length}, short answer correct ${ansOk}/${ansN}`);
