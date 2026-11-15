// Compare retrieval setups on eval/questions.json: recall@1/3/5/10 (is an expected article in the top k?)
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env } from '@huggingface/transformers';
import fs from 'node:fs';
import { buildIndex, tokens } from '../site/engine.js';
env.cacheDir = '/home/claude/hf-cache';
const P = JSON.parse(fs.readFileSync('site/data/passages.json', 'utf8')).passages;
const Q = JSON.parse(fs.readFileSync('eval/questions.json', 'utf8'));
const IDX = buildIndex(P);
function bm25(q) { const k1 = 1.2, b = 0.75, out = new Float32Array(IDX.n);
  for (const t of new Set(tokens(q))) { const df = IDX.df.get(t); if (!df) continue; const idf = Math.log(1 + (IDX.n - df + 0.5) / (df + 0.5));
    for (let i = 0; i < IDX.n; i++) { const f = IDX.tf[i].get(t); if (f) out[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * IDX.len[i] / IDX.avg)); } } return out; }
const EMB = {
  minilm: ['Xenova/all-MiniLM-L6-v2', '', ''],
  bge: ['Xenova/bge-small-en-v1.5', 'Represent this sentence for searching relevant passages: ', ''],
  gte: ['Xenova/gte-small', '', ''],
  arctic: ['Snowflake/snowflake-arctic-embed-s', 'Represent this sentence for searching relevant passages: ', ''],
  mxbai: ['mixedbread-ai/mxbai-embed-xsmall-v1', 'Represent this sentence for searching relevant passages: ', ''],
};
const which = process.argv.slice(2);
const rer = which.includes('rerank');
let rtok, rmod;
if (rer) { rtok = await AutoTokenizer.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2'); rmod = await AutoModelForSequenceClassification.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2', { dtype: 'q8' }); }
for (const name of which.filter((w) => EMB[w])) {
  const [id, qp, dp] = EMB[name];
  const e = await pipeline('feature-extraction', id, { dtype: 'q8' });
  const pool = name === 'minilm' || name === 'gte' ? 'mean' : 'cls';
  const D = []; for (const p of P) D.push((await e(`${dp}${p.title}. ${p.text}`, { pooling: pool, normalize: true })).data);
  for (const kwW of [0, 0.3]) {
    const hits = { 1: 0, 3: 0, 5: 0, 10: 0 }, rh = { 1: 0, 3: 0, 5: 0 }; const miss = [];
    for (const item of Q) {
      const v = (await e(qp + item.q, { pooling: pool, normalize: true })).data; const kw = bm25(item.q); const mk = Math.max(...kw) || 1;
      const sc = D.map((d, i) => { let s = 0; for (let j = 0; j < d.length; j++) s += v[j] * d[j]; return [s + kwW * kw[i] / mk, i]; }).sort((a, b) => b[0] - a[0]);
      const ok = (k) => sc.slice(0, k).some(([, i]) => item.a.includes(P[i].article));
      for (const k of [1, 3, 5, 10]) hits[k] += ok(k);
      if (!ok(5)) miss.push(item.q);
      if (rer) {
        const cand = sc.slice(0, 20).map(([, i]) => i);
        const inp = rtok(new Array(cand.length).fill(item.q), { text_pair: cand.map((i) => `${P[i].title}. ${P[i].text}`), padding: true, truncation: true });
        const lg = (await rmod(inp)).logits.data;
        const rr = cand.map((i, j) => [lg[j], i]).sort((a, b) => b[0] - a[0]);
        for (const k of [1, 3, 5]) rh[k] += rr.slice(0, k).some(([, i]) => item.a.includes(P[i].article));
      }
    }
    console.log(`${name} kw=${kwW}: R@1 ${hits[1]} R@3 ${hits[3]} R@5 ${hits[5]} R@10 ${hits[10]} / ${Q.length}` + (rer ? ` | rerank top20: R@1 ${rh[1]} R@3 ${rh[3]} R@5 ${rh[5]}` : ''));
    if (kwW === 0.3) console.log('   missed@5:', miss.join(' | '));
  }
}
