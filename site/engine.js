// Hybrid retrieval: meaning (embedding similarity) + keywords (BM25). Shared by the browser app and the Node test script.
const STOP = new Set('a an and are as at be by can do does for from has have how i if in is it its many may much of on or per the their there this to was what when where which who will with within'.split(' '));
const stem = (w) => w.length > 4 && w.endsWith('ies') ? w.slice(0, -3) + 'y' : w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w;
export const tokens = (s) => (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => !STOP.has(w)).map(stem);

export function buildIndex(passages) {
  const docs = passages.map((p) => tokens(`${p.title} ${p.title} ${p.text}`));
  const df = new Map();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) || 0) + 1);
  const avg = docs.reduce((a, d) => a + d.length, 0) / docs.length;
  const tf = docs.map((d) => { const m = new Map(); for (const t of d) m.set(t, (m.get(t) || 0) + 1); return m; });
  return { n: docs.length, df, avg, tf, len: docs.map((d) => d.length) };
}

function bm25(index, q) {
  const k1 = 1.2, b = 0.75, out = new Float32Array(index.n);
  for (const t of new Set(tokens(q))) {
    const df = index.df.get(t); if (!df) continue;
    const idf = Math.log(1 + (index.n - df + 0.5) / (df + 0.5));
    for (let i = 0; i < index.n; i++) {
      const f = index.tf[i].get(t); if (!f) continue;
      out[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * index.len[i] / index.avg));
    }
  }
  return out;
}

export const KEYWORD_WEIGHT = 0.3;

// Returns the top k passages as [{ i, sim, kw, score }]
export function rank(qvec, question, matrix, dims, index, k = 5) {
  const kw = bm25(index, question);
  const maxKw = Math.max(...kw) || 1;
  const res = [];
  for (let i = 0; i < index.n; i++) {
    let s = 0; const off = i * dims;
    for (let d = 0; d < dims; d++) s += qvec[d] * matrix[off + d];
    res.push({ i, sim: s, kw: kw[i] / maxKw, score: s + KEYWORD_WEIGHT * kw[i] / maxKw });
  }
  return res.sort((a, b) => b.score - a.score).slice(0, k);
}
