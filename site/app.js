// F1 Rulebook Assistant: semantic search + extractive question answering, entirely in the browser.
import { pipeline, env } from './lib/transformers.min.js';
import { buildIndex, rank } from './engine.js';

// Everything is served from this site: no calls to Hugging Face or any CDN.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL('./models/', location.href).href;
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', location.href).href;

const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const QA_MODEL = 'Xenova/distilbert-base-cased-distilled-squad';
const TOP_K = 5;          // passages shown/considered
const QA_ON = 3;          // passages the QA model reads
const MIN_QA_SCORE = 0.08;

const $ = (id) => document.getElementById(id);
const statusEl = $('status'), input = $('q'), button = $('go'), results = $('results');
const progress = $('progress'), bar = $('bar'), progressText = $('progress-text');

let embed, qa, passages, matrix, dims, index;

const EXAMPLES = [
  'What is the pit lane speed limit?',
  'How many engines may each driver use?',
  'What is the maximum race distance?',
  'How long is the sprint?',
  'What is the maximum duration of a race?',
  'How long is the first qualifying session?',
];

function setStatus(text, cls = '') { statusEl.textContent = text; statusEl.className = 'badge ' + cls; }
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

// Track download progress across all model files
const fileProgress = new Map();
function onProgress(p) {
  if (p.status === 'progress' && p.total) {
    fileProgress.set(p.file + p.name, [p.loaded, p.total]);
    let loaded = 0, total = 0;
    for (const [l, t] of fileProgress.values()) { loaded += l; total += t; }
    progress.hidden = false;
    bar.style.width = `${(100 * loaded / total).toFixed(1)}%`;
    progressText.textContent = `Loading AI models into your browser: ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`;
  }
}

async function loadData() {
  const meta = await (await fetch('data/passages.json')).json();
  const buf = await (await fetch('data/embeddings.bin')).arrayBuffer();
  passages = meta.passages; dims = meta.dims;
  matrix = new Float32Array(buf);
  index = buildIndex(passages);
  const link = $('source-link');
  link.href = meta.source.url; link.textContent = meta.source.title;
}

async function init() {
  try {
    await loadData();
    [embed, qa] = await Promise.all([
      pipeline('feature-extraction', EMBED_MODEL, { dtype: 'q8', progress_callback: onProgress }),
      pipeline('question-answering', QA_MODEL, { dtype: 'q8', progress_callback: onProgress }),
    ]);
    progress.hidden = true;
    setStatus(`Ready · ${passages.length} rule passages`, 'ready');
    input.disabled = button.disabled = false;
    document.querySelectorAll('#examples button').forEach((b) => (b.disabled = false));
    input.focus();
  } catch (err) {
    console.error(err);
    setStatus('Could not load the models', 'error');
    results.innerHTML = `<div class="card"><p>Something went wrong while loading: ${escapeHtml(String(err.message || err))}</p></div>`;
  }
}

// The QA model's tokenizer re-spaces its answer ("80km / h"), so find the exact span in the rule text instead
function locate(text, answer) {
  const chars = answer.replace(/\s+/g, '').split('').map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  if (!chars.length) return null;
  const m = new RegExp(chars.join('\\s*'), 'i').exec(text);
  return m ? [m.index, m.index + m[0].length] : null;
}
const cleanAnswer = (a) => a.replace(/^\s*(?:[a-z]|[ivx]+)\.\s+(?=\S)/, '').trim();

function highlight(text, span) {
  if (!span) return escapeHtml(text);
  const [a, b] = span;
  return escapeHtml(text.slice(0, a)) + '<mark>' + escapeHtml(text.slice(a, b)) + '</mark>' + escapeHtml(text.slice(b));
}

function ruleHtml(p, span) {
  const ref = /^B\d+$/.test(p.article) ? `Appendix ${p.article}` : `Article ${p.article}`;
  return `<span class="ref">${escapeHtml(ref)} · page ${escapeHtml(p.page)}</span>
    <p class="rule-title">${escapeHtml(p.title)}</p>
    <p class="rule-text">${highlight(p.text, span)}</p>`;
}

async function ask(question) {
  question = question.trim();
  if (!question || !embed) return;
  input.value = question;
  button.disabled = true;
  results.innerHTML = '<div class="card"><p class="low">Searching the rulebook…</p></div>';
  const t0 = performance.now();

  const out = await embed(question, { pooling: 'mean', normalize: true });
  const hits = rank(out.data, question, matrix, dims, index, TOP_K);

  // Let the QA model read the best few passages and keep the most confident answer
  let best = null;
  for (const h of hits.slice(0, QA_ON)) {
    const r = await qa(question, passages[h.i].text);
    const score = r.score * (0.5 + h.score / 1.3);   // blend answer confidence with how relevant the passage is
    if (!best || score > best.score) best = { ...h, answer: r.answer, qaScore: r.score, score };
  }
  const ms = Math.round(performance.now() - t0);

  const top = passages[best.i];
  const confident = best.qaScore >= MIN_QA_SCORE;
  const span = confident ? locate(top.text, cleanAnswer(best.answer)) : null;
  const answerText = span ? top.text.slice(...span) : cleanAnswer(best.answer);
  const others = hits.filter((h) => h.i !== best.i);
  results.innerHTML = `
    <article class="card">
      ${confident ? `<p class="answer-label">Answer</p><p class="answer">${escapeHtml(answerText)}</p>`
                  : `<p class="low">No short answer found, so here is the most relevant rule.</p>`}
      ${ruleHtml(top, span)}
      <p class="meta">Answer confidence ${(best.qaScore * 100).toFixed(0)}% · meaning match ${(Math.max(0, best.sim) * 100).toFixed(0)}% · ${ms} ms, computed on your device</p>
    </article>
    <details class="card"><summary>Other relevant rules (${others.length})</summary>
      ${others.map((h) => `<div class="rule">${ruleHtml(passages[h.i], null)}<p class="meta">meaning match ${(Math.max(0, h.sim) * 100).toFixed(0)}%</p></div>`).join('')}
    </details>`;
  button.disabled = false;
}

$('ask').addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });
for (const q of EXAMPLES) {
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = q; b.disabled = true;
  b.addEventListener('click', () => ask(q));
  $('examples').appendChild(b);
}
init();
