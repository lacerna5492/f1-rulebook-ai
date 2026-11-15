// F1 Rulebook Assistant: search + rerank + a small language model that writes the answer, all in the browser.
import { pipeline, env, AutoTokenizer, AutoModelForSequenceClassification, TextStreamer } from './lib/transformers.min.js';
import { buildIndex, rank } from './engine.js';

// Everything is served from this site: no calls to Hugging Face or any CDN.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = 'models/';   // relative: transformers.js v4 only checks local files for non-URL paths
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', location.href).href;

const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';          // finds candidate rules by meaning
const RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';   // re-scores the candidates against the question
const ANSWER_MODEL = window.ANSWER_MODEL || 'onnx-community/LFM2-1.2B-ONNX';   // reads the best rules and writes the answer
const CANDIDATES = 20, SHOWN = 5, CONTEXT = 3;

const SYSTEM = 'You answer questions about the 2026 FIA Formula 1 regulations using only the rule extracts provided. The first extract is usually the most relevant.\n- Answer in one or two short sentences, using the exact numbers and wording from the rules.\n- Cite the article number in square brackets, e.g. [B1.6.3].\n- If the extracts only partly answer the question, say what they do say.\n- Only if none of the extracts relate to the question, reply exactly: Not covered in these rules.';

const $ = (id) => document.getElementById(id);
const statusEl = $('status'), input = $('q'), button = $('go'), results = $('results');
const progress = $('progress'), bar = $('bar'), progressText = $('progress-text');

let embed, rtok, rmod, gen, passages, matrix, dims, index, device = 'wasm', busy = false, noGpu = false;

const EXAMPLES = [
  'How many points for a 3rd place finish in the Grand Prix?',
  'What is the pit lane speed limit?',
  'How many engines can each driver use?',
  'How long does Q1 last and how many cars are knocked out?',
  'What happens to Overtake mode when the safety car is out?',
  'Which car number is not allowed?',
];

function setStatus(text, cls = '') { statusEl.textContent = text; statusEl.className = 'badge ' + cls; }
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

// Download progress across all files of one loading phase
function tracker(label) {
  const files = new Map();
  return (p) => {
    if (p.status !== 'progress' || !p.total) return;
    files.set(p.file + p.name, [p.loaded, p.total]);
    let loaded = 0, total = 0;
    for (const [l, t] of files.values()) { loaded += l; total += t; }
    progress.hidden = false;
    bar.style.width = `${(100 * loaded / total).toFixed(1)}%`;
    progressText.textContent = `${label}: ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`;
  };
}

async function loadData() {
  const meta = await (await fetch('data/passages.json')).json();
  const buf = await (await fetch('data/embeddings.bin')).arrayBuffer();
  passages = meta.passages; dims = meta.dims;
  matrix = new Float32Array(buf);
  index = buildIndex(passages);
  $('sources').innerHTML = meta.sources.map((s) => `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a>`).join('; ');
}

async function pickDevice() {
  try { if (navigator.gpu && await navigator.gpu.requestAdapter()) return 'webgpu'; } catch { /* no WebGPU */ }
  return 'wasm';
}

async function init() {
  try {
    await loadData();
    // Phase 1: the two small search models (about 46 MB). Search works as soon as these are in.
    const onSearch = tracker('Loading search models');
    [embed, rtok, rmod] = await Promise.all([
      pipeline('feature-extraction', EMBED_MODEL, { dtype: 'q8', progress_callback: onSearch }),
      AutoTokenizer.from_pretrained(RERANK_MODEL),
      AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, { dtype: 'q8', progress_callback: onSearch }),
    ]);
    input.disabled = button.disabled = false;
    document.querySelectorAll('#examples button').forEach((b) => (b.disabled = false));
    input.focus();
    // Phase 2: the answer model (larger). Uses the graphics card when the browser supports WebGPU.
    device = await pickDevice();
    if (device !== 'webgpu') {   // the 4-bit answer model needs WebGPU; WebAssembly lacks one of its operators
      noGpu = true;
      setStatus('Search only · AI answers need WebGPU', 'error');
      return;
    }
    setStatus('Search ready · loading AI answers…');
    gen = await pipeline('text-generation', ANSWER_MODEL, { dtype: 'q4', device, progress_callback: tracker('Loading the AI answer model (one time, then cached)') });
    progress.hidden = true;
    setStatus(`Ready · ${passages.length} rule passages · ${device === 'webgpu' ? 'GPU' : 'CPU'}`, 'ready');
  } catch (err) {
    console.error(err);
    progress.hidden = true;
    if (embed) setStatus('Search only (AI answers unavailable)', 'error');
    else {
      setStatus('Could not load the models', 'error');
      results.innerHTML = `<div class="card"><p>Something went wrong while loading: ${escapeHtml(String(err.message || err))}</p></div>`;
    }
  }
}

// Search: meaning + keywords -> top 20, then the cross-encoder re-scores them
async function retrieve(question) {
  const v = (await embed(question, { pooling: 'mean', normalize: true })).data;
  const hits = rank(v, question, matrix, dims, index, CANDIDATES);
  const enc = rtok(new Array(hits.length).fill(question), {
    text_pair: hits.map((h) => `${passages[h.i].title}. ${passages[h.i].text}`), padding: true, truncation: true });
  const logits = (await rmod(enc)).logits.data;
  return hits.map((h, j) => ({ ...h, rr: logits[j] })).sort((a, b) => b.rr - a.rr).slice(0, SHOWN);
}

const refOf = (p) => (/^[AB]\d+$/.test(p.article) ? `Appendix ${p.article}` : `Article ${p.article}`);
const anchor = (article) => 'rule-' + article.replace(/\./g, '-');

function ruleHtml(p, open) {
  return `<details class="rule" id="${anchor(p.article)}" ${open ? 'open' : ''}>
    <summary><span class="ref">${escapeHtml(refOf(p))} · page ${escapeHtml(p.page)}</span> <span class="rule-title">${escapeHtml(p.title)}</span></summary>
    <p class="rule-text">${escapeHtml(p.text)}</p></details>`;
}

// Turn [B1.6.3] citations in the answer into links to the rule cards below
function answerHtml(text) {
  return escapeHtml(text).replace(/\[?((?:Article |Rule )?([AB]\d+(?:\.\d+){0,3}))\]?/g, (m, label, art) =>
    document.getElementById(anchor(art)) ? `<a class="cite" href="#${anchor(art)}">${art}</a>` : m);
}

async function ask(question) {
  question = question.trim();
  if (!question || !embed || busy) return;
  busy = true; input.value = question; button.disabled = true;
  results.innerHTML = '<div class="card"><p class="low">Searching the rulebook…</p></div>';
  const t0 = performance.now();
  const hits = await retrieve(question);
  const weak = hits[0].rr < -4;   // the best rule is a poor match for the question
  results.innerHTML = `
    <article class="card answer-card">
      <p class="answer-label">Answer</p>
      <p class="answer" id="answer">${gen ? '<span class="typing">Reading the rules…</span>' : noGpu
        ? 'AI-written answers need a browser with WebGPU (Chrome or Edge on a desktop or laptop). The best matching rules are below.'
        : 'The AI answer model is still loading. The best matching rules are below.'}</p>
      <p class="meta" id="meta"></p>
    </article>
    <section class="card"><p class="answer-label">${weak ? 'Closest rules found (may not match your question)' : 'Matching rules'}</p>
      ${hits.map((h, k) => ruleHtml(passages[h.i], k < 2)).join('')}
    </section>`;
  if (gen) {
    const ctx = hits.slice(0, CONTEXT).map((h) => `[${passages[h.i].article}] ${passages[h.i].title}\n${passages[h.i].text}`).join('\n\n');
    const box = $('answer');
    let text = '';
    const streamer = new TextStreamer(gen.tokenizer, { skip_prompt: true, skip_special_tokens: true,
      callback_function: (t) => { text += t; box.innerHTML = answerHtml(text) + '<span class="cursor"></span>'; } });
    await gen([{ role: 'system', content: SYSTEM }, { role: 'user', content: `Rule extracts:\n\n${ctx}\n\nQuestion: ${question}` }],
      { max_new_tokens: 140, do_sample: false, streamer });
    text = text.trim();
    box.innerHTML = /^not covered/i.test(text)
      ? 'Not covered in the rules this app has (Sections A and B of the 2026 regulations). The closest rules are below.'
      : answerHtml(text);
  }
  $('meta').textContent = `Answered in ${((performance.now() - t0) / 1000).toFixed(1)} s on your device` +
    (gen ? ` (${device === 'webgpu' ? 'GPU' : 'CPU'})` : '') + '. Small AI models can make mistakes: check the rule text below.';
  busy = false; button.disabled = false;
}

$('ask').addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });
for (const q of EXAMPLES) {
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = q; b.disabled = true;
  b.addEventListener('click', () => ask(q));
  $('examples').appendChild(b);
}
init();
