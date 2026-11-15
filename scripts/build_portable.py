"""Build the portable, double-click version of the app: one index.html plus a files/ folder.

Usage (after `bash scripts/netlify-build.sh` has filled site/lib and site/models):
    python3 scripts/build_portable.py <out dir>

Browsers block fetch() and ES-module imports for pages opened from the file system (file://),
so every asset (runtime, models, rule data) is written as a plain .js file that holds the bytes
as base64. index.html loads them with ordinary <script> tags, then answers transformers.js'
requests for those files from memory. Nothing is downloaded and no web server is needed.
"""
import base64
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'
OUT = pathlib.Path(sys.argv[1]).resolve()
shutil.rmtree(OUT, ignore_errors=True)
(OUT / 'files').mkdir(parents=True)

# Every file the web version serves: runtime, rule data and models
ASSETS = sorted(str(f.relative_to(SITE)) for d in ('lib', 'data', 'models') for f in (SITE / d).rglob('*') if f.is_file())
CHUNK = 40_000_000   # bytes per piece: keeps each base64 string far below browser string limits

# Small files share one script; large files are split into pieces across several scripts
scripts, small = [], []
for a in ASSETS:
    data = (SITE / a).read_bytes()
    if len(data) < 2_000_000:
        small.append((a, data))
        continue
    stem = re.sub(r'[^A-Za-z0-9]+', '-', a.split('/', 1)[-1]).strip('-')[-60:]
    for n, start in enumerate(range(0, len(data), CHUNK)):
        name = f'{stem}-{n + 1:02d}.js'
        with open(OUT / 'files' / name, 'w') as f:
            f.write('window.F1_ASSETS = window.F1_ASSETS || {};\n')
            f.write(f'(F1_ASSETS["{a}"] = F1_ASSETS["{a}"] || []).push("{base64.b64encode(data[start:start + CHUNK]).decode()}");\n')
        scripts.append(name)
with open(OUT / 'files' / 'small-files.js', 'w') as f:
    f.write('window.F1_ASSETS = window.F1_ASSETS || {};\n')
    for a, data in small:
        f.write(f'F1_ASSETS["{a}"] = ["{base64.b64encode(data).decode()}"];\n')
scripts.insert(0, 'small-files.js')

# ---- index.html: same page, styles inlined, assets loaded with classic script tags
html = (SITE / 'index.html').read_text()
css = (SITE / 'style.css').read_text()
engine = re.sub(r'^export ', '', (SITE / 'engine.js').read_text(), flags=re.M)
app = (SITE / 'app.js').read_text()
app = re.sub(r"^import \{[^}]*\} from './lib/transformers.min.js';\n", '', app, flags=re.M)
app = app.replace("import { buildIndex, rank } from './engine.js';\n", '')
app = app.replace("env.backends.onnx.wasm.wasmPaths = new URL('./lib/', location.href).href;",
                  "env.backends.onnx.wasm.wasmPaths = { mjs: assetUrl('lib/ort-wasm-simd-threaded.asyncify.mjs', 'text/javascript'),\n"
                  "                                     wasm: assetUrl('lib/ort-wasm-simd-threaded.asyncify.wasm', 'application/wasm') };\n"
                  "env.useBrowserCache = false;   // the files are already on this computer")
assert 'import ' not in app.split('\n', 3)[1] and 'assetUrl' in app

bootstrap = r"""
// Serve the bundled files (files/*.js) to transformers.js from memory instead of the network.
const TYPES = { js: 'text/javascript', mjs: 'text/javascript', wasm: 'application/wasm', json: 'application/json' };
const cache = new Map();
function assetBytes(key) {
  if (!cache.has(key)) {
    const pieces = window.F1_ASSETS[key].map((b64) => {     // each file may be split into several base64 pieces
      const bin = atob(b64), out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    });
    const bytes = new Uint8Array(pieces.reduce((n, p) => n + p.length, 0));
    let off = 0;
    for (const p of pieces) { bytes.set(p, off); off += p.length; }
    cache.set(key, bytes);
    delete window.F1_ASSETS[key];   // free the text copy
  }
  return cache.get(key);
}
function assetUrl(key, type) { return URL.createObjectURL(new Blob([assetBytes(key)], { type })); }
const realFetch = window.fetch.bind(window);
window.fetch = async (input, init) => {
  const url = decodeURIComponent(typeof input === 'string' ? input : input.url);
  const key = Object.keys(window.F1_ASSETS).concat([...cache.keys()]).find((k) => url.endsWith(k));
  if (!key) return realFetch(input, init);
  const bytes = assetBytes(key);
  const type = TYPES[key.split('.').pop()] || 'application/octet-stream';
  return new Response(bytes, { status: 200, headers: { 'Content-Type': type, 'Content-Length': String(bytes.length) } });
};
const { pipeline, env, AutoTokenizer, AutoModelForSequenceClassification, TextStreamer } = await import(assetUrl('lib/transformers.min.js', 'text/javascript'));
"""

script_tags = '\n'.join(f'<script src="files/{n}"></script>' for n in scripts)
html = html.replace('<link rel="stylesheet" href="style.css">', f'<style>\n{css}</style>')
html = html.replace('<script type="module" src="app.js"></script>',
                    f'{script_tags}\n<script type="module">\n{bootstrap}\n{engine}\n{app}\n</script>')
html = html.replace('<title>F1 Rulebook Assistant</title>', '<title>F1 Rulebook Assistant (offline)</title>')
html = html.replace('After the first load they are cached by your browser.',
                    'This copy runs from your own computer: double-click <code>index.html</code>, no internet or server needed.')
(OUT / 'index.html').write_text(html)
(OUT / 'README.txt').write_text(
    'F1 Rulebook Assistant (offline version)\n\n'
    'Double-click index.html to open it in your browser. Keep index.html and the files folder together.\n'
    'Everything (the AI models, the transformers.js runtime and the rulebook) is inside the files folder,\n'
    'so it works with no internet connection and no web server.\n\n'
    'Online version: https://f1-rulebook-ai.netlify.app\n'
    'Source code: https://github.com/lacerna5492/f1-rulebook-ai\n')
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        print(f'{p.relative_to(OUT)}  {p.stat().st_size / 1e6:.1f} MB')
