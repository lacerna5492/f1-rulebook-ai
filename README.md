# F1 Rulebook Assistant

Ask questions about the **2026 FIA Formula 1 Regulations** in plain English: Section A (General Regulatory Provisions, Issue 03) and Section B (Sporting, Issue 09).
The app finds the matching articles and an AI model writes a short answer that cites them. **All three AI models run entirely in the browser**: there is no server, no API key, and nothing you type leaves your device.

Ryan Lacerna · CAI2300C (Dr. Lee) · "Build an AI model that runs in the browser"

Live: https://f1-rulebook-ai.netlify.app

## How it works

| Step | What happens | Model / method |
|---|---|---|
| 1. Find candidates | The question is turned into a 384-number vector and compared with all 696 rule passages, blended with a keyword (BM25) search. A small glossary maps fan words to rulebook terms ("Grand Prix" → Race, "DRS" → Driver Adjustable Bodywork, "red flag" → suspension). | `Xenova/all-MiniLM-L6-v2` (23 MB, q8) |
| 2. Rank them | A cross-encoder reads the question together with each of the top 20 candidates and re-orders them. | `Xenova/ms-marco-MiniLM-L-6-v2` (23 MB, q8) |
| 3. Write the answer | A small language model reads the 3 best passages and writes a one- or two-sentence answer with the article number, streaming word by word. It says "Not covered" when the rules don't answer the question. Runs on the graphics card through WebGPU (Chrome or Edge on a desktop or laptop); in browsers without WebGPU the app still works as a rule search. | `onnx-community/LFM2-1.2B-ONNX` (850 MB, 4-bit) |
| 4. Show the source | The matching rules are always shown under the answer, so every answer can be checked. | |

Runtime: [transformers.js](https://huggingface.co/docs/transformers.js) 4.3.2 (ONNX Runtime Web).

### Choosing the answer model
I graded answer models on about 60 realistic questions (`eval/`):
- The first version used an extractive QA model (DistilBERT-SQuAD), which copies a phrase from a passage. It was right about 55-60% of the time and its answers were fragments.
- Small generative models (Qwen2.5-0.5B, LFM2-1.2B, Qwen2.5-1.5B) were tested as the answer step. LFM2-1.2B gave the most accurate and readable answers at a size a browser can handle. It is still wrong on roughly 1 question in 3, which is why the rule text is always shown.

## Fully local

- `env.allowRemoteModels = false`: transformers.js never contacts Hugging Face at run time.
- The models, the WebAssembly runtime and the rulebook data are all served from this site (`site/models`, `site/lib`, `site/data`).
- After the first visit (about 930 MB, mostly the answer model) the browser caches everything.

You can check this in DevTools → Network: every request goes to the site's own domain.

## Preparing the rulebook (`scripts/process_pdf.py`, `scripts/merge_sections.py`)

The official PDFs mark changes in magenta and **deleted rules with a strikethrough**. The script (run once per section):

1. Reads every character with PyMuPDF and drops the ones crossed by a magenta strike line, so deleted rules (e.g. the old "Rain Hazard" article) are not in the app.
2. Removes headers, footers, the contents pages, blank forms and the appendices of approved changes for 2027 onward (A9, B5).
3. Turns the championship points tables (Article A2.2) into plain sentences, e.g. "3rd place 15 points", with the normal full-distance table separate from the shortened-race tables.
4. Splits the text by article number (e.g. `B1.6.3`), keeps each article's title and page, and breaks long articles into passages of at most 200 words so they fit the models.

`scripts/embed.mjs` then computes the passage vectors with the same model and runtime the browser uses, saved as `site/data/embeddings.bin`.

## Deploying (Netlify)

The models are too large for a GitHub web upload, so the Netlify build fetches them:

- `netlify.toml` runs `scripts/netlify-build.sh` and publishes `site/`.
- The build copies the transformers.js runtime from npm into `site/lib` and downloads the three models into `site/models`.
- The published site is self-contained.

Run it locally:

```bash
npm install
bash scripts/netlify-build.sh
npx serve site        # or: python3 -m http.server -d site
```

## Offline version (double-click index.html)

`scripts/build_portable.py` builds a folder with `index.html` and a `files/` folder that runs straight from your computer, with no server and no internet.

- Browsers block `fetch()` and module imports for pages opened from the file system, so each asset (runtime, models, rule data) is stored as a plain `.js` file holding the bytes in base64.
- `index.html` loads those files with ordinary `<script>` tags and hands them to transformers.js from memory.

```bash
python3 scripts/build_portable.py F1-Rulebook-Assistant   # after netlify-build.sh
```

## Limits

- The answer model is small (1.2 billion parameters). It sometimes picks the wrong detail or mixes up two rules, so always check the rule text shown under the answer.
- Only Sections A and B are included. Technical (C), financial and power-unit rules are not.
- This is an unofficial study tool. Always check the official FIA document. Rule text © Fédération Internationale de l'Automobile.
