# F1 Rulebook Assistant

Ask questions about the **2026 FIA Formula 1 Sporting Regulations** (Section B, Issue 09, 1 October 2026) in plain English.
The app finds the matching articles and highlights the answer. **Both AI models run entirely in the browser**: there is no server, no API key, and nothing you type leaves your device.

Ryan Lacerna · CAI2300C (Dr. Lee) · "Build an AI model that runs in the browser"

## How it works

| Step | What happens | Model / method |
|---|---|---|
| 1. Find the rules | The question is turned into a 384-number vector and compared with all 423 rule passages. A keyword score (BM25) is blended in so exact F1 terms like "Parc Fermé" still match. | `Xenova/all-MiniLM-L6-v2` (sentence embeddings, 23 MB, q8) |
| 2. Find the answer | A question-answering model reads the top 3 passages and extracts the exact words that answer the question. The most confident answer, weighted by how relevant its passage is, wins. | `Xenova/distilbert-base-cased-distilled-squad` (66 MB, q8) |
| 3. Show the source | The answer is shown with its article number, page and the full rule text, plus other relevant rules. | |

The models run with [transformers.js](https://huggingface.co/docs/transformers.js) 3.7.6 (ONNX Runtime Web, WebAssembly).

## Fully local

- `env.allowRemoteModels = false`: transformers.js never contacts Hugging Face at run time.
- The model files, the WebAssembly runtime and the rulebook data are all served from this site (`site/models`, `site/lib`, `site/data`).
- After the first visit (about 110 MB) the browser caches everything, so later visits load in a few seconds.

You can check this in DevTools → Network: every request goes to the site's own domain.

## Preparing the rulebook (`scripts/process_pdf.py`)

The official PDF marks changes in magenta and **deleted rules with a strikethrough**. The script:

1. Reads every character with PyMuPDF and drops the ones crossed by a magenta strike line, so deleted rules (e.g. the old "Rain Hazard" article) are not in the app.
2. Removes headers, footers, the contents pages and Appendix B5 (approved changes for 2027 onward).
3. Splits the text by article number (e.g. `B1.6.3`), keeps each article's title and page, and breaks long articles into passages of at most 200 words so they fit the QA model.

`scripts/embed.mjs` then computes the passage vectors with the same model and runtime the browser uses, saved as `site/data/embeddings.bin`.

## Deploying (Netlify)

The models are too large for a GitHub web upload, so the Netlify build fetches them:

- `netlify.toml` runs `scripts/netlify-build.sh` and publishes `site/`.
- The build copies the transformers.js runtime from npm into `site/lib` and downloads the two models into `site/models`.
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

- These are small extractive models. They copy an answer from the rule text and do not reason across several articles.
- Questions whose answer is not in Section B (for example championship points or penalty-point bans, which are in other sections) will return the closest rule rather than an answer.
- This is an unofficial study tool. Always check the official FIA document. Rule text © Fédération Internationale de l'Automobile.
