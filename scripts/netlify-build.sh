#!/usr/bin/env bash
# Netlify build step: copies the transformers.js runtime and downloads the two models INTO the site,
# so the published app never contacts Hugging Face or a CDN at run time.
set -euo pipefail
cd "$(dirname "$0")/.."

DIST=node_modules/@huggingface/transformers/dist
mkdir -p site/lib
cp "$DIST/transformers.min.js" "$DIST/ort-wasm-simd-threaded.jsep.mjs" "$DIST/ort-wasm-simd-threaded.jsep.wasm" site/lib/

FILES="config.json tokenizer.json tokenizer_config.json special_tokens_map.json onnx/model_quantized.onnx"
for MODEL in Xenova/all-MiniLM-L6-v2 Xenova/distilbert-base-cased-distilled-squad; do
  for F in $FILES; do
    OUT="site/models/$MODEL/$F"
    [ -s "$OUT" ] && continue
    mkdir -p "$(dirname "$OUT")"
    curl -fsSL --retry 3 "https://huggingface.co/$MODEL/resolve/main/$F" -o "$OUT"
  done
done
du -sh site/lib site/models
