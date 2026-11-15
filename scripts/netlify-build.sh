#!/usr/bin/env bash
# Netlify build step: copies the transformers.js runtime and downloads the AI models INTO the site,
# so the published app never contacts Hugging Face or a CDN at run time.
set -euo pipefail
cd "$(dirname "$0")/.."

TJS=node_modules/@huggingface/transformers/dist
ORT=node_modules/onnxruntime-web/dist
[ -d "$ORT" ] || ORT=node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist
mkdir -p site/lib
cp "$TJS/transformers.min.js" site/lib/
cp "$ORT/ort-wasm-simd-threaded.asyncify.mjs" "$ORT/ort-wasm-simd-threaded.asyncify.wasm" site/lib/

fetch() {   # fetch <model> <file>...
  local MODEL=$1; shift
  for F in "$@"; do
    local OUT="site/models/$MODEL/$F"
    [ -s "$OUT" ] && continue
    mkdir -p "$(dirname "$OUT")"
    curl -fsSL --retry 3 "https://huggingface.co/$MODEL/resolve/main/$F" -o "$OUT"
  done
}
fetch Xenova/all-MiniLM-L6-v2        config.json tokenizer.json tokenizer_config.json special_tokens_map.json onnx/model_quantized.onnx
fetch Xenova/ms-marco-MiniLM-L-6-v2  config.json tokenizer.json tokenizer_config.json special_tokens_map.json onnx/model_quantized.onnx
fetch onnx-community/LFM2-1.2B-ONNX  config.json generation_config.json tokenizer.json tokenizer_config.json onnx/model_q4.onnx onnx/model_q4.onnx_data
du -sh site/lib site/models
