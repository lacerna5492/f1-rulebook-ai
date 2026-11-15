"""Combine per-section passage files into site/data/passages.json.
Usage: python3 -I scripts/merge_sections.py out.json sectionA.json sectionB.json ..."""
import json
import sys

out, parts = sys.argv[1], [json.load(open(f)) for f in sys.argv[2:]]
passages = [p for part in parts for p in part['passages']]
for i, p in enumerate(passages):
    p['id'] = i
json.dump({'sources': [part['source'] for part in parts], 'passages': passages}, open(out, 'w'), ensure_ascii=False)
print(len(passages), 'passages from', len(parts), 'sections')
