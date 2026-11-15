"""Turn the FIA 2026 F1 Sporting Regulations PDF into article-sized passages.

Usage: python3 -I process_pdf.py <regs.pdf> <out passages.json> [section letter, default B]

- Removes text the FIA struck through (deleted in this issue): magenta strike lines drawn over the characters.
- Removes page headers/footers, the contents pages, governance notes, blank forms (A4-A6) and the approved changes for future years (Appendix A9, B5).
- Splits the text by article number (e.g. B1.6.3) and keeps long articles under ~200 words per passage.
"""
import json
import re
import sys

import pymupdf

SRC, OUT = sys.argv[1], sys.argv[2]
SEC = sys.argv[3] if len(sys.argv) > 3 else 'B'
doc = pymupdf.open(SRC)

TOP, BOTTOM = 60, 780          # header / footer bands (points)
FIRST_PAGE = 3                 # pages 0-2 are the cover and contents
ID_RE = re.compile(rf'^({SEC}\d+(?:\.\d+){{1,3}})\b\.?\s*(.*)$')
ART_RE = re.compile(rf'^(ARTICLE|APPENDIX)\s+({SEC}\d+)\s*:\s*(.+)$')
SKIP_LINE = re.compile(rf'^\[?(Advisory Committee:|Governance:|SECTION {SEC}: [A-Z ]+$|{SEC}$)')


def strike_lines(page):
    """Thin magenta rectangles = strikethrough marks."""
    out = []
    for d in page.get_drawings():
        col = d.get('fill') or d.get('color')
        r = d['rect']
        if col and col[0] > 0.9 and col[1] < 0.3 and col[2] > 0.9 and r.height < 1.6 and r.width > 2:
            out.append(r)
    return out


def struck(ch_bbox, strikes):
    x0, y0, x1, y1 = ch_bbox
    cx, h = (x0 + x1) / 2, y1 - y0
    for r in strikes:
        ry = (r.y0 + r.y1) / 2
        if r.x0 - 0.5 <= cx <= r.x1 + 0.5 and y0 + 0.2 * h <= ry <= y1 - 0.1 * h:
            return True
    return False


lines = []   # (page_label, text, ends_short): article number and its text sit at the same height, so merge by y
for pno in range(FIRST_PAGE, doc.page_count):
    page = doc[pno]
    strikes = strike_lines(page)
    rows = {}
    for b in page.get_text('rawdict')['blocks']:
        for l in b.get('lines', []):
            y0, y1 = l['bbox'][1], l['bbox'][3]
            if y1 < TOP or y0 > BOTTOM:
                continue
            chars, x1 = [], None
            for s in l['spans']:
                if s['color'] == 0xFFFFFF:
                    continue
                for c in s['chars']:
                    if not struck(c['bbox'], strikes):
                        chars.append(c['c'])
                        if c['c'].strip():
                            x1 = c['bbox'][2]
            text = ''.join(chars)
            if not text.strip():
                continue
            key = round((y0 + y1) / 2)
            row = next((k for k in rows if abs(k - key) <= 2), key)
            rows.setdefault(row, []).append((l['bbox'][0], text, x1))
    for y in sorted(rows):
        parts = sorted(rows[y])
        text = re.sub(r'\s+', ' ', ' '.join(t for _, t, _ in parts)).strip()
        right = max(x for _, _, x in parts if x is not None)
        if text and not SKIP_LINE.match(text):
            lines.append((f'{SEC}{pno + 1}', text, right < 470, parts[0][0]))

# ---- Walk the lines, tracking article / section titles
passages, cur = [], None
article_title, section_title = '', ''


def flush():
    global cur
    if cur and len(cur['text'].split()) >= 6:
        passages.append(cur)
    cur = None


skip_appendix = False
for page, text, short, left in lines:
    m_art = ART_RE.match(text)
    if m_art:
        flush()
        if m_art.group(1) == 'APPENDIX' and m_art.group(2) in ('B5', 'A9'):
            break                              # approved changes for future years: not current rules
        article_title = f"{m_art.group(1).title()} {m_art.group(2)}: {m_art.group(3).title()}"
        section_title = ''
        skip_appendix = m_art.group(1) == 'APPENDIX' and m_art.group(2) in ('A4', 'A5', 'A6')   # blank entry/registration forms
        if m_art.group(1) == 'APPENDIX' and not skip_appendix:
            cur = {'article': m_art.group(2), 'title': article_title, 'page': page, 'text': ''}
        continue
    if skip_appendix:
        continue
    m_id = ID_RE.match(text) if left < 60 else None   # article numbers sit in the left margin
    if m_id:
        num, rest = m_id.group(1), m_id.group(2)
        level = num.count('.')
        flush()
        heading = bool(rest) and short and len(rest) < 90 and not rest.rstrip().endswith(('.', ':', ';'))
        if level == 1:                    # e.g. B1.6 Pit Entry Road, Pit Lane And Pit Exit Road
            section_title = rest
            cur = None
            continue
        title = rest if heading else section_title
        cur = {'article': num, 'title': title, 'page': page, 'text': '' if heading else rest}
        continue
    if cur is None:
        continue
    cur['text'] = (cur['text'] + '\n' + text).strip() if cur['text'] else text
flush()

# Join hyphen/line breaks inside sentences; keep list items on their own lines
for p in passages:
    t = p['text']
    t = re.sub(r'\n(?=[a-z]\.\s|[ivx]+\.\s|\(?[a-z]\)\s)', '\n\n', t)       # list items
    t = re.sub(r'(?<![.:;])\n(?!\n)', ' ', t)                                  # wrapped lines
    t = re.sub(r'[ \t]+', ' ', t).replace(' \n', '\n')
    p['text'] = t.strip()

# Points tables come out as rows like "3rd 3 8 12 15". Add a plain-English reading of each column
# (e.g. "3rd place: 15 points") so the QA model can answer "how many points for 3rd?".
ORD = re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)\b((?:\s+\d{1,2}\b)+)')
tables = []
SUFFIX = lambda n: 'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'
for p in passages:
    rows = [(int(m.group(1)), [int(x) for x in m.group(2).split()]) for m in ORD.finditer(p['text'])]
    if len(rows) < 5 or rows[0][0] != 1:
        continue
    ncol = max(len(v) for _, v in rows)
    cond = {int(c): w.strip() for w, c in re.findall(
        r'If the leader has completed ([^.]*?), points will be awarded in accordance with column (\d)', p['text'])}
    session = 'Sprint' if 'Sprint' in p['title'] else 'Race (Grand Prix)'
    def column(c):
        vals = [(pos, v[c - (ncol - len(v))]) for pos, v in rows if c - (ncol - len(v)) >= 0]   # short rows are right-aligned
        return ', '.join(f'{pos}{SUFFIX(pos)} place {v} point{"s" if v != 1 else ""}' for pos, v in vals)
    # The normal case (the last column) comes first; shortened-race columns follow
    text = f'Championship points for each finishing position in a {session} that runs its normal distance'
    text += f' (leader completes {cond[ncol]})' if ncol in cond else ''
    text += f': {column(ncol - 1)}. No points for lower positions.'
    tables.append({'article': p['article'], 'title': p['title'] + ' (points table)', 'page': p['page'], 'text': text})
    if ncol > 1:
        short = ' '.join(f'If the leader has completed only {cond.get(c + 1, "less distance")}: {column(c)}.' for c in range(ncol - 1))
        tables.append({'article': p['article'], 'title': p['title'] + ' (reduced points for a shortened race)', 'page': p['page'],
                       'text': f'Reduced championship points when a {session} is shortened. {short}'})
passages += tables

# Split long articles into overlapping windows of whole sentences
MAX_W, OVERLAP_W = 200, 35
final = []
for p in passages:
    words = p['text'].split()
    if len(words) <= MAX_W:
        final.append(p)
        continue
    sents = re.split(r'(?<=[.;:])\s+', p['text'])
    chunk, n, part = [], 0, 1
    for s in sents:
        w = len(s.split())
        if chunk and n + w > MAX_W:
            final.append({**p, 'text': ' '.join(chunk), 'part': part})
            part += 1
            tail, tn = [], 0
            for prev in reversed(chunk):           # carry the last sentence(s) over for context
                if tn + len(prev.split()) > OVERLAP_W:
                    break
                tail.insert(0, prev); tn += len(prev.split())
            chunk, n = tail, tn
        chunk.append(s); n += w
    if chunk:
        final.append({**p, 'text': ' '.join(chunk), 'part': part})

SOURCES = {
    'A': {'title': 'FIA 2026 F1 Regulations, Section A [General Regulatory Provisions], Issue 03 (25 Jun 2026)',
          'url': 'https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_03_-_2026-06-25.pdf'},
    'B': {'title': 'FIA 2026 F1 Regulations, Section B [Sporting], Issue 09 (1 Oct 2026)',
          'url': 'https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf'},
}
json.dump({'source': SOURCES[SEC], 'passages': final}, open(OUT, 'w'), ensure_ascii=False, indent=0)
lens = [len(p['text'].split()) for p in final]
print(f'{len(final)} passages from {len(passages)} articles · words min {min(lens)} median {sorted(lens)[len(lens)//2]} max {max(lens)}')
