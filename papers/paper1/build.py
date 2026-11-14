"""Fill Manuscript.md from values.json and build Manuscript_v2.docx.

Order: compute_values.py -> make_figures.py -> build.py [--anon] -> verify.py
Unfilled placeholders are listed and left visible as ⟦key⟧, so a partial build still works.
"""
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from docx_helpers import add_table, bullet, caption, numbered, para, setup

HERE = Path(__file__).resolve().parent
MD = HERE / 'Manuscript.md'
FIG = HERE / 'figures'

# Authors as on the Telco paper (user, 2026-10-24); details copied from ../../../NEw REsearch/manuscript/build_manuscript.py
AUTHOR = 'Emmanuel Kwame Amoako'
EMAIL = 'amoakopuyol123@gmail.com'
AUTHOR_LINES = (
    'Emmanuel Kwame Amoakoᵃ, Mahamudu Amaduᵇ, James Quaysonᵇ, Kingsford Donkorᶜ, Daniel Asamoah Aprakuᵈ',
    'ᵃ Independent researcher, P.O. Box 121, Dormaa Ahenkro, Ghana',
    'ᵇ Department of Information Technology and Decision Sciences, University of Energy and Natural Resources, Sunyani, Ghana',
    'ᶜ Department of Computer Engineering, University of Energy and Natural Resources, Sunyani, Ghana',
    'ᵈ Department of Computer Science, University of Energy and Natural Resources, Sunyani, Ghana',
)
AUTHOR_NAMES = ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku')
EXTRA = {
    'REPO': 'https://github.com/Amoakoemma123/iot-ids-pipeline',
    # Elsevier wording; user 2026-10-24: Claude used "for writing codes and drafting paper"
    'AI_DECLARATION': ('During the preparation of this work the authors used Claude (Anthropic) to write and edit the manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. The released scripts compute every reported number from the result files, and an automated check compares each quantitative statement in the text with those files.'),
}

# Elsevier: 3–5 highlights, each ≤ 85 characters including spaces (separate file). Written once results exist
# (verify.py reports them as pending while this list is empty).
HIGHLIGHTS = [      # final-only values stay ⟦…⟧ until the p1final queue has run (values.json p1final_complete)
    '{{hl_budget}}',
    'From {{TON_IoT_tree_over_net_sig_kb}} KB (TON-IoT), {{UNSW_NB15_tree_over_net_sig_kb}} KB (UNSW-NB15) a tree significantly beats every network',
    'Int8 cost CNNs at most {{cnn_int8_drop_max_all}} and MLPs up to {{mlp_int8_drop_max_all}} macro-F1 (seed means)',
    'Tied-best TON-IoT tree models took {{TON_IoT_lat_tied_range}} ms/record on one thread, best CNN {{TON_IoT_lat_best_net_ms}} ms',
    'Early-stopped scikit-learn GBDTs keep 20 extra, sometimes diverged, iterations',
]
ABSTRACT_MAX_WORDS = 250

FIGURES = {
    '5.2': [(FIG / 'fig1_accuracy_size.png', 16.0,
             'Test macro-F1 against model size (log scale), one panel per dataset. Filled markers: float32 networks and tree '
             'models (compact encoding, Section 4.2); open markers: the same networks after int8 quantisation. Bars: '
             'standard deviation over five seeds. Dashed line: best mean macro-F1 minus 0.01. Logistic regression, where it '
             'lies below the plotted range, is shown on the lower edge with its value. The ringed point is the forest '
             'configuration of the earlier draft. All GBDTs are rolled back to their best internal-validation iteration; '
             'squares: main grid (both learning rates), crosses: GBDTs with capped iterations; downward triangles: single '
             'decision trees.')],
}
TABLE_CAPTIONS = {
    '3.1': ['Datasets. Training records before and after SMOTE oversampling (N-BaIoT is not oversampled); validation and '
            'test sets are never resampled.'],
    '5.1': ['Test macro-F1 (mean ± standard deviation over five seeds) and model size for selected configurations, including '
            'those named in Section 5.2. Encoding: tree sizes use the compact encoding of Section 4.2, networks four bytes '
            'per parameter (float32) or the TensorFlow Lite file (int8). GBDTs are rolled back to their best '
            'internal-validation iteration. The full grid is given in Appendix A.'],
    '5.4': ['Inference time on one CPU thread (all libraries limited to one thread): per record (batch of one) and per '
            '1,000 records in one call. GBDTs: seed-0 model (the spread over five seeds is given in the text); other '
            'models: seed 0. Networks: TensorFlow (float32) and the TensorFlow Lite interpreter (int8, float input and output).',
            'TensorFlow Lite variants of the seed-0 networks: per-record time of the float32 file, of the int8 model of '
            'Table 2 (float fallback, float input and output) with and without the default XNNPACK delegate, and of a '
            'full-integer int8 model (int8 operations, input and output); file sizes; seed-0 test macro-F1.'],
    '5.6': ['TON-IoT with and without the source and destination port numbers: test macro-F1 (mean over five seeds), '
            'difference (computed before rounding) and Welch p-value. GBDTs rolled back to their best iteration.'],
    'A': ['All configurations, sorted by compact size within each dataset: test macro-F1 (mean ± standard deviation over '
          'five seeds), validation macro-F1 (mean over seeds; * seed 0 only), benign false-positive rate, attack detection '
          'rate, macro PR-AUC (average precision), compact size and pickled scikit-learn size. Compact size: Section 4.2 '
          'encoding for tree models, four bytes per parameter for float32 networks, the TensorFlow Lite file for int8 '
          'networks. GBDTs at both learning rates and with capped iterations are included, all rolled back.',
          'GBDTs of the main grid: test macro-F1 over five seeds (mean and lowest seed) of the learning-rate-0.1 models as '
          'first run (evaluated 20 iterations past their best point) and of both learning rates after rollback to the '
          'best internal-validation iteration, and the range of iterations kept.'],
}

def fill(text, V):
    missing = set()

    def tab(m):
        k = m.group(1)
        if k in V['tables']:
            return V['tables'][k]
        missing.add('table:' + k)
        return f'⟦table:{k}⟧'

    def val(m):
        k = m.group(1)
        if k in V:
            return str(V[k])
        missing.add(k)
        return f'⟦{k}⟧'

    text = re.sub(r'\{\{table:(\w+)\}\}', tab, text)
    text = re.sub(r'\{\{(\w+)\}\}', val, text)
    return text, sorted(missing)


def line_numbers(doc):
    sectPr = doc.sections[0]._sectPr
    ln = OxmlElement('w:lnNumType')
    ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); ln.set(qn('w:distance'), '283')
    sectPr.find(qn('w:pgMar')).addnext(ln)


def section_key(h):
    if h.startswith('Appendix '):
        return h.split()[1].rstrip('.')
    m = re.match(r'(\d+\.\d+|\d+\.)', h)
    return m.group(1) if m else None


def build(md_text, out, authors=True):
    lines = md_text.splitlines()
    doc = Document(); setup(doc)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.5
    line_numbers(doc)
    title_at = lines.index('## Title') + 2
    title = lines[title_at].strip('*')
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph(style='Title'); p.add_run(title); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(24)
    if authors:
        for line in (*AUTHOR_LINES, '', f'Corresponding author: {AUTHOR}, {EMAIL}'):
            para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    else:
        doc.core_properties.author = doc.core_properties.last_modified_by = ''
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    current, pending, in_refs, used = None, [], False, {}

    def flush():
        nonlocal pending
        for path, width, cap in pending:
            q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.keep_with_next = True
            if not FINAL_RESULTS:   # the figure on disk predates the final GBDT re-runs (rollback): mark it pending
                q.add_run('⟦figure pending: regenerate with make_figures.py after the p1final runs⟧')
            elif Path(path).exists():
                q.add_run().add_picture(str(path), width=Cm(width))
            else:   # partial build before the results exist; verify.py reports the marker
                q.add_run(f'⟦figure {Path(path).name} not yet generated⟧')
            caption(doc, 'Figure', cap)
        pending = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if i in (0, title_at) or line.startswith('## Title'):
            i += 1; continue
        if line.startswith('## ') or line.startswith('### '):
            flush()
            level = 1 if line.startswith('## ') else 2
            text = line.lstrip('# ').strip()
            current = section_key(text)
            pending = list(FIGURES.get(current, []))
            if text in ('1. Introduction', 'References'):
                doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            doc.add_paragraph(text, style=f'Heading {level}')
            in_refs = text == 'References'
            i += 1; continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            rows = [[c for c in re.split(r'(?<!\\)\|', r.strip())[1:-1]] for r in block]
            rows = [r for r in rows if not all(re.fullmatch(r'\s*:?-{3,}:?\s*', c) for c in r)]
            caps = TABLE_CAPTIONS.get(current)
            assert caps and used.get(current, 0) < len(caps), f'table without caption in section {current}'
            caption(doc, 'Table', caps[used.get(current, 0)], above=True,
                    label=f'{current}.{used.get(current, 0) + 1}' if current == 'A' else None)
            used[current] = used.get(current, 0) + 1
            add_table(doc, rows)
            continue
        if not line.strip() or line.startswith('<!--'):
            i += 1; continue
        if re.match(r'^\s*- ', line):
            bullet(doc, line.strip()[2:], 1 if line.startswith('  ') else 0); i += 1; continue
        m = re.match(r'^(\d+)\. (.*)', line)
        if m:
            numbered(doc, m.group(1), m.group(2)); i += 1; continue
        q = para(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT if in_refs else WD_ALIGN_PARAGRAPH.JUSTIFY)
        if in_refs:
            q.paragraph_format.left_indent = Cm(1.27); q.paragraph_format.first_line_indent = Cm(-1.27)
        i += 1
    flush()
    doc.save(out)
    return used


FINAL_RESULTS = True


def main():
    global FINAL_RESULTS
    anon = '--anon' in sys.argv
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    FINAL_RESULTS = bool(V.get('p1final_complete'))
    V.update(EXTRA)
    md = MD.read_text(encoding='utf-8')
    if anon:   # double-anonymised copy: no CRediT names, no author-owned repository link
        md = re.sub(r'\*\*CRediT authorship contribution statement\.\*\*[^\n]*\n\n', '', md)
        V['REPO'] = '[repository link removed for anonymous review]'
    text, missing = fill(md, V)
    (HERE / ('Manuscript_filled_anon.md' if anon else 'Manuscript_filled.md')).write_text(text, encoding='utf-8')
    out = HERE / ('Manuscript_v2_anon.docx' if anon else 'Manuscript_v2.docx')
    used = build(text, out, authors=not anon)
    unused = {k: len(c) - used.get(k, 0) for k, c in TABLE_CAPTIONS.items() if used.get(k, 0) != len(c)}
    hl = [fill(h, V)[0] for h in HIGHLIGHTS]
    hd = Document(); setup(hd)
    hd.add_paragraph('Highlights', style='Heading 1')
    for h in hl:
        bullet(hd, h)
    hd.save(HERE / 'Highlights.docx')
    for h in hl:
        if len(h) > 85:
            print(f'WARNING highlight > 85 chars ({len(h)}): {h}')
    abstract = text.split('## Abstract')[1].split('**Keywords')[0]
    aw = len(abstract.split())
    print(f'abstract: {aw} words (limit {ABSTRACT_MAX_WORDS})' + ('  <-- TOO LONG' if aw > ABSTRACT_MAX_WORDS else ''))
    words = len(re.findall(r'\w+', text.split('## References')[0]))
    print(f'saved {out.name}  (~{words} words before references)')
    if unused:
        print('WARNING unused table captions:', unused)
    if missing:
        print(f'WARNING {len(missing)} unfilled placeholders:', ', '.join(missing))


if __name__ == '__main__':
    main()
