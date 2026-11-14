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

# Elsevier: 3–5 highlights, each ≤ 85 characters including spaces (separate file)
HIGHLIGHTS = [
    'Fusing two flow-feature streams helps little more than using all features',
    'Late fusion beat one CNN on all features by only {{TON_IoT_d_late_early}} macro-F1 on TON-IoT',
    'Tuned trees beat untuned CNNs in macro-F1; a {{TON_IoT_rf_t25_d12_mb}} MB forest fell below late fusion',
    'Attention fusion: no significant gain; its weights tracked stream usefulness loosely',
    'Min–max inputs silently broke one CNN branch and reversed the stream ranking',
]
ABSTRACT_MAX_WORDS = 250

FIGURES = {
    '4.1': [(FIG / 'fig1_architectures.png', 16.0,
             'The five CNN architectures and one capacity control. Every encoder is the same compact 1D-CNN (two convolutional blocks with batch '
             'normalisation, global average pooling, 64-unit embedding); every model ends with dropout and a softmax layer. Early fusion is one encoder on the concatenated '
             'features; late fusion concatenates two stream embeddings and passes them through a 128-unit dense layer; attention '
             'fusion weights them with a learned two-way softmax, sums them and passes the sum through a 64-unit dense layer. '
             'Early fusion with head (right) adds the 128-unit dense layer of late fusion to early fusion. Not drawn: wide early fusion '
             '(every layer of early fusion widened by 1.6) and late fusion on all features (the late-fusion network with both encoders fed all features).')],
    '5.3': [(FIG / 'fig2_random_splits.png', 16.0,
             'Semantic against random feature splits. Grey dots: test macro-F1 for each of 10 random splits of the features '
             'into two streams of the semantic sizes (one seed each); red diamonds: semantic split (mean of five seeds). Because the grey dots include seed '
             'noise and the diamonds average it out, counts of dots below a diamond slightly favour the semantic split; the within-seed differences in Table 6 '
             'are the like-for-like comparison. s1, s2 = stream 1, stream 2. The y-axis ranges differ '
             'between panels; each spans at least 0.05 macro-F1.')],
    '5.4': [(FIG / 'fig3_attention.png', 16.0,
             'Attention weights against stream usefulness. Each point is a class: x = per-class F1 of the stream-1 CNN minus '
             'that of the stream-2 CNN (mean over seeds); y = mean attention weight on stream 1 for the test records of the '
             'class (mean over seeds; bars = range over seeds). The x-axis ranges differ between panels: on N-BaIoT all '
             'stream advantages lie within about 0.013, so its horizontal spread is tiny.')],
}
TABLE_CAPTIONS = {
    '3.1': ['Datasets and semantic streams. Training records before and after SMOTE oversampling; validation and test sets are not resampled.'],
    '5.1': ['Test macro-F1 (mean ± standard deviation over five seeds) and benign false-positive rate (FPR, mean) on the semantic split. Params = trainable parameters on TON-IoT; with other input lengths the counts differ by up to {{params_dev_max_pct}}.'],
    '5.2': ['Paired differences in test macro-F1 over five seeds: mean, t-based 95% confidence interval and number of seeds with a positive difference. "Better single stream" is stream 1 on all three datasets. The confidence intervals are unadjusted; the Holm-adjusted p-values of the confirmatory family are in Table B.1. Tree rows: difference of means with Welch\'s p.',
            'Per-class test F1 on TON-IoT (mean over five seeds).',
            'Per-class test F1 on UNSW-NB15 (mean over five seeds).'],
    '5.3': ['Semantic against random splits: macro-F1 of the semantic split (mean of five seeds) and distribution over 10 random splits of the same sizes (one seed each). Within-seed difference: semantic run minus the mean of the two random splits trained with the same seed; differences below 10⁻⁶ count as ties.'],
    '5.5': ['Effect of input scaling on the CNNs (test macro-F1, mean over five seeds). The first run used the min–max scaled features; the second row of each dataset repeats it with the quantile transform and nothing else changed (60-epoch cap, best weights restored only when early stopping fired). The third row is the final run (Table 2: 120-epoch budget, best weights always restored). RF, stream 1: untuned RF in the first two rows, tuned RF in the final run.'],
    '5.6': ['CPU inference time per record (batch of one; median over five repeats of the mean of 200 calls) and per 1,000 records (mean of 20 calls), measured in isolation on the laptop CPU. TensorFlow and the GBDT use all cores; the RF uses one thread for single records and all cores for batches. Model sizes are given in Section 5.6.'],
    'A': ['Fusion, multi-view and attention-based IDS studies examined for Section 2.1. Source: FT = full text read, Abs = abstract only (absence of a control in an abstract is weak evidence). Columns: random grouping of features; single network with matched parameters; tree ensemble trained on the same data; results reported over several seeds or repeated runs.'],
    'B': ['Confirmatory family (Section 4.3): paired differences in test macro-F1 over five seeds with unadjusted t-based 95% confidence intervals, unadjusted and Holm-adjusted p-values (eight tests), and the reading against the post hoc ±0.01 margin.',
          'Test records with and without a training match on the selected features (semantic split, mean over five seeds). A match is a training record with the same selected-feature vector; its label was not saved. Accuracy is used for the two subsets because the matched subset lacks or nearly lacks several classes. On TON-IoT only {{TON_IoT_dup_n}} test records have a match.',
          'Macro-F1, macro PR-AUC (one-vs-rest), attack detection rate (attack records flagged as any attack) and benign FPR on the semantic split (mean over five seeds).'],
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
    text = text.replace('= < 0.001', '< 0.001')
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
            assert Path(path).exists(), path
            q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.keep_with_next = True
            q.add_run().add_picture(str(path), width=Cm(width))
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
                    label=f'{current}.{used.get(current, 0) + 1}' if current in ('A', 'B') else None)
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


def main():
    anon = '--anon' in sys.argv
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    V.update(EXTRA)
    md = MD.read_text(encoding='utf-8')
    if anon:   # double-anonymised copy: no CRediT names, no author-owned repository link
        md = re.sub(r'\*\*CRediT authorship contribution statement\.\*\*[^\n]*\n\n', '', md)
        V['REPO'] = '[repository link removed for anonymous review]'
    text, missing = fill(md, V)
    for k, caps in TABLE_CAPTIONS.items():   # captions may hold placeholders too
        TABLE_CAPTIONS[k] = [fill(c, V)[0] for c in caps]
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
