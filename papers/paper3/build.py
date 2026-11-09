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

# Elsevier (JNCA): 3–5 highlights, each ≤ 85 characters including spaces (separate file).
# Check the JNCA Guide for Authors (abstract length, reference style) before the final build.
HIGHLIGHTS = [
    'Replay of 2,000 records recovered {{UNSW_er2000_orders_gap_closed}}–{{TON_er2000_gap_closed}}% of the gap to full retraining',
    'A random forest retrained on the same replay memory beat the CNN with replay',
    'Forgotten attack classes were mostly still flagged as attacks',
    'Replaying memory once per task, not per batch, cut TON-IoT false alarms {{TON_er2000_fpr_final}}→{{P3F_TON_er2000p_fpr_final}}',
    'A random-forest reference showed the CNN needs compressed heavy-tailed inputs',
]
ABSTRACT_MAX_WORDS = 250

FIGURES = {   # section key -> [(path, width_cm, caption)]
    '5.3': [(FIG / 'fig1_task0_retention.png', 16.0,
             'The first task (Task 0) as later tasks are learned, semantic order: class-level retention (mean recall of '
             'Task 0’s attack classes), detection-level retention (share of Task 0’s attack records flagged as any attack) '
             'and the benign false-positive rate. Lines are means over five seeds, bands the range over seeds. Online EWC '
             'and ER-500 are omitted for legibility (Table 2); EWC follows fine-tuning closely. Where fine-tuning and LwF '
             'coincide, the wider LwF line lies underneath. The detection-level axis starts at {{fig1_det_lo}}.'
             '{{P3F_fig1_agem}}')],
}
TABLE_CAPTIONS = {
    '3.2': ['Class-incremental tasks in the semantic order. Training records are those of the task itself, including its '
            'share of the benign training records; attack test records are counted on the full test set, which is never '
            'divided between tasks.'],
    '5.1': ['Results after the last task, semantic order. Final macro-F1 over all classes including benign: mean ± '
            'standard deviation over five seeds; the other columns are means over the seeds. A, FM and plasticity: '
            'average retention, forgetting and retention of each task when it was learned, at class level (mean recall '
            'of a task’s attack classes); A and FM also at detection level (share of a task’s attack records flagged as '
            'any attack). A negative FM means that final retention exceeds the earlier maximum (backward transfer). '
            'Benign FPR: share of benign test records flagged as an attack after the last task. The random forest is '
            'retrained on all task data with the same Task-0 pre-processing and evaluated at the final stage only. '
            'A-GEM: corrected version (Section 4.2).'],
    '5.2': ['Class-level retention of each task after the last task (mean recall of the task’s attack classes, five '
            'seeds), experience replay with 2,000 records against joint retraining.'],
    '5.4': ['Not-yet-learned attack families flagged as attacks (share of their test records) and benign false-positive '
            'rate (FPR), mean over five seeds. After Task 0 all strategies share the same model. {{P3F_ranges_note}}'],
    '5.5': ['Final macro-F1 over the four random task orders (one run each, with seed equal to the order index): mean '
            '(minimum–maximum).',
            'Pre-processing fitted on Task 0 only against fitted on all training data (oracle): final macro-F1 and '
            'difference in not-yet-learned flagging, mean over five seeds.'],
    '5.6': ['Joint retraining on TON-IoT with each combination of input map and training budget: final macro-F1, '
            'mean ± standard deviation over five seeds. Joint retraining uses the Task-0 budget at every stage: at most '
            '40 epochs (patience 5) under the earlier budget and at most 100 epochs (patience 15) under the present one.'],
}


def fill(text, V):
    """Replace {{key}} and {{table:key}} with values.json entries (flat keys; tables stored as 'table:<key>')."""
    missing = set()

    def val(m):
        k = m.group(1)
        if k in V:
            return str(V[k])
        missing.add(k)
        return f'⟦{k}⟧'

    text = re.sub(r'\{\{((?:table:)?\w+)\}\}', val, text)
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


def build(md_text, out, authors=True, V=None):
    """V: values for the {{key}} placeholders in table and figure captions (missing keys stay visible as ⟦key⟧)."""
    capf = (lambda c: fill(c, V)[0]) if V is not None else (lambda c: c)
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
            caption(doc, 'Figure', capf(cap))
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
            caption(doc, 'Table', capf(caps[used.get(current, 0)]), above=True,
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


def main():
    anon = '--anon' in sys.argv
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    V.update(EXTRA)
    meta = FIG / 'fig1_meta.json'   # written by make_figures.py
    if meta.exists():
        V.update({k: v for k, v in json.loads(meta.read_text()).items() if isinstance(v, str)})
    md = MD.read_text(encoding='utf-8')
    if anon:   # double-anonymised copy: no CRediT names, no author-owned repository link
        md = re.sub(r'\*\*CRediT authorship contribution statement\.\*\*[^\n]*\n\n', '', md)
        V['REPO'] = '[repository link removed for anonymous review]'
    text, missing = fill(md, V)
    (HERE / ('Manuscript_filled_anon.md' if anon else 'Manuscript_filled.md')).write_text(text, encoding='utf-8')
    out = HERE / ('Manuscript_v2_anon.docx' if anon else 'Manuscript_v2.docx')
    used = build(text, out, authors=not anon, V=V)
    unused = {k: len(c) - used.get(k, 0) for k, c in TABLE_CAPTIONS.items() if used.get(k, 0) != len(c)}
    cap_missing = sorted({k for caps in list(TABLE_CAPTIONS.values()) + [[c for *_, c in FIGURES[s]] for s in FIGURES]
                          for c in caps for k in fill(c, V)[1]})
    if cap_missing:
        print(f'WARNING {len(cap_missing)} unfilled caption placeholders:', ', '.join(cap_missing))
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
