"""Paper 7: fill Manuscript.md from values.json and build Manuscript_v2.docx (draft layout; Springer layout at submission).

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
    'AI_DECLARATION': ('During the preparation of this work the authors used Claude (Anthropic) to write the analysis '
                       'code and to draft the paper. After using this tool, the authors reviewed and edited the content '
                       'as needed and take full responsibility for the content of the published article. The released scripts '
                       'compute every reported number from the result files, and an automated check compares each '
                       'quantitative statement in the text with those files.'),
}
EXTRA['P7:ai_declaration'] = (
    'An AI model (Claude, Anthropic) wrote most of the analysis code of the corrected studies and drafted this paper '
    "under the authors' direction;"
    ' separate instances of the same model acted as the reviewers described in Section 3.2. '
    'Section 5.2 describes the failures this assistance introduced and caught. The authors reviewed and edited all content '
    'and take full responsibility for it. Every number in the paper is computed by released scripts from result files, '
    'and an automated check compares the quantitative statements with those files.')

ABSTRACT_MAX_WORDS = 250
FIGURES = {}
TABLE_CAPTIONS = {
    '1': ['How the main conclusion of each study changed. Original conclusions are paraphrased or quoted from the original manuscripts; numbers after the re-run are from the corrected studies (TON_IoT unless stated).'],
    '4': ['The failure modes. Studies: where the failure occurred ("re-run": introduced during the re-run and caught before submission). Prior coverage: Arp = pitfall of Arp et al. [2022]; K&N = leakage type of Kapoor and Narayanan [2023]; Lones = Lones [2024]; REFORMS = Kapoor et al. [2024]; H = fault category of Humbatova et al. [2020]; New = not listed by the evaluation-pitfall taxonomies.',
          'Prevalence, effect and consequence of each failure mode. "Not isolated": the mode was present but its effect on the conclusion could not be separated from other corrections. "New analysis": the conclusion changed because the re-run asked a new question, not because a failure was corrected.'],
    '5.2': ['Outcomes of the points raised by AI reviewer instances in the review reports kept for Studies 4 to 6 (Study 4 round-1 numbers audit excluded: superseded by a full rebuild). Counting rules and per-report detail are in the released review table.'],
    '6': ['Checklist, ordered by the stage of a study at which each check applies. Cost: approximate effort per study. Flags: failure modes the check would have flagged in this series (authors\' judgement; Section 5.1).'],
}


def minus(text):
    """Use the minus sign for negative numbers in the body (not in the reference list, URLs or DOIs)."""
    body, sep, refs = text.partition('## References')
    body = re.sub(r'(?<![\w/.\-–])-(?=\d)', '−', body)
    return body + sep + refs


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
    text = re.sub(r'\{\{([\w.:\-]+)\}\}', val, text)   # keys may contain '-' (P7:rev_methods-code_*)
    text = text.replace('= < 0.001', '< 0.001')
    if text.startswith('# Paper 7'):           # drop the build note on line 1 from the filled outputs
        text = text.split('\n', 1)[1].lstrip('\n')
    return minus(text), sorted(missing)


def line_numbers(doc):
    sectPr = doc.sections[0]._sectPr
    ln = OxmlElement('w:lnNumType')
    ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); ln.set(qn('w:distance'), '283')
    sectPr.find(qn('w:pgMar')).addnext(ln)


def section_key(h):
    if h.startswith('Appendix '):
        return h.split()[1].rstrip('.')
    m = re.match(r'([0-9]+(?:[.][0-9]+)?) ', h)
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
            if text in ('1 Introduction', 'References'):
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
