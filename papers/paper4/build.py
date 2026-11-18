"""Fill Manuscript.md from values.json and build Manuscript_v2.docx (Paper 4, IEEE IoT Journal draft).

Order: compute_values.py -> make_figures.py -> build.py [--anon] -> verify.py
Unfilled placeholders are listed and left visible as ⟦key⟧. Tables come only from {{table:X}}
placeholders and get the caption TABLE_CAPTIONS[X]; figures are inserted after the paragraph that
first cites them (FIGURES, keyed by figure number).
"""
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from docx_helpers import add_inline, add_table, bullet, numbered, para, setup

ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']


def ieee_caption(doc, label, text, table=False):
    """IEEE style: 'TABLE II' centred above tables, 'Fig. 3.' below figures (static numbers)."""
    p = doc.add_paragraph(style='Caption')
    r = p.add_run(label + ('' if table else '. ')); r.bold = not table
    if table:
        p.add_run().add_break()
    add_inline(p, text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4 if table else 10)
    p.paragraph_format.space_before = Pt(6 if table else 2)
    p.paragraph_format.keep_with_next = table
    return p

HERE = Path(__file__).resolve().parent
MD = HERE / 'Manuscript.md'
SUPP = HERE / 'Supplement.md'
FIG = HERE / 'figures'

AUTHOR = 'Emmanuel Kwame Amoako'
EMAIL = 'amoakopuyol123@gmail.com'
AUTHOR_LINES = (   # as in ../Paper5/build.py
    'Emmanuel Kwame Amoakoᵃ, Mahamudu Amaduᵇ, James Quaysonᵇ, Kingsford Donkorᶜ, Daniel Asamoah Aprakuᵈ',
    'ᵃ Independent researcher, P.O. Box 121, Dormaa Ahenkro, Ghana',
    'ᵇ Department of Information Technology and Decision Sciences, University of Energy and Natural Resources, Sunyani, Ghana',
    'ᶜ Department of Computer Engineering, University of Energy and Natural Resources, Sunyani, Ghana',
    'ᵈ Department of Computer Science, University of Energy and Natural Resources, Sunyani, Ghana',
)
AUTHOR_NAMES = ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku')
CODE_COMMIT = '2a1c16a'   # commit of the public repository that holds the code and results
EXTRA = {
    'REPO': 'https://github.com/Amoakoemma123/iot-ids-pipeline',
    'CODE_COMMIT': f' (commit {CODE_COMMIT})',
    'AI_DECLARATION': ('During the preparation of this work the authors used Claude (Anthropic) to write and edit the manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.'),
}
ABSTRACT_MAX_WORDS = 250
TABNUM = {}

TABLE_CAPTIONS = {
    'TON_main': 'Core results on TON-IoT (N = 4 clients, 50 rounds, five repetitions with redrawn partitions): test macro-F1 at the validation-selected and at the final round (mean ± 95% CI half-width), mean selected round (for the centralised model, in units of two epochs) and benign false-positive rate; local-only rows are means over clients.',
    'UNSW_main': 'Core results on UNSW-NB15, as in Table {{tabref:TON_main}}.',
    'TON_extra': 'Number of clients and normalisation on TON-IoT (α = 0.5 unless stated). The N = 4 row repeats Table I; the scaling rows and the first batch-normalisation rows use three repetitions, the last four rows a second set of five repetitions scored with server-averaged and with federatedly re-estimated statistics (Supplement S6).',
    'UNSW_extra': 'Number of clients and normalisation on UNSW-NB15, as in Table {{tabref:TON_extra}}.',
    'TON_dp': 'Record-level DP-SGD on TON-IoT (α = 0.5; three repetitions, except non-private FedAvg at 50 × 2, which repeats Table I with five). ε: for each repetition the maximum over clients of the RDP-accounted privacy loss for δ = 10⁻⁵, shown as mean (minimum–maximum) over repetitions; ± is the 95% CI half-width. The 50 × 2 rows use the full training budget.',
    'UNSW_dp': 'Record-level DP-SGD on UNSW-NB15, as in Table {{tabref:TON_dp}}.',
    'paired': 'Primary paired comparisons (difference in test macro-F1 at the validation-selected round, same partition in both arms), with p-values of paired t-tests and Holm-adjusted p-values within each dataset; † Holm-adjusted p < 0.05. Five repetitions.',
    'TON_nat': "Host partition of TON-IoT (each destination host assigned to one of four clients, assignment redrawn in each of five repetitions): test macro-F1 on the common test set (mean ± 95% CI half-width, SD over repetitions) and, on each site's own test traffic, macro-F1, attack detection and benign false-positive rate (mean over sites per repetition, then mean ± 95% CI half-width over repetitions). The centralised row repeats Table I.",
    'NB_main': 'N-BaIoT with one client per device (N = 9) and with IID and Dirichlet partitions over nine clients: test macro-F1 on the common test set (mean ± 95% CI half-width over five repetitions, SD) and benign false-positive rate. The algorithms on the device partition are in Supplementary Table S3.',
    'NB_full': 'All N-BaIoT settings, including the algorithms on the device partition, as in Table III of the paper.',
    'pers': 'Personalisation on the device-level partitions (five repetitions; Supplement S5). Own-traffic difference: adapted model minus the site\'s local-only model (mean ± 95% CI half-width over repetitions on TON-IoT, over devices on N-BaIoT); common test: mean macro-F1 over sites; the alert columns score the federated and the adapted model run together (alert if either predicts an attack) on the site\'s own test traffic and on the common test set; the first two rows are single models.',
    'NB_dev': 'N-BaIoT per device: macro-F1 on the device\'s own test traffic of its local model and of FedAvg with one client per device (mean over five repetitions), and of the local model on the common test set.',
}
FIGURES = {
    1: (FIG / 'fig1_paired_differences.png', 9.0, 'Paired differences in test macro-F1 at the validation-selected round (mean and 95% CI over five repetitions; where both arms have a client partition, they share it) on TON-IoT, UNSW-NB15 and N-BaIoT. "Device" is the host partition of TON-IoT and the device partition of N-BaIoT (Section V-C). † marks comparisons significant after Holm correction (Supplementary Table S1).'),
    2: (FIG / 'fig2_normalisation.png', 9.0, 'Validation macro-F1 per round (mean over repetitions, band from minimum to maximum) for FedAvg at α = 0.5 with group normalisation and with batch normalisation, scored with server-averaged running statistics and after federated re-estimation of the statistics (the same five runs), and for the centralised model with batch normalisation. Mean round-to-round swing on TON-IoT (rounds 11–50): {{TON_fedavg_a05_swing}} (group normalisation), {{TON_bnfr_a05_swing}} (server-averaged), {{TON_bnfr_a05_fr_swing}} (re-estimated).'),
    3: (FIG / 'fig3_dp_per_class.png', 9.0, 'Per-class test F1 on TON-IoT with record-level DP-SGD (mean over three repetitions) for the sanity runs and the noise levels of the short schedule (20 rounds × 1 local epoch) and for the full budget (50 × 2); ε as in Table {{tabref:TON_dp}}. UNSW-NB15: Supplementary Fig. S1.'),
}
SUPP_FIGURES = {
    'S1': (FIG / 'figS1_dp_per_class_unsw.png', 9.0, 'Per-class test F1 on UNSW-NB15 with record-level DP-SGD (20 rounds × 1 local epoch, mean over three repetitions); ε as in Table {{tabref:UNSW_dp}}.'),
}




def ieee_layout(doc):
    """IEEE journal look: US letter, IEEE margins, 10-pt Times, single spacing (title block in one column)."""
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
    sec.left_margin = sec.right_margin = Cm(1.57); sec.top_margin = Cm(1.9); sec.bottom_margin = Cm(2.54)
    st = doc.styles
    st['Normal'].font.size = Pt(10); st['Normal'].paragraph_format.line_spacing = 1.0
    st['Normal'].paragraph_format.space_after = Pt(2)
    for name, size in (('Heading 1', 10), ('Heading 2', 10), ('Title', 22), ('Caption', 8)):
        st[name].font.size = Pt(size)
    st['Heading 1'].paragraph_format.space_before = Pt(8); st['Heading 2'].paragraph_format.space_before = Pt(6)


def two_columns(sec):
    cols = sec._sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols'); sec._sectPr.append(cols)
    cols.set(qn('w:num'), '2'); cols.set(qn('w:space'), str(int(0.42 / 2.54 * 1440)))


def minus(t):
    """Typographic minus sign for negative numbers (−0.121, [−0.284, −0.074])."""
    return re.sub(r'(?<![\w.])-(?=\d)', '−', t)


def table_order(path):
    return re.findall(r'\{\{table:(\w+)\}\}', path.read_text(encoding='utf-8'))


def fill(text, V, supp=False):
    """{{tabref:X}}: number of table X in this document (falls back to the main paper for the supplement);
    {{stabref:X}}: supplement table number (S1, S2, ...)."""
    missing = set()

    def tab(m):
        k = m.group(1)
        if k in V['tables']:
            return f'<!--TABLE:{k}-->\n' + minus(V['tables'][k])
        missing.add('table:' + k)
        return f'⟦table:{k}⟧'

    def val(m):
        k = m.group(1)
        if k in V:
            return minus(str(V[k]))
        missing.add(k)
        return f'⟦{k}⟧'

    main_num = {k: ROMAN[i] for i, k in enumerate(table_order(MD))}
    supp_num = {k: f'S{i + 1}' for i, k in enumerate(table_order(SUPP))}
    V['_tabnum'] = {**main_num, **supp_num} if supp else main_num
    text = re.sub(r'\{\{tabref:(\w+)\}\}', lambda m: V['_tabnum'].get(m.group(1), '⟦tabref:' + m.group(1) + '⟧'), text)
    text = re.sub(r'\{\{stabref:(\w+)\}\}', lambda m: supp_num.get(m.group(1), '⟦stabref:' + m.group(1) + '⟧'), text)
    text = re.sub(r'\{\{table:(\w+)\}\}', tab, text)
    text = re.sub(r'\{\{(\w+)\}\}', val, text)
    return text, sorted(missing)


def build(md_text, out, authors=True, ieee=False, supp=False, V=None):
    lines = md_text.splitlines()
    doc = Document(); setup(doc)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.5
    if ieee:
        ieee_layout(doc)
    title_at = lines.index('## Title') + 2
    p = doc.add_paragraph(style='Title'); p.add_run(lines[title_at].strip('*')); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if authors:
        for line in (*AUTHOR_LINES, f'Corresponding author: {AUTHOR}, {EMAIL}'):
            para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    else:
        doc.core_properties.author = doc.core_properties.last_modified_by = ''
    in_refs, next_caption, n_tab = False, None, 0
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if i in (0, title_at) or line.startswith('## Title'):
            i += 1; continue
        m = re.match(r'<!--TABLE:(\w+)-->', line)
        if m:
            cap = re.sub(r'\{\{tabref:(\w+)\}\}', lambda mm: TABNUM.get(mm.group(1), '?'), TABLE_CAPTIONS.get(m.group(1), m.group(1)))
            next_caption = (TABNUM.get(m.group(1), '?'), cap); i += 1; continue
        if line.startswith('## ') or line.startswith('### '):
            level = 1 if line.startswith('## ') else 2
            text = line.lstrip('# ').strip()
            if ieee and text.startswith('I. '):
                two_columns(doc.add_section(WD_SECTION.CONTINUOUS))
            doc.add_paragraph(text, style=f'Heading {level}')
            in_refs = text == 'References'
            i += 1; continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            rows = [[c for c in re.split(r'(?<!\\)\|', r.strip())[1:-1]] for r in block]
            rows = [r for r in rows if not all(re.fullmatch(r'\s*:?-{3,}:?\s*', c) for c in r)]
            assert next_caption, 'table without caption'
            ieee_caption(doc, f'TABLE {next_caption[0]}', next_caption[1], table=True); next_caption = None; n_tab += 1
            add_table(doc, rows, total_cm=8.6 if ieee else 16.0, size=7 if ieee else None)
            continue
        if not line.strip() or line.startswith('<!--'):
            i += 1; continue
        mnum = re.match(r'^(\d+)\. (.*)', line)
        if mnum and not in_refs:
            numbered(doc, mnum.group(1), mnum.group(2)); i += 1; continue
        q = para(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT if in_refs else WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=(1 if in_refs else 2) if ieee else 6,
                 size=8 if (ieee and in_refs) else None)   # IEEE: references in 8 pt
        if in_refs:
            q.paragraph_format.left_indent = Cm(0.9); q.paragraph_format.first_line_indent = Cm(-0.9)
        figs = SUPP_FIGURES if supp else FIGURES
        for fno, val in list(figs.items()):
            if str(fno).startswith('_done'):
                continue
            path, width, cap = val
            if re.search(r'Fig\. ' + str(fno) + r'\b', line) and not figs.get(f'_done{fno}'):
                fp = doc.add_paragraph(); fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                fp.add_run().add_picture(str(path), width=Cm(8.6 if ieee else width))
                cap = re.sub(r'\{\{tabref:(\w+)\}\}', lambda mm: TABNUM.get(mm.group(1), '?'), cap)
                cap = minus(re.sub(r'\{\{(\w+)\}\}', lambda mm: str((V or {}).get(mm.group(1), '⟦' + mm.group(1) + '⟧')), cap))
                ieee_caption(doc, f'Fig. {fno}', cap)
                figs[f'_done{fno}'] = True
        i += 1
    doc.save(out)
    return n_tab


def main():
    anon = '--anon' in sys.argv
    ieee = '--ieee' in sys.argv
    supp = '--supp' in sys.argv
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    V.update(EXTRA)
    md = (SUPP if supp else MD).read_text(encoding='utf-8')
    if anon:
        V['REPO'] = '[repository link removed for anonymous review]'; V['CODE_COMMIT'] = ''
    text, missing = fill(md, V, supp=supp)
    global TABNUM
    TABNUM = V.pop('_tabnum')
    stem = 'Supplement' if supp else 'Manuscript'
    (HERE / (f'{stem}_filled_anon.md' if anon else f'{stem}_filled.md')).write_text(text, encoding='utf-8')
    out = HERE / (f'{stem}_v2_anon.docx' if anon else f'{stem}_v2_ieee.docx' if ieee else f'{stem}_v2.docx')
    n_tab = build(text, out, authors=not anon, ieee=ieee, supp=supp, V=V)
    if not supp:
        abstract = text.split('## Abstract')[1].split('**Index Terms')[0]
        aw = len(abstract.split())
        print(f'abstract: {aw} words (limit {ABSTRACT_MAX_WORDS})' + ('  <-- TOO LONG' if aw > ABSTRACT_MAX_WORDS else ''))
    words = len(re.findall(r'\w+', text.split('## References')[0]))
    print(f'saved {out.name}  ({n_tab} tables, ~{words} words before references)')
    if missing:
        print(f'WARNING {len(missing)} unfilled placeholders:', ', '.join(missing))

if __name__ == '__main__':
    main()
