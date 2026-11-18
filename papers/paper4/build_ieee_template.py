# -*- coding: utf-8 -*-
"""Build Paper 4 in the official IEEE Transactions Word template (downloaded by the user from IEEE).

Run after build.py (same values.json, same filled text):  python build_ieee_template.py
Output: Manuscript_IEEE_template.docx. The template's page set-up, styles, header, first-page footnote and
section layout (one-column title block, two-column body) are kept; its instruction text is replaced by the
paper. Paragraph formats are copied from the template's own example paragraphs (title, authors, abstract,
index terms, heading 1/2, drop-cap first paragraph, body text, table title, references).
"""
import copy
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import build as B                                    # noqa: E402  (captions, figures, fill)
from docx_helpers import add_inline, add_table       # noqa: E402
from title_pages import AFFIL, AUTHOR_INFO           # noqa: E402  (names, affiliations, e-mails)

TEMPLATE = HERE / 'ieee_template' / 'Transactions-template-and-instructions-on-how-to-create-your-article-formatted (4).docx'
OUT = HERE / 'Manuscript_IEEE_template.docx'
RUNNING_HEAD = 'FEDERATED INTRUSION DETECTION FOR IOT NETWORKS REVISITED'
W = qn


def strip_runs(p_el):
    """keep only the paragraph properties"""
    for ch in list(p_el):
        if ch.tag != W('w:pPr'):
            p_el.remove(ch)
    return p_el


def ppr_of(par):
    ppr = par._p.find(W('w:pPr'))
    return copy.deepcopy(ppr) if ppr is not None else OxmlElement('w:pPr')


def no_numbering(ppr):
    num = ppr.find(W('w:numPr'))
    if num is not None:
        ppr.remove(num)
    num = OxmlElement('w:numPr')
    ilvl = OxmlElement('w:ilvl'); ilvl.set(W('w:val'), '0')
    nid = OxmlElement('w:numId'); nid.set(W('w:val'), '0')
    num.append(ilvl); num.append(nid)
    # numPr must follow pStyle/keepNext/... ; inserting right after pStyle (or first) is schema-valid enough for Word
    ps = ppr.find(W('w:pStyle'))
    (ps.addnext(num) if ps is not None else ppr.insert(0, num))
    return ppr


def main():
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    V.update(B.EXTRA)
    text, missing = B.fill(B.MD.read_text(encoding='utf-8'), V)
    assert not [m for m in missing if not m.startswith('P4F_')], missing
    B.TABNUM = V.pop('_tabnum')
    lines = text.splitlines()

    doc = Document(str(TEMPLATE))
    if 'Table Grid' not in [s.name for s in doc.styles]:     # the template has no grid table style
        grid = next(s for s in Document().styles if s.name == 'Table Grid')
        doc.styles.element.append(copy.deepcopy(grid.element))
    P = doc.paragraphs
    expect = {0: 'Preparation of Articles', 2: 'First A. Author', 6: 'Abstract', 8: 'Index Terms', 10: 'I. INTRODUCTION',
              11: 'T', 12: 'HIS document', 13: 'This template is a guide', 39: 'TABLE I', 40: 'This is a Sample'}
    for i, s in expect.items():
        assert P[i].text.startswith(s), (i, P[i].text[:40])
    h2 = next(p for p in P if p.style.name == 'Heading 2')
    ref = next(p for p in P if p.text.startswith('J. U. Duncombe'))
    PPR = {'h1': ppr_of(P[10]), 'drop': ppr_of(P[11]), 'first': ppr_of(P[12]), 'body': ppr_of(P[13]),
           'h2': no_numbering(ppr_of(h2)), 'tablabel': ppr_of(P[39]), 'tabtitle': ppr_of(P[40]), 'ref': ppr_of(ref)}
    no_numbering(PPR['ref'])
    ind = PPR['ref'].find(W('w:ind'))
    ind.set(W('w:left'), '400'); ind.set(W('w:hanging'), '400')

    # ---- title block: replace the text of the template's own paragraphs
    title_at = lines.index('## Title') + 2
    title = lines[title_at].strip('*')
    t = strip_runs(P[0]._p); r = P[0].add_run(title)
    names = [a[0] for a in AUTHOR_INFO]
    strip_runs(P[2]._p)
    r = P[2].add_run(', '.join(names[:-1]) + ', and ' + names[-1]); r.font.size = Pt(11)
    ab = text.split('## Abstract')[1].split('**Index Terms')[0].strip()
    strip_runs(P[6]._p)
    r = P[6].add_run('Abstract—'); r.bold = r.italic = True; r.font.size = Pt(9)
    add_inline(P[6], ab, size=9, bold=True)
    kw = re.search(r'\*\*Index Terms:\*\*\s*(.+)', text).group(1).strip()
    strip_runs(P[8]._p)
    r = P[8].add_run('Index Terms—'); r.bold = r.italic = True; r.font.size = Pt(9)
    r = P[8].add_run(kw[0].upper() + kw[1:] + '.'); r.bold = True; r.font.size = Pt(9)

    # ---- delete the template's instruction text (everything after the index terms paragraph and its spacer)
    body = doc.element.body
    keep_until = P[9]._p
    after = False
    for el in list(body):
        if after and el.tag != W('w:sectPr'):
            body.remove(el)
        if el is keep_until:
            after = True

    def newp(key, txt='', size=None):
        p = doc.add_paragraph()
        old = p._p.find(W('w:pPr'))
        if old is not None:
            p._p.remove(old)
        p._p.insert(0, copy.deepcopy(PPR[key]))
        if txt:
            add_inline(p, txt, size=size)
        return p

    def caption(label, txt, table):
        if table:
            p = newp('tablabel'); r = p.add_run(label); r.font.size = Pt(8)
            p.paragraph_format.keep_with_next = True; p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.first_line_indent = Cm(0)
            q = newp('tabtitle', txt, size=8)
            q.paragraph_format.keep_with_next = True; q.paragraph_format.first_line_indent = Cm(0)
            q.paragraph_format.keep_together = True; p.paragraph_format.keep_together = True
            q.paragraph_format.space_after = Pt(3)
            for run in q.runs:
                run.font.small_caps = True
        else:
            q = newp('body'); r = q.add_run(label + '. '); r.font.size = Pt(8)
            add_inline(q, txt, size=8)
            q.paragraph_format.first_line_indent = Cm(0); q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            q.paragraph_format.space_after = Pt(8)

    def ieee_table(rows):
        tb = add_table(doc, rows, total_cm=8.6, size=7)
        for tc in tb._tbl.iter(W('w:tc')):           # IEEE tables: no shading
            pr = tc.find(W('w:tcPr'))
            if pr is not None:
                for shd in pr.findall(W('w:shd')):
                    pr.remove(shd)
        for k, row in enumerate(tb.rows):             # keep the whole table (and its caption) in one column
            for c in row.cells:
                for q in c.paragraphs:
                    q.paragraph_format.keep_with_next = k < len(tb.rows) - 1
            trpr = row._tr.get_or_add_trPr()
            cs = OxmlElement('w:cantSplit'); cs.set(W('w:val'), 'true'); trpr.append(cs)
        return tb

    start = next(i for i, l in enumerate(lines) if l.startswith('## I. '))
    i, in_refs, next_cap, first_after_intro = start, False, None, False
    figs = {k: v for k, v in B.FIGURES.items()}
    done = set()
    while i < len(lines):
        line = lines[i].rstrip()
        m = re.match(r'<!--TABLE:(\w+)-->', line)
        if m:
            cap = re.sub(r'\{\{tabref:(\w+)\}\}', lambda mm: B.TABNUM.get(mm.group(1), '?'), B.TABLE_CAPTIONS[m.group(1)])
            next_cap = (B.TABNUM[m.group(1)], cap); i += 1; continue
        if line.startswith('## ') or line.startswith('### '):
            txt = line.lstrip('# ').strip()
            if txt == 'Declarations':
                newp('h1', 'Declarations'); i += 1; continue
            newp('h1' if line.startswith('## ') else 'h2', txt)
            in_refs = txt == 'References'
            first_after_intro = txt.startswith('I. ')
            i += 1; continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            rows = [[c for c in re.split(r'(?<!\\)\|', r.strip())[1:-1]] for r in block]
            rows = [r for r in rows if not all(re.fullmatch(r'\s*:?-{3,}:?\s*', c) for c in r)]
            assert next_cap, 'table without caption'
            caption(f'TABLE {next_cap[0]}', next_cap[1], table=True); next_cap = None
            ieee_table(rows)
            continue
        if not line.strip() or line.startswith('<!--'):
            i += 1; continue
        if in_refs:
            mref = re.match(r'^(\[\d+\])\s+(.*)', line)
            p = newp('ref'); r = p.add_run(mref.group(1) + '\t'); r.font.size = Pt(8)
            add_inline(p, mref.group(2), size=8)
            i += 1; continue
        mnum = re.match(r'^(\d+)\. (.*)', line)
        if first_after_intro:                         # IEEE drop cap + small caps for the rest of the first word
            first_after_intro = False
            word, rest = line.split(' ', 1)
            d = newp('drop'); d.add_run(word[0])
            for rr in d.runs:
                rr.font.small_caps = True; rr.font.size = Pt(48.5)
                rpr = rr._r.get_or_add_rPr(); pos = OxmlElement('w:position'); pos.set(W('w:val'), '-9'); rpr.append(pos)
            p = newp('first'); r = p.add_run(word[1:].upper() + ' '); r.font.small_caps = True
            add_inline(p, rest)
        elif mnum:
            p = newp('body', f'{mnum.group(1)}) {mnum.group(2)}')
        else:
            p = newp('body', line)
        for fno, (path, width, cap) in figs.items():
            if fno not in done and re.search(r'Fig\. ' + str(fno) + r'\b', line):
                fp = newp('body'); fp.paragraph_format.first_line_indent = Cm(0)
                fp.alignment = WD_ALIGN_PARAGRAPH.CENTER; fp.paragraph_format.keep_with_next = True
                fp.add_run().add_picture(str(path), width=Cm(8.6))
                cap = re.sub(r'\{\{tabref:(\w+)\}\}', lambda mm: B.TABNUM.get(mm.group(1), '?'), cap)
                cap = B.minus(re.sub(r'\{\{(\w+)\}\}', lambda mm: str(V.get(mm.group(1), '⟦' + mm.group(1) + '⟧')), cap))
                caption(f'Fig. {fno}', cap, table=False)
                done.add(fno)
        i += 1
    assert done == set(figs), done

    # ---- first-page footnote (affiliations) and running head
    fpart = next(rel.target_part for rel in doc.part.rels.values() if rel.reltype.endswith('/footnotes'))
    fx = fpart.blob.decode('utf-8')
    m = re.search(r'(<w:footnote [^>]*w:id="1"[^>]*>)(.*?)(</w:footnote>)', fx, flags=re.S)
    paras = re.findall(r'<w:p[ >].*?</w:p>', m.group(2), flags=re.S)

    def fn_para(proto, txt, keep_ref=False):
        ppr = re.search(r'<w:pPr>.*?</w:pPr>', proto, flags=re.S)
        refrun = re.search(r'<w:r>(?:(?!</w:r>).)*?<w:footnoteRef/>.*?</w:r>', proto, flags=re.S) if keep_ref else None
        esc = txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        run = f'<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr><w:t xml:space="preserve">{esc}</w:t></w:r>'
        head = re.match(r'<w:p[^>]*>', proto).group(0)
        return head + (ppr.group(0) if ppr else '') + (refrun.group(0) if refrun else '') + run + '</w:p>'

    by_aff = {}
    for name, key, _, mail, _ in AUTHOR_INFO:
        by_aff.setdefault(key, []).append((name, mail))
    fn = ['This work did not receive any specific grant from funding agencies in the public, commercial, or '
          f'not-for-profit sectors. (Corresponding author: {AUTHOR_INFO[0][0]}.)']
    for key, people in by_aff.items():
        who = ' and '.join(n for n, _ in people)
        verb = 'are' if len(people) > 1 else 'is'
        where = AFFIL[key]
        where = ('an independent researcher, ' + where.split(', ', 1)[1]) if where.startswith('Independent') else 'with the ' + where
        mails = ', '.join(mm for _, mm in people)
        fn.append(f'{who} {verb} {where} (e-mail: {mails}).')
    fn.append('Supplementary material for this article is available online.')
    new = ''.join(fn_para(paras[min(k, len(paras) - 1)], s, keep_ref=(k == 0)) for k, s in enumerate(fn))
    fx = fx[:m.start(2)] + new + fx[m.end(2):]
    fpart._blob = fx.encode('utf-8')

    for sec in doc.sections:
        for hp in sec.header.paragraphs:
            ts = hp._p.findall('.//' + W('w:t'))
            joined = ''.join(x.text or '' for x in ts)
            if 'REPLACE THIS LINE' in joined:
                hit = False
                for x in ts:
                    if 'REPLACE' in (x.text or ''):
                        close = x.text.rstrip().endswith('<')
                        x.text = x.text.split('REPLACE')[0] + RUNNING_HEAD + (' <' if close else ''); hit = not close
                    elif hit and '<' not in (x.text or ''):
                        x.text = ''
                    elif hit:
                        x.text = ' <' if x.text.strip().startswith('<') else x.text; hit = False

    cp = doc.core_properties
    cp.title, cp.author, cp.last_modified_by, cp.subject, cp.keywords, cp.comments = title, AUTHOR_INFO[0][0], '', '', kw, ''
    doc.save(str(OUT))
    print('saved', OUT.name)

    # ---- same content as the IEEE build of build.py?
    def words(dname):
        d = Document(str(HERE / dname))
        allp = [x.text for x in d.paragraphs]
        k = next(j for j, s in enumerate(allp) if re.match(r'I\. (Introduction|INTRODUCTION)$', s.strip()))
        txt = ' '.join(allp[k:]) + ' ' + ' '.join(c.text for t in d.tables for r in t.rows for c in r.cells)
        return re.findall(r'[\w.−±%-]+', txt)
    a, b = words('Manuscript_v2_ieee.docx'), words(OUT.name)
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    diff = (ca - cb) + (cb - ca)
    print(f'content check vs Manuscript_v2_ieee.docx: {len(a)} vs {len(b)} tokens; differing tokens: {dict(diff) if diff else "none"}')


if __name__ == '__main__':
    main()
