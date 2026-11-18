"""Word-building helpers, copied from the Telco paper build_docx.py (E:/Research_Works/NEw REsearch)."""
import re

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
INK = RGBColor(0x0B, 0x0B, 0x0B)
HEAD_FILL = "DCE7F5"

def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def add_field(paragraph, instr, placeholder="1"):
    """Insert a complex field (e.g. 'SEQ Figure \\* ARABIC', 'PAGE')."""
    def r(tag, **attrs):
        run = OxmlElement("w:r")
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        run.append(el)
        return run
    paragraph._p.append(r("w:fldChar", **{"w:fldCharType": "begin"}))
    instr_run = OxmlElement("w:r")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "
    instr_run.append(it); paragraph._p.append(instr_run)
    paragraph._p.append(r("w:fldChar", **{"w:fldCharType": "separate"}))
    t_run = paragraph.add_run(placeholder)
    paragraph._p.append(r("w:fldChar", **{"w:fldCharType": "end"}))
    return t_run


INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`|\*[^*]+?\*)")


def add_inline(paragraph, text, size=None, bold=None):
    text = text.replace("\\|", "|")
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2]); run.bold = True
            # nested italics inside bold, e.g. **...*x*...**
            if "*" in run.text:
                run.text = run.text.replace("*", "")
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1]); run.font.name = "Consolas"
            run.font.size = Pt((size or 12) - 1.5)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = paragraph.add_run(part[1:-1]); run.italic = True
        else:
            run = paragraph.add_run(part)
        if size and run.font.name != "Consolas":
            run.font.size = Pt(size)
        if bold is not None and bold:
            run.bold = True
    return paragraph


def para(doc, text="", style=None, align=None, space_after=6, size=None):
    p = doc.add_paragraph(style=style)
    if text:
        add_inline(p, text, size=size)
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    return p


_SEQ = {}


def caption(doc, kind, text, above=False, label=None):
    """Numbered caption. The SEQ field's cached result is the true running number, so the numbers are right even if
    Word never updates fields. label (e.g. 'A.1') gives a fixed number instead (appendix tables)."""
    p = doc.add_paragraph(style="Caption")
    r = p.add_run(f"{kind} "); r.bold = True
    if label:
        num = p.add_run(label)
    else:
        n = _SEQ.setdefault(id(doc), {}).get(kind, 0) + 1
        _SEQ[id(doc)][kind] = n
        num = add_field(p, f"SEQ {kind} \\* ARABIC", placeholder=str(n))
    num.bold = True
    r2 = p.add_run(". "); r2.bold = True
    add_inline(p, text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if not above else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(4 if above else 10)
    p.paragraph_format.space_before = Pt(6 if above else 2)
    p.paragraph_format.keep_with_next = above
    return p


def add_table(doc, rows, total_cm=16.0):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    size = 10 if ncol <= 6 else 9 if ncol <= 8 else 8   # wide tables: smaller type so words do not break
    # width weighting by content length
    # weight = longest single word (can't wrap) vs. average cell length (wraps fine)
    lens = []
    for i in range(ncol):
        cells = [r[i].replace("*", "").replace("`", "") for r in rows]
        longest_word = max(len(w) for c in cells for w in (c.split() or [""]))
        lens.append(max(longest_word * 1.15, sum(map(len, cells)) / len(cells), 7))
    widths = [total_cm * l / sum(lens) for l in lens]
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j])
            c.paragraphs[0].text = ""
            add_inline(c.paragraphs[0], val.strip(), size=size, bold=(i == 0))
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            # short tables stay on one page
            c.paragraphs[0].paragraph_format.keep_with_next = len(rows) <= 10 and i < len(rows) - 1
            if i == 0:
                set_cell_shading(c, HEAD_FILL)
    # repeat header row
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def setup(doc):
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.5))
    sec.top_margin, sec.bottom_margin = Cm(2.3), Cm(2.3)
    st = doc.styles
    normal = st["Normal"]
    normal.font.name = FONT; normal.font.size = Pt(12); normal.font.color.rgb = INK
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.line_spacing = 1.15
    for name, size in (("Heading 1", 14), ("Heading 2", 12.5), ("Heading 3", 12), ("Title", 18)):
        s = st[name]; s.font.name = FONT; s.font.size = Pt(size); s.font.bold = True
        s.font.color.rgb = INK; s.font.italic = False
        rpr = s.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rfonts.get(qn(a)) is not None:
                del rfonts.attrib[qn(a)]
        s.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        s.paragraph_format.space_after = Pt(6)
    cap = st["Caption"]; cap.font.name = FONT; cap.font.size = Pt(10); cap.font.italic = False
    cap.font.color.rgb = INK
    # footer page number
    fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(fp, "PAGE")


def numbered(doc, num, text, indent_cm=0.9):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(indent_cm); pf.first_line_indent = Cm(-0.6); pf.space_after = Pt(4)
    p.add_run(f"{num}.\t")
    pf.tab_stops.add_tab_stop(Cm(indent_cm))
    add_inline(p, text)
    return p


def bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    add_inline(p, text)
    p.paragraph_format.space_after = Pt(3)
    return p


# ---------------------------------------------------------------- main
