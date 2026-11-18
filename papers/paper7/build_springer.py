# -*- coding: utf-8 -*-
"""Paper 7 in the Springer svjour3 LaTeX class (International Journal of Information Security).

Reads the same Manuscript.md + values.json as build.py (so text and numbers cannot drift), writes
springer/Manuscript_Springer[_anon].tex, compiles it with pdflatex and checks the log.
- citations: ACM author-year text ("Arp et al. [2022]", "[Kapoor et al. 2024]") -> natbib \\citet / \\citep
- references: ACM format -> Springer basic author-year format ("Abadi M, Chu A, ... (2016) Title. Venue")
- CCS concepts (ACM only) dropped; declarations in Springer's "Statements and Declarations" order.
Run: python build_springer.py [--anon]
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build as B                                       # noqa: E402  (fill(), EXTRA, TABLE_CAPTIONS)
from title_pages import AUTHOR_INFO, AFFIL              # noqa: E402

PKG = Path(r'E:\Research_Works\IDS_IOT_Research\Submissions\Paper7_Int_J_Information_Security'
           r'\1633537_svjour3-Latex-package')
OUT = HERE / 'springer'
PDFLATEX = Path.home() / 'AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe'
JOURNAL = 'International Journal of Information Security'
RUNNING_TITLE = 'Failure modes found by re-running an ML IoT intrusion detection series'
KEYWORDS = ['Intrusion detection', 'Internet of Things', 'Machine learning', 'Evaluation methodology',
            'Reproducibility', 'Data leakage']      # Springer asks for 4-6 keywords (Manuscript.md lists 7)
ANON = '--anon' in sys.argv
TEXTWIDTH_PT = 11.9 / 2.54 * 72.27                    # svglov3.clo, smallextended (\@smallexttrue): \textwidth = 11.9 cm
CHAR_PT = 4.9                                         # width of one character at \footnotesize (bold, generous)
PARTICLES = {'van', 'de', 'der', 'von', 'da', 'del', 'di', 'le', 'la', 'De', 'Van', 'Von'}   # "Filip De Turck" -> De Turck F

UNICODE = {   # characters pdflatex/T1 does not handle by itself
    'ε': r'\ensuremath{\varepsilon}', 'α': r'\ensuremath{\alpha}', 'β': r'\ensuremath{\beta}',
    'δ': r'\ensuremath{\delta}', 'λ': r'\ensuremath{\lambda}', '×': r'\ensuremath{\times}',
    '±': r'\ensuremath{\pm}', '→': r'\ensuremath{\rightarrow}', '−': r'\ensuremath{-}',
    '∞': r'\ensuremath{\infty}', '≤': r'\ensuremath{\leq}', '≥': r'\ensuremath{\geq}',
    '⁵': r'\textsuperscript{5}', '⁻': r'\textsuperscript{\ensuremath{-}}', '†': r'\dag{}', '…': r'\ldots{}',
    '⟦': r'\textbf{[[', '⟧': r']]}',
}
SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
           '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '|': r'\textbar{}'}


def esc(s):
    s = re.sub(r'"([^"\n]+)"', '\x01\\1\x02', s)                 # "quoted" -> ``quoted''
    s = ''.join(SPECIAL.get(c, c) for c in s)
    return s.replace('\x01', '``').replace('\x02', "''")


# ---------------------------------------------------------------- references
def split_names(authors):
    authors = authors.replace(', and ', ', ').replace(' and ', ', ')
    return [a.strip() for a in authors.split(', ') if a.strip()]


def surname_of(name):
    toks = name.split()
    if 'y' in toks[1:-1]:                              # Spanish compound surname: "Blaise Agüera y Arcas"
        i = toks.index('y') - 1
        return ' '.join(toks[i:]), toks[:i]
    for i, t in enumerate(toks[:-1]):
        if t in PARTICLES and i > 0:                   # never the first given name
            return ' '.join(toks[i:]), toks[:i]
    return toks[-1], toks[:-1]


def springer_name(name):
    if ' ' not in name:                    # corporate author, e.g. "Keras"
        return name
    sur, given = surname_of(name)
    ini = ''.join(p[0] for g in given for p in g.replace('.', ' ').replace('-', ' ').split())
    return f'{sur} {ini}'


def parse_ref(r):
    m = re.match(r'(.+?)\. ((?:19|20)\d\d)\. (.+)$', r)
    assert m, r
    authors, year, rest = m.groups()
    cands = [i for i in (rest.find('? In '), rest.find('. ')) if i >= 0]
    cut = min(cands)
    title = rest[:cut + 1] if rest[cut] == '?' else rest[:cut]
    venue = rest[cut + 2:].strip()
    return split_names(authors), year, title, venue


def springer_venue(v):
    if v.startswith('In '):
        v = 'In: ' + v[3:]
    v = re.sub(r'^(.+?) (\d+), ([\d–]+) \((\d{4})\), ([\w–]+)\.', r'\1 \2(\3):\5.', v)      # journal, vol(no):pp
    v = re.sub(r'^(.+?) (\d+) \((\d{4})\), ([\w–]+)\.', r'\1 \2:\4.', v)                    # journal, vol:pp
    v = re.sub(r'^(.+?) (\d+), ([\d–]+) \((\d{4})\)\.', r'\1 \2(\3).', v)                   # journal, no pages
    v = re.sub(r'([,.]) (\d+)–(\d+)\.(?= |$)', r', pp \2–\3.', v)                           # proceedings pages
    v = re.sub(r'^arXiv:(\S+)\.', r'arXiv preprint arXiv:\1.', v)
    return v


def tex_ref_text(s):
    """Escape a reference string but keep URLs as \\url{}."""
    parts = re.split(r'(https?://\S+?)(?=\.?$|\s)', s)
    return ''.join(rf'\url{{{p}}}' if p.startswith('http') else latex_inline(p) for p in parts if p)


def build_refs(ref_lines):
    keys, items = {}, []
    for r in ref_lines:
        names, year, title, venue = parse_ref(r)
        sur = surname_of(names[0])[0] if ' ' in names[0] else names[0]
        key = re.sub(r'[^a-z]', '', sur.lower()) + year
        assert key not in items and (sur.split()[-1], year) not in keys, f'duplicate citation key {key}'
        keys[(sur.split()[-1], year)] = key
        sur2 = (surname_of(names[1])[0] if ' ' in names[1] else names[1]) if len(names) == 2 else ''
        label = sur if len(names) == 1 else f'{sur} and {sur2}' if len(names) == 2 else f'{sur} et~al.'
        sn = [springer_name(n) for n in names]
        auth = ', '.join(sn[:3]) + ', et al' if len(sn) > 10 else ', '.join(sn)
        end = '' if title.endswith(('?', '!')) else '.'
        text = f'{auth} ({year}) {title}{end} {springer_venue(venue)}'
        items.append((key, label.replace('~', ' '), year, text))
        LABELS[key] = (latex_inline(label.replace('~', ' ')).replace('et al.', 'et~al.'), year)
    return keys, items


# ---------------------------------------------------------------- inline text and citations
CITES = []
LABELS = {}   # key -> (author label, year)


def cite_sub(text, keys):
    """Replace ACM author-year citations by sentinels \x00n\x00 holding \\citet / \\citep commands."""
    def key_for(name, year):
        sur = name.split(' et al.')[0].split(' and ')[0].strip()
        sur = sur.split()[-1]
        return keys.get((sur, year))

    def store(cmd):
        CITES.append(cmd)
        return f'\x00{len(CITES) - 1}\x00'

    def paren(m):
        out = []
        for c in m.group(1).split(';'):
            mm = re.match(r"\s*(.+?) (\d{4})\s*$", c)
            k = mm and key_for(mm.group(1), mm.group(2))
            if not k:
                return m.group(0)
            out.append(k)
        return store('(' + '; '.join(rf'\hyperlink{{{k}}}{{{LABELS[k][0]} {LABELS[k][1]}}}' for k in out) + ')')

    def narr(m):
        k = key_for(m.group(1), m.group(2))
        return store(rf'\hyperlink{{{k}}}{{{LABELS[k][0]} ({LABELS[k][1]})}}') if k else m.group(0)

    text = re.sub(r'\[([^\[\]]*?[A-Za-z][^\[\]]*? \d{4}(?:; [^\[\]]+? \d{4})*)\]', paren, text)
    name = r"(?:van de )?[A-Z][\w'’\-]+"
    text = re.sub(rf"((?:{name})(?: et al\.| and {name})?) \[(\d{{4}})\]", narr, text)
    return text


INLINE = re.compile(r'(\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*])|https?://[^\s)]+[^\s).,;:])')


def latex_inline(s):
    out = []
    for part in INLINE.split(s):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            out.append(r'\textbf{' + latex_inline(part[2:-2].replace('*', '')) + '}')
        elif part.startswith('`') and part.endswith('`'):
            out.append(r'\texttt{' + esc(part[1:-1]).replace('/', r'/\allowbreak{}') + '}')
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            out.append(r'\emph{' + latex_inline(part[1:-1]) + '}')
        elif part.startswith('http'):
            out.append(rf'\url{{{part}}}')
        else:
            out.append(esc(part))
    s = ''.join(out).replace('\\textbackslash{}|', r'\textbar{}')
    for ch, rep in UNICODE.items():
        s = s.replace(ch, rep)
    return re.sub('\x00(\\d+)\x00', lambda m: CITES[int(m.group(1))], s)


# ---------------------------------------------------------------- tables
def table_tex(rows, cap, n):
    ncol = len(rows[0])
    avail = TEXTWIDTH_PT - 2 * ncol * 4 - 4           # minus \tabcolsep (set to 4pt below) on both sides of each column
    share, minw = [], []
    for i in range(ncol):
        cells = [re.sub(r'[*`]', '', r[i]) for r in rows]
        longest = max(len(w) for c in cells for w in (c.split() or ['']))
        minw.append(min(longest * CHAR_PT + 2, 0.4 * avail))   # longest word must fit; cap for long tokens
        share.append(max(sum(map(len, cells)) / len(cells), 1))
    fixed = [False] * ncol                            # columns held at their minimum (sticky)
    while True:                                       # raise narrow columns to their minimum, take it from the rest
        free = avail - sum(m for m, f in zip(minw, fixed) if f)
        tot = sum(s for s, f in zip(share, fixed) if not f)
        widths = [m if f else free * s / tot for m, s, f in zip(minw, share, fixed)]
        short = [not f and w < m for w, m, f in zip(widths, minw, fixed)]
        if not any(short):
            break
        fixed = [f or s for f, s in zip(fixed, short)]
    spec = ''.join(rf'>{{\raggedright\arraybackslash}}p{{{w:.1f}pt}}' for w in widths)
    head = ' & '.join(r'\textbf{' + latex_inline(c.strip()) + '}' for c in rows[0]) + r' \\'
    body = '\n'.join(' & '.join(latex_inline(c.strip()) for c in r) + r' \\' for r in rows[1:])
    # svjour3's \caption collapses to zero width inside longtable: set the caption by hand in the journal style
    return (f'\\par\\medskip\\noindent\\refstepcounter{{table}}\\label{{tab:{n}}}'
            f'\\parbox{{\\linewidth}}{{\\small\\textbf{{Table \\thetable}}\\enspace {latex_inline(cap)}}}\\par\\nopagebreak\n'
            f'{{\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}\n\\begin{{longtable}}{{{spec}}}\n'
            f'\\hline\\noalign{{\\smallskip}}\n{head}\n\\noalign{{\\smallskip}}\\hline\\noalign{{\\smallskip}}\n\\endfirsthead\n'
            f'\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Table {n} (continued)}}}}\\\\\n\\hline\\noalign{{\\smallskip}}\n{head}\n'
            f'\\noalign{{\\smallskip}}\\hline\\noalign{{\\smallskip}}\n\\endhead\n\\noalign{{\\smallskip}}\\hline\n\\endlastfoot\n'
            f'{body}\n\\end{{longtable}}\n}}\n')


# ---------------------------------------------------------------- document
def body_tex(lines, caps):
    out, i, current, used, ntab = [], 0, None, {}, 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith('## ') or line.startswith('### '):
            text = line.lstrip('# ').strip()
            current = B.section_key(text)
            title = re.sub(r'^[0-9.]+ ', '', text)
            cmd = 'section' if line.startswith('## ') else 'subsection'
            out.append(f'\n\\{cmd}{{{latex_inline(title)}}}\\label{{sec:{current}}}\n')
            i += 1
            continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            rows = [[c for c in re.split(r'(?<!\\)\|', r.strip())[1:-1]] for r in block]
            rows = [r for r in rows if not all(re.fullmatch(r'\s*:?-{3,}:?\s*', c) for c in r)]
            k = used.get(current, 0)
            ntab += 1
            out.append(table_tex(rows, caps[current][k], ntab))
            used[current] = k + 1
            continue
        if not line.strip() or line.startswith('<!--'):
            i += 1
            continue
        if re.match(r'^- ', line) or re.match(r'^\d+\. ', line):
            env = 'itemize' if line.startswith('- ') else 'enumerate'
            items = []
            while i < len(lines) and (re.match(r'^- ', lines[i]) if env == 'itemize' else re.match(r'^\d+\. ', lines[i])):
                items.append(re.sub(r'^(- |\d+\. )', '', lines[i].rstrip())); i += 1
            out.append(f'\\begin{{{env}}}\n' + '\n'.join(r'\item ' + latex_inline(t) for t in items) +
                       f'\n\\end{{{env}}}\n')
            continue
        out.append(latex_inline(line) + '\n')
        i += 1
    return '\n'.join(out), ntab


def front_matter(title, abstract):
    if ANON:
        authors = r'\author{Anonymous authors}'
        inst = r'\institute{Author details withheld for anonymous review}'
        running = r'\authorrunning{Anonymous}'
    else:
        authors = r'\author{' + ' \\and '.join(esc(a[0]).replace(' ', '~') for a in AUTHOR_INFO) + '}'
        blocks = []
        for k, (name, aff, orcid, email, _) in enumerate(AUTHOR_INFO):
            initials = '.'.join(p[0] for p in name.split()[:-1]) + '. ' + name.split()[-1]
            who = initials + (' (corresponding author)' if k == 0 else '')
            extra = (r' \\ ORCID: ' + orcid) if orcid else ''
            blocks.append(f'{esc(who)} \\at {esc(AFFIL[aff])} \\\\ \\email{{{email}}}{extra}')
        inst = '\\institute{' + '\n\\and '.join(blocks) + '}'
        running = r'\authorrunning{E.K. Amoako et al.}'
    return (f'\\title{{{esc(title)}}}\n\\titlerunning{{{RUNNING_TITLE}}}\n{authors}\n{running}\n{inst}\n'
            '\\date{Received: date / Accepted: date}\n\\maketitle\n\n'
            f'\\begin{{abstract}}\n{abstract}\n\\keywords{{' + ' \\and '.join(KEYWORDS) + '}\n\\end{abstract}\n')


DECL_ORDER = [   # Springer "Statements and Declarations": md lead-in -> heading used here
    ('Funding', 'Funding'), ('Conflict of interest', 'Competing interests'),
    ('ETHICS', 'Ethics approval and consent to participate'),
    ('CRediT authorship contribution statement', 'Author contributions'),
    ('Data and code availability', 'Data and code availability'), ('Use of generative AI', 'Use of generative AI'),
]
ETHICS = 'Not applicable. The study involves no human participants and no personal data.'


def declarations_tex(decl_md):
    found = dict(re.findall(r'^\*\*(.+?)\.\*\* (.+)$', decl_md, re.M))
    found['ETHICS'] = ETHICS
    out = ['\\section*{Statements and Declarations}\n']
    for key, head in DECL_ORDER:
        if key in found:
            out.append(f'\\noindent\\textbf{{{head}.}} {latex_inline(found[key])}\n')
    missing = set(found) - {k for k, _ in DECL_ORDER}
    assert not missing, f'declaration(s) not placed: {missing}'
    return '\n'.join(out)


def main():
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    V.update(B.EXTRA)
    md = B.MD.read_text(encoding='utf-8')
    if ANON:   # as build.py --anon
        md = re.sub(r'\*\*CRediT authorship contribution statement\.\*\*[^\n]*\n\n?', '', md)
        V['REPO'] = '[repository link removed for anonymous review]'
    text, missing = B.fill(md, V)
    caps = {k: [B.fill(c, V)[0] for c in cs] for k, cs in B.TABLE_CAPTIONS.items()}

    head, _, refs_md = text.partition('## References')
    body_md, _, decl_md = head.partition('## Declarations')
    ref_lines = [r.strip() for r in refs_md.strip().splitlines() if r.strip()]
    keys, items = build_refs(ref_lines)

    lines = body_md.splitlines()
    title = lines[lines.index('## Title') + 2].strip('*')
    a0 = lines.index('## Abstract') + 1
    a1 = next(i for i in range(a0, len(lines)) if lines[i].startswith('**CCS') or lines[i].startswith('**Keywords'))
    abstract = latex_inline(cite_sub(' '.join(l for l in lines[a0:a1] if l.strip()), keys))
    start = next(i for i, l in enumerate(lines) if l.startswith('## 1 '))
    body_lines = cite_sub('\n'.join(lines[start:]), keys).splitlines()
    caps = {k: [cite_sub(c, keys) for c in cs] for k, cs in caps.items()}
    body, ntab = body_tex(body_lines, caps)
    decl = declarations_tex(cite_sub(decl_md, keys))

    bib = [r'\section*{References}', r'{\small',
           r'\begin{list}{}{\setlength{\leftmargin}{1.5em}\setlength{\itemindent}{-1.5em}'
           r'\setlength{\itemsep}{2pt}\setlength{\parsep}{0pt}}']
    for key, label, year, t in items:
        bib.append(rf'\item \hypertarget{{{key}}}{{}}{tex_ref_text(t)}')
    bib += [r'\end{list}', '}']
    tex = rf"""%% Paper 7 - generated by Revision_v2/Paper7/build_springer.py from Manuscript.md and values.json; do not edit by hand.
\RequirePackage{{fix-cm}}
\documentclass[smallextended]{{svjour3}}
\smartqed
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{textcomp}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage[hyphens]{{url}}
\urlstyle{{same}}
\makeatletter\g@addto@macro\UrlBreaks{{\do0\do1\do2\do3\do4\do5\do6\do7\do8\do9\do a\do b\do c\do d\do e\do f}}\makeatother
\setlength{{\emergencystretch}}{{3em}}
\providecommand{{\hyperlink}}[2]{{#2}}
\providecommand{{\hypertarget}}[2]{{#2}}
\setlength{{\LTcapwidth}}{{\linewidth}}
\journalname{{{JOURNAL}}}
\begin{{document}}
{front_matter(title, abstract)}
{body}
{decl}
{chr(10).join(bib)}
\end{{document}}
"""
    OUT.mkdir(exist_ok=True)
    for f in ('svjour3.cls', 'svglov3.clo'):
        shutil.copy2(PKG / f, OUT / f)
    stem = 'Manuscript_Springer_anon' if ANON else 'Manuscript_Springer'
    (OUT / f'{stem}.tex').write_text(tex, encoding='utf-8')
    for _ in range(3):
        r = subprocess.run([str(PDFLATEX), '--disable-installer', '-interaction=nonstopmode', '-halt-on-error',
                            f'{stem}.tex'], cwd=OUT, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=180)   # never wait for MiKTeX's package installer
        if r.returncode:
            print(r.stdout[-3000:]); sys.exit('pdflatex failed')
    log = (OUT / f'{stem}.log').read_text(encoding='utf-8', errors='replace')
    pages = re.search(r'Output written on .*?\((\d+) pages', log)
    undef = sorted(set(re.findall(r"name\{([^}]+)\} has been referenced but does not exist", log)))
    overfull = re.findall(r'Overfull \\hbox \((\d+\.\d+)pt', log)
    left = sorted(set(re.findall(r'\[(?:[A-Z][^\[\]]*? )?\d{4}\]', body + abstract)))
    print(f'{stem}.pdf: {pages.group(1) if pages else "?"} pages, {ntab} tables, {len(items)} references')
    print('undefined citations:', undef or 'none')
    print('unconverted bracket citations:', left or 'none')
    print('overfull boxes > 2pt:', [o for o in overfull if float(o) > 2] or 'none')
    if missing:
        print(f'WARNING {len(missing)} unfilled placeholders:', ', '.join(missing))


if __name__ == '__main__':
    main()
