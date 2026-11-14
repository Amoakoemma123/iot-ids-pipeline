"""Cross-check in-text author–year citations against the reference list.

Checks Manuscript_filled.md (the text as submitted) and Manuscript.md; the body includes Appendix A and B but not the
Declarations. Year suffixes (2024a, 2024b) are kept, and "Dinh et al. (2024a, 2024b)" counts as two citations.
Exit code = number of mismatches over both files.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOT_AUTHORS = {'Table', 'Figure', 'Section', 'In', 'TON', 'UNSW', 'CICIDS', 'CIC', 'CIC-IoT', 'IoT', 'Appendix', 'Edge', 'NeurIPS'}
AUTHOR = r"([A-Z][A-Za-zÀ-ž'\-]+)(?: et al\.)?(?: (?:and|&) [A-Z][A-Za-zÀ-ž\-]+)?"
YEAR = r"((?:19|20)\d\d[a-z]?)"


def check(path):
    t = path.read_text(encoding='utf-8')
    body, refs = t.split('## References')
    decl = body.find('## Declarations')
    app = body.find('## Appendix A')
    body = body[:decl] + (body[app:] if app > decl >= 0 else '')
    cited = set()
    for a, y, more in re.findall(AUTHOR + r",? \(?" + YEAR + r"((?:, (?:19|20)\d\d[a-z]?)*)\b", body):
        if a in NOT_AUTHORS:
            continue
        cited.add((a, y))
        for y2 in re.findall(r'(?:19|20)\d\d[a-z]?', more):    # "(2024a, 2024b)"
            cited.add((a, y2))
    reflist = set()
    for r in refs.strip().splitlines():
        if r.strip():
            m = re.search(r'\(((?:19|20)\d\d[a-z]?)\)', r)
            assert m, f'no year in reference: {r[:80]}'
            reflist.add((r.split(',')[0].strip().split()[-1], m.group(1)))
    a = sorted(c for c in cited if c not in reflist)
    b = sorted(r for r in reflist if r not in cited)
    print(f'{path.name}: {len(reflist)} references, {len(cited)} distinct citations')
    print('  cited but not in references:', a)
    print('  in references but not cited:', b)
    return len(a) + len(b)


n = sum(check(HERE / f) for f in ('Manuscript_filled.md', 'Manuscript.md') if (HERE / f).exists())
sys.exit(n)
