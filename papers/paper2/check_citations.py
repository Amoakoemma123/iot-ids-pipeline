"""Cross-check in-text author–year citations against the reference list of Manuscript.md."""
import re
import sys
from pathlib import Path

t = Path(__file__).with_name('Manuscript.md').read_text(encoding='utf-8')
body, refs = t.split('## References')
body = body.split('## Declarations')[0]
cited = set()
for a, y in re.findall(r"([A-Z][A-Za-zÀ-ž'\-]*)(?: et al\.)?(?: (?:and|&) [A-Z][A-Za-zÀ-ž\-]+)?,? \(?((?:19|20)\d\d)[a-z]?\b", body):
    if a not in ('Table', 'Figure', 'Section', 'In', 'TON', 'UNSW', 'CICIDS'):
        cited.add((a, y))
reflist = set()
for r in refs.strip().splitlines():
    if r.strip():
        reflist.add((r.split(',')[0].strip().split()[-1], re.search(r'\((\d{4})[a-z]?\)', r).group(1)))
a = sorted(c for c in cited if c not in reflist)
b = sorted(r for r in reflist if r not in cited)
print('cited but not in references:', a)
print('in references but not cited:', b)
sys.exit(len(a) + len(b))
