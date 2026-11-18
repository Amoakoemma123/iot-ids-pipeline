"""Numeric (IEEE) citation check for Manuscript.md: every reference [1]..[N] is cited, references are first cited in
numerical order, and nothing is cited that is not in the list. (The earlier author-year version crashed on the
numeric list and checked nothing; review_final_refs_code.md R1.)"""
import re
import sys
from pathlib import Path

md = (Path(__file__).resolve().parent / 'Manuscript.md').read_text(encoding='utf-8')
body, refs = md.split('## References')
listed = [int(n) for n in re.findall(r'^\[(\d+)\] ', refs, re.M)]
body = re.sub(r'\[(\d+)\]\s*[–-]\s*\[(\d+)\]', r'[\1–\2]', body)     # IEEE ranges "[4]–[6]"
order = []
for grp in re.findall(r'\[(\d+(?:\s*[–-]\s*\d+)?)\]', body):
    if '–' in grp or '-' in grp:
        a, b = (int(x) for x in re.split(r'\s*[–-]\s*', grp))
        nums = range(a, b + 1)
    else:
        nums = [int(grp)]
    for n in nums:
        if n not in order:
            order.append(n)
problems = []
if sorted(order) != listed:
    problems.append(f'cited {sorted(set(order) - set(listed))} not listed; listed {sorted(set(listed) - set(order))} not cited')
if order != sorted(order):
    problems.append(f'first citations out of order: {order}')
print('\n'.join(problems) if problems else f'ok: {len(listed)} references, all cited, first cited in order')
sys.exit(len(problems))
